use crate::anthropic::convert::map_finish_reason;
use bytes::Bytes;
use futures_util::Stream;
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Stateful converter that consumes parsed OpenAI SSE chunk JSON values and
/// emits Anthropic Messages SSE events. Mirrors the textgen `StreamConverter`
/// behavior — defers `message_delta` + `message_stop` until the optional
/// usage-only chunk arrives so output_tokens is accurate when the upstream
/// emits one. Otherwise we flush on stream close.
pub struct StreamConverter {
    msg_id: String,
    model: String,
    block_index: u32,
    in_thinking: bool,
    in_text: bool,
    input_tokens: u64,
    output_tokens: u64,
    tool_calls_accum: BTreeMap<u32, AccumTool>,
    stop_reason: String,
    pending_finish: bool,
    started: bool,
}

struct AccumTool {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
pub struct AnthropicEvent {
    pub event: &'static str,
    pub data: String,
}

impl AnthropicEvent {
    pub fn to_sse_bytes(&self) -> Bytes {
        let mut buf = String::with_capacity(self.event.len() + self.data.len() + 16);
        buf.push_str("event: ");
        buf.push_str(self.event);
        buf.push('\n');
        buf.push_str("data: ");
        buf.push_str(&self.data);
        buf.push_str("\n\n");
        Bytes::from(buf)
    }
}

impl StreamConverter {
    pub fn new(model: String) -> Self {
        let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        Self {
            msg_id,
            model,
            block_index: 0,
            in_thinking: false,
            in_text: false,
            input_tokens: 0,
            output_tokens: 0,
            tool_calls_accum: BTreeMap::new(),
            stop_reason: "end_turn".to_string(),
            pending_finish: false,
            started: false,
        }
    }

    pub fn process_chunk(&mut self, chunk: &Value) -> Vec<AnthropicEvent> {
        let mut events = Vec::new();

        if let Some(usage) = chunk.get("usage").and_then(Value::as_object) {
            self.input_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.input_tokens);
            self.output_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.output_tokens);
        }

        let choices = chunk
            .get("choices")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        if choices.is_empty() {
            // Usage-only chunk — emit deferred finish events.
            if self.pending_finish {
                events.extend(self.finish());
            }
            return events;
        }

        // Emit message_start lazily on the first usable chunk so the upstream
        // model name is reflected.
        let first = &choices[0];
        let delta = first.get("delta").cloned().unwrap_or(Value::Null);
        let finish_reason = first
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(str::to_owned);

        if !self.started {
            self.started = true;
            let model_name = chunk
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or(self.model.as_str());
            self.model = model_name.to_string();
            events.push(AnthropicEvent {
                event: "message_start",
                data: json!({
                    "type": "message_start",
                    "message": {
                        "id": self.msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {
                            "input_tokens": self.input_tokens,
                            "output_tokens": 0
                        }
                    }
                })
                .to_string(),
            });
            events.push(AnthropicEvent {
                event: "ping",
                data: json!({"type": "ping"}).to_string(),
            });
        }

        // Reasoning / thinking content.
        if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str)
            && !reasoning.is_empty()
        {
            if !self.in_thinking {
                self.in_thinking = true;
                events.push(AnthropicEvent {
                    event: "content_block_start",
                    data: json!({
                        "type": "content_block_start",
                        "index": self.block_index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                    })
                    .to_string(),
                });
            }
            events.push(AnthropicEvent {
                event: "content_block_delta",
                data: json!({
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning}
                })
                .to_string(),
            });
        }

        // Text content.
        if let Some(text) = delta.get("content").and_then(Value::as_str)
            && !text.is_empty()
        {
            if self.in_thinking {
                events.push(AnthropicEvent {
                    event: "content_block_stop",
                    data: json!({
                        "type": "content_block_stop",
                        "index": self.block_index
                    })
                    .to_string(),
                });
                self.in_thinking = false;
                self.block_index += 1;
            }
            if !self.in_text {
                self.in_text = true;
                events.push(AnthropicEvent {
                    event: "content_block_start",
                    data: json!({
                        "type": "content_block_start",
                        "index": self.block_index,
                        "content_block": {"type": "text", "text": ""}
                    })
                    .to_string(),
                });
            }
            events.push(AnthropicEvent {
                event: "content_block_delta",
                data: json!({
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "text_delta", "text": text}
                })
                .to_string(),
            });
        }

        // Tool call deltas.
        if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
            for tc in tcs {
                let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
                let id = tc.get("id").and_then(Value::as_str).unwrap_or("");
                let func = tc.get("function").cloned().unwrap_or(Value::Null);
                let name = func
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let arg_chunk = func
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();

                if !id.is_empty() {
                    self.tool_calls_accum.insert(
                        idx,
                        AccumTool {
                            id: id.to_string(),
                            name,
                            arguments: arg_chunk,
                        },
                    );
                } else if let Some(existing) = self.tool_calls_accum.get_mut(&idx) {
                    existing.arguments.push_str(&arg_chunk);
                }
            }
        }

        // Final chunk: close any open content blocks, emit tool_use blocks,
        // defer message_delta / message_stop until the optional usage chunk.
        if let Some(reason) = finish_reason {
            self.stop_reason =
                map_finish_reason(Some(reason.as_str())).unwrap_or_else(|| "end_turn".to_string());

            if self.in_thinking {
                events.push(AnthropicEvent {
                    event: "content_block_stop",
                    data: json!({
                        "type": "content_block_stop",
                        "index": self.block_index
                    })
                    .to_string(),
                });
                self.in_thinking = false;
                self.block_index += 1;
            }
            if self.in_text {
                events.push(AnthropicEvent {
                    event: "content_block_stop",
                    data: json!({
                        "type": "content_block_stop",
                        "index": self.block_index
                    })
                    .to_string(),
                });
                self.in_text = false;
                self.block_index += 1;
            }

            let tool_calls = std::mem::take(&mut self.tool_calls_accum);
            for (_, tc) in tool_calls {
                let arguments_str = if tc.arguments.is_empty() {
                    "{}".to_string()
                } else {
                    tc.arguments
                };
                events.push(AnthropicEvent {
                    event: "content_block_start",
                    data: json!({
                        "type": "content_block_start",
                        "index": self.block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": {}
                        }
                    })
                    .to_string(),
                });
                events.push(AnthropicEvent {
                    event: "content_block_delta",
                    data: json!({
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": {"type": "input_json_delta", "partial_json": arguments_str}
                    })
                    .to_string(),
                });
                events.push(AnthropicEvent {
                    event: "content_block_stop",
                    data: json!({
                        "type": "content_block_stop",
                        "index": self.block_index
                    })
                    .to_string(),
                });
                self.block_index += 1;
            }

            self.pending_finish = true;
        }

        events
    }

    /// Final input/output token counts as observed from the upstream usage chunk.
    pub fn token_usage(&self) -> (u64, u64) {
        (self.input_tokens, self.output_tokens)
    }

    /// Emit deferred message_delta + message_stop. Idempotent.
    pub fn finish(&mut self) -> Vec<AnthropicEvent> {
        if !self.pending_finish && self.started {
            self.pending_finish = true;
        }
        if !self.pending_finish {
            return Vec::new();
        }
        self.pending_finish = false;
        vec![
            AnthropicEvent {
                event: "message_delta",
                data: json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": self.stop_reason, "stop_sequence": null},
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": self.output_tokens
                    }
                })
                .to_string(),
            },
            AnthropicEvent {
                event: "message_stop",
                data: json!({"type": "message_stop"}).to_string(),
            },
        ]
    }
}

/// Wrap an upstream OpenAI-style SSE byte stream into an Anthropic-style SSE
/// byte stream. Buffers across chunk boundaries to ensure each `data:` line is
/// parsed completely before being run through the converter.
pub fn anthropic_event_stream(
    upstream: reqwest::Response,
    model: String,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> {
    let mut bytes_stream = upstream.bytes_stream();
    let span = tracing::Span::current();
    async_stream::try_stream! {
        let mut converter = StreamConverter::new(model);
        let mut buffer: Vec<u8> = Vec::with_capacity(8192);
        let mut done_marker_seen = false;
        let mut chunks: u64 = 0;
        let mut events_emitted: u64 = 0;

        while let Some(item) = bytes_stream.next().await {
            let chunk = item.map_err(std::io::Error::other)?;
            chunks += 1;
            tracing::trace!(chunk_size = chunk.len(), "sse byte chunk");
            buffer.extend_from_slice(&chunk);

            // Drain complete events delimited by \n\n.
            while let Some(pos) = find_double_newline(&buffer) {
                let raw_event: Vec<u8> = buffer.drain(..pos + 2).collect();
                let event_str = String::from_utf8_lossy(&raw_event);
                for line in event_str.lines() {
                    let Some(payload) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let payload = payload.trim();
                    if payload.is_empty() {
                        continue;
                    }
                    if payload == "[DONE]" {
                        done_marker_seen = true;
                        for ev in converter.finish() {
                            events_emitted += 1;
                            yield ev.to_sse_bytes();
                        }
                        continue;
                    }
                    let Ok(value) = serde_json::from_str::<Value>(payload) else {
                        continue;
                    };
                    for ev in converter.process_chunk(&value) {
                        events_emitted += 1;
                        yield ev.to_sse_bytes();
                    }
                }
            }
        }

        // Flush any trailing partial event line (rare, but handle gracefully).
        if !buffer.is_empty()
            && let Some(payload) = std::str::from_utf8(&buffer).ok().and_then(|s| s.lines().find_map(|l| l.strip_prefix("data:")))
        {
            let payload = payload.trim();
            if !payload.is_empty() && payload != "[DONE]"
                && let Ok(value) = serde_json::from_str::<Value>(payload)
            {
                for ev in converter.process_chunk(&value) {
                    events_emitted += 1;
                    yield ev.to_sse_bytes();
                }
            }
        }

        if !done_marker_seen {
            for ev in converter.finish() {
                events_emitted += 1;
                yield ev.to_sse_bytes();
            }
        }

        let (input_tokens, output_tokens) = converter.token_usage();
        let _e = span.enter();
        tracing::debug!(
            chunks,
            events = events_emitted,
            input_tokens,
            output_tokens,
            "anthropic stream complete"
        );
    }
}

pub(crate) fn find_double_newline(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}
