//! Translate an OpenAI Responses API SSE stream into an Anthropic Messages
//! SSE stream. Sister module to `stream.rs` (which translates Chat Completions
//! SSE). Used for models whose GHCP catalog only exposes `/responses`.

use crate::anthropic::convert_responses::derive_stop_reason;
use crate::anthropic::stream::{AnthropicEvent, find_double_newline};
use bytes::Bytes;
use futures_util::Stream;
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
enum BlockKind {
    Text,
    ToolUse,
    Thinking,
}

#[derive(Debug)]
struct OpenBlock {
    block_index: u32,
    kind: BlockKind,
    /// `content_block_start` emitted? (For text we defer until `content_part.added`.)
    emitted_start: bool,
}

pub struct ResponsesStreamConverter {
    msg_id: String,
    model: String,
    input_tokens: u64,
    output_tokens: u64,
    stop_reason: String,
    started: bool,
    pending_finish: bool,
    finalized: bool,
    next_block_index: u32,
    /// Map Responses `output_index` → currently open Anthropic block.
    blocks: BTreeMap<u64, OpenBlock>,
}

impl ResponsesStreamConverter {
    pub fn new(model: String) -> Self {
        let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        Self {
            msg_id,
            model,
            input_tokens: 0,
            output_tokens: 0,
            stop_reason: "end_turn".to_string(),
            started: false,
            pending_finish: false,
            finalized: false,
            next_block_index: 0,
            blocks: BTreeMap::new(),
        }
    }

    pub fn token_usage(&self) -> (u64, u64) {
        (self.input_tokens, self.output_tokens)
    }

    pub fn process_event(&mut self, event: &Value) -> Vec<AnthropicEvent> {
        let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "response.created" => self.on_response_created(event),
            "response.in_progress" => Vec::new(),
            "response.output_item.added" => self.on_output_item_added(event),
            "response.content_part.added" => self.on_content_part_added(event),
            "response.output_text.delta" => self.on_text_delta(event),
            "response.output_text.done" => Vec::new(),
            "response.content_part.done" => Vec::new(),
            "response.function_call_arguments.delta" => self.on_function_args_delta(event),
            "response.function_call_arguments.done" => Vec::new(),
            "response.reasoning_summary_text.delta" => self.on_reasoning_delta(event),
            "response.reasoning_summary_text.done" => Vec::new(),
            "response.reasoning_summary_part.added" => Vec::new(),
            "response.reasoning_summary_part.done" => Vec::new(),
            "response.output_item.done" => self.on_output_item_done(event),
            "response.completed" | "response.failed" | "response.incomplete" => {
                self.on_response_completed(event)
            }
            "error" => self.on_error(event),
            _ => Vec::new(),
        }
    }

    /// Emit any deferred `message_delta` + `message_stop`. Idempotent.
    pub fn finish(&mut self) -> Vec<AnthropicEvent> {
        if self.finalized {
            return Vec::new();
        }
        if !self.pending_finish && self.started {
            self.pending_finish = true;
        }
        if !self.pending_finish {
            return Vec::new();
        }
        self.pending_finish = false;
        self.finalized = true;

        let mut events = Vec::new();
        // Close any block still open (defensive — well-formed Responses streams
        // always send output_item.done before response.completed).
        let leftover: Vec<u64> = self.blocks.keys().copied().collect();
        for key in leftover {
            if let Some(block) = self.blocks.remove(&key)
                && block.emitted_start
            {
                events.push(AnthropicEvent {
                    event: "content_block_stop",
                    data: json!({
                        "type": "content_block_stop",
                        "index": block.block_index,
                    })
                    .to_string(),
                });
            }
        }
        events.push(AnthropicEvent {
            event: "message_delta",
            data: json!({
                "type": "message_delta",
                "delta": {"stop_reason": self.stop_reason, "stop_sequence": null},
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                }
            })
            .to_string(),
        });
        events.push(AnthropicEvent {
            event: "message_stop",
            data: json!({"type": "message_stop"}).to_string(),
        });
        events
    }

    fn on_response_created(&mut self, event: &Value) -> Vec<AnthropicEvent> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        let resp = event.get("response");
        if let Some(model) = resp.and_then(|r| r.get("model")).and_then(Value::as_str) {
            self.model = model.to_string();
        }
        if let Some(id) = resp.and_then(|r| r.get("id")).and_then(Value::as_str) {
            self.msg_id = if id.starts_with("msg_") {
                id.to_string()
            } else {
                format!("msg_{}", id.trim_start_matches("resp_"))
            };
        }
        if let Some(usage) = resp.and_then(|r| r.get("usage")).and_then(Value::as_object) {
            self.input_tokens = usage
                .get("input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.input_tokens);
        }
        vec![
            AnthropicEvent {
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
                            "output_tokens": 0,
                        }
                    }
                })
                .to_string(),
            },
            AnthropicEvent {
                event: "ping",
                data: json!({"type": "ping"}).to_string(),
            },
        ]
    }

    fn on_output_item_added(&mut self, event: &Value) -> Vec<AnthropicEvent> {
        let Some(output_index) = event.get("output_index").and_then(Value::as_u64) else {
            return Vec::new();
        };
        let item = event.get("item");
        let item_type = item
            .and_then(|i| i.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("");

        let block_index = self.next_block_index;
        self.next_block_index += 1;

        match item_type {
            "message" => {
                self.blocks.insert(
                    output_index,
                    OpenBlock {
                        block_index,
                        kind: BlockKind::Text,
                        emitted_start: false,
                    },
                );
                Vec::new()
            }
            "function_call" => {
                let id = item
                    .and_then(|i| i.get("call_id").or_else(|| i.get("id")))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .and_then(|i| i.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                self.blocks.insert(
                    output_index,
                    OpenBlock {
                        block_index,
                        kind: BlockKind::ToolUse,
                        emitted_start: true,
                    },
                );
                vec![AnthropicEvent {
                    event: "content_block_start",
                    data: json!({
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": {},
                        }
                    })
                    .to_string(),
                }]
            }
            "reasoning" => {
                self.blocks.insert(
                    output_index,
                    OpenBlock {
                        block_index,
                        kind: BlockKind::Thinking,
                        emitted_start: true,
                    },
                );
                vec![AnthropicEvent {
                    event: "content_block_start",
                    data: json!({
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                        }
                    })
                    .to_string(),
                }]
            }
            _ => Vec::new(),
        }
    }

    fn on_content_part_added(&mut self, event: &Value) -> Vec<AnthropicEvent> {
        let Some(output_index) = event.get("output_index").and_then(Value::as_u64) else {
            return Vec::new();
        };
        let part_type = event
            .get("part")
            .and_then(|p| p.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if part_type != "output_text" {
            return Vec::new();
        }
        let Some(block) = self.blocks.get_mut(&output_index) else {
            return Vec::new();
        };
        if block.emitted_start {
            return Vec::new();
        }
        block.emitted_start = true;
        vec![AnthropicEvent {
            event: "content_block_start",
            data: json!({
                "type": "content_block_start",
                "index": block.block_index,
                "content_block": {"type": "text", "text": ""},
            })
            .to_string(),
        }]
    }

    fn on_text_delta(&mut self, event: &Value) -> Vec<AnthropicEvent> {
        let Some(output_index) = event.get("output_index").and_then(Value::as_u64) else {
            return Vec::new();
        };
        let delta = event.get("delta").and_then(Value::as_str).unwrap_or("");
        if delta.is_empty() {
            return Vec::new();
        }
        let Some(block) = self.blocks.get(&output_index) else {
            return Vec::new();
        };
        if !matches!(block.kind, BlockKind::Text) {
            return Vec::new();
        }
        let block_index = block.block_index;
        // Defensive: some upstreams skip content_part.added.
        let mut events = Vec::new();
        if !block.emitted_start
            && let Some(block) = self.blocks.get_mut(&output_index)
        {
            block.emitted_start = true;
            events.push(AnthropicEvent {
                event: "content_block_start",
                data: json!({
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
                .to_string(),
            });
        }
        events.push(AnthropicEvent {
            event: "content_block_delta",
            data: json!({
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": delta},
            })
            .to_string(),
        });
        events
    }

    fn on_function_args_delta(&mut self, event: &Value) -> Vec<AnthropicEvent> {
        let Some(output_index) = event.get("output_index").and_then(Value::as_u64) else {
            return Vec::new();
        };
        let delta = event.get("delta").and_then(Value::as_str).unwrap_or("");
        if delta.is_empty() {
            return Vec::new();
        }
        let Some(block) = self.blocks.get(&output_index) else {
            return Vec::new();
        };
        if !matches!(block.kind, BlockKind::ToolUse) {
            return Vec::new();
        }
        vec![AnthropicEvent {
            event: "content_block_delta",
            data: json!({
                "type": "content_block_delta",
                "index": block.block_index,
                "delta": {"type": "input_json_delta", "partial_json": delta},
            })
            .to_string(),
        }]
    }

    fn on_reasoning_delta(&mut self, event: &Value) -> Vec<AnthropicEvent> {
        let Some(output_index) = event.get("output_index").and_then(Value::as_u64) else {
            return Vec::new();
        };
        let delta = event.get("delta").and_then(Value::as_str).unwrap_or("");
        if delta.is_empty() {
            return Vec::new();
        }
        let Some(block) = self.blocks.get(&output_index) else {
            return Vec::new();
        };
        if !matches!(block.kind, BlockKind::Thinking) {
            return Vec::new();
        }
        vec![AnthropicEvent {
            event: "content_block_delta",
            data: json!({
                "type": "content_block_delta",
                "index": block.block_index,
                "delta": {"type": "thinking_delta", "thinking": delta},
            })
            .to_string(),
        }]
    }

    fn on_output_item_done(&mut self, event: &Value) -> Vec<AnthropicEvent> {
        let Some(output_index) = event.get("output_index").and_then(Value::as_u64) else {
            return Vec::new();
        };
        let Some(block) = self.blocks.remove(&output_index) else {
            return Vec::new();
        };
        if !block.emitted_start {
            return Vec::new();
        }
        vec![AnthropicEvent {
            event: "content_block_stop",
            data: json!({
                "type": "content_block_stop",
                "index": block.block_index,
            })
            .to_string(),
        }]
    }

    fn on_response_completed(&mut self, event: &Value) -> Vec<AnthropicEvent> {
        let resp = event.get("response");
        if let Some(usage) = resp.and_then(|r| r.get("usage")).and_then(Value::as_object) {
            self.input_tokens = usage
                .get("input_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.input_tokens);
            self.output_tokens = usage
                .get("output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.output_tokens);
        }

        let saw_function_call = resp
            .and_then(|r| r.get("output"))
            .and_then(Value::as_array)
            .is_some_and(|items| {
                items
                    .iter()
                    .any(|i| i.get("type").and_then(Value::as_str) == Some("function_call"))
            });

        self.stop_reason = resp
            .and_then(|r| derive_stop_reason(r, saw_function_call))
            .unwrap_or_else(|| "end_turn".to_string());

        self.pending_finish = true;
        self.finish()
    }

    fn on_error(&mut self, event: &Value) -> Vec<AnthropicEvent> {
        let message = event
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| event.get("error").and_then(|e| e.get("message")?.as_str()))
            .unwrap_or("upstream error")
            .to_string();
        vec![AnthropicEvent {
            event: "error",
            data: json!({
                "type": "error",
                "error": {"type": "api_error", "message": message},
            })
            .to_string(),
        }]
    }
}

/// Wrap an upstream Responses-API SSE byte stream into an Anthropic-style SSE
/// byte stream. Buffers across chunk boundaries.
pub fn anthropic_event_stream_responses(
    upstream: reqwest::Response,
    model: String,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> {
    let mut bytes_stream = upstream.bytes_stream();
    let span = tracing::Span::current();
    async_stream::try_stream! {
        let mut converter = ResponsesStreamConverter::new(model);
        let mut buffer: Vec<u8> = Vec::with_capacity(8192);
        let mut done_marker_seen = false;
        let mut chunks: u64 = 0;
        let mut events_emitted: u64 = 0;

        while let Some(item) = bytes_stream.next().await {
            let chunk = item.map_err(std::io::Error::other)?;
            chunks += 1;
            tracing::trace!(chunk_size = chunk.len(), "responses sse byte chunk");
            buffer.extend_from_slice(&chunk);

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
                    for ev in converter.process_event(&value) {
                        events_emitted += 1;
                        yield ev.to_sse_bytes();
                    }
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
            "anthropic responses stream complete"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Drive a sequence of Responses events through the converter and collect
    /// (event_name, parsed_data) pairs for assertion.
    fn drive(events: &[Value]) -> Vec<(String, Value)> {
        let mut converter = ResponsesStreamConverter::new("gpt-5.5".to_string());
        let mut out = Vec::new();
        for event in events {
            for ev in converter.process_event(event) {
                let parsed: Value = serde_json::from_str(&ev.data).unwrap();
                out.push((ev.event.to_string(), parsed));
            }
        }
        for ev in converter.finish() {
            let parsed: Value = serde_json::from_str(&ev.data).unwrap();
            out.push((ev.event.to_string(), parsed));
        }
        out
    }

    fn names(events: &[(String, Value)]) -> Vec<&str> {
        events.iter().map(|(n, _)| n.as_str()).collect()
    }

    #[test]
    fn text_only_stream() {
        let events = vec![
            json!({"type": "response.created", "response": {
                "id": "resp_1", "model": "gpt-5.5", "status": "in_progress"
            }}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "message", "role": "assistant", "id": "msg_1"
            }}),
            json!({"type": "response.content_part.added", "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": ""}}),
            json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
                "delta": "Hello"}),
            json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
                "delta": " world"}),
            json!({"type": "response.output_text.done", "output_index": 0, "content_index": 0,
                "text": "Hello world"}),
            json!({"type": "response.content_part.done", "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": "Hello world"}}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": {
                "type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Hello world"}
                ]
            }}),
            json!({"type": "response.completed", "response": {
                "id": "resp_1", "model": "gpt-5.5", "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Hello world"}
                ]}],
                "usage": {"input_tokens": 5, "output_tokens": 2}
            }}),
        ];
        let out = drive(&events);
        assert_eq!(
            names(&out),
            vec![
                "message_start",
                "ping",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let (_, msg_start) = &out[0];
        assert_eq!(msg_start["message"]["model"], "gpt-5.5");
        let (_, delta1) = &out[3];
        assert_eq!(delta1["delta"]["type"], "text_delta");
        assert_eq!(delta1["delta"]["text"], "Hello");
        let (_, msg_delta) = out.iter().find(|(n, _)| n == "message_delta").unwrap();
        assert_eq!(msg_delta["delta"]["stop_reason"], "end_turn");
        assert_eq!(msg_delta["usage"]["input_tokens"], 5);
        assert_eq!(msg_delta["usage"]["output_tokens"], 2);
    }

    #[test]
    fn tool_call_stream_emits_input_json_delta_and_tool_use_stop_reason() {
        let events = vec![
            json!({"type": "response.created", "response": {
                "id": "resp_2", "model": "gpt-5.5", "status": "in_progress"
            }}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "function_call", "id": "fc_1", "call_id": "call_x",
                "name": "search", "arguments": ""
            }}),
            json!({"type": "response.function_call_arguments.delta",
                "output_index": 0, "delta": "{\"q\":"}),
            json!({"type": "response.function_call_arguments.delta",
                "output_index": 0, "delta": "\"rust\"}"}),
            json!({"type": "response.function_call_arguments.done",
                "output_index": 0, "arguments": "{\"q\":\"rust\"}"}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": {
                "type": "function_call", "call_id": "call_x", "name": "search",
                "arguments": "{\"q\":\"rust\"}"
            }}),
            json!({"type": "response.completed", "response": {
                "id": "resp_2", "model": "gpt-5.5", "status": "completed",
                "output": [{"type": "function_call", "call_id": "call_x",
                            "name": "search", "arguments": "{\"q\":\"rust\"}"}],
                "usage": {"input_tokens": 3, "output_tokens": 4}
            }}),
        ];
        let out = drive(&events);
        let (_, start) = out
            .iter()
            .find(|(n, _)| n == "content_block_start")
            .unwrap();
        assert_eq!(start["content_block"]["type"], "tool_use");
        assert_eq!(start["content_block"]["id"], "call_x");
        assert_eq!(start["content_block"]["name"], "search");

        let json_deltas: Vec<&Value> = out
            .iter()
            .filter(|(n, _)| n == "content_block_delta")
            .map(|(_, v)| v)
            .collect();
        assert_eq!(json_deltas.len(), 2);
        for d in &json_deltas {
            assert_eq!(d["delta"]["type"], "input_json_delta");
        }
        let combined: String = json_deltas
            .iter()
            .map(|d| d["delta"]["partial_json"].as_str().unwrap())
            .collect();
        assert_eq!(combined, "{\"q\":\"rust\"}");

        let (_, msg_delta) = out.iter().find(|(n, _)| n == "message_delta").unwrap();
        assert_eq!(msg_delta["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn reasoning_maps_to_thinking_block() {
        let events = vec![
            json!({"type": "response.created", "response": {
                "id": "resp_3", "model": "gpt-5.5", "status": "in_progress"
            }}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "reasoning", "id": "rs_1"
            }}),
            json!({"type": "response.reasoning_summary_text.delta",
                "output_index": 0, "summary_index": 0, "delta": "thinking..."}),
            json!({"type": "response.output_item.done", "output_index": 0, "item": {
                "type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]
            }}),
            json!({"type": "response.completed", "response": {
                "id": "resp_3", "model": "gpt-5.5", "status": "completed",
                "output": [{"type": "reasoning", "summary": [
                    {"type": "summary_text", "text": "thinking..."}
                ]}],
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }}),
        ];
        let out = drive(&events);
        let (_, start) = out
            .iter()
            .find(|(n, _)| n == "content_block_start")
            .unwrap();
        assert_eq!(start["content_block"]["type"], "thinking");
        let (_, delta) = out
            .iter()
            .find(|(n, _)| n == "content_block_delta")
            .unwrap();
        assert_eq!(delta["delta"]["type"], "thinking_delta");
        assert_eq!(delta["delta"]["thinking"], "thinking...");
    }

    #[test]
    fn incomplete_max_tokens_maps_stop_reason() {
        let events = vec![
            json!({"type": "response.created", "response": {
                "id": "resp_4", "model": "gpt-5.5", "status": "in_progress"
            }}),
            json!({"type": "response.output_item.added", "output_index": 0, "item": {
                "type": "message", "role": "assistant"
            }}),
            json!({"type": "response.content_part.added", "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": ""}}),
            json!({"type": "response.output_text.delta",
                "output_index": 0, "content_index": 0, "delta": "abc"}),
            json!({"type": "response.incomplete", "response": {
                "id": "resp_4", "model": "gpt-5.5", "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "abc"}
                ]}],
                "usage": {"input_tokens": 2, "output_tokens": 100}
            }}),
        ];
        let out = drive(&events);
        let (_, msg_delta) = out.iter().find(|(n, _)| n == "message_delta").unwrap();
        assert_eq!(msg_delta["delta"]["stop_reason"], "max_tokens");
    }

    #[test]
    fn error_event_emits_error_event() {
        let events = vec![
            json!({"type": "response.created", "response": {
                "id": "resp_5", "model": "gpt-5.5", "status": "in_progress"
            }}),
            json!({"type": "error", "message": "boom"}),
        ];
        let out = drive(&events);
        let names = names(&out);
        assert!(names.contains(&"error"));
        let (_, err) = out.iter().find(|(n, _)| n == "error").unwrap();
        assert_eq!(err["error"]["message"], "boom");
    }
}
