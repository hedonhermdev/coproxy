//! Translate between Anthropic Messages and OpenAI Responses payloads.
//!
//! Used for models whose GHCP catalog entry exposes only `/responses`
//! (e.g. `gpt-5.5`). For models that also expose `/chat/completions` we keep
//! the existing `convert.rs` path because it is the more battle-tested route.

use crate::anthropic::convert::{flatten_system, stringify};
use crate::anthropic::types::{ContentBlock, CreateMessageRequest, MessageResponse, Usage};
use serde_json::{Map, Value, json};

/// Convert an Anthropic Messages request into a Responses API JSON body.
///
/// The Responses API uses a single mixed `input` array carrying both
/// conversational messages and tool-call / tool-result items, and a separate
/// top-level `instructions` field for the system prompt.
pub fn anthropic_to_responses(req: &CreateMessageRequest) -> Value {
    let mut body = Map::new();

    if let Some(model) = req.model.as_ref() {
        body.insert("model".to_string(), Value::String(model.clone()));
    }

    if let Some(text) = flatten_system(req.system.as_ref())
        && !text.is_empty()
    {
        body.insert("instructions".to_string(), Value::String(text));
    }

    let mut input: Vec<Value> = Vec::new();
    for msg in &req.messages {
        push_message_items(msg, &mut input);
    }
    body.insert("input".to_string(), Value::Array(input));

    if let Some(tools) = req.tools.as_ref() {
        let translated: Vec<Value> = tools.iter().filter_map(convert_tool_definition).collect();
        if !translated.is_empty() {
            body.insert("tools".to_string(), Value::Array(translated));
        }
    }

    if let Some(choice) = req.tool_choice.as_ref()
        && let Some(translated) = convert_tool_choice(choice)
    {
        body.insert("tool_choice".to_string(), translated);
    }

    if let Some(max) = req.max_tokens {
        // Anthropic accepts max_tokens >= 1, but OpenAI /responses requires
        // max_output_tokens >= 16. Clamp so probe-style requests
        // (e.g. Claude Code's model-switch sanity check) survive the bridge.
        body.insert("max_output_tokens".to_string(), json!(max.max(16)));
    }
    if let Some(temp) = req.temperature {
        body.insert("temperature".to_string(), json!(temp));
    }
    if let Some(top_p) = req.top_p {
        body.insert("top_p".to_string(), json!(top_p));
    }
    if let Some(stream) = req.stream {
        body.insert("stream".to_string(), Value::Bool(stream));
    }

    // Anthropic clients always send the full conversation; never opt into
    // server-side state on the Responses side.
    body.insert("store".to_string(), Value::Bool(false));

    Value::Object(body)
}

/// Convert a finished Responses API JSON object into an Anthropic
/// `MessageResponse`. Used for non-streaming requests.
pub fn responses_to_anthropic(resp: &Value, fallback_model: &str) -> MessageResponse {
    let model = resp
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(fallback_model)
        .to_string();

    let id = resp
        .get("id")
        .and_then(Value::as_str)
        .map(|s| {
            if s.starts_with("msg_") {
                s.to_string()
            } else {
                format!("msg_{}", s.trim_start_matches("resp_"))
            }
        })
        .unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4().simple()));

    let mut content: Vec<ContentBlock> = Vec::new();

    if let Some(items) = resp.get("output").and_then(Value::as_array) {
        for item in items {
            match item.get("type").and_then(Value::as_str) {
                Some("message") => {
                    if let Some(parts) = item.get("content").and_then(Value::as_array) {
                        for part in parts {
                            if part.get("type").and_then(Value::as_str) == Some("output_text")
                                && let Some(text) =
                                    part.get("text").and_then(Value::as_str).map(str::to_owned)
                                && !text.is_empty()
                            {
                                content.push(ContentBlock::Text { text });
                            }
                        }
                    }
                }
                Some("function_call") => {
                    let id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let arguments = item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    let input = serde_json::from_str::<Value>(arguments)
                        .unwrap_or_else(|_| Value::Object(Default::default()));
                    content.push(ContentBlock::ToolUse { id, name, input });
                }
                Some("reasoning") => {
                    if let Some(summaries) = item.get("summary").and_then(Value::as_array) {
                        for part in summaries {
                            if part.get("type").and_then(Value::as_str) == Some("summary_text")
                                && let Some(text) =
                                    part.get("text").and_then(Value::as_str).map(str::to_owned)
                                && !text.is_empty()
                            {
                                content.push(ContentBlock::Thinking {
                                    thinking: text,
                                    signature: String::new(),
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let saw_function_call = content
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolUse { .. }));
    let stop_reason = derive_stop_reason(resp, saw_function_call);

    let usage = resp.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| u.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    MessageResponse {
        id,
        kind: "message",
        role: "assistant",
        content,
        model,
        stop_reason,
        stop_sequence: None,
        usage: Usage {
            input_tokens,
            output_tokens,
        },
    }
}

/// Derive an Anthropic `stop_reason` from a Responses object.
pub(crate) fn derive_stop_reason(resp: &Value, saw_function_call: bool) -> Option<String> {
    if saw_function_call {
        return Some("tool_use".to_string());
    }
    let status = resp.get("status").and_then(Value::as_str).unwrap_or("");
    match status {
        "completed" => Some("end_turn".to_string()),
        "incomplete" => {
            let reason = resp
                .get("incomplete_details")
                .and_then(|d| d.get("reason"))
                .and_then(Value::as_str)
                .unwrap_or("");
            match reason {
                "max_output_tokens" => Some("max_tokens".to_string()),
                "stop_sequence" => Some("stop_sequence".to_string()),
                _ => Some("end_turn".to_string()),
            }
        }
        _ => None,
    }
}

fn push_message_items(msg: &Value, out: &mut Vec<Value>) {
    let Some(obj) = msg.as_object() else {
        return;
    };
    let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
    let Some(content) = obj.get("content") else {
        return;
    };

    match content {
        Value::String(s) => {
            let part_type = if role == "assistant" {
                "output_text"
            } else {
                "input_text"
            };
            out.push(json!({
                "type": "message",
                "role": role,
                "content": [{"type": part_type, "text": s}],
            }));
        }
        Value::Array(blocks) => match role {
            "assistant" => push_assistant_blocks(blocks, out),
            "user" => push_user_blocks(blocks, out),
            _ => {
                out.push(json!({
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": stringify(content)}],
                }));
            }
        },
        Value::Null => {
            out.push(json!({
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": ""}],
            }));
        }
        other => {
            out.push(json!({
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": stringify(other)}],
            }));
        }
    }
}

fn push_assistant_blocks(blocks: &[Value], out: &mut Vec<Value>) {
    let mut text_parts: Vec<Value> = Vec::new();
    let mut function_calls: Vec<Value> = Vec::new();

    for block in blocks {
        let Some(obj) = block.as_object() else {
            continue;
        };
        let Some(kind) = obj.get("type").and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "text" => {
                if let Some(text) = obj.get("text").and_then(Value::as_str) {
                    text_parts.push(json!({"type": "output_text", "text": text}));
                }
            }
            "tool_use" => {
                let id = obj.get("id").and_then(Value::as_str).unwrap_or("");
                let name = obj.get("name").and_then(Value::as_str).unwrap_or("");
                let input = obj
                    .get("input")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));
                let arguments = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
                function_calls.push(json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    "arguments": arguments,
                }));
            }
            // Anthropic thinking blocks have no faithful mapping to Responses
            // input items — drop on the way out, same as convert.rs does for
            // chat/completions.
            "thinking" => {}
            _ => {}
        }
    }

    if !text_parts.is_empty() {
        out.push(json!({
            "type": "message",
            "role": "assistant",
            "content": text_parts,
        }));
    }
    out.extend(function_calls);
}

fn push_user_blocks(blocks: &[Value], out: &mut Vec<Value>) {
    let mut regular: Vec<Value> = Vec::new();

    let flush_regular = |regular: &mut Vec<Value>, sink: &mut Vec<Value>| {
        if regular.is_empty() {
            return;
        }
        sink.push(json!({
            "type": "message",
            "role": "user",
            "content": std::mem::take(regular),
        }));
    };

    for block in blocks {
        let Some(obj) = block.as_object() else {
            continue;
        };
        let Some(kind) = obj.get("type").and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "tool_result" => {
                flush_regular(&mut regular, out);
                let call_id = obj
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let raw = obj.get("content").cloned().unwrap_or(Value::Null);
                let output = match &raw {
                    Value::String(s) => s.clone(),
                    Value::Array(items) => items
                        .iter()
                        .filter_map(|item| {
                            let o = item.as_object()?;
                            if o.get("type").and_then(Value::as_str) == Some("text") {
                                o.get("text").and_then(Value::as_str).map(str::to_owned)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Value::Null => String::new(),
                    other => stringify(other),
                };
                out.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }));
            }
            "text" => {
                if let Some(text) = obj.get("text").and_then(Value::as_str) {
                    regular.push(json!({"type": "input_text", "text": text}));
                }
            }
            "image" => {
                if let Some(source) = obj.get("source").and_then(Value::as_object) {
                    let kind_src = source.get("type").and_then(Value::as_str).unwrap_or("");
                    if kind_src == "base64" {
                        let media_type = source
                            .get("media_type")
                            .and_then(Value::as_str)
                            .unwrap_or("image/png");
                        let data = source.get("data").and_then(Value::as_str).unwrap_or("");
                        let url = format!("data:{media_type};base64,{data}");
                        regular.push(json!({"type": "input_image", "image_url": url}));
                    } else if kind_src == "url"
                        && let Some(url) = source.get("url").and_then(Value::as_str)
                    {
                        regular.push(json!({"type": "input_image", "image_url": url}));
                    }
                }
            }
            "thinking" => {}
            _ => {}
        }
    }

    flush_regular(&mut regular, out);
}

fn convert_tool_definition(tool: &Value) -> Option<Value> {
    let obj = tool.as_object()?;
    let name = obj.get("name").and_then(Value::as_str)?.to_string();
    let description = obj
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let parameters = obj
        .get("input_schema")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    let mut out = Map::new();
    out.insert("type".to_string(), Value::String("function".to_string()));
    out.insert("name".to_string(), Value::String(name));
    if let Some(desc) = description {
        out.insert("description".to_string(), Value::String(desc));
    }
    out.insert("parameters".to_string(), parameters);
    Some(Value::Object(out))
}

fn convert_tool_choice(choice: &Value) -> Option<Value> {
    let obj = choice.as_object()?;
    let kind = obj.get("type").and_then(Value::as_str)?;
    match kind {
        "auto" => Some(Value::String("auto".to_string())),
        "any" => Some(Value::String("required".to_string())),
        "none" => Some(Value::String("none".to_string())),
        "tool" => {
            let name = obj.get("name").and_then(Value::as_str)?.to_string();
            Some(json!({"type": "function", "name": name}))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::types::ContentBlock;
    use serde_json::json;

    fn req_from(value: Value) -> CreateMessageRequest {
        serde_json::from_value(value).expect("valid CreateMessageRequest fixture")
    }

    #[test]
    fn translates_text_only_conversation() {
        let req = req_from(json!({
            "model": "gpt-5.5",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "how are you?"},
            ],
        }));
        let body = anthropic_to_responses(&req);
        assert_eq!(body["model"], "gpt-5.5");
        assert_eq!(body["max_output_tokens"], 256);
        assert_eq!(body["store"], false);
        assert!(body.get("instructions").is_none());

        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "hi");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[1]["content"][0]["text"], "hello");
    }

    #[test]
    fn clamps_tiny_max_tokens_to_responses_minimum() {
        let req = req_from(json!({
            "model": "gpt-5.5",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }));
        let body = anthropic_to_responses(&req);
        assert_eq!(body["max_output_tokens"], 16);
    }

    #[test]
    fn flattens_system_into_instructions() {
        let req = req_from(json!({
            "model": "gpt-5.5",
            "system": [
                {"type": "text", "text": "be terse"},
                {"type": "text", "text": "stay on topic"},
            ],
            "messages": [{"role": "user", "content": "go"}],
        }));
        let body = anthropic_to_responses(&req);
        assert_eq!(body["instructions"], "be terse\nstay on topic");
    }

    #[test]
    fn maps_tool_use_and_tool_result() {
        let req = req_from(json!({
            "model": "gpt-5.5",
            "messages": [
                {"role": "user", "content": "search the docs"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "looking..."},
                    {"type": "tool_use", "id": "call_42", "name": "search", "input": {"q": "rust"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_42", "content": "found it"},
                ]},
            ],
        }));
        let body = anthropic_to_responses(&req);
        let input = body["input"].as_array().unwrap();
        // user msg, assistant msg (text), function_call item, function_call_output item.
        assert_eq!(input.len(), 4);
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["text"], "looking...");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_42");
        assert_eq!(input[2]["name"], "search");
        // arguments must be a JSON string, not an object.
        let args_str = input[2]["arguments"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(parsed, json!({"q": "rust"}));
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "call_42");
        assert_eq!(input[3]["output"], "found it");
    }

    #[test]
    fn maps_image_blocks() {
        let req = req_from(json!({
            "model": "gpt-5.5",
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}},
                    {"type": "text", "text": "describe"},
                ]},
            ],
        }));
        let body = anthropic_to_responses(&req);
        let parts = body["input"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "input_image");
        assert!(
            parts[0]["image_url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,AAA")
        );
        assert_eq!(parts[1]["type"], "input_text");
    }

    #[test]
    fn maps_tool_choice_variants() {
        let auto = req_from(json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "auto"},
        }));
        assert_eq!(anthropic_to_responses(&auto)["tool_choice"], "auto");

        let any = req_from(json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "any"},
        }));
        assert_eq!(anthropic_to_responses(&any)["tool_choice"], "required");

        let named = req_from(json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "tool", "name": "search"},
        }));
        let body = anthropic_to_responses(&named);
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["name"], "search");
    }

    #[test]
    fn translates_tool_definition() {
        let req = req_from(json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{
                "name": "search",
                "description": "search the web",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }],
        }));
        let body = anthropic_to_responses(&req);
        let tool = &body["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "search");
        assert_eq!(tool["description"], "search the web");
        assert_eq!(tool["parameters"]["type"], "object");
    }

    #[test]
    fn responses_to_anthropic_extracts_text_and_tool_use() {
        let resp = json!({
            "id": "resp_abc",
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "hello there"}
                ]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_x",
                 "name": "search", "arguments": "{\"q\":\"rust\"}"},
            ],
            "usage": {"input_tokens": 12, "output_tokens": 7},
        });
        let msg = responses_to_anthropic(&resp, "fallback");
        assert_eq!(msg.id, "msg_abc");
        assert_eq!(msg.model, "gpt-5.5");
        assert_eq!(msg.usage.input_tokens, 12);
        assert_eq!(msg.usage.output_tokens, 7);
        assert_eq!(msg.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(msg.content.len(), 2);
        match &msg.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "hello there"),
            _ => panic!("expected text block"),
        }
        match &msg.content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_x");
                assert_eq!(name, "search");
                assert_eq!(input["q"], "rust");
            }
            _ => panic!("expected tool_use block"),
        }
    }

    #[test]
    fn responses_to_anthropic_max_tokens_status() {
        let resp = json!({
            "id": "resp_abc",
            "model": "gpt-5.5",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "..."}
            ]}],
            "usage": {"input_tokens": 5, "output_tokens": 1},
        });
        let msg = responses_to_anthropic(&resp, "gpt-5.5");
        assert_eq!(msg.stop_reason.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn responses_to_anthropic_reasoning_to_thinking() {
        let resp = json!({
            "id": "resp_abc",
            "model": "gpt-5.5",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [
                    {"type": "summary_text", "text": "let me think"}
                ]},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "answer"}
                ]},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        });
        let msg = responses_to_anthropic(&resp, "gpt-5.5");
        match &msg.content[0] {
            ContentBlock::Thinking { thinking, .. } => assert_eq!(thinking, "let me think"),
            _ => panic!("expected thinking block first"),
        }
    }
}
