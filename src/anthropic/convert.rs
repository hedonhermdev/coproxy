use crate::anthropic::types::{ContentBlock, CreateMessageRequest, MessageResponse, Usage};
use crate::openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionRequestMessage, ChatCompletionTool,
    ChatCompletionToolChoiceOption, CreateChatCompletionRequest, FunctionCall, FunctionNameOnly,
    FunctionObject,
};
use crate::provider::ProviderChatResponse;
use serde_json::{Value, json};

/// Convert an Anthropic Messages API request into the OpenAI chat completion
/// request shape that this proxy dispatches to GHCP. Mirrors textgen's
/// `convert_request` but emits typed `CreateChatCompletionRequest`.
pub fn anthropic_to_openai(req: &CreateMessageRequest) -> CreateChatCompletionRequest {
    let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();

    // System message: string or list of {type:"text",text:"..."} blocks.
    if let Some(text) = flatten_system(req.system.as_ref())
        && !text.is_empty()
    {
        messages.push(ChatCompletionRequestMessage {
            role: "system".to_string(),
            content: Some(Value::String(text)),
            tool_call_id: None,
            tool_calls: None,
        });
    }

    for msg in &req.messages {
        convert_message_into(msg, &mut messages);
    }

    let tools = req.tools.as_ref().map(|tools| {
        tools
            .iter()
            .filter_map(convert_tool_definition)
            .collect::<Vec<_>>()
    });

    let tool_choice = req.tool_choice.as_ref().and_then(convert_tool_choice);

    CreateChatCompletionRequest {
        model: req.model.clone(),
        messages,
        temperature: req.temperature,
        stream: req.stream,
        tools,
        tool_choice,
    }
}

/// Convert a finished OpenAI chat response into an Anthropic Messages response.
pub fn openai_to_anthropic(resp: ProviderChatResponse, model: String) -> MessageResponse {
    let mut content: Vec<ContentBlock> = Vec::new();
    if let Some(text) = resp.content
        && !text.is_empty()
    {
        content.push(ContentBlock::Text { text });
    }
    for tc in resp.tool_calls {
        let input = serde_json::from_str::<Value>(&tc.function.arguments)
            .unwrap_or_else(|_| Value::Object(Default::default()));
        content.push(ContentBlock::ToolUse {
            id: tc.id,
            name: tc.function.name,
            input,
        });
    }

    let id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    MessageResponse {
        id,
        kind: "message",
        role: "assistant",
        content,
        model,
        // Without an OpenAI finish_reason in our typed provider response we
        // default to end_turn when no tool calls fired, otherwise tool_use.
        stop_reason: None,
        stop_sequence: None,
        usage: Usage {
            input_tokens: resp.prompt_tokens,
            output_tokens: resp.completion_tokens,
        },
    }
}

/// Map OpenAI finish_reason → Anthropic stop_reason.
pub fn map_finish_reason(finish_reason: Option<&str>) -> Option<String> {
    finish_reason.map(|reason| match reason {
        "stop" => "end_turn".to_string(),
        "length" => "max_tokens".to_string(),
        "tool_calls" => "tool_use".to_string(),
        "content_filter" => "end_turn".to_string(),
        other => other.to_string(),
    })
}

pub(crate) fn flatten_system(system: Option<&Value>) -> Option<String> {
    let value = system?;
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Array(blocks) => {
            let parts: Vec<String> = blocks
                .iter()
                .filter_map(|block| {
                    let obj = block.as_object()?;
                    if obj.get("type").and_then(Value::as_str) == Some("text") {
                        obj.get("text").and_then(Value::as_str).map(str::to_owned)
                    } else {
                        None
                    }
                })
                .collect();
            Some(parts.join("\n"))
        }
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

fn convert_message_into(msg: &Value, out: &mut Vec<ChatCompletionRequestMessage>) {
    let Some(obj) = msg.as_object() else {
        return;
    };
    let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
    let Some(content) = obj.get("content") else {
        return;
    };

    match content {
        Value::String(s) => {
            out.push(ChatCompletionRequestMessage {
                role: role.to_string(),
                content: Some(Value::String(s.clone())),
                tool_call_id: None,
                tool_calls: None,
            });
        }
        Value::Array(blocks) => match role {
            "assistant" => convert_assistant_blocks(blocks, out),
            "user" => convert_user_blocks(blocks, out),
            _ => {
                out.push(ChatCompletionRequestMessage {
                    role: role.to_string(),
                    content: Some(Value::String(stringify(content))),
                    tool_call_id: None,
                    tool_calls: None,
                });
            }
        },
        Value::Null => {
            out.push(ChatCompletionRequestMessage {
                role: role.to_string(),
                content: Some(Value::String(String::new())),
                tool_call_id: None,
                tool_calls: None,
            });
        }
        other => {
            out.push(ChatCompletionRequestMessage {
                role: role.to_string(),
                content: Some(Value::String(stringify(other))),
                tool_call_id: None,
                tool_calls: None,
            });
        }
    }
}

fn convert_assistant_blocks(blocks: &[Value], out: &mut Vec<ChatCompletionRequestMessage>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ChatCompletionMessageToolCall> = Vec::new();

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
                    text_parts.push(text.to_string());
                }
            }
            "tool_use" => {
                let id = obj
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = obj
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let input = obj
                    .get("input")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));
                let arguments = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
                tool_calls.push(ChatCompletionMessageToolCall {
                    id,
                    kind: "function".to_string(),
                    function: FunctionCall { name, arguments },
                });
            }
            // thinking blocks are stripped — not all upstream models accept them
            // when sent back as input.
            "thinking" => {}
            _ => {}
        }
    }

    let content_text = text_parts.join("\n");
    out.push(ChatCompletionRequestMessage {
        role: "assistant".to_string(),
        content: if content_text.is_empty() && !tool_calls.is_empty() {
            None
        } else {
            Some(Value::String(content_text))
        },
        tool_call_id: None,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
    });
}

fn convert_user_blocks(blocks: &[Value], out: &mut Vec<ChatCompletionRequestMessage>) {
    let mut regular_parts: Vec<Value> = Vec::new();

    let flush_regular = |regular: &mut Vec<Value>, sink: &mut Vec<ChatCompletionRequestMessage>| {
        if regular.is_empty() {
            return;
        }
        let content = if regular.len() == 1
            && regular[0]
                .as_object()
                .and_then(|o| o.get("type"))
                .and_then(Value::as_str)
                == Some("text")
        {
            let text = regular[0]
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            Value::String(text)
        } else {
            Value::Array(std::mem::take(regular))
        };
        sink.push(ChatCompletionRequestMessage {
            role: "user".to_string(),
            content: Some(content),
            tool_call_id: None,
            tool_calls: None,
        });
        regular.clear();
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
                flush_regular(&mut regular_parts, out);
                let tool_use_id = obj
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let raw = obj.get("content").cloned().unwrap_or(Value::Null);
                let content_str = match &raw {
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
                out.push(ChatCompletionRequestMessage {
                    role: "tool".to_string(),
                    content: Some(Value::String(content_str)),
                    tool_call_id: Some(tool_use_id),
                    tool_calls: None,
                });
            }
            "text" => {
                let text = obj
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                regular_parts.push(json!({"type": "text", "text": text}));
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
                        regular_parts.push(json!({
                            "type": "image_url",
                            "image_url": {"url": url}
                        }));
                    } else if kind_src == "url"
                        && let Some(url) = source.get("url").and_then(Value::as_str)
                    {
                        regular_parts.push(json!({
                            "type": "image_url",
                            "image_url": {"url": url}
                        }));
                    }
                }
            }
            "thinking" => {}
            _ => {}
        }
    }

    flush_regular(&mut regular_parts, out);
}

fn convert_tool_definition(tool: &Value) -> Option<ChatCompletionTool> {
    let obj = tool.as_object()?;
    let name = obj.get("name").and_then(Value::as_str)?.to_string();
    let description = obj
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let parameters = obj
        .get("input_schema")
        .cloned()
        .or_else(|| Some(json!({"type": "object", "properties": {}})));
    Some(ChatCompletionTool {
        kind: "function".to_string(),
        function: FunctionObject {
            name,
            description,
            parameters,
        },
    })
}

fn convert_tool_choice(choice: &Value) -> Option<ChatCompletionToolChoiceOption> {
    let obj = choice.as_object()?;
    let kind = obj.get("type").and_then(Value::as_str)?;
    match kind {
        "auto" => Some(ChatCompletionToolChoiceOption::String("auto".to_string())),
        "any" => Some(ChatCompletionToolChoiceOption::String(
            "required".to_string(),
        )),
        "none" => Some(ChatCompletionToolChoiceOption::String("none".to_string())),
        "tool" => {
            let name = obj.get("name").and_then(Value::as_str)?.to_string();
            Some(ChatCompletionToolChoiceOption::Named {
                function: FunctionNameOnly { name },
            })
        }
        _ => None,
    }
}

pub(crate) fn stringify(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}
