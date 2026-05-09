use crate::logging::redact_serializable;
use crate::openai::error::ApiError;
use crate::openai::types::{
    ChatCompletionChoice, ChatCompletionResponseMessage, CompletionUsage,
    CreateChatCompletionRequest, CreateChatCompletionResponse,
};
use crate::provider::ModelProvider;
use crate::server::routes::auth;
use crate::state::AppState;
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;

pub async fn create_chat_completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateChatCompletionRequest>,
) -> Result<Response, ApiError> {
    auth::authorize(&headers, state.api_key.as_deref())?;

    let stream_flag = request.stream.unwrap_or(false);
    let model_hint = request.model.clone();
    let span = tracing::Span::current();
    span.record("stream", stream_flag);
    if let Some(m) = model_hint.as_deref() {
        span.record("model", m);
    }
    tracing::debug!(
        model = ?model_hint,
        stream = stream_flag,
        n_messages = request.messages.len(),
        "chat completion dispatch"
    );
    tracing::trace!(body = %redact_serializable(&request), "request body");

    if stream_flag {
        let upstream = state
            .provider
            .create_chat_completion_stream(request, state.default_model.as_deref())
            .await
            .map_err(ApiError::from_provider_error)?;

        let upstream_stream = upstream.bytes_stream();
        let span_for_stream = tracing::Span::current();
        let counted = async_stream::stream! {
            let mut chunks: u64 = 0;
            let mut bytes: u64 = 0;
            futures_util::pin_mut!(upstream_stream);
            while let Some(item) = upstream_stream.next().await {
                if let Ok(b) = &item {
                    chunks += 1;
                    bytes += b.len() as u64;
                    tracing::trace!(chunk_size = b.len(), "sse byte chunk");
                }
                yield item;
            }
            let _e = span_for_stream.enter();
            tracing::debug!(chunks, bytes, "openai stream complete");
        };
        let mut response = Response::new(Body::from_stream(counted));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        response
            .headers_mut()
            .insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
        return Ok(response);
    }

    let result = state
        .provider
        .create_chat_completion(request, state.default_model.as_deref())
        .await
        .map_err(ApiError::from_provider_error)?;

    tracing::Span::current().record("model", result.model.as_str());
    tracing::debug!(
        model = %result.model,
        prompt_tokens = result.prompt_tokens,
        completion_tokens = result.completion_tokens,
        "chat completion ok"
    );

    let finish_reason = if result.tool_calls.is_empty() {
        Some("stop".to_string())
    } else {
        Some("tool_calls".to_string())
    };

    let response = CreateChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
        object: "chat.completion",
        created: chrono::Utc::now().timestamp(),
        model: result.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                role: "assistant",
                content: result.content,
                tool_calls: result.tool_calls,
            },
            finish_reason,
        }],
        usage: CompletionUsage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.prompt_tokens + result.completion_tokens,
        },
    };

    Ok(Json(response).into_response())
}
