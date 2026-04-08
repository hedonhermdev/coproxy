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

pub async fn create_chat_completion(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateChatCompletionRequest>,
) -> Result<Response, ApiError> {
    auth::authorize(&headers, state.api_key.as_deref())?;

    if request.stream.unwrap_or(false) {
        let upstream = state
            .provider
            .create_chat_completion_stream(request, state.default_model.as_deref())
            .await
            .map_err(ApiError::from_provider_error)?;

        let stream = upstream.bytes_stream();
        let mut response = Response::new(Body::from_stream(stream));
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
