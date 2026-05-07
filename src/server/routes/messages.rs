use crate::anthropic::convert::{anthropic_to_openai, map_finish_reason, openai_to_anthropic};
use crate::anthropic::stream::anthropic_event_stream;
use crate::anthropic::types::CreateMessageRequest;
use crate::openai::error::ApiError;
use crate::provider::ModelProvider;
use crate::state::AppState;
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};

/// `POST /v1/messages` — Anthropic Messages API surface.
///
/// Accepts either `x-api-key: <key>` (Anthropic SDK default) or
/// `Authorization: Bearer <key>` (compatibility) when an API key is configured.
pub async fn create_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateMessageRequest>,
) -> Result<Response, ApiError> {
    authorize_anthropic(&headers, state.api_key.as_deref())?;

    let stream_requested = request.stream.unwrap_or(false);
    let openai_request = anthropic_to_openai(&request);

    if stream_requested {
        // Resolve a model name we can echo back in message_start.
        let model_for_stream = openai_request
            .model
            .clone()
            .or_else(|| state.default_model.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let upstream = state
            .provider
            .create_chat_completion_stream(openai_request, state.default_model.as_deref())
            .await
            .map_err(ApiError::from_provider_error)?;

        let stream = anthropic_event_stream(upstream, model_for_stream);
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
        .create_chat_completion(openai_request, state.default_model.as_deref())
        .await
        .map_err(ApiError::from_provider_error)?;

    let stop_reason = if result.tool_calls.is_empty() {
        map_finish_reason(Some("stop"))
    } else {
        map_finish_reason(Some("tool_calls"))
    };

    let model = result.model.clone();
    let mut anthropic_response = openai_to_anthropic(result, model);
    anthropic_response.stop_reason = stop_reason;

    Ok(Json(anthropic_response).into_response())
}

fn authorize_anthropic(
    headers: &HeaderMap,
    expected_api_key: Option<&str>,
) -> Result<(), ApiError> {
    let Some(expected) = expected_api_key else {
        return Ok(());
    };

    if let Some(value) = headers.get("x-api-key").and_then(|v| v.to_str().ok())
        && value == expected
    {
        return Ok(());
    }

    if let Some(auth) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        && let Some(provided) = auth.strip_prefix("Bearer ")
        && provided == expected
    {
        return Ok(());
    }

    Err(ApiError::unauthorized(
        "missing or invalid authentication; expected x-api-key or Authorization: Bearer",
    ))
}
