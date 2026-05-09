use crate::anthropic::convert::{anthropic_to_openai, map_finish_reason, openai_to_anthropic};
use crate::anthropic::convert_responses::{anthropic_to_responses, responses_to_anthropic};
use crate::anthropic::stream::anthropic_event_stream;
use crate::anthropic::stream_responses::anthropic_event_stream_responses;
use crate::anthropic::types::CreateMessageRequest;
use crate::logging::redact_serializable;
use crate::openai::error::ApiError;
use crate::provider::ModelProvider;
use crate::provider::ghcp::sanitize_error_body;
use crate::state::AppState;
use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

#[derive(Debug, Clone, Copy)]
enum Surface {
    ChatCompletions,
    Responses,
}

/// `POST /v1/messages` — Anthropic Messages API surface.
///
/// Accepts either `x-api-key: <key>` (Anthropic SDK default) or
/// `Authorization: Bearer <key>` (compatibility) when an API key is configured.
pub async fn create_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut request): Json<CreateMessageRequest>,
) -> Result<Response, ApiError> {
    authorize_anthropic(&headers, state.api_key.as_deref())?;

    let stream_requested = request.stream.unwrap_or(false);

    // Anthropic-only: rewrite the request model via longest-prefix match
    // against the GHCP catalog so e.g. `claude-haiku-4-5-20251001` becomes
    // `claude-haiku-4.5`. OpenAI-surface routes deliberately don't do this.
    let endpoints = if let Some(requested) = request.model.as_deref() {
        let (canonical, endpoints) = state.provider.resolve_model_and_endpoints(requested).await;
        if canonical != requested {
            request.model = Some(canonical);
        }
        endpoints
    } else {
        None
    };

    let resolved_model = request.model.as_deref().or(state.default_model.as_deref());

    let span = tracing::Span::current();
    span.record("stream", stream_requested);
    if let Some(m) = resolved_model {
        span.record("model", m);
    }
    tracing::debug!(
        model = ?resolved_model,
        stream = stream_requested,
        n_messages = request.messages.len(),
        "anthropic message dispatch"
    );
    tracing::trace!(body = %redact_serializable(&request), "request body");

    let surface = pick_surface(resolved_model, endpoints.as_deref())?;
    tracing::debug!(?surface, "resolved upstream surface");

    match surface {
        Surface::ChatCompletions => dispatch_chat_completions(state, request).await,
        Surface::Responses => dispatch_responses(state, request).await,
    }
}

const ENDPOINT_CHAT_COMPLETIONS: &str = "/chat/completions";
const ENDPOINT_RESPONSES: &str = "/responses";

/// Pick the upstream surface from the model's `supported_endpoints` slice.
///
/// Prefers `/chat/completions` when both are listed — keeps well-tested
/// models on the existing converter. Falls back to `/chat/completions` when
/// no endpoints were resolved (private aliases, offline-cache scenarios).
fn pick_surface(model: Option<&str>, endpoints: Option<&[String]>) -> Result<Surface, ApiError> {
    let Some(endpoints) = endpoints else {
        return Ok(Surface::ChatCompletions);
    };
    if endpoints.iter().any(|e| e == ENDPOINT_CHAT_COMPLETIONS) {
        Ok(Surface::ChatCompletions)
    } else if endpoints.iter().any(|e| e == ENDPOINT_RESPONSES) {
        Ok(Surface::Responses)
    } else {
        let model = model.unwrap_or("<unknown>");
        Err(ApiError::bad_request(format!(
            "model `{model}` does not expose `{ENDPOINT_CHAT_COMPLETIONS}` or `{ENDPOINT_RESPONSES}`"
        )))
    }
}

async fn dispatch_chat_completions(
    state: AppState,
    request: CreateMessageRequest,
) -> Result<Response, ApiError> {
    let stream_requested = request.stream.unwrap_or(false);
    let openai_request = anthropic_to_openai(&request);

    if stream_requested {
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
        return Ok(sse_response(Body::from_stream(stream)));
    }

    let result = state
        .provider
        .create_chat_completion(openai_request, state.default_model.as_deref())
        .await
        .map_err(ApiError::from_provider_error)?;

    tracing::Span::current().record("model", result.model.as_str());
    tracing::debug!(
        model = %result.model,
        prompt_tokens = result.prompt_tokens,
        completion_tokens = result.completion_tokens,
        "anthropic message ok"
    );

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

async fn dispatch_responses(
    state: AppState,
    request: CreateMessageRequest,
) -> Result<Response, ApiError> {
    let stream_requested = request.stream.unwrap_or(false);
    let model_for_stream = request
        .model
        .clone()
        .or_else(|| state.default_model.clone())
        .unwrap_or_else(|| "unknown".to_string());

    let mut payload = anthropic_to_responses(&request);
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("stream".to_string(), Value::Bool(stream_requested));
    }

    let upstream = state
        .provider
        .create_response(payload, state.default_model.as_deref())
        .await
        .map_err(ApiError::from_provider_error)?;

    if !upstream.status().is_success() {
        let status = upstream.status();
        let body = upstream.text().await.unwrap_or_default();
        let sanitized = sanitize_error_body(&body);
        tracing::warn!(%status, body = %sanitized, "upstream /responses returned non-2xx");
        return Err(ApiError::bad_request(format!(
            "upstream /responses returned {status}: {sanitized}"
        )));
    }

    if stream_requested {
        let stream = anthropic_event_stream_responses(upstream, model_for_stream);
        return Ok(sse_response(Body::from_stream(stream)));
    }

    let bytes = upstream
        .bytes()
        .await
        .map_err(|e| ApiError::internal(format!("reading upstream /responses body: {e}")))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|e| ApiError::internal(format!("parsing upstream /responses body: {e}")))?;
    let anthropic_response = responses_to_anthropic(&value, &model_for_stream);
    tracing::Span::current().record("model", anthropic_response.model.as_str());
    tracing::debug!(
        model = %anthropic_response.model,
        input_tokens = anthropic_response.usage.input_tokens,
        output_tokens = anthropic_response.usage.output_tokens,
        "anthropic message ok (responses)"
    );
    Ok(Json(anthropic_response).into_response())
}

fn sse_response(body: Body) -> Response {
    let mut response = Response::new(body);
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
    response
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

    tracing::warn!(reason = "anthropic_missing_or_invalid", "auth rejected");
    Err(ApiError::unauthorized(
        "missing or invalid authentication; expected x-api-key or Authorization: Bearer",
    ))
}
