use crate::openai::error::ApiError;
use crate::server::routes::auth;
use crate::state::AppState;
use axum::Json;
use axum::body::Body;
use axum::extract::{Path, RawQuery, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;

pub async fn create_response(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(_payload): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    auth::authorize(&headers, state.api_key.as_deref())?;

    let upstream = state
        .provider
        .create_response(_payload, state.default_model.as_deref())
        .await
        .map_err(ApiError::from_provider_error)?;

    Ok(proxy_upstream_response(upstream))
}

pub async fn get_response(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(response_id): Path<String>,
    RawQuery(raw_query): RawQuery,
) -> Result<Response, ApiError> {
    auth::authorize(&headers, state.api_key.as_deref())?;

    let upstream = state
        .provider
        .get_response(&response_id, raw_query.as_deref())
        .await
        .map_err(ApiError::from_provider_error)?;

    Ok(proxy_upstream_response(upstream))
}

fn proxy_upstream_response(upstream: reqwest::Response) -> Response {
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let headers = upstream.headers().clone();
    let stream = upstream.bytes_stream();

    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;

    for (name, value) in &headers {
        if *name == header::CONTENT_LENGTH
            || *name == header::TRANSFER_ENCODING
            || *name == header::CONNECTION
        {
            continue;
        }
        response.headers_mut().insert(name, value.clone());
    }

    response
}
