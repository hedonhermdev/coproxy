use crate::openai::error::ApiError;
use axum::http::HeaderMap;

pub fn authorize(headers: &HeaderMap, expected_api_key: Option<&str>) -> Result<(), ApiError> {
    let Some(expected) = expected_api_key else {
        return Ok(());
    };

    let auth = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| ApiError::unauthorized("missing Authorization header"))?;

    let provided = auth
        .strip_prefix("Bearer ")
        .ok_or_else(|| ApiError::unauthorized("expected Bearer token"))?;

    if provided != expected {
        return Err(ApiError::unauthorized("invalid API key"));
    }

    Ok(())
}
