use std::time::Duration;

use bytes::Bytes;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde::Deserialize;

pub(crate) fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let raw = headers.get(RETRY_AFTER)?.to_str().ok()?;
    raw.parse::<u64>().ok().map(Duration::from_secs)
}

#[derive(Deserialize)]
struct WireErrorBody {
    error: WireErrorPayload,
}

#[derive(Deserialize)]
struct WireErrorPayload {
    message: String,
}

pub(crate) fn parse_error_message(bytes: &Bytes) -> Option<String> {
    serde_json::from_slice::<WireErrorBody>(bytes)
        .ok()
        .map(|b| b.error.message)
}

pub(crate) trait FailureFromStatus: Sized {
    fn auth(message: Option<String>) -> Self;
    fn rate_limited(retry_after: Option<Duration>, message: Option<String>) -> Self;
    fn invalid_request(message: String) -> Self;
    fn server_error(status: u16, message: Option<String>) -> Self;
}

pub(crate) fn map_status_to_failure<F: FailureFromStatus>(
    status: u16,
    headers: &HeaderMap,
    bytes: &Bytes,
) -> F {
    let message = parse_error_message(bytes);
    match status {
        401 | 403 => F::auth(message),
        429 => F::rate_limited(parse_retry_after(headers), message),
        400..=499 => F::invalid_request(message.unwrap_or_else(|| format!("HTTP {status}"))),
        500..=599 => F::server_error(status, message),
        // 1xx/3xx or any other non-success status surfaces as a typed ServerError so the
        // caller still gets a concrete status code instead of a silent collapse.
        _ => F::server_error(status, message),
    }
}
