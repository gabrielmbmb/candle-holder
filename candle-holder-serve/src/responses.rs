use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use serde_json::json;

/// A generic error response.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct ErrorResponse {
    /// The HTTP status code.
    code: u16,
    /// The description of the error.
    message: String,
}

impl ErrorResponse {
    pub fn new(code: u16, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": {
                "message": self.message,
            }
        }));

        (StatusCode::from_u16(self.code).unwrap(), body).into_response()
    }
}
