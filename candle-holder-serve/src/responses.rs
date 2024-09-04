use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// A generic error response.
#[derive(Debug)]
pub(crate) struct ErrorResponse {
    /// The HTTP status code.
    code: StatusCode,
    /// The description of the error.
    message: String,
}

impl ErrorResponse {
    pub fn new(code: StatusCode, message: impl Into<String>) -> Self {
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
                "code": self.code.as_u16(),
                "message": self.message,
            }
        }));

        (self.code, body).into_response()
    }
}
