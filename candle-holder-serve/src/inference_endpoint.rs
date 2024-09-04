use axum::{extract::State, Json};
use tokio::sync::oneshot;

use crate::{
    responses::ErrorResponse,
    workers::{InferenceState, InferenceTask},
};

pub(crate) async fn inference<I, O>(
    State(state): State<InferenceState<I, Result<O, ErrorResponse>>>,
    Json(req): Json<I>,
) -> Result<Json<O>, ErrorResponse> {
    let (resp_tx, resp_rx) = oneshot::channel();
    let task = InferenceTask { req, resp_tx };

    if let Err(e) = state.tx.send(task).await {
        tracing::error!("Failed to send task to worker: {}", e);
        return Err(ErrorResponse::new(500, "Failed to process request"));
    }

    match resp_rx.await {
        Ok(response) => match response {
            Ok(response) => Ok(Json(response)),
            Err(e) => Err(e),
        },
        Err(e) => {
            tracing::error!("Failed to receive response from worker: {}", e);
            Err(ErrorResponse::new(500, "Failed to process request"))
        }
    }
}
