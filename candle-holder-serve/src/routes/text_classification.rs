use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::{routing::post, Json, Router};
use candle_holder_pipelines::TextClassificationPipeline;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use crate::cli::Cli;
use crate::responses::ErrorResponse;
use crate::workers::{task_distributor, InferenceState, InferenceTask, ProcessFn};

#[derive(Debug, Clone, Deserialize)]
struct TextClassificationInferenceParams {
    #[serde(default)]
    top_k: usize,
}

impl Default for TextClassificationInferenceParams {
    fn default() -> Self {
        Self { top_k: 1 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum Inputs {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
struct TextClassificationInferenceRequest {
    inputs: Inputs,
    parameters: Option<TextClassificationInferenceParams>,
}

#[derive(Debug, Clone, Serialize)]
struct TextClassificationResult {
    label: String,
    score: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum TextClassificationInferenceResponse {
    Single(Vec<TextClassificationResult>),
    Multiple(Vec<Vec<TextClassificationResult>>),
}

pub fn router(args: &Cli) -> Result<Router> {
    let model = args.model();
    let device = args.device()?;

    tracing::info!(
        "Loading text classification pipeline for model '{}' on device {:?}",
        model,
        device
    );

    let pipeline = Arc::new(TextClassificationPipeline::new(
        &args.model(),
        &args.device()?,
        None,
        None,
    )?);

    let (tx, rx) = mpsc::channel::<
        InferenceTask<TextClassificationInferenceRequest, TextClassificationInferenceResponse>,
    >(32);

    tokio::spawn(task_distributor::<
        TextClassificationPipeline,
        TextClassificationInferenceRequest,
        TextClassificationInferenceResponse,
        ProcessFn<
            TextClassificationPipeline,
            TextClassificationInferenceRequest,
            TextClassificationInferenceResponse,
        >,
    >(
        rx,
        pipeline,
        args.num_workers(),
        Arc::new(process_text_classification),
    ));

    let state = InferenceState { tx };

    Ok(Router::new().route("/", post(inference)).with_state(state))
}

async fn inference(
    State(state): State<
        InferenceState<TextClassificationInferenceRequest, TextClassificationInferenceResponse>,
    >,
    Json(req): Json<TextClassificationInferenceRequest>,
) -> Result<Json<TextClassificationInferenceResponse>, ErrorResponse> {
    let (resp_tx, resp_rx) = oneshot::channel();
    let task = InferenceTask { req, resp_tx };

    if let Err(e) = state.tx.send(task).await {
        tracing::error!("Failed to send task to worker: {}", e);
        return Err(ErrorResponse::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to process request",
        ));
    }

    match resp_rx.await {
        Ok(response) => Ok(Json(response)),
        Err(e) => {
            tracing::error!("Failed to receive response from worker: {}", e);
            Err(ErrorResponse::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to process request",
            ))
        }
    }
}

fn process_text_classification(
    pipeline: &TextClassificationPipeline,
    request: TextClassificationInferenceRequest,
) -> TextClassificationInferenceResponse {
    let params = request.parameters.unwrap_or_default();

    match request.inputs {
        Inputs::Single(text) => {
            let outputs = pipeline.run(text, Some(params.top_k)).unwrap();
            let results = outputs
                .into_iter()
                .map(|(label, score)| TextClassificationResult { label, score })
                .collect();
            TextClassificationInferenceResponse::Single(results)
        }
        Inputs::Multiple(texts) => {
            let outputs = pipeline.run_batch(texts, Some(params.top_k)).unwrap();
            let results = outputs
                .into_iter()
                .map(|output| {
                    output
                        .into_iter()
                        .map(|(label, score)| TextClassificationResult { label, score })
                        .collect()
                })
                .collect();
            TextClassificationInferenceResponse::Multiple(results)
        }
    }
}
