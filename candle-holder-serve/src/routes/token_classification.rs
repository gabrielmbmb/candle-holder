use anyhow::Result;
use axum::{routing::post, Router};
use candle_holder_pipelines::TokenClassificationPipeline;
use std::sync::Arc;

use crate::cli::Cli;

pub fn router(args: &Cli) -> Result<Router> {
    let model = args.model();
    let device = args.device()?;

    tracing::info!(
        "Loading token classification pipeline for model '{}' on device {:?}",
        model,
        device
    );

    let pipeline = Arc::new(TokenClassificationPipeline::new(
        &args.model(),
        &args.device()?,
        None,
        None,
    )?);

    Ok(Router::new()
        .route("/", post(inference))
        .with_state(pipeline))
}

async fn inference() -> &'static str {
    "inference"
}
