mod cli;
mod routes;

use anyhow::Result;
use axum::Router;
use clap::Parser;

use crate::cli::{Cli, Pipeline};
use crate::routes::{
    feature_extraction, fill_mask, text_classification, text_generation, token_classification,
    zero_shot_classification,
};

#[tokio::main]
async fn main() -> Result<()> {
    // initialize tracing
    tracing_subscriber::fmt::init();

    // Parse the command line arguments
    let args = Cli::parse();

    // Initialize the router based on the pipeline
    let inference_router = match args.pipeline() {
        Pipeline::FeatureExtraction => feature_extraction::router(&args)?,
        Pipeline::FillMask => fill_mask::router(&args)?,
        Pipeline::TextClassification => text_classification::router(&args)?,
        Pipeline::TextGeneration => text_generation::router(&args)?,
        Pipeline::TokenClassification => token_classification::router(&args)?,
        Pipeline::ZeroShotClassification => zero_shot_classification::router(&args)?,
    };
    let router = Router::new().nest("/", inference_router);

    tracing::info!("Listening on {}", args.host());
    let listener = tokio::net::TcpListener::bind(args.host()).await.unwrap();
    axum::serve(listener, router).await.unwrap();

    Ok(())
}
