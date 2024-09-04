use candle_holder_pipelines::{
    AggregationStrategy, TokenClassificationOptions, TokenClassificationPipeline,
};
use serde::{Deserialize, Serialize};

use crate::generate_router;

#[derive(Debug, Clone, Deserialize)]
struct TokenClassificationInferenceParams {
    aggregation_strategy: Option<AggregationStrategy>,
    ignore_labels: Option<Vec<String>>,
}

impl Default for TokenClassificationInferenceParams {
    fn default() -> Self {
        Self {
            aggregation_strategy: None,
            ignore_labels: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum Inputs {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TokenClassificationInferenceRequest {
    inputs: Inputs,
    parameters: Option<TokenClassificationInferenceParams>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TokenClassificationResult {
    entity: String,
    score: f32,
    index: usize,
    word: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum TokenClassificationInferenceResponse {
    Single(Vec<TokenClassificationResult>),
    Multiple(Vec<Vec<TokenClassificationResult>>),
}

generate_router!(
    TokenClassificationPipeline,
    TokenClassificationInferenceRequest,
    TokenClassificationInferenceResponse,
    process_token_classification
);

pub(crate) fn process_token_classification(
    pipeline: &TokenClassificationPipeline,
    request: TokenClassificationInferenceRequest,
) -> Result<TokenClassificationInferenceResponse, ErrorResponse> {
    let params = request.parameters.unwrap_or_default();

    match request.inputs {
        Inputs::Single(text) => {
            let outputs = pipeline
                .run(
                    text,
                    Some(TokenClassificationOptions {
                        aggregation_strategy: params
                            .aggregation_strategy
                            .unwrap_or(AggregationStrategy::None),
                        ignore_labels: params.ignore_labels.unwrap_or_default(),
                    }),
                )
                .map_err(|e| {
                    tracing::error!("Failed to run pipeline: {}", e);
                    ErrorResponse::new(500, "Failed to process request")
                })?;
            let results = outputs
                .into_iter()
                .map(|output| TokenClassificationResult {
                    entity: output.entity().to_string(),
                    score: output.score(),
                    index: output.index(),
                    word: output.word().to_string(),
                    start: output.start(),
                    end: output.end(),
                })
                .collect::<Vec<_>>();
            Ok(TokenClassificationInferenceResponse::Single(results))
        }
        Inputs::Multiple(texts) => {
            let outputs = pipeline
                .run_batch(
                    texts,
                    Some(TokenClassificationOptions {
                        aggregation_strategy: params
                            .aggregation_strategy
                            .unwrap_or(AggregationStrategy::None),
                        ignore_labels: params.ignore_labels.unwrap_or_default(),
                    }),
                )
                .map_err(|e| {
                    tracing::error!("Failed to run pipeline: {}", e);
                    ErrorResponse::new(500, "Failed to process request")
                })?;
            let results = outputs
                .into_iter()
                .map(|outputs| {
                    outputs
                        .into_iter()
                        .map(|output| TokenClassificationResult {
                            entity: output.entity().to_string(),
                            score: output.score(),
                            index: output.index(),
                            word: output.word().to_string(),
                            start: output.start(),
                            end: output.end(),
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            Ok(TokenClassificationInferenceResponse::Multiple(results))
        }
    }
}
