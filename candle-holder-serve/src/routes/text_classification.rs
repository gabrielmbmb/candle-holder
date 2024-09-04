use candle_holder_pipelines::TextClassificationPipeline;
use serde::{Deserialize, Serialize};

use crate::generate_router;

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
pub(crate) struct TextClassificationInferenceRequest {
    inputs: Inputs,
    parameters: Option<TextClassificationInferenceParams>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TextClassificationResult {
    label: String,
    score: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum TextClassificationInferenceResponse {
    Single(Vec<TextClassificationResult>),
    Multiple(Vec<Vec<TextClassificationResult>>),
}

generate_router!(
    TextClassificationPipeline,
    TextClassificationInferenceRequest,
    TextClassificationInferenceResponse,
    process_text_classification
);

pub(crate) fn process_text_classification(
    pipeline: &TextClassificationPipeline,
    request: TextClassificationInferenceRequest,
) -> Result<TextClassificationInferenceResponse, ErrorResponse> {
    let params = request.parameters.unwrap_or_default();

    match request.inputs {
        Inputs::Single(text) => {
            let outputs = pipeline.run(text, Some(params.top_k)).map_err(|e| {
                tracing::error!("Failed to run pipeline: {}", e);
                ErrorResponse::new(500, "Failed to process request")
            })?;
            let results = outputs
                .into_iter()
                .map(|(label, score)| TextClassificationResult { label, score })
                .collect();
            Ok(TextClassificationInferenceResponse::Single(results))
        }
        Inputs::Multiple(texts) => {
            let outputs = pipeline.run_batch(texts, Some(params.top_k)).map_err(|e| {
                tracing::error!("Failed to run pipeline on batch: {}", e);
                ErrorResponse::new(500, "Failed to process request")
            })?;
            let results = outputs
                .into_iter()
                .map(|output| {
                    output
                        .into_iter()
                        .map(|(label, score)| TextClassificationResult { label, score })
                        .collect()
                })
                .collect();
            Ok(TextClassificationInferenceResponse::Multiple(results))
        }
    }
}
