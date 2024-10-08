use candle_holder_pipelines::{FillMaskOptions, FillMaskPipeline};
use serde::{Deserialize, Serialize};

use crate::generate_router;

#[derive(Debug, Clone, Deserialize)]
struct FillMaskInferenceParams {
    #[serde(default)]
    top_k: usize,
}

impl Default for FillMaskInferenceParams {
    fn default() -> Self {
        Self { top_k: 5 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum Inputs {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FillMaskInferenceRequest {
    inputs: Inputs,
    parameters: Option<FillMaskInferenceParams>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct FillMaskResult {
    score: f32,
    token: u32,
    token_str: String,
    sequence: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum FillMaskInferenceResponse {
    Single(Vec<Vec<FillMaskResult>>),
    Multiple(Vec<Vec<Vec<FillMaskResult>>>),
}

generate_router!(
    FillMaskPipeline,
    FillMaskInferenceRequest,
    FillMaskInferenceResponse,
    process_feature_extraction,
    fill_mask_warm_up
);

pub(crate) fn process_feature_extraction(
    pipeline: &FillMaskPipeline,
    request: FillMaskInferenceRequest,
) -> Result<FillMaskInferenceResponse, ErrorResponse> {
    let params = request.parameters.unwrap_or_default();

    match request.inputs {
        Inputs::Single(text) => {
            let output = pipeline
                .run(
                    text,
                    Some(FillMaskOptions {
                        top_k: params.top_k,
                    }),
                )
                .map_err(|e| {
                    tracing::error!("Failed to run pipeline: {}", e);
                    ErrorResponse::new(400, e.to_string())
                })?;
            let result: Vec<Vec<FillMaskResult>> = output
                .into_iter()
                .map(|mask_results| {
                    mask_results
                        .into_iter()
                        .map(|(score, token, token_str, sequence)| FillMaskResult {
                            score,
                            token,
                            token_str,
                            sequence,
                        })
                        .collect()
                })
                .collect();
            Ok(FillMaskInferenceResponse::Single(result))
        }
        Inputs::Multiple(texts) => {
            let outputs = pipeline
                .run_batch(
                    texts,
                    Some(FillMaskOptions {
                        top_k: params.top_k,
                    }),
                )
                .map_err(|e| {
                    tracing::error!("Failed to run pipeline on batch: {}", e);
                    ErrorResponse::new(400, e.to_string())
                })?;
            let results: Vec<Vec<Vec<FillMaskResult>>> = outputs
                .into_iter()
                .map(|input_mask_results| {
                    input_mask_results
                        .into_iter()
                        .map(|mask_results| {
                            mask_results
                                .into_iter()
                                .map(|(score, token, token_str, sequence)| FillMaskResult {
                                    score,
                                    token,
                                    token_str,
                                    sequence,
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();
            Ok(FillMaskInferenceResponse::Multiple(results))
        }
    }
}

pub(crate) fn fill_mask_warm_up(pipeline: &FillMaskPipeline) -> Result<()> {
    pipeline.run("Hello, my name is [MASK].", None)?;
    Ok(())
}
