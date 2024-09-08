use candle_holder_pipelines::{FeatureExtractionOptions, FeatureExtractionPipeline, Pooling};
use serde::{Deserialize, Serialize};

use crate::generate_router;

#[derive(Debug, Clone, Deserialize)]
struct FeatureExtractionInferenceParams {
    pooling: Option<Pooling>,
    normalize: Option<bool>,
}

impl Default for FeatureExtractionInferenceParams {
    fn default() -> Self {
        Self {
            pooling: None,
            normalize: None,
        }
    }
}

impl From<FeatureExtractionInferenceParams> for FeatureExtractionOptions {
    fn from(params: FeatureExtractionInferenceParams) -> Self {
        let default = FeatureExtractionOptions::default();
        let pooling = default.pooling;
        Self {
            pooling: params.pooling.or(pooling),
            normalize: params.normalize.unwrap_or(default.normalize),
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
pub(crate) struct FeatureExtractionInferenceRequest {
    inputs: Inputs,
    parameters: Option<FeatureExtractionInferenceParams>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum FeatureExtractionInferenceResponse {
    Single(Vec<f32>),
    Multiple(Vec<Vec<f32>>),
}

generate_router!(
    FeatureExtractionPipeline,
    FeatureExtractionInferenceRequest,
    FeatureExtractionInferenceResponse,
    process_feature_extraction,
    feature_extraction_warm_up
);

pub(crate) fn process_feature_extraction(
    pipeline: &FeatureExtractionPipeline,
    request: FeatureExtractionInferenceRequest,
) -> Result<FeatureExtractionInferenceResponse, ErrorResponse> {
    let params = request.parameters.unwrap_or_default();

    match request.inputs {
        Inputs::Single(text) => {
            let output = pipeline.run(text, Some(params.into())).map_err(|e| {
                tracing::error!("Failed to run pipeline on batch: {}", e);
                ErrorResponse::new(500, format!("Failed to run pipeline: {}", e))
            })?;
            let result = output.to_vec1::<f32>().unwrap();
            Ok(FeatureExtractionInferenceResponse::Single(result))
        }
        Inputs::Multiple(texts) => {
            let outputs = pipeline
                .run_batch(texts, Some(params.into()))
                .map_err(|e| {
                    tracing::error!("Failed to run pipeline on batch: {}", e);
                    ErrorResponse::new(500, format!("Failed to run pipeline on batch: {}", e))
                })?;
            let results = outputs.to_vec2::<f32>().unwrap();
            Ok(FeatureExtractionInferenceResponse::Multiple(results))
        }
    }
}

pub(crate) fn feature_extraction_warm_up(pipeline: &FeatureExtractionPipeline) -> Result<()> {
    pipeline.run(
        "This is a sample text to warm up the feature extraction pipeline.",
        None,
    )?;
    Ok(())
}
