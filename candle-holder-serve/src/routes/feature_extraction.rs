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
    process_feature_extraction
);

pub(crate) fn process_feature_extraction(
    pipeline: &FeatureExtractionPipeline,
    request: FeatureExtractionInferenceRequest,
) -> FeatureExtractionInferenceResponse {
    let params = request.parameters.unwrap_or_default();

    match request.inputs {
        Inputs::Single(text) => {
            let output = pipeline.run(text, Some(params.into())).unwrap();
            let result = output.to_vec1::<f32>().unwrap();
            FeatureExtractionInferenceResponse::Single(result)
        }
        Inputs::Multiple(texts) => {
            let outputs = pipeline.run_batch(texts, Some(params.into())).unwrap();
            let results = outputs.to_vec2::<f32>().unwrap();
            FeatureExtractionInferenceResponse::Multiple(results)
        }
    }
}
