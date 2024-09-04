use candle_holder_pipelines::{ZeroShotClassificationOptions, ZeroShotClassificationPipeline};
use serde::{Deserialize, Serialize};

use crate::generate_router;

#[derive(Debug, Clone, Deserialize)]
struct ZeroShotClassificationInferenceParams {
    candidate_labels: Vec<String>,
    #[serde(default)]
    hypothesis_template: Option<String>,
    #[serde(default)]
    multi_label: Option<bool>,
}

impl Default for ZeroShotClassificationInferenceParams {
    fn default() -> Self {
        Self {
            candidate_labels: vec![],
            hypothesis_template: None,
            multi_label: None,
        }
    }
}

impl From<ZeroShotClassificationInferenceParams> for ZeroShotClassificationOptions {
    fn from(params: ZeroShotClassificationInferenceParams) -> Self {
        let default = ZeroShotClassificationOptions::default();
        Self {
            hypothesis_template: params
                .hypothesis_template
                .unwrap_or(default.hypothesis_template),
            multi_label: params.multi_label.unwrap_or(default.multi_label),
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
pub(crate) struct ZeroShotClassificationInferenceRequest {
    inputs: Inputs,
    parameters: Option<ZeroShotClassificationInferenceParams>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ZeroShotClassificationResult {
    sequence: String,
    labels: Vec<String>,
    scores: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum ZeroShotClassificationInferenceResponse {
    Single(ZeroShotClassificationResult),
    Multiple(Vec<ZeroShotClassificationResult>),
}

generate_router!(
    ZeroShotClassificationPipeline,
    ZeroShotClassificationInferenceRequest,
    ZeroShotClassificationInferenceResponse,
    process_text_classification
);

pub(crate) fn process_text_classification(
    pipeline: &ZeroShotClassificationPipeline,
    request: ZeroShotClassificationInferenceRequest,
) -> ZeroShotClassificationInferenceResponse {
    let params = request.parameters.unwrap_or_default();

    match request.inputs {
        Inputs::Single(text) => {
            let output = pipeline
                .run(
                    text.clone(),
                    params.candidate_labels.clone(),
                    Some(params.into()),
                )
                .unwrap();
            let (labels, scores): (Vec<String>, Vec<f32>) = output.into_iter().unzip();
            let result = ZeroShotClassificationResult {
                sequence: text,
                labels,
                scores,
            };
            ZeroShotClassificationInferenceResponse::Single(result)
        }
        Inputs::Multiple(texts) => {
            let outputs = pipeline
                .run_batch(
                    texts.clone(),
                    params.candidate_labels.clone(),
                    Some(params.into()),
                )
                .unwrap();
            let results = outputs
                .into_iter()
                .zip(texts.into_iter())
                .map(|(output, text)| {
                    let (labels, scores): (Vec<String>, Vec<f32>) = output.into_iter().unzip();
                    ZeroShotClassificationResult {
                        sequence: text,
                        labels,
                        scores,
                    }
                })
                .collect();
            ZeroShotClassificationInferenceResponse::Multiple(results)
        }
    }
}
