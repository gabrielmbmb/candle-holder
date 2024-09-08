use candle_holder_models::{GenerationConfig, GenerationParams};
use candle_holder_pipelines::TextGenerationPipeline;
use serde::{Deserialize, Serialize};

use crate::generate_router;

#[derive(Debug, Clone, Deserialize)]
struct TextGenerationInferenceParams {
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    repetition_penalty: Option<f64>,
    #[serde(default)]
    max_new_tokens: Option<usize>,
    #[serde(default)]
    do_sample: Option<bool>,
    #[serde(default)]
    num_return_sequences: Option<usize>,
}

impl Default for TextGenerationInferenceParams {
    fn default() -> Self {
        Self {
            top_k: None,
            top_p: None,
            temperature: None,
            repetition_penalty: None,
            max_new_tokens: Some(50),
            do_sample: Some(true),
            num_return_sequences: Some(1),
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
pub(crate) struct TextGenerationInferenceRequest {
    inputs: Inputs,
    parameters: Option<TextGenerationInferenceParams>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TextGenerationResult {
    generated_text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum TextGenerationInferenceResponse {
    Single(Vec<TextGenerationResult>),
    Multiple(Vec<Vec<TextGenerationResult>>),
}

generate_router!(
    TextGenerationPipeline,
    TextGenerationInferenceRequest,
    TextGenerationInferenceResponse,
    process_text_generation,
    text_generation_warm_up
);

pub(crate) fn process_text_generation(
    pipeline: &TextGenerationPipeline,
    request: TextGenerationInferenceRequest,
) -> Result<TextGenerationInferenceResponse, ErrorResponse> {
    let params = request.parameters.unwrap_or_default();
    let generation_params = GenerationParams {
        generation_config: Some(GenerationConfig {
            top_k: params.top_k,
            top_p: params.top_p,
            temperature: params.temperature.unwrap_or(0.7),
            repetition_penalty: params.repetition_penalty,
            max_new_tokens: params.max_new_tokens,
            do_sample: params.do_sample.unwrap_or(false),
            num_return_sequences: params.num_return_sequences.unwrap_or(1).max(1),
            ..Default::default()
        }),
        tokenizer: None,
        stopping_criteria: None,
        token_streamer: None,
        seed: None,
    };

    match request.inputs {
        Inputs::Single(text) => {
            let outputs = pipeline.run(text, Some(generation_params)).map_err(|e| {
                tracing::error!("Failed to run pipeline: {}", e);
                ErrorResponse::new(500, "Failed to process request")
            })?;
            let results = outputs
                .into_iter()
                .map(|output| TextGenerationResult {
                    generated_text: output.get_text().unwrap_or("".to_string()),
                })
                .collect();
            Ok(TextGenerationInferenceResponse::Single(results))
        }
        Inputs::Multiple(texts) => {
            let outputs = pipeline
                .run_batch(texts, Some(generation_params))
                .map_err(|e| {
                    tracing::error!("Failed to run pipeline: {}", e);
                    ErrorResponse::new(500, "Failed to process request")
                })?;
            let results: Vec<Vec<TextGenerationResult>> = outputs
                .into_iter()
                .map(|output| {
                    output
                        .into_iter()
                        .map(|output| TextGenerationResult {
                            generated_text: output.get_text().unwrap_or("".to_string()),
                        })
                        .collect()
                })
                .collect();
            Ok(TextGenerationInferenceResponse::Multiple(results))
        }
    }
}

pub(crate) fn text_generation_warm_up(pipeline: &TextGenerationPipeline) -> Result<()> {
    pipeline.run("warm up", None)?;
    Ok(())
}
