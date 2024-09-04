#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_holder_examples::get_device_from_args;
use candle_holder_models::{GenerationConfig, GenerationParams};
use candle_holder_pipelines::TextGenerationPipeline;
use candle_holder_tokenizers::Message;

fn main() -> Result<()> {
    let device = get_device_from_args()?;
    println!("Device: {:?}", device);

    let pipeline =
        TextGenerationPipeline::new("meta-llama/Meta-Llama-3.1-8B-Instruct", &device, None, None)?;

    let generations = pipeline.run(
        vec![Message::user("How much is 2 + 2?")],
        Some(GenerationParams {
            generation_config: Some(GenerationConfig {
                do_sample: true,
                max_new_tokens: Some(256),
                top_p: Some(0.9),
                top_k: None,
                temperature: 0.6,
                ..GenerationConfig::default()
            }),
            ..Default::default()
        }),
    )?;

    println!("`pipeline.run` results: {:?}", generations);

    let generations = pipeline.run_batch(
        vec![
            vec![Message::user("How much is 2 + 2?")],
            vec![Message::user("How much is 2 x 3?")],
        ],
        Some(GenerationParams {
            generation_config: Some(GenerationConfig {
                do_sample: true,
                max_new_tokens: Some(256),
                top_p: Some(0.9),
                top_k: None,
                temperature: 0.6,
                ..GenerationConfig::default()
            }),
            ..Default::default()
        }),
    )?;

    println!("`pipeline.run_batch` results: {:?}", generations);

    Ok(())
}
