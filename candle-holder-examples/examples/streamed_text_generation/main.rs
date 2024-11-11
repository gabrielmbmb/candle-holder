#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::sync::Arc;

use anyhow::{Error, Result};
use candle_holder_examples::Cli;
use candle_holder_models::{
    AutoModelForCausalLM, GenerationConfig, GenerationParams, TextStreamer
};
use candle_holder_tokenizers::{AutoTokenizer, BatchEncoding, Message, Tokenizer};
use clap::Parser;

#[derive(Debug, Parser)]
pub struct GenerationCli {
    #[command(flatten)]
    pub base: Cli,

    #[arg(long, default_value = "meta-llama/Meta-Llama-3.1-8B-Instruct")]
    pub model: String,

    #[arg(long, default_value = "0.6")]
    pub temperature: f64,

    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    #[arg(long, default_value = "50")]
    pub top_k: usize,

    #[arg(long, default_value = "1024")]
    pub max_new_tokens: usize,

    #[arg(long)]
    pub system_prompt: Option<String>,

    #[arg(long, required = true)]
    pub prompt: String,

    #[arg(long, default_value = "false")]
    pub apply_chat_template: bool,
}

fn encode(
    tokenizer: Arc<dyn Tokenizer>,
    prompt: String,
    system_prompt: Option<String>,
    apply_chat_template: bool,
    device: &candle_core::Device,
) -> Result<BatchEncoding> {
    let mut encodings = if apply_chat_template {
        let mut messages = vec![Message::user(prompt)];
        if let Some(system_prompt) = system_prompt {
            messages.insert(0, Message::system(system_prompt))
        }
        tokenizer
            .apply_chat_template_and_encode(messages, true)
            .map_err(Error::msg)?
    } else {
        tokenizer
            .encode(vec![prompt], true, None)
            .map_err(Error::msg)?
    };
    encodings.to_device(device)?;
    Ok(encodings)
}

fn main() -> Result<()> {
    let args = GenerationCli::parse();

    let device = args.base.get_device()?;
    println!("Device: {:?}", device);

    let tokenizer = AutoTokenizer::from_pretrained(args.model.clone(), None, None)?;
    let model = AutoModelForCausalLM::from_pretrained(args.model, &device, None, None)?;

    let start = std::time::Instant::now();

    let token_streamer = Box::new(TextStreamer::new(
        tokenizer.clone(),
        args.apply_chat_template,
        true,
    ));

    let encodings = encode(
        tokenizer.clone(),
        args.prompt,
        args.system_prompt,
        args.apply_chat_template,
        &device,
    )?;

    let input_ids = encodings.get_input_ids();

    model.generate(
        input_ids,
        GenerationParams {
            generation_config: Some(GenerationConfig {
                do_sample: true,
                top_p: Some(args.top_p),
                top_k: Some(args.top_k),
                temperature: args.temperature,
                max_new_tokens: Some(args.max_new_tokens),
                ..GenerationConfig::default()
            }),
            tokenizer: Some(tokenizer.clone()),
            token_streamer: Some(token_streamer),
            ..Default::default()
        },
    )?;

    println!("\nTook: {:?}", start.elapsed());

    Ok(())
}
