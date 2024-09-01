use anyhow::{Error, Result};
use candle_holder_examples::Cli;
use candle_holder_models::{
    AutoModelForCausalLM, GenerationConfig, GenerationParams, TextStreamer, TokenStreamer,
};
use candle_holder_tokenizers::{AutoTokenizer, Message};
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

fn main() -> Result<()> {
    let args = GenerationCli::parse();

    let device = args.base.get_device()?;
    println!("Device: {:?}", device);

    let tokenizer = AutoTokenizer::from_pretrained(args.model.clone(), None, None)?;
    let model = AutoModelForCausalLM::from_pretrained(args.model, &device, None, None)?;

    let mut encodings = if args.apply_chat_template {
        tokenizer
            .apply_chat_template_and_encode(vec![Message::user(args.prompt)], true)
            .map_err(Error::msg)?
    } else {
        tokenizer
            .encode(vec![args.prompt], true, None)
            .map_err(Error::msg)?
    };
    encodings.to_device(&device)?;

    let start = std::time::Instant::now();

    let token_streamer: Box<dyn TokenStreamer> = Box::new(TextStreamer::new(
        &tokenizer,
        args.apply_chat_template,
        true,
    ));

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
            tokenizer: Some(&tokenizer),
            token_streamer: Some(token_streamer),
            ..Default::default()
        },
    )?;

    println!("\nTook: {:?}", start.elapsed());

    Ok(())
}
