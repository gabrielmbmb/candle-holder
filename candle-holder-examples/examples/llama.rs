use anyhow::{Error, Result};
use candle_holder_examples::get_device_from_args;
use candle_holder_models::{
    config::GenerationConfig,
    generation::token_streamer::{TextStreamer, TokenStreamer},
    AutoModelForCausalLM,
};
use candle_holder_tokenizers::LlamaTokenizer;

fn main() -> Result<()> {
    let device = get_device_from_args()?;
    println!("Device: {:?}", device);

    let tokenizer = LlamaTokenizer::from_pretrained("meta-llama/Meta-Llama-3-8B", None, None)?;
    let model =
        AutoModelForCausalLM::from_pretrained("meta-llama/Meta-Llama-3-8B", &device, None, None)?;

    let mut encodings = tokenizer
        .encode(vec!["Once upon a time, in a small village nestled deep within the misty mountains, there lived an old clockmaker named Tobias.".to_string()], true, None)
        .map_err(Error::msg)?;
    encodings.to_device(&device)?;

    let start = std::time::Instant::now();

    let token_streamer: Box<dyn TokenStreamer> = Box::new(TextStreamer::new(&tokenizer, true));

    let input_ids = encodings.get_input_ids();
    let generation = model.generate(
        input_ids,
        Some(GenerationConfig {
            do_sample: true,
            top_p: Some(0.9),
            top_k: Some(50),
            temperature: 0.7,
            max_new_tokens: Some(5000),
            ..GenerationConfig::default()
        }),
        Some(token_streamer),
        Some(1234),
    )?;

    let sequence = &generation[0][..];
    tokenizer.decode(sequence, true)?;

    println!("Took: {:?}", start.elapsed());

    Ok(())
}
