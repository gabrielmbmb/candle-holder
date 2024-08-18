use anyhow::{Error, Result};
use candle_holder_examples::get_device_from_args;
use candle_holder_models::{config::GenerationConfig, LlamaForCausalLM, PreTrainedModel};
use candle_holder_tokenizers::LlamaTokenizer;

fn main() -> Result<()> {
    let device = get_device_from_args()?;
    println!("Device: {:?}", device);

    let tokenizer = LlamaTokenizer::from_pretrained("meta-llama/Meta-Llama-3-8B", None, None)?;
    let model =
        LlamaForCausalLM::from_pretrained("meta-llama/Meta-Llama-3-8B", &device, None, None)?;

    let mut encodings = tokenizer
        .encode(vec!["Hello, how are you?".to_string()], true, None)
        .map_err(Error::msg)?;

    encodings.to_device(&device)?;

    let start = std::time::Instant::now();

    let input_ids = encodings.get_input_ids().clone();

    let generation = model.generate(
        &input_ids,
        Some(GenerationConfig {
            do_sample: true,
            top_p: Some(1.0),
            top_k: None,
            temperature: 0.7,
            max_new_tokens: Some(500),
            ..GenerationConfig::default()
        }),
        Some(1234),
    )?;

    let sequence = &generation[0][..];
    let text = tokenizer.decode(sequence, true)?;

    println!("Generated text: {}", text);
    println!("Took: {:?}", start.elapsed());

    Ok(())
}
