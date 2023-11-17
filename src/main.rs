use anyhow::{Error, Result};
use candle_core::{Device, Tensor};
use candle_holder::{models::factories::AutoModelForSequenceClassification, AutoModel};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Load tokenizer and model
    let tokenizer =
        Tokenizer::from_pretrained("bhadresh-savani/bert-base-uncased-emotion", None).unwrap();
    let model = AutoModelForSequenceClassification::from_pretrained(
        "bhadresh-savani/bert-base-uncased-emotion",
        &device,
        None,
    )?;
    let tokens = tokenizer
        .encode("What's up?", true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();

    // Tokenize text
    let token_ids = Tensor::new(tokens, &device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;

    // Inference
    let start = std::time::Instant::now();
    let ys = model.forward(&token_ids, &token_type_ids)?;
    println!("Took: {:?}", start.elapsed());
    println!("{}", ys);

    Ok(())
}
