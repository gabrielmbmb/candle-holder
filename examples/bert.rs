use anyhow::{Error, Result};
use candle_core::{Device, Tensor};
use candle_holder::model::PreTrainedModel;
use candle_holder::{BertForMaskedLM, BertTokenizer};

fn main() -> Result<()> {
    let device = Device::Cpu;
    // let device = Device::new_metal(0)?;

    // Load tokenizer and model
    let model = BertForMaskedLM::from_pretrained("gabrielmbmb/bert-test", &device, None)?;
    // let model = BertModel::from_pretrained("dslim/bert-base-NER", &device, None)?;
    let tokenizer = BertTokenizer::from_pretrained("gabrielmbmb/bert-test", None)?;

    let tokens = tokenizer
        .encode("Paris is the [MASK] of France", true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();

    // Tokenize text
    let token_ids = Tensor::new(tokens, &device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;

    println!("Token ids: {}", token_ids);

    // Inference
    let start = std::time::Instant::now();
    let ys = model.forward(&token_ids, &token_type_ids)?;
    println!("Took: {:?}", start.elapsed());
    println!("{}", ys);

    Ok(())
}
