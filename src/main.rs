use std::time::Instant;

use anyhow::{Error, Result};
use candle_core::{Device, Tensor};
use candle_holder::models::bert::{BertConfig, BertForSequenceClassification};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::DTYPE;
use hf_hub::{api::sync::Api, Repo, RepoType};
// use tokenizers::Tokenizer;
use tokenizers::tokenizer::Tokenizer;

fn build_model_and_tokenizer(
    device: &Device,
) -> Result<(BertForSequenceClassification, Tokenizer)> {
    let repo = Repo::with_revision(
        "bhadresh-savani/bert-base-uncased-emotion".into(),
        RepoType::Model,
        "main".into(),
    );
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("pytorch_model.bin")?;
        (config, tokenizer, weights)
    };

    let config = std::fs::read_to_string(config_filename)?;
    let config: BertConfig = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;

    let vb = VarBuilder::from_pth(&weights_filename, DTYPE, device)?;
    let model = BertForSequenceClassification::load(vb, &config)?;
    Ok((model, tokenizer))
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    println!("Device: {:?} ", device);
    let (model, tokenizer) = build_model_and_tokenizer(&device)?;
    let tokens = tokenizer
        .encode("What's up?", true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
    println!("Token ids: {}", token_ids);
    let token_type_ids = token_ids.zeros_like()?;

    let start = Instant::now();
    let ys = model.forward(&token_ids, &token_type_ids)?;
    println!("Took: {:?}", start.elapsed());
    Ok(())
}
