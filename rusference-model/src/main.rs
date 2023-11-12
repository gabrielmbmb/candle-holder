use std::time::Instant;

use anyhow::{Error, Result};
use candle_core::{utils::metal_is_available, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

fn build_model_and_tokenizer(device: &Device) -> Result<(BertModel, Tokenizer)> {
    let repo = Repo::with_revision(
        "sentence-transformers/all-MiniLM-L6-v2".into(),
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
    let config: Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;

    let vb = VarBuilder::from_pth(&weights_filename, DTYPE, &device)?;
    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer))
}

fn main() -> Result<()> {
    let device = device(false)?;
    print!("Device: {:?} ", device);
    let (model, tokenizer) = build_model_and_tokenizer(&device)?;
    let tokens = tokenizer
        .encode("What's up?", true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;

    let start = Instant::now();
    for idx in 0..5 {
        let ys = model.forward(&token_ids, &token_type_ids)?;
        if idx == 0 {
            println!("{ys}");
        }
    }
    println!("Took: {:?}", start.elapsed());
    Ok(())
}
