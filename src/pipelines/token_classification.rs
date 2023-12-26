use anyhow::Result;
use candle_core::Device;
use tokenizers::Tokenizer;

use crate::{
    model::PreTrainedModel, AutoModelForTokenClassification, AutoTokenizer,
    FromPretrainedParameters,
};

pub struct TokenClassificationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Tokenizer,
    device: Device,
}

impl TokenClassificationPipeline {
    pub fn new<S: AsRef<str> + Copy>(
        identifier: S,
        device: &Device,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let model =
            AutoModelForTokenClassification::from_pretrained(identifier, device, params.clone())?;
        let tokenizer = AutoTokenizer::from_pretrained(identifier, params)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }
}
