use anyhow::{Error, Result};
use candle_core::{Device, D};
use candle_nn::ops::softmax;
use tokenizers::Tokenizer;

use crate::{
    model_factories::PreTrainedModel, tokenizer::TensorEncoding, AutoModelForSequenceClassification,
};

pub enum ClassificationFunction {
    Softmax,
    Sigmoid,
    None,
}

pub struct TextClassificationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Tokenizer,
    device: Device,
}

impl TextClassificationPipeline {
    pub fn new<S: AsRef<str> + Copy>(identifier: S, device: &Device) -> Result<Self> {
        let model = AutoModelForSequenceClassification::from_pretrained(identifier, device, None)?;
        let tokenizer = Tokenizer::from_pretrained(identifier, None).unwrap();
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    pub fn forward<I: AsRef<str>>(
        &self,
        input: I,
        add_special_tokens: bool,
    ) -> Result<Vec<(String, f32)>> {
        // Preprocessing
        let encoding = self
            .tokenizer
            .encode(input.as_ref(), add_special_tokens)
            .map_err(Error::msg)?;
        let input_ids = encoding.get_ids_tensor(&self.device)?;
        let token_type_ids = encoding.get_type_ids_tensor(&self.device)?;

        // Forward
        let output = self.model.forward(&input_ids, &token_type_ids)?;
        let scores = softmax(&output, D::Minus1)?.squeeze(0)?.to_vec1::<f32>()?;

        // Postprocessing
        let config = self.model.config()?;
        let mut scores = scores
            .iter()
            .enumerate()
            .map(move |(i, score)| {
                let label = config["id2label"][i.to_string()].as_str().unwrap();
                (label.to_string(), *score)
            })
            .collect::<Vec<(String, f32)>>();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(scores)
    }
}
