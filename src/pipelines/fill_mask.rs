use anyhow::{Error, Result};
use candle_core::{error::zip, Device, Tensor, D};
use candle_nn::ops::softmax;

use crate::{
    model::PreTrainedModel,
    tokenizer::{BatchEncoding, Tokenizer},
    utils::FromPretrainedParameters,
    AutoModelForMaskedLM, AutoTokenizer, Padding,
};

pub type FillMaskResult = (f32, u32, String, String);

pub struct FillMaskOptions {
    pub top_k: usize,
}

impl Default for FillMaskOptions {
    fn default() -> Self {
        Self { top_k: 5 }
    }
}

pub struct FillMaskPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Box<dyn Tokenizer>,
    device: Device,
}

impl FillMaskPipeline {
    pub fn new<S: AsRef<str> + Copy>(
        identifier: S,
        device: &Device,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let model = AutoModelForMaskedLM::from_pretrained(identifier, device, params.clone())?;
        let tokenizer = AutoTokenizer::from_pretrained(identifier, params)?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    fn ensure_at_least_one_mask_token(&self, masked_index: &[Vec<u32>]) -> Result<()> {
        for (i, input_ids) in masked_index.iter().enumerate() {
            if input_ids.is_empty() {
                return Err(Error::msg(format!(
                    "Sentence {} doesn't contain any mask token id.",
                    i
                )));
            }
        }

        Ok(())
    }

    fn preprocess(&mut self, inputs: Vec<String>) -> Result<(BatchEncoding, Vec<Vec<u32>>)> {
        let mut encodings = self
            .tokenizer
            .encode(inputs, true, Some(Padding::Longest))?;
        encodings.to_device(&self.device)?;
        let masked_index = self.get_masked_index(encodings.get_input_ids())?;
        self.ensure_at_least_one_mask_token(&masked_index)?;
        Ok((encodings, masked_index))
    }

    fn postprocess(
        &self,
        output: &Tensor,
        encodings: BatchEncoding,
        masked_index: Vec<Vec<u32>>,
        top_k: usize,
    ) -> Result<Vec<Vec<Vec<FillMaskResult>>>> {
        let pad_token_id = self
            .tokenizer
            .get_pad_token_id()
            .ok_or_else(|| Error::msg("The tokenizer doesn't have a pad token id."))?;

        let input_ids = encodings.get_input_ids().to_vec2::<u32>()?;

        let mut results = Vec::new();
        for (indexes, input_ids) in masked_index.into_iter().zip(input_ids) {
            let single_mask = indexes.len() == 1;

            // Select the logits for the masked token
            let logits =
                output.index_select(&Tensor::new(indexes.clone(), &self.device)?, D::Minus2)?;

            // Apply softmax to the logits to get the probabilities
            let probs = softmax(&logits, D::Minus1)?;
            let probs = probs.to_vec3::<f32>()?;
            let probs = probs[0].clone();

            let mut sentence_results = Vec::new();
            for (_probs, index) in probs.into_iter().zip(indexes) {
                // For each mask token in the sentence, get the top k predictions
                let topk = topk(&_probs, top_k);

                let mut mask_results = Vec::new();
                for (score, token_id) in topk.into_iter() {
                    // Remove pad tokens from the input_ids
                    let mut input_ids = input_ids.clone();
                    input_ids.retain(|&token_id| token_id != pad_token_id);

                    // Replace the mask token with the predicted token
                    input_ids[index as usize] = token_id;
                    let token = self.tokenizer.decode(&[token_id], false)?;
                    let sentence = self.tokenizer.decode(&input_ids, single_mask)?;
                    mask_results.push((score, token_id, token, sentence));
                }
                sentence_results.push(mask_results);
            }
            results.push(sentence_results);
        }

        Ok(results)
    }

    fn get_masked_index(&self, input_ids: &Tensor) -> Result<Vec<Vec<u32>>> {
        let mask_token_id = self
            .tokenizer
            .get_mask_token_id()
            .ok_or_else(|| Error::msg("The tokenizer doesn't have a mask token id."))?;

        let batch_input_ids = input_ids.to_vec2::<u32>()?;

        let mut masked_index: Vec<Vec<u32>> = Vec::new();
        for input_ids in batch_input_ids.into_iter() {
            let indexes = input_ids
                .into_iter()
                .enumerate()
                .filter(|(_, item)| *item == mask_token_id)
                .map(|(idx, _)| idx as u32)
                .collect();
            masked_index.push(indexes);
        }

        Ok(masked_index)
    }

    pub fn run<I: Into<String>>(
        &mut self,
        input: I,
        options: Option<FillMaskOptions>,
    ) -> Result<Vec<Vec<FillMaskResult>>> {
        let options = options.unwrap_or_default();
        let (encodings, masked_index) = self.preprocess(vec![input.into()])?;
        let output = self.model.forward(&encodings)?;
        Ok(self.postprocess(&output, encodings, masked_index, options.top_k)?[0].clone())
    }

    pub fn run_batch<I: Into<String>>(
        &mut self,
        inputs: Vec<I>,
        options: Option<FillMaskOptions>,
    ) -> Result<Vec<Vec<Vec<FillMaskResult>>>> {
        let options = options.unwrap_or_default();
        let inputs: Vec<String> = inputs.into_iter().map(|x| x.into()).collect();
        let (encodings, masked_index) = self.preprocess(inputs)?;
        let output = self.model.forward(&encodings)?;
        self.postprocess(&output, encodings, masked_index, options.top_k)
    }
}

fn topk<N: PartialOrd + Clone>(vec: &[N], k: usize) -> Vec<(N, u32)> {
    if k == 0 {
        return vec![];
    }

    let mut indexed_vec: Vec<(N, u32)> = vec
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i as u32))
        .collect();
    indexed_vec.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    indexed_vec.truncate(k);
    indexed_vec
}