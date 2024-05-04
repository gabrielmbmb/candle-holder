use anyhow::{Error, Result};
use candle_core::{Device, Tensor, D};
use candle_nn::ops::{sigmoid, softmax};

use crate::{
    config::ProblemType,
    model::PreTrainedModel,
    tokenizer::{BatchEncoding, Tokenizer},
    utils::FromPretrainedParameters,
    AutoModelForSequenceClassification, AutoTokenizer, Padding,
};

pub struct TextClassificationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Box<dyn Tokenizer>,
    device: Device,
}

impl TextClassificationPipeline {
    pub fn new<S: AsRef<str> + Copy>(
        identifier: S,
        device: &Device,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let model = AutoModelForSequenceClassification::from_pretrained(
            identifier,
            device,
            params.clone(),
        )?;
        let tokenizer = AutoTokenizer::from_pretrained(identifier, params)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    fn preprocess(&mut self, inputs: Vec<String>) -> Result<BatchEncoding> {
        let mut encodings = self
            .tokenizer
            .encode(inputs, true, Some(Padding::Longest))?;
        encodings.to_device(&self.device)?;
        Ok(encodings)
    }

    fn postprocess(
        &self,
        model_outputs: Tensor,
        top_k: Option<usize>,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let config = self.model.config();
        // TODO: move this to `new` method to avoid cloning every time?
        let id2label = config
            .clone()
            .id2label
            .ok_or_else(|| Error::msg("id2label not found in model config"))?;

        let scores = {
            if config.problem_type == ProblemType::MultiLabelClassification
                || config.num_labels() == 1
            {
                sigmoid(&model_outputs)?
            } else if config.problem_type == ProblemType::SingleLabelClassification
                || config.num_labels() > 1
            {
                softmax(&model_outputs, D::Minus1)?
            } else {
                model_outputs
            }
        }
        .to_vec2::<f32>()?;
        let mut results = Vec::new();
        for inner in &scores {
            let mut scores_with_labels = inner
                .iter()
                .enumerate()
                .map(|(i, score)| {
                    let label = id2label.get(&i).unwrap();
                    (label.to_string(), *score)
                })
                .collect::<Vec<(String, f32)>>();
            scores_with_labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if let Some(top_k) = top_k {
                scores_with_labels.truncate(top_k);
            }
            results.push(scores_with_labels);
        }
        Ok(results)
    }
    pub fn run<I: Into<String>>(
        &mut self,
        input: I,
        top_k: Option<usize>,
    ) -> Result<Vec<(String, f32)>> {
        let encodings = self.preprocess(vec![input.into()])?;
        let output = self.model.forward(&encodings)?;
        Ok(self.postprocess(output, top_k)?[0].clone())
    }
    pub fn run_batch<I: Into<String>>(
        &mut self,
        inputs: Vec<I>,
        top_k: Option<usize>,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let inputs: Vec<String> = inputs.into_iter().map(|x| x.into()).collect();
        let encodings = self.preprocess(inputs)?;
        let output = self.model.forward(&encodings)?;
        self.postprocess(output, top_k)
    }
}
