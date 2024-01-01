use anyhow::{Error, Result};
use candle_core::{Device, Tensor, D};
use candle_nn::ops::{sigmoid, softmax};
use tokenizers::{EncodeInput, Tokenizer};

use crate::{
    config::ProblemType, model::PreTrainedModel, utils::FromPretrainedParameters,
    AutoModelForSequenceClassification, AutoTokenizer,
};

pub struct TextClassificationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Tokenizer,
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

    fn preprocess<'s, E>(&self, inputs: Vec<E>) -> Result<(Tensor, Tensor)>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(Error::msg)?;

        let mut input_ids: Vec<Vec<u32>> = Vec::new();
        let mut token_type_ids: Vec<Vec<u32>> = Vec::new();

        for encoding in encodings {
            input_ids.push(encoding.get_ids().to_vec());
            token_type_ids.push(encoding.get_type_ids().to_vec());
        }

        let input_ids = Tensor::new(input_ids, &self.device)?;
        let token_type_ids = Tensor::new(token_type_ids, &self.device)?;

        Ok((input_ids, token_type_ids))
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
                scores_with_labels.truncate(top_k as usize);
            }
            results.push(scores_with_labels);
        }

        Ok(results)
    }

    pub fn run<I: AsRef<str>>(&self, input: I, top_k: Option<usize>) -> Result<Vec<(String, f32)>> {
        let (input_ids, token_type_ids) = self.preprocess(vec![input.as_ref()])?;
        let output = self.model.forward(&input_ids, &token_type_ids)?;
        Ok(self.postprocess(output, top_k)?[0].clone())
    }

    pub fn run_batch<'s, E>(
        &self,
        inputs: Vec<E>,
        top_k: Option<usize>,
    ) -> Result<Vec<Vec<(String, f32)>>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let (input_ids, token_type_ids) = self.preprocess(inputs)?;
        let output = self.model.forward(&input_ids, &token_type_ids)?;
        self.postprocess(output, top_k)
    }
}
