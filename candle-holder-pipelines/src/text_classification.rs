use std::sync::Arc;

use candle_core::{DType, Device, Tensor, D};
use candle_holder::{Error, FromPretrainedParameters, Result};
use candle_holder_models::{
    AutoModelForSequenceClassification, ForwardParams, PreTrainedModel, ProblemType,
};
use candle_holder_tokenizers::{AutoTokenizer, BatchEncoding, Padding, PaddingOptions, Tokenizer};
use candle_nn::ops::{sigmoid, softmax};

/// A pipeline for doing text classification.
pub struct TextClassificationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Arc<dyn Tokenizer>,
    device: Device,
}

impl TextClassificationPipeline {
    /// Creates a new `TextClassificationPipeline`.
    ///
    /// # Arguments
    ///
    /// * `identifier` - The repository id of the model to load.
    /// * `device` - The device to run the model on.
    /// * `params` - Optional parameters to specify the revision, user agent, and auth token.
    ///
    /// # Returns
    ///
    /// The `TextClassificationPipeline` instance.
    pub fn new<S: AsRef<str> + Copy>(
        identifier: S,
        device: &Device,
        dtype: Option<DType>,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let model = AutoModelForSequenceClassification::from_pretrained(
            identifier,
            device,
            dtype,
            params.clone(),
        )?;
        let tokenizer = AutoTokenizer::from_pretrained(identifier, None, params)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    fn preprocess(&self, inputs: Vec<String>) -> Result<BatchEncoding> {
        let mut encodings = self.tokenizer.encode(
            inputs,
            true,
            Some(PaddingOptions::new(Padding::Longest, None)),
        )?;
        encodings.to_device(&self.device)?;
        Ok(encodings)
    }

    fn postprocess(
        &self,
        logits: &Tensor,
        top_k: Option<usize>,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let config = self.model.get_config();
        // TODO: move this to `new` method to avoid cloning every time?
        let id2label = config
            .get_id2label()
            .ok_or_else(|| Error::msg("id2label not found in model config"))?;

        let problem_type = config.get_problem_type();
        let scores = {
            if *problem_type == ProblemType::MultiLabelClassification || config.num_labels() == 1 {
                sigmoid(logits)?
            } else if *problem_type == ProblemType::SingleLabelClassification
                || config.num_labels() > 1
            {
                softmax(logits, D::Minus1)?
            } else {
                logits.clone()
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

    /// Classifies a single sequence.
    ///
    /// # Arguments
    ///
    /// * `input` - The input sequence to classify.
    /// * `candidate_labels` - The candidate labels to classify the input against.
    /// * `options` - Optional parameters for the pipeline..
    ///
    /// # Returns
    ///
    /// A list containing the predicted label and the confidence score.
    pub fn run<I: Into<String>>(
        &self,
        input: I,
        top_k: Option<usize>,
    ) -> Result<Vec<(String, f32)>> {
        let encodings = self.preprocess(vec![input.into()])?;
        let output = self.model.forward(ForwardParams::from(&encodings))?;
        let logits = output.get_logits().unwrap();
        Ok(self.postprocess(logits, top_k)?[0].clone())
    }

    /// Classifies a list of sequences.
    ///
    /// # Arguments
    ///
    /// * `input` - The input sequence to classify.
    /// * `candidate_labels` - The candidate labels to classify the input against.
    /// * `options` - Optional parameters for the pipeline..
    ///
    /// # Returns
    ///
    /// A list containing the predicted label and the confidence score for each sequence.
    pub fn run_batch<I: Into<String>>(
        &self,
        inputs: Vec<I>,
        top_k: Option<usize>,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let inputs: Vec<String> = inputs.into_iter().map(|x| x.into()).collect();
        let encodings = self.preprocess(inputs)?;
        let output = self.model.forward(ForwardParams::from(&encodings))?;
        let logits = output.get_logits().unwrap();
        self.postprocess(logits, top_k)
    }
}
