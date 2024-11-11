use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_holder::{FromPretrainedParameters, Result};
use candle_holder_models::{AutoModel, ForwardParams, PreTrainedModel};
use candle_holder_tokenizers::{AutoTokenizer, BatchEncoding, Padding, PaddingOptions, Tokenizer};
use serde::Deserialize;

/// The pooling strategy that will be used to pool the outputs of the model.
#[derive(Debug, Clone, Deserialize)]
pub enum Pooling {
    /// Mean pooling.
    #[serde(alias = "mean")]
    Mean,
    /// Use the CLS token.
    #[serde(alias = "cls")]
    Cls,
}

/// Options for the [`FeatureExtractionPipeline`].
pub struct FeatureExtractionOptions {
    /// The pooling strategy that will be used to pool the outputs of the model.
    pub pooling: Option<Pooling>,
    /// Whether to normalize the outputs.
    pub normalize: bool,
}

impl Default for FeatureExtractionOptions {
    fn default() -> Self {
        Self {
            pooling: Some(Pooling::Mean),
            normalize: true,
        }
    }
}

/// A pipeline for generating sentence embeddings from input texts.
pub struct FeatureExtractionPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Arc<dyn Tokenizer>,
    device: Device,
}

impl FeatureExtractionPipeline {
    /// Creates a new `FeatureExtractionPipeline`.
    ///
    /// # Arguments
    ///
    /// * `identifier` - The repository id of the model to load.
    /// * `device` - The device to run the model on.
    /// * `params` - Optional parameters to specify the revision, user agent, and auth token.
    ///
    /// # Returns
    ///
    /// The `ZeroShotClassificationPipeline` instance.
    pub fn new<S: AsRef<str> + Copy>(
        identifier: S,
        device: &Device,
        dtype: Option<DType>,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let model = AutoModel::from_pretrained(identifier, device, dtype, params.clone())?;
        let tokenizer = AutoTokenizer::from_pretrained(identifier, None, params)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    fn preprocess(&self, sequences: Vec<String>) -> Result<BatchEncoding> {
        let mut encodings = self.tokenizer.encode(
            sequences,
            true,
            Some(PaddingOptions::new(Padding::Longest, None)),
        )?;
        encodings.to_device(&self.device)?;
        Ok(encodings)
    }

    fn postprocess(
        &self,
        outputs: &Tensor,
        encodings: BatchEncoding,
        options: FeatureExtractionOptions,
    ) -> Result<Tensor> {
        let mut outputs = outputs.clone();

        if let Some(pooling) = options.pooling {
            outputs = match pooling {
                Pooling::Mean => mean_pooling(&outputs, encodings.get_attention_mask())?,
                Pooling::Cls => outputs.i((.., 0))?,
            };
        }

        if options.normalize {
            outputs = outputs.broadcast_div(&outputs.powf(2.)?.sum(1)?.sqrt()?.unsqueeze(1)?)?;
        }

        Ok(outputs)
    }

    /// Generates a sentence embedding for the input text.
    ///
    /// # Arguments
    ///
    /// * `input` - The input text.
    /// * `options` - Optional feature extraction options.
    ///
    /// # Returns
    ///
    /// The sentence embedding tensor.
    pub fn run<I: Into<String>>(
        &self,
        input: I,
        options: Option<FeatureExtractionOptions>,
    ) -> Result<Tensor> {
        let options = options.unwrap_or_default();
        let inputs = vec![input.into()];
        let encodings = self.preprocess(inputs)?;
        let outputs = self.model.forward(ForwardParams::from(&encodings))?;
        let last_hidden_states = outputs.get_last_hidden_state().unwrap();
        Ok(self
            .postprocess(last_hidden_states, encodings, options)?
            .i(0)?)
    }

    /// Generates sentence embeddings for a batch of input texts.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The input texts.
    /// * `options` - Optional feature extraction options.
    ///
    /// # Returns
    ///
    /// A tensor containing the sentence embeddings.
    pub fn run_batch<I: Into<String>>(
        &self,
        inputs: Vec<I>,
        options: Option<FeatureExtractionOptions>,
    ) -> Result<Tensor> {
        let options = options.unwrap_or_default();
        let inputs: Vec<String> = inputs.into_iter().map(|x| x.into()).collect();
        let encodings = self.preprocess(inputs)?;
        let outputs = self.model.forward(ForwardParams::from(&encodings))?;
        let last_hidden_states = outputs.get_last_hidden_state().unwrap();
        self.postprocess(last_hidden_states, encodings, options)
    }
}

/// Computes the mean pooling of the outputs.
///
/// # Arguments
///
/// * `outputs` - The model outputs.
/// * `attention_mask` - The attention mask.
///
/// # Returns
///
/// The mean pooled tensor.
fn mean_pooling(outputs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let input_mask_expanded = attention_mask
        .unsqueeze(D::Minus1)?
        .repeat((1, 1, outputs.dims3()?.2))?
        .to_dtype(outputs.dtype())?;
    Ok(outputs
        .mul(&input_mask_expanded)?
        .sum(1)?
        .broadcast_div(&input_mask_expanded.sum(1)?.clamp(f64::MIN, f64::MAX)?)?)
}
