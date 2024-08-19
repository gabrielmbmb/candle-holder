use candle_core::{DType, Device, IndexOp, Tensor};
use candle_holder::Error;
use candle_holder::{bail, utils::FromPretrainedParameters, Result};
use candle_nn::VarBuilder;

use crate::config::{GenerationConfig, PretrainedConfig};
use crate::from_pretrained::from_pretrained;
use crate::generation::LogitSampler;
use crate::models::bert::{
    BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification, BertModel,
    BERT_DTYPE,
};
use crate::models::llama::modeling::{LlamaForCausalLM, LlamaModel, LLAMA_DTYPE};
use crate::utils::cache::DynamicCache;

/// Parameters for the `forward` method of a `PreTrainedModel`.
pub struct ForwardParams<'a> {
    pub input_ids: Option<&'a Tensor>,
    pub attention_mask: Option<&'a Tensor>,
    pub token_type_ids: Option<&'a Tensor>,
    pub position_ids: Option<&'a Tensor>,
    pub cache: Option<&'a mut DynamicCache>,
}

impl<'a> ForwardParams<'a> {
    pub fn new(
        input_ids: Option<&'a Tensor>,
        attention_mask: Option<&'a Tensor>,
        token_type_ids: Option<&'a Tensor>,
        position_ids: Option<&'a Tensor>,
        cache: Option<&'a mut DynamicCache>,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            cache,
        }
    }

    pub fn get_input_ids(&self) -> Option<&'a Tensor> {
        self.input_ids
    }

    pub fn get_attention_mask(&self) -> Option<&'a Tensor> {
        self.attention_mask
    }

    pub fn get_token_type_ids(&self) -> Option<&'a Tensor> {
        self.token_type_ids
    }

    pub fn get_position_ids(&self) -> Option<&'a Tensor> {
        self.position_ids
    }

    pub fn get_cache(&mut self) -> Option<&mut DynamicCache> {
        self.cache.as_deref_mut()
    }
}

impl<'a> Default for ForwardParams<'a> {
    fn default() -> Self {
        Self::new(None, None, None, None, None)
    }
}

#[cfg(feature = "tokenizers")]
impl<'a> From<&'a candle_holder_tokenizers::BatchEncoding> for ForwardParams<'a> {
    fn from(encodings: &'a candle_holder_tokenizers::BatchEncoding) -> Self {
        Self::new(
            Some(encodings.get_input_ids()),
            Some(encodings.get_attention_mask()),
            Some(encodings.get_token_type_ids()),
            None,
            None,
        )
    }
}

/// Trait for a pre-trained model.
pub trait PreTrainedModel {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self>
    where
        Self: Sized;
    fn load_with_generation_config(
        _vb: VarBuilder,
        _config: serde_json::Value,
        _generation_config: Option<GenerationConfig>,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        unimplemented!("`load_with_generation_config` method not implemented for this model");
    }
    fn get_generation_config(&self) -> &GenerationConfig {
        unimplemented!("`get_generation_config` method not implemented for this model");
    }
    fn get_config(&self) -> &PretrainedConfig;
    fn forward(&self, params: ForwardParams) -> Result<Tensor>;

    /// Generates a completion of the input sequences.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - The input sequences.
    /// * `generation_config` - Optional generation configuration. If not provided, the model's
    ///  default generation configuration will be used (if available). Otherwise, default
    ///  generation configuration will be used.
    /// * `seed` - Optional seed for the random number generator.
    ///
    /// # Returns
    ///
    /// A vector containing vectors of token ids for each input sequence.
    fn generate(
        &self,
        input_ids: &Tensor,
        generation_config: Option<GenerationConfig>,
        seed: Option<u64>,
    ) -> Result<Vec<Vec<u32>>> {
        let generation_config =
            generation_config.unwrap_or_else(|| self.get_generation_config().clone());

        let mut output = input_ids.to_vec2::<u32>()?;
        let mut input_ids = input_ids.clone();
        let input_ids_dims = input_ids.dims2()?;

        // Calculate the number of max new tokens to be generated if `max_new_tokens` not provided
        let input_seq_len = input_ids.dims2()?.1;
        let max_new_tokens = generation_config
            .get_max_new_tokens()
            .unwrap_or_else(|| generation_config.get_max_length() - input_seq_len);

        let mut cache = if generation_config.get_use_cache() {
            Some(DynamicCache::new())
        } else {
            None
        };
        let mut sampling_config = LogitSampler::from_generation_config(generation_config, seed);
        // TODO: update to try to get from generation config first before failing
        let eos_token_id = self
            .get_config()
            .get_eos_token_id()
            .ok_or_else(|| Error::MissingSpecialTokenId("eos_token_id".to_string()))?;

        // Initialize a vector to store the next token ids for each sequence
        let num_sequences = input_ids_dims.0;
        let mut sequences_next_tokens: Vec<Vec<u32>> = vec![Vec::new(); num_sequences];
        let mut active_sequences = num_sequences;

        // TODO: if `generation_config.num_return_sequences>1` then we need to expand the
        // `input_ids` tensor to have `num_return_sequences` times the number of sequences

        // Generation loop
        for _index in 0..max_new_tokens {
            if active_sequences == 0 {
                break;
            }

            let logits = self.forward(ForwardParams {
                input_ids: Some(&input_ids),
                cache: cache.as_mut(),
                ..Default::default()
            })?;
            let dims = logits.dims3()?;

            let last_token_logits = logits.i((.., dims.1 - 1, ..))?;

            // TODO: apply repeat penalty, frequency penalty

            // Sample the next token for each sequence
            for i in 0..dims.0 {
                let seq_logits = last_token_logits.i((i, ..))?;
                let next_token_id = sampling_config.sample(&seq_logits)?;
                sequences_next_tokens[i].push(next_token_id);

                // TODO: check for other stop conditions
                if next_token_id == eos_token_id {
                    active_sequences -= 1;
                }
            }

            // Build the next `input_ids` vectors with the last token of each sequence
            let sequences_last_tokens = sequences_next_tokens
                .iter()
                .map(|seq| seq.last().unwrap().clone())
                .collect::<Vec<_>>();
            input_ids =
                Tensor::new(&sequences_last_tokens[..], input_ids.device())?.unsqueeze(0)?;
        }

        // Append the generated sequences to the input sequences
        for (i, seq) in sequences_next_tokens.iter().enumerate() {
            output[i].extend(seq);
        }
        Ok(output)
    }
}

/// Implement `from_pretrained` method for a model struct.
#[macro_export]
macro_rules! impl_from_pretrained_method {
    ($model_struct:ident, $default_dtype:expr, $load_generation_config:expr) => {
        impl $model_struct {
            /// Loads a model from the Hugging Face Hub.
            ///
            /// # Arguments
            ///
            /// * `identifier` - The repository id of the model to load.
            /// * `device` - The device to run the model on.
            /// * `dtype` - The numeric type in which the model parameters should be loaded.
            /// * `params` - Optional parameters to specify the revision, user agent, and auth token.
            ///
            /// # Returns
            ///
            /// The loaded model.
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                device: &Device,
                dtype: Option<DType>,
                params: Option<FromPretrainedParameters>,
            ) -> Result<Self> {
                let model_info = from_pretrained(repo_id, params)?;
                let config = model_info
                    .get_config()
                    .expect("Model config not found. Cannot load the model.")
                    .clone();
                let dtype = dtype.unwrap_or($default_dtype);
                let vb = model_info.get_var_builder(dtype, device)?;
                if $load_generation_config {
                    Self::load_with_generation_config(
                        vb,
                        config,
                        model_info.get_generation_config().cloned(),
                    )
                } else {
                    Self::load(vb, config)
                }
            }
        }
    };
}

/// Implement `from_pretrained` method for the `AutoModel` struct.
#[macro_export]
macro_rules! impl_auto_model_from_pretrained_method {
    ($auto_model_struct:ident, $(($model_type:expr, $model_struct:ident, $default_dtype:expr, $load_generation_config:expr)), *) => {
        impl $auto_model_struct {
            /// Loads a model from the Hugging Face Hub.
            ///
            /// # Arguments
            ///
            /// * `identifier` - The repository id of the model to load.
            /// * `device` - The device to run the model on.
            /// * `dtype` - The numeric type in which the model parameters should be loaded.
            /// * `params` - Optional parameters to specify the revision, user agent, and auth token.
            ///
            /// # Returns
            ///
            /// The loaded model.
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                device: &Device,
                dtype: Option<DType>,
                params: Option<FromPretrainedParameters>,
            ) -> Result<Box<dyn PreTrainedModel>> {
                let model_info = from_pretrained(repo_id, params)?;
                let config = model_info
                    .get_config()
                    .expect("Model config not found. Cannot load the model.")
                    .clone();
                let model_type = config["model_type"].as_str().unwrap();

                let model: Result<Box<dyn PreTrainedModel>> = match model_type {
                    $(
                        $model_type => {
                            let dtype = dtype.unwrap_or($default_dtype);
                            let vb = model_info.get_var_builder(dtype, device)?;
                            if $load_generation_config {
                                Ok(Box::new($model_struct::load(vb, config)?))
                            } else {
                                Ok(Box::new($model_struct::load_with_generation_config(vb, config, model_info.get_generation_config().cloned())?))
                            }
                        },
                    )*
                    _ => bail!(format!("Model '{}' type not supported", model_type)),
                };

                model
            }
        }
    };
}

/// Allows to automatically load a `PreTrainedModel` from a Hugging Face Hub repository.
#[derive(Debug)]
pub struct AutoModel {}

impl_auto_model_from_pretrained_method!(
    AutoModel,
    ("bert", BertModel, BERT_DTYPE, false),
    ("llama", LlamaModel, LLAMA_DTYPE, false)
);

/// Alows to automatically load a `PreTrainedModel` for sequence classification from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForSequenceClassification {}

impl_auto_model_from_pretrained_method!(
    AutoModelForSequenceClassification,
    ("bert", BertForSequenceClassification, BERT_DTYPE, false)
);

/// Alows to automatically load a `PreTrainedModel` for token classification from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForTokenClassification {}

impl_auto_model_from_pretrained_method!(
    AutoModelForTokenClassification,
    ("bert", BertForTokenClassification, BERT_DTYPE, false)
);

/// Alows to automatically load a `PreTrainedModel` for masked language modeling from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForMaskedLM {}

impl_auto_model_from_pretrained_method!(
    AutoModelForMaskedLM,
    ("bert", BertForMaskedLM, BERT_DTYPE, false)
);

/// Alows to automatically load a `PreTrainedModel` for causal language modeling from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForCausalLM {}

impl_auto_model_from_pretrained_method!(
    AutoModelForCausalLM,
    ("llama", LlamaForCausalLM, LLAMA_DTYPE, true)
);

// Bert
impl_from_pretrained_method!(BertModel, BERT_DTYPE, false);
impl_from_pretrained_method!(BertForSequenceClassification, BERT_DTYPE, false);
impl_from_pretrained_method!(BertForTokenClassification, BERT_DTYPE, false);
impl_from_pretrained_method!(BertForMaskedLM, BERT_DTYPE, false);

// Llama
impl_from_pretrained_method!(LlamaModel, LLAMA_DTYPE, false);
impl_from_pretrained_method!(LlamaForCausalLM, LLAMA_DTYPE, true);
