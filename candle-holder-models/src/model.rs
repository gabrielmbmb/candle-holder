use candle_core::{DType, Device, Tensor};
use candle_holder::{bail, utils::FromPretrainedParameters, Result};
use candle_nn::VarBuilder;

use crate::config::{GenerationConfig, PretrainedConfig};
use crate::from_pretrained::from_pretrained;
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
    fn config(&self) -> &PretrainedConfig;
    fn forward(&self, params: ForwardParams) -> Result<Tensor>;
    fn generate(
        &self,
        input_ids: &Tensor,
        generation_config: Option<GenerationConfig>,
    ) -> Result<Tensor> {
        let generation_config =
            generation_config.unwrap_or_else(|| self.get_generation_config().clone());

        let input_seq_len = input_ids.dims2()?.1;

        // Calculate the number of max new tokens to be generated
        let max_new_tokens = generation_config
            .get_max_new_tokens()
            .unwrap_or_else(|| generation_config.get_max_length() - input_seq_len);

        Ok(Tensor::new(&[[1u32]], &Device::Cpu)?)
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
