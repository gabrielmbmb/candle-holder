use candle_core::{Device, Tensor};
use candle_holder::{bail, utils::FromPretrainedParameters, Result};
use candle_nn::VarBuilder;

use crate::config::PretrainedConfig;
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
    fn config(&self) -> &PretrainedConfig;
    fn forward(&self, params: ForwardParams) -> Result<Tensor>;
    fn generate(&self, _params: ForwardParams) -> Result<Tensor> {
        unimplemented!("generate method not implemented for this model");
    }
}

/// Implement `from_pretrained` method for a model struct.
#[macro_export]
macro_rules! impl_from_pretrained_method {
    ($model_struct:ident, $dtype:expr) => {
        impl $model_struct {
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                device: &Device,
                params: Option<FromPretrainedParameters>,
            ) -> Result<Self> {
                let model_info = from_pretrained(repo_id, params)?;
                let config = model_info
                    .get_config()
                    .expect("Model config not found. Cannot load the model.")
                    .clone();
                let vb = model_info.get_var_builder($dtype, device)?;
                Self::load(vb, config)
            }
        }
    };
}

/// Implement `from_pretrained` method for the `AutoModel` struct.
#[macro_export]
macro_rules! impl_auto_model_from_pretrained_method {
    ($auto_model_struct:ident, $(($model_type:expr, $model_struct:ident, $dtype:expr)), *) => {
        impl $auto_model_struct {
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                device: &Device,
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
                            let vb = model_info.get_var_builder($dtype, device)?;
                            Ok(Box::new($model_struct::load(vb, config)?))
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
    ("bert", BertModel, BERT_DTYPE),
    ("llama", LlamaModel, LLAMA_DTYPE)
);

/// Alows to automatically load a `PreTrainedModel` for sequence classification from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForSequenceClassification {}

impl_auto_model_from_pretrained_method!(
    AutoModelForSequenceClassification,
    ("bert", BertForSequenceClassification, BERT_DTYPE)
);

/// Alows to automatically load a `PreTrainedModel` for token classification from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForTokenClassification {}

impl_auto_model_from_pretrained_method!(
    AutoModelForTokenClassification,
    ("bert", BertForTokenClassification, BERT_DTYPE)
);

/// Alows to automatically load a `PreTrainedModel` for masked language modeling from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForMaskedLM {}

impl_auto_model_from_pretrained_method!(
    AutoModelForMaskedLM,
    ("bert", BertForMaskedLM, BERT_DTYPE)
);

/// Alows to automatically load a `PreTrainedModel` for causal language modeling from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForCausalLM {}

impl_auto_model_from_pretrained_method!(
    AutoModelForCausalLM,
    ("llama", LlamaForCausalLM, LLAMA_DTYPE)
);

// Bert
impl_from_pretrained_method!(BertModel, BERT_DTYPE);
impl_from_pretrained_method!(BertForSequenceClassification, BERT_DTYPE);
impl_from_pretrained_method!(BertForTokenClassification, BERT_DTYPE);
impl_from_pretrained_method!(BertForMaskedLM, BERT_DTYPE);

// Llama
impl_from_pretrained_method!(LlamaModel, LLAMA_DTYPE);
impl_from_pretrained_method!(LlamaForCausalLM, LLAMA_DTYPE);
