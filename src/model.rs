use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::config::PretrainedConfig;
use crate::model_utils::DynamicCache;
use crate::models::bert::{
    BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification, BertModel,
    BERT_DTYPE,
};

use crate::models::llama::modeling::{LlamaModel, LLAMA_DTYPE};
use crate::tokenizer::BatchEncoding;
use crate::utils::{from_pretrained, FromPretrainedParameters};
use crate::LlamaForCausalLM;

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
                let config = model_info.config()?;
                let vb = model_info.vb($dtype, device)?;
                Self::load(vb, config)
            }
        }
    };
}

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
                let config = model_info.config()?;
                let model_type = config["model_type"].as_str().unwrap();

                let model: Result<Box<dyn PreTrainedModel>, anyhow::Error> = match model_type {
                    $(
                        $model_type => {
                            let vb = model_info.vb($dtype, device)?;
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

pub trait PreTrainedModel {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self>
    where
        Self: Sized;
    fn config(&self) -> &PretrainedConfig;
    fn forward(&self, encodings: &BatchEncoding) -> Result<Tensor>;
    fn forward_with_cache(
        &self,
        encodings: &BatchEncoding,
        index_pos: usize,
        cache: &mut DynamicCache,
    ) -> Result<Tensor> {
        unimplemented!("forward_with_cache not implemented")
    }
}

pub struct AutoModel {}

impl_auto_model_from_pretrained_method!(
    AutoModel,
    ("bert", BertModel, BERT_DTYPE),
    ("llama", LlamaModel, LLAMA_DTYPE)
);

pub struct AutoModelForSequenceClassification {}

impl_auto_model_from_pretrained_method!(
    AutoModelForSequenceClassification,
    ("bert", BertForSequenceClassification, BERT_DTYPE)
);

pub struct AutoModelForTokenClassification {}

impl_auto_model_from_pretrained_method!(
    AutoModelForTokenClassification,
    ("bert", BertForTokenClassification, BERT_DTYPE)
);

pub struct AutoModelForMaskedLM {}

impl_auto_model_from_pretrained_method!(
    AutoModelForMaskedLM,
    ("bert", BertForMaskedLM, BERT_DTYPE)
);

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
