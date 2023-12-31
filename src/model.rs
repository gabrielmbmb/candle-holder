use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::config::PretrainedConfig;
use crate::models::bert::{BertModel, BERT_DTYPE};
use crate::BertForTokenClassification;

use crate::{
    utils::{from_pretrained, FromPretrainedParameters},
    BertForSequenceClassification,
};

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
    fn config(&self) -> PretrainedConfig;
    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor>;
}

pub struct AutoModel {}

impl_auto_model_from_pretrained_method!(AutoModel, ("bert", BertModel, BERT_DTYPE));

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

// Implement `from_pretrained` method for each model
impl_from_pretrained_method!(BertModel, BERT_DTYPE);
impl_from_pretrained_method!(BertForSequenceClassification, BERT_DTYPE);
impl_from_pretrained_method!(BertForTokenClassification, BERT_DTYPE);
