use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::bail;
use crate::config::PretrainedConfig;
use crate::error::Result;
use crate::model_utils::DynamicCache;
use crate::models::bert::{
    BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification, BertModel,
    BERT_DTYPE,
};

use crate::models::llama::modeling::{LlamaModel, LLAMA_DTYPE};
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

                let model: Result<Box<dyn PreTrainedModel>> = match model_type {
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

pub trait PreTrainedModel {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self>
    where
        Self: Sized;
    fn config(&self) -> &PretrainedConfig;
    fn forward(&self, params: ForwardParams) -> Result<Tensor>;
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
