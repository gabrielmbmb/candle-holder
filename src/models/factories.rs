use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use super::{
    bert::{PreTrainedBertModel, BERT_DTYPE},
    roberta::{PreTrainedRobertaModel, ROBERTA_DTYPE},
};

use crate::{
    utils::{from_pretrained, FromPretrainedParameters},
    BertForSequenceClassification,
};

macro_rules! impl_from_pretrained_method {
    ($factory_struct:ident, $(($model_type:expr, $model_struct:ident, $dtype:expr)), *) => {
        impl $factory_struct {
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                device: &Device,
                params: Option<FromPretrainedParameters>,
            ) -> Result<Box<dyn PreTrainedModel>> {
                let model_info = from_pretrained(repo_id, params)?;
                let config = model_info.config()?;

                let model: Box<dyn PreTrainedModel> = match config["model_type"].as_str().unwrap() {
                    $(
                        $model_type => {
                            let vb = model_info.vb($dtype, device)?;
                            Box::new($model_struct::load(vb, config).unwrap())
                        },
                    )*
                    _ => panic!("")
                };

                Ok(model)
            }
        }
    };
}

pub trait PreTrainedModel {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self>
    where
        Self: Sized;
    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor>;
}

pub struct AutoModel {}

impl_from_pretrained_method!(
    AutoModel,
    ("bert", PreTrainedBertModel, BERT_DTYPE),
    ("roberta", PreTrainedRobertaModel, ROBERTA_DTYPE)
);

pub struct AutoModelForSequenceClassification {}

impl_from_pretrained_method!(
    AutoModelForSequenceClassification,
    ("bert", BertForSequenceClassification, BERT_DTYPE)
);
