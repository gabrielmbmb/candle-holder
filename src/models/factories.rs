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

pub trait PreTrainedModel {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self>
    where
        Self: Sized;
    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor>;
}

// TODO: can we have a macro for the `from_pretrained` method?

pub struct AutoModel {}

impl AutoModel {
    /// Load a pretrained model from the Hugging Face Hub.
    ///
    /// # Arguments
    ///
    /// * `repo_id` - The model id of the model to load.
    /// * `device` - The device to load the model on.
    pub fn from_pretrained<S: AsRef<str>>(
        repo_id: S,
        device: &Device,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Box<dyn PreTrainedModel>> {
        let model_info = from_pretrained(repo_id, params)?;
        let config = model_info.config()?;

        let model: Box<dyn PreTrainedModel> = match config["model_type"].as_str().unwrap() {
            "bert" => {
                let vb = model_info.vb(BERT_DTYPE, device)?;
                Box::new(PreTrainedBertModel::load(vb, config).unwrap())
            }
            "roberta" => {
                let vb = model_info.vb(ROBERTA_DTYPE, device)?;
                Box::new(PreTrainedRobertaModel::load(vb, config).unwrap())
            }
            _ => panic!("Model type not supported"),
        };

        Ok(model)
    }
}

pub struct AutoModelForSequenceClassification {}

impl AutoModelForSequenceClassification {
    /// Loads a pretrained model for sequence classification from the Hugging Face Hub.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model id of the model to load.
    /// * `device` - The device to load the model on.
    pub fn from_pretrained<S: AsRef<str>>(
        repo_id: S,
        device: &Device,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Box<dyn PreTrainedModel>> {
        let model_info = from_pretrained(repo_id, params)?;
        let config = model_info.config()?;

        let model: Box<dyn PreTrainedModel> = match config["model_type"].as_str().unwrap() {
            "bert" => {
                let vb = model_info.vb(BERT_DTYPE, device)?;
                Box::new(BertForSequenceClassification::load(vb, config).unwrap())
            }
            _ => panic!("Model type not supported"),
        };

        Ok(model)
    }
}
