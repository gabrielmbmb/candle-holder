use candle_core::Tensor;
use candle_holder::{Error, Result};
use candle_nn::{embedding, layer_norm, Dropout, Embedding, LayerNorm, Module, VarBuilder};

use super::config::RobertaConfig;
use crate::{model::ModelOutput, ForwardParams, PreTrainedModel, PretrainedConfig};

pub struct RobertaEmbeddings {
    pub word_embeddings: Embedding,
    pub position_embeddings: Option<Embedding>,
    pub token_type_embeddings: Embedding,
    pub layer_norm: LayerNorm,
    pub dropout: Dropout,
}

impl RobertaEmbeddings {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let (_bsize, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            // TODO: Proper absolute positions?
            let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
            let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
        }
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings, false)?;
        Ok(embeddings)
    }
}

pub struct Roberta {}

impl Roberta {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        Ok(Roberta {})
    }
}

pub struct RobertaModel {
    model: Roberta,
    config: RobertaConfig,
}

impl PreTrainedModel for RobertaModel {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self> {
        let config: RobertaConfig = serde_json::from_value(config)?;
        let model = Roberta::load(vb.pp("roberta"), &config)?;
        Ok(Self { model, config })
    }

    fn forward(&self, params: ForwardParams) -> Result<ModelOutput> {
        self.model.forward(
            params
                .get_input_ids()
                .ok_or(Error::MissingForwardParam("input_ids".to_string()))?,
            params
                .get_attention_mask()
                .ok_or(Error::MissingForwardParam("attention_mask".to_string()))?,
            params
                .get_token_type_ids()
                .ok_or(Error::MissingForwardParam("token_type_ids".to_string()))?,
        )
    }

    fn get_config(&self) -> &PretrainedConfig {
        &self.config.pretrained_config
    }
}
