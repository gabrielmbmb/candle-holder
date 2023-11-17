use super::factories::PreTrainedModel;
use anyhow::Result;
use candle_core::{DType, Module, Tensor};
use candle_nn::{embedding, layer_norm, Dropout, Embedding, LayerNorm, VarBuilder};
use serde::Deserialize;

pub const ROBERTA_DTYPE: DType = DType::F32;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RobertaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    // pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    // #[serde(default)]
    // pub position_embedding_type: PositionEmbeddingType,
    // #[serde(default)]
    pub use_cache: bool,
    pub classifier_dropout: Option<f64>,
    pub model_type: Option<String>,
}

pub struct RobertaEmbeddings {
    pub word_embeddings: Embedding,
    pub position_embeddings: Option<Embedding>,
    pub token_type_embeddings: Embedding,
    pub layer_norm: LayerNorm,
    pub dropout: Dropout,
}

impl RobertaEmbeddings {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> candle_core::Result<Self> {
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

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
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

pub struct RobertaModel {
    pub embeddings: RobertaEmbeddings,
}

impl RobertaModel {
    pub fn load(vb: VarBuilder, config: &RobertaConfig) -> Result<Self> {
        let embeddings = RobertaEmbeddings::load(vb.pp("embeddings"), config)?;
        Ok(Self { embeddings })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        Ok(embedding_output)
    }
}

pub struct PreTrainedRobertaModel {
    model: RobertaModel,
}

impl PreTrainedModel for PreTrainedRobertaModel {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self> {
        let config: RobertaConfig = serde_json::from_value(config)?;
        let model = RobertaModel::load(vb, &config)?;
        Ok(Self { model })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids, token_type_ids)
    }
}
