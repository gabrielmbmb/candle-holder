use serde::{Deserialize, Serialize};

use crate::config::PretrainedConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    Relu,
    Silu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
    RelativeKey,
    RelativeKeyQuery,
}

#[derive(Serialize, Deserialize)]
pub struct RobertaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub pad_token_id: usize,
    #[serde(default)]
    pub bos_token_id: usize,
    #[serde(default)]
    pub eos_token_id: usize,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub is_decoder: bool,
    #[serde(default)]
    pub use_cache: bool,
    pub classifier_dropout: Option<f32>,
    pub model_type: Option<String>,

    #[serde(flatten, default)]
    pub pretrained_config: PretrainedConfig,
}

impl Default for RobertaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            position_embedding_type: PositionEmbeddingType::Absolute,
            is_decoder: false,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("roberta".to_string()),
            pretrained_config: PretrainedConfig::default(),
        }
    }
}
