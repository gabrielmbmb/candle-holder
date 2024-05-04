use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::PretrainedConfig;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Silu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeScalingStrategy {
    Linear,
    Dynamic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub strategy: RopeScalingStrategy,
    pub factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub initializer_range: f64,
    pub rms_norm_eps: f64,
    pub use_cache: Option<bool>,
    pub pad_token_id: Option<usize>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub pretraining_tp: Option<usize>,
    pub tie_word_embeddings: Option<bool>,
    pub rope_theta: Option<f32>,
    pub rope_scaling: Option<HashMap<usize, f64>>,
    pub attention_bias: Option<bool>,
    pub attention_dropout: Option<f32>,

    #[serde(flatten, default)]
    pub pretrained_config: PretrainedConfig,
}

impl LlamaConfig {
    pub fn get_num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            hidden_act: HiddenAct::Silu,
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: Some(true),
            pad_token_id: None,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            pretraining_tp: Some(1),
            tie_word_embeddings: Some(true),
            rope_theta: Some(10000.0),
            rope_scaling: None,
            attention_bias: Some(false),
            attention_dropout: Some(0.0),
            pretrained_config: PretrainedConfig::default(),
        }
    }
}
