use std::collections::HashMap;

use anyhow::Result;
use candle_core::{IndexOp, Module, Result as CandleResult, Tensor};
use candle_nn::{linear, Linear, VarBuilder};
use candle_transformers::models::bert::{BertModel as CandleBertModel, Config as CandleBertConfig};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum HiddenAct {
    Gelu,
    Relu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BertConfig {
    vocab_size: usize,
    hidden_size: usize,
    id2label: Option<HashMap<usize, String>>,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: HiddenAct,
    hidden_dropout_prob: f64,
    attention_probs_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_eps: f64,
    pad_token_id: usize,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    classifier_dropout: Option<f64>,
    model_type: Option<String>,
}

impl BertConfig {
    fn to_candle_config(&self) -> CandleBertConfig {
        let serialized = serde_json::to_string(self).unwrap();
        serde_json::from_str(&serialized).unwrap()
    }

    fn num_labels(&self) -> usize {
        if let Some(id2label) = &self.id2label {
            return id2label.len();
        }
        0
    }
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            id2label: None,
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
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        }
    }
}

#[derive(Debug)]
struct BertPooler(Linear);

impl BertPooler {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        Ok(Self(dense))
    }
}

impl Module for BertPooler {
    fn forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let first_token_tensor = hidden_states.i((.., 0))?;
        let pooled_output = self.0.forward(&first_token_tensor)?;
        pooled_output.tanh()
    }
}

pub struct BertModel {
    bert: CandleBertModel,
    pooler: BertPooler,
}

impl BertModel {
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.bert.forward(input_ids, token_type_ids)?;
        let outputs = self.pooler.forward(&hidden_states)?;
        Ok(outputs)
    }

    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let pooler = BertPooler::load(vb.pp("pooler"), config)?;
        let bert = CandleBertModel::load(vb, &config.to_candle_config())?;
        Ok(Self { bert, pooler })
    }
}

pub struct BertForSequenceClassification {
    bert: BertModel,
    classifier: Linear,
}

impl BertForSequenceClassification {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let classifier = linear(config.hidden_size, config.num_labels(), vb.pp("classifier"))?;
        let bert_model_with_pooler = BertModel::load(vb.pp("bert"), config)?;

        Ok(Self {
            bert: bert_model_with_pooler,
            classifier,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let pooled_output = self.bert.forward(input_ids, token_type_ids)?;
        let logits = self.classifier.forward(&pooled_output)?;
        Ok(logits)
    }
}
