use crate::{config::PretrainedConfig, model::PreTrainedModel};
use anyhow::Result;
use candle_core::{DType, IndexOp, Module, Tensor};
use candle_nn::{
    embedding, layer_norm, linear,
    ops::{dropout, softmax},
    Dropout, Embedding, LayerNorm, Linear, VarBuilder,
};
use serde::{Deserialize, Serialize};

pub const BERT_DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Gelu,
    Relu,
}

pub struct HiddenActLayer {
    act: HiddenAct,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        Self { act }
    }

    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        match self.act {
            HiddenAct::Gelu => hidden_states.gelu_erf(),
            HiddenAct::Relu => hidden_states.relu(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BertConfig {
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
    pub pad_token_id: usize,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    pub use_cache: bool,
    pub classifier_dropout: Option<f32>,
    pub model_type: Option<String>,

    #[serde(flatten, default)]
    pub pretrained_config: PretrainedConfig,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
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
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
            pretrained_config: PretrainedConfig::default(),
        }
    }
}

pub struct BertEmbeddings {
    pub word_embeddings: Embedding,
    pub position_embeddings: Option<Embedding>,
    pub token_type_embeddings: Embedding,
    pub layer_norm: LayerNorm,
    pub dropout: Dropout,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
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

pub struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.attention_probs_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }
}

impl Module for BertSelfAttention {
    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_probs = softmax(&attention_scores, candle_core::D::Minus1)?;
        let attention_probs = self.dropout.forward(&attention_probs, false)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;
        Ok(context_layer)
    }
}

pub struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertSelfOutput {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

pub struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
}

impl BertAttention {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let self_attention = BertSelfAttention::load(vb.pp("self"), config)?;
        let self_output = BertSelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
        })
    }
}

impl Module for BertAttention {
    fn forward(&self, input_tensor: &Tensor) -> candle_core::Result<Tensor> {
        let self_outputs = self.self_attention.forward(input_tensor)?;
        let attention_output = self.self_output.forward(&self_outputs, input_tensor)?;
        Ok(attention_output)
    }
}

pub struct BertIntermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
}

impl BertIntermediate {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
        })
    }
}

impl Module for BertIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

pub struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let attention = BertAttention::load(vb.pp("attention"), config)?;
        let intermediate = BertIntermediate::load(vb.pp("intermediate"), config)?;
        let output = BertOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }
}

impl Module for BertLayer {
    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

pub struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }
}

impl Module for BertEncoder {
    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states)?;
        }
        Ok(hidden_states)
    }
}

pub struct BertPooler {
    dense: Linear,
}

impl BertPooler {
    fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }
}

impl Module for BertPooler {
    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let first_token_tensor = hidden_states.i((.., 0))?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;
        pooled_output.tanh()
    }
}

pub struct Bert {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: BertPooler,
}

impl Bert {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let embeddings = BertEmbeddings::load(vb.pp("embeddings"), config)?;
        let encoder = BertEncoder::load(vb.pp("encoder"), config)?;
        let pooler = BertPooler::load(vb.pp("pooler"), config)?;
        Ok(Self {
            embeddings,
            encoder,
            pooler,
        })
    }

    pub fn forward_return_sequence(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;
        Ok(sequence_output)
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let sequence_output = self.forward_return_sequence(input_ids, token_type_ids)?;
        let pooled_output = self.pooler.forward(&sequence_output)?;
        Ok(pooled_output)
    }
}

pub struct BertModel {
    model: Bert,
    config: BertConfig,
}

impl PreTrainedModel for BertModel {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self> {
        let config: BertConfig = serde_json::from_value(config)?;
        let model = Bert::load(vb.pp("bert"), &config)?;
        Ok(Self { model, config })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids, token_type_ids)
    }

    fn config(&self) -> PretrainedConfig {
        self.config.pretrained_config.clone()
    }
}

pub struct BertForSequenceClassification {
    model: Bert,
    dropout: Dropout,
    classifier: Linear,
    config: BertConfig,
}

impl PreTrainedModel for BertForSequenceClassification {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self> {
        let config: BertConfig = serde_json::from_value(config)?;
        let model = Bert::load(vb.pp("bert"), &config)?;
        let dropout = Dropout::new(config.classifier_dropout.unwrap_or(0.1));
        let classifier = linear(
            config.hidden_size,
            config.pretrained_config.num_labels(),
            vb.pp("classifier"),
        )?;

        Ok(Self {
            model,
            dropout,
            classifier,
            config,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let pooled_output = self.model.forward(input_ids, token_type_ids)?;
        let pooled_output = self.dropout.forward(&pooled_output, false)?;
        let logits = self.classifier.forward(&pooled_output)?;
        Ok(logits)
    }

    fn config(&self) -> PretrainedConfig {
        self.config.pretrained_config.clone()
    }
}

pub struct BertForTokenClassification {
    model: Bert,
    dropout: Dropout,
    classifier: Linear,
    config: BertConfig,
}

impl PreTrainedModel for BertForTokenClassification {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self> {
        let config: BertConfig = serde_json::from_value(config)?;
        let model = Bert::load(vb.pp("bert"), &config)?;
        let dropout = Dropout::new(config.classifier_dropout.unwrap_or(0.1));
        let classifier = linear(
            config.hidden_size,
            config.pretrained_config.num_labels(),
            vb.pp("classifier"),
        )?;

        Ok(Self {
            model,
            dropout,
            classifier,
            config,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let sequence_output = self
            .model
            .forward_return_sequence(input_ids, token_type_ids)?;
        let sequence_output = self.dropout.forward(&sequence_output, false)?;
        let logits = self.classifier.forward(&sequence_output)?;
        Ok(logits)
    }

    fn config(&self) -> PretrainedConfig {
        self.config.pretrained_config.clone()
    }
}
