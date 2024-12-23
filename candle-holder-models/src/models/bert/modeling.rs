use candle_core::{DType, IndexOp, Module, Tensor};
use candle_holder::{Error, Result};
use candle_nn::{
    embedding, layer_norm, linear, ops::softmax, Dropout, Embedding, LayerNorm, Linear, VarBuilder,
};
use std::sync::Arc;

use super::config::{BertConfig, HiddenAct};

use crate::{
    config::PretrainedConfig,
    model::{ForwardParams, ModelOutput, PreTrainedModel},
    utils::attn_mask::get_extended_attention_mask,
};

pub const BERT_DTYPE: DType = DType::F32;

#[derive(Debug)]
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

#[derive(Debug)]
pub struct BertEmbeddings {
    pub word_embeddings: Arc<Embedding>,
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
            word_embeddings: Arc::new(word_embeddings),
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

#[derive(Debug)]
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

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(&attention_mask)?;

        let attention_probs = softmax(&attention_scores, candle_core::D::Minus1)?;
        let attention_probs = self.dropout.forward(&attention_probs, false)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;
        Ok(context_layer)
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
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

    fn forward(
        &self,
        input_tensor: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let self_outputs = self.self_attention.forward(input_tensor, attention_mask)?;
        let attention_output = self.self_output.forward(&self_outputs, input_tensor)?;
        Ok(attention_output)
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

#[derive(Debug)]
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

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

#[derive(Debug)]
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
        let first_token_tensor = hidden_states.i((.., 0))?.contiguous()?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;
        pooled_output.tanh()
    }
}

#[derive(Debug)]
pub struct BertPredictionHeadTransform {
    dense: Linear,
    transform_act_fn: HiddenActLayer,
    layer_norm: LayerNorm,
}

impl BertPredictionHeadTransform {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let transform_act_fn = HiddenActLayer::new(config.hidden_act);
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            dense,
            transform_act_fn,
            layer_norm,
        })
    }
}

impl Module for BertPredictionHeadTransform {
    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.transform_act_fn.forward(&hidden_states)?;
        self.layer_norm.forward(&hidden_states)
    }
}

#[derive(Debug)]
pub struct BertLMPredictionHead {
    transform: BertPredictionHeadTransform,
    // The decoder weights are tied with the embeddings weights so the model only learns one
    // representation
    decoder_weight: Arc<Embedding>,
    decoder_bias: Tensor,
}

impl BertLMPredictionHead {
    pub fn load(
        vb: VarBuilder,
        decoder_weight: Arc<Embedding>,
        config: &BertConfig,
    ) -> Result<Self> {
        let transform = BertPredictionHeadTransform::load(vb.pp("transform"), config)?;
        let decoder_bias =
            vb.get_with_hints((config.vocab_size,), "bias", candle_nn::Init::Const(0.))?;
        Ok(Self {
            transform,
            decoder_weight,
            decoder_bias,
        })
    }
}

impl Module for BertLMPredictionHead {
    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let hidden_states = self.transform.forward(hidden_states)?;
        hidden_states
            .broadcast_matmul(&self.decoder_weight.embeddings().t()?)?
            .broadcast_add(&self.decoder_bias)
    }
}

#[derive(Debug)]
pub struct BertOnlyMLMHead {
    predictions: BertLMPredictionHead,
}

impl BertOnlyMLMHead {
    pub fn load(
        vb: VarBuilder,
        decoder_weight: Arc<Embedding>,
        config: &BertConfig,
    ) -> Result<Self> {
        let predictions = BertLMPredictionHead::load(vb.pp("predictions"), decoder_weight, config)?;
        Ok(Self { predictions })
    }
}

impl Module for BertOnlyMLMHead {
    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        self.predictions.forward(hidden_states)
    }
}

#[derive(Debug)]
pub struct Bert {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: BertPooler,
    config: BertConfig,
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
            config: config.clone(),
        })
    }

    pub fn forward_return_sequence(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<Tensor> {
        let extended_attention_mask =
            get_extended_attention_mask(attention_mask, BERT_DTYPE, self.config.is_decoder)?;
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let sequence_output = self
            .encoder
            .forward(&embedding_output, &extended_attention_mask)?;
        Ok(sequence_output)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<ModelOutput> {
        let sequence_output =
            self.forward_return_sequence(input_ids, attention_mask, token_type_ids)?;
        let pooled_output = self.pooler.forward(&sequence_output)?;
        Ok(ModelOutput::new(
            None,
            Some(sequence_output),
            Some(pooled_output),
        ))
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
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

    fn forward(&self, params: ForwardParams) -> Result<ModelOutput> {
        let output = self.model.forward(
            params
                .get_input_ids()
                .ok_or(Error::MissingForwardParam("input_ids".to_string()))?,
            params
                .get_attention_mask()
                .ok_or(Error::MissingForwardParam("attention_mask".to_string()))?,
            params
                .get_token_type_ids()
                .ok_or(Error::MissingForwardParam("token_type_ids".to_string()))?,
        )?;
        let pooled_output = output.get_pooled_output().unwrap();
        let pooled_output = self.dropout.forward(pooled_output, false)?;
        let logits = self.classifier.forward(&pooled_output)?;
        Ok(ModelOutput::new(Some(logits), None, None))
    }

    fn get_config(&self) -> &PretrainedConfig {
        &self.config.pretrained_config
    }
}

#[derive(Debug)]
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

    fn forward(&self, params: ForwardParams) -> Result<ModelOutput> {
        let output = self.model.forward(
            params
                .get_input_ids()
                .ok_or(Error::MissingForwardParam("input_ids".to_string()))?,
            params
                .get_attention_mask()
                .ok_or(Error::MissingForwardParam("attention_mask".to_string()))?,
            params
                .get_token_type_ids()
                .ok_or(Error::MissingForwardParam("token_type_ids".to_string()))?,
        )?;
        let sequence_output = self
            .dropout
            .forward(&output.get_last_hidden_state().unwrap(), false)?;
        let logits = self.classifier.forward(&sequence_output)?;
        Ok(ModelOutput::new(Some(logits), None, None))
    }

    fn get_config(&self) -> &PretrainedConfig {
        &self.config.pretrained_config
    }
}

#[derive(Debug)]
pub struct BertForMaskedLM {
    model: Bert,
    cls: BertOnlyMLMHead,
    config: BertConfig,
}

impl PreTrainedModel for BertForMaskedLM {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self> {
        let config: BertConfig = serde_json::from_value(config)?;
        let model = Bert::load(vb.pp("bert"), &config)?;
        let cls = BertOnlyMLMHead::load(
            vb.pp("cls"),
            model.embeddings.word_embeddings.clone(),
            &config,
        )?;
        Ok(Self { model, cls, config })
    }

    fn forward(&self, params: ForwardParams) -> Result<ModelOutput> {
        let output = self.model.forward(
            params
                .get_input_ids()
                .ok_or(Error::MissingForwardParam("input_ids".to_string()))?,
            params
                .get_attention_mask()
                .ok_or(Error::MissingForwardParam("attention_mask".to_string()))?,
            params
                .get_token_type_ids()
                .ok_or(Error::MissingForwardParam("token_type_ids".to_string()))?,
        )?;
        let logits = self.cls.forward(&output.get_last_hidden_state().unwrap())?;
        Ok(ModelOutput::new(Some(logits), None, None))
    }

    fn get_config(&self) -> &PretrainedConfig {
        &self.config.pretrained_config
    }
}
