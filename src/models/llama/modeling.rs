use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{
    embedding, linear_no_bias, ops::softmax_last_dim, rms_norm, rotary_emb::rope, Dropout,
    Embedding, Linear, RmsNorm, VarBuilder,
};

use crate::{
    config::PretrainedConfig,
    model::PreTrainedModel,
    model_utils::{prepare_4d_causal_attention_mask, repeat_kv, DynamicCache},
    tokenizer::BatchEncoding,
};

use super::config::{HiddenAct, LlamaConfig};

pub const LLAMA_DTYPE: DType = DType::F16;

pub struct LlamaRotaryEmbedding {
    cos_cached: Tensor,
    sin_cached: Tensor,
}

impl LlamaRotaryEmbedding {
    fn new(dim: usize, max_position_embeddings: usize, base: f32, device: &Device) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let emb = Tensor::arange(0, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?
            .matmul(&inv_freq.reshape((1, inv_freq.elem_count()))?)?;
        let cos_cached = emb.cos()?.to_dtype(LLAMA_DTYPE)?;
        let sin_cached = emb.sin()?.to_dtype(LLAMA_DTYPE)?;

        Ok(Self {
            cos_cached,
            sin_cached,
        })
    }

    fn apply_rotary_pos_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos_cached.narrow(0, index_pos, seq_len)?;
        let sin = self.sin_cached.narrow(0, index_pos, seq_len)?;
        let q_embed = rope(q, &cos, &sin)?;
        let k_embed = rope(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

pub struct LlamaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: Arc<LlamaRotaryEmbedding>,
    dropout: Dropout,
    head_dim: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
}

impl LlamaAttention {
    fn load(
        vb: VarBuilder,
        rotary_emb: Arc<LlamaRotaryEmbedding>,
        config: &LlamaConfig,
    ) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.get_num_key_value_heads();
        let num_key_value_groups = num_attention_heads / num_key_value_heads;
        let size_q = head_dim * num_attention_heads;
        let size_kv = head_dim * num_key_value_heads;
        let q_proj = linear_no_bias(config.hidden_size, size_q, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(config.hidden_size, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(config.hidden_size, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(size_q, config.hidden_size, vb.pp("o_proj"))?;
        let dropout = Dropout::new(config.attention_dropout.unwrap_or_default());

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            dropout,
            rotary_emb,
            head_dim,
            num_attention_heads,
            num_key_value_heads,
            num_key_value_groups,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        causal_mask: &Tensor,
        index_pos: usize,
        layer_idx: usize,
        cache: Option<&mut DynamicCache>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = hidden_states.dims3()?;

        let query_states = self.q_proj.forward(hidden_states)?;
        let key_states = self.k_proj.forward(hidden_states)?;
        let value_states = self.v_proj.forward(hidden_states)?;

        let query_states = query_states
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let key_states = key_states
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut value_states = value_states
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (query_states, mut key_states) =
            self.rotary_emb
                .apply_rotary_pos_emb(&query_states, &key_states, index_pos)?;

        if let Some(cache) = cache {
            (key_states, value_states) =
                cache.update_key_states(key_states, value_states, layer_idx)?;
        }

        let key_states = repeat_kv(key_states, self.num_key_value_groups)?;
        let value_states = repeat_kv(value_states, self.num_key_value_groups)?;

        // Upcast to f32 for sensible operations (softmax)
        let dtype = query_states.dtype();
        let query_states = query_states.to_dtype(DType::F32)?;
        let key_states = key_states.to_dtype(DType::F32)?;
        let value_states = value_states.to_dtype(DType::F32)?;

        let attn_weights =
            (query_states.matmul(&key_states.t()?)? / (self.head_dim as f64).sqrt())?;

        // Apply causal attention mask
        let attn_weights = attn_weights.broadcast_add(causal_mask)?;

        let attn_weights = softmax_last_dim(&attn_weights)?;
        let attn_weights = self.dropout.forward(&attn_weights, false)?;

        let attn_output = attn_weights.matmul(&value_states)?.to_dtype(dtype)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, hidden_size))?;
        let attn_output = self.o_proj.forward(&attn_output)?;

        Ok(attn_output)
    }
}
pub struct HiddenActLayer {
    act: HiddenAct,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        Self { act }
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self.act {
            HiddenAct::Silu => candle_nn::ops::silu(x),
        }
    }
}

pub struct LlamaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: HiddenActLayer,
}

impl LlamaMLP {
    fn load(vb: VarBuilder, config: &LlamaConfig) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        let act = HiddenActLayer::new(config.hidden_act);
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self
            .act
            .forward(&self.gate_proj.forward(x)?)?
            .mul(&self.up_proj.forward(x)?)?;
        let x = self.down_proj.forward(&x)?;
        Ok(x)
    }
}

pub struct LlamaDecoderLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LlamaDecoderLayer {
    fn load(
        vb: VarBuilder,
        rotary_emb: Arc<LlamaRotaryEmbedding>,
        config: &LlamaConfig,
    ) -> Result<Self> {
        let self_attn = LlamaAttention::load(vb.pp("self_attn"), rotary_emb, config)?;
        let mlp = LlamaMLP::load(vb.pp("mlp"), config)?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        causal_mask: &Tensor,
        index_pos: usize,
        layer_idx: usize,
        cache: Option<&mut DynamicCache>,
    ) -> Result<Tensor> {
        let residual = hidden_states;

        // Self Attention
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states =
            self.self_attn
                .forward(&hidden_states, causal_mask, index_pos, layer_idx, cache)?;
        let hidden_states = (residual + hidden_states)?;

        // Fully connected
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        Ok(hidden_states)
    }
}

pub struct Llama {
    embed_tokens: Embedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
}

impl Llama {
    fn load(vb: VarBuilder, config: &LlamaConfig) -> Result<Self> {
        let embed_tokens: Embedding =
            embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(LlamaRotaryEmbedding::new(
            config.hidden_size / config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta.unwrap_or_default(),
            vb.device(),
        )?);
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                LlamaDecoderLayer::load(
                    vb.pp(&format!("layers.{index}")),
                    rotary_emb.clone(),
                    config,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        index_pos: usize,
        mut cache: Option<&mut DynamicCache>,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let causal_mask = prepare_4d_causal_attention_mask(attention_mask, DType::F32)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                &causal_mask,
                index_pos,
                layer_idx,
                cache.as_deref_mut(),
            )?;
        }
        let hidden_states = self.norm.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

pub struct LlamaModel {
    model: Llama,
    config: LlamaConfig,
}

impl PreTrainedModel for LlamaModel {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self> {
        let config: LlamaConfig = serde_json::from_value(config)?;
        let model = Llama::load(vb.pp("model"), &config)?;
        Ok(Self { model, config })
    }

    fn forward(&self, encodings: &BatchEncoding) -> Result<Tensor> {
        self.model.forward(
            encodings.get_input_ids(),
            encodings.get_attention_mask(),
            // TODO: hardcoded for now
            0,
            None,
        )
    }

    fn forward_with_cache(
        &self,
        encodings: &BatchEncoding,
        index_pos: usize,
        cache: &mut DynamicCache,
    ) -> Result<Tensor> {
        self.model.forward(
            encodings.get_input_ids(),
            encodings.get_attention_mask(),
            index_pos,
            Some(cache),
        )
    }

    fn config(&self) -> &PretrainedConfig {
        &self.config.pretrained_config
    }
}

pub struct LlamaForCausalLM {
    model: Llama,
    lm_head: Linear,
    config: LlamaConfig,
}

impl PreTrainedModel for LlamaForCausalLM {
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self> {
        let config: LlamaConfig = serde_json::from_value(config)?;
        let model = Llama::load(vb.pp("model"), &config)?;
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            model,
            lm_head,
            config,
        })
    }

    fn forward(&self, encodings: &BatchEncoding) -> Result<Tensor> {
        let outputs = self.model.forward(
            encodings.get_input_ids(),
            encodings.get_attention_mask(),
            // TODO: hardcoded for now
            0,
            None,
        )?;
        let logits = self.lm_head.forward(&outputs)?;
        Ok(logits)
    }

    fn forward_with_cache(
        &self,
        encodings: &BatchEncoding,
        index_pos: usize,
        cache: &mut DynamicCache,
    ) -> Result<Tensor> {
        let outputs = self.model.forward(
            encodings.get_input_ids(),
            encodings.get_attention_mask(),
            index_pos,
            Some(cache),
        )?;
        let logits = self.lm_head.forward(&outputs)?;
        Ok(logits)
    }

    fn config(&self) -> &PretrainedConfig {
        &self.config.pretrained_config
    }
}
