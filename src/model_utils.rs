use candle_core::{DType, Tensor, D};

use crate::error::Result;

pub fn repeat_kv(kv: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(kv);
    }

    let (b_sz, n_kv_heads, seq_len, head_dim) = kv.dims4()?;
    Ok(Tensor::cat(&vec![kv; n_rep], 2)?.reshape((b_sz, n_kv_heads * n_rep, seq_len, head_dim))?)
}

pub fn prepare_4d_causal_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let (b_sz, seq_len) = attention_mask.dims2()?;
    let inverted_mask = attention_mask
        .ones_like()?
        .sub(attention_mask)?
        .unsqueeze(1)?
        .unsqueeze(1)?
        .to_dtype(DType::F32)?;
    let inverted_mask = Tensor::cat(&vec![inverted_mask; seq_len], D::Minus2)?;
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
        .collect();
    let mask = Tensor::from_slice(&mask, (seq_len, seq_len), attention_mask.device())?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    let mask = Tensor::cat(&vec![mask; b_sz], 0)?;
    let on_true =
        Tensor::new(f32::NEG_INFINITY, attention_mask.device())?.broadcast_as(mask.shape())?;
    let causal_mask = mask.where_cond(&on_true, &inverted_mask)?.to_dtype(dtype)?;
    Ok(causal_mask)
}

pub struct DynamicCache {
    key_states: Vec<Tensor>,
    value_states: Vec<Tensor>,
}

impl DynamicCache {
    pub fn new() -> Self {
        Self {
            key_states: vec![],
            value_states: vec![],
        }
    }

    pub fn update_key_states(
        &mut self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        if self.key_states.len() <= layer_idx {
            self.key_states.push(key_states);
            self.value_states.push(value_states);
        } else {
            self.key_states[layer_idx] =
                Tensor::cat(&[&self.key_states[layer_idx], &key_states], D::Minus2)?;
            self.value_states[layer_idx] =
                Tensor::cat(&[&self.value_states[layer_idx], &key_states], D::Minus2)?;
        }

        Ok((
            self.key_states[layer_idx].clone(),
            self.value_states[layer_idx].clone(),
        ))
    }
}
