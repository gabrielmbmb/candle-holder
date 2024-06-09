use candle_core::{DType, Tensor, D};
use candle_holder::Result;

/// Repeats the values of each head the specified number of times. This function is meant to be used
/// with the key and value tensors of an attention layer that used Grouped Query Attention (GQA).
///
/// # Arguments
///
/// * `kv` - The key or value tensor of an attention layer that used GQA.
/// * `n_rep` - The number of times to repeat the values of each head.
///
/// # Returns
///
/// The tensor with the values of each head repeated `n_rep` times.
pub fn repeat_kv(kv: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(kv);
    }

    let (b_sz, n_kv_heads, seq_len, head_dim) = kv.dims4()?;
    Ok(Tensor::cat(&vec![kv; n_rep], 2)?.reshape((b_sz, n_kv_heads * n_rep, seq_len, head_dim))?)
}

/// Creates a broadcastable attention mask to ignore future, padding and masked tokens.
///
/// # Arguments
///
/// * `attention_mask` - The attention mask tensor with shape `(batch_size, seq_len)`.
/// * `dtype` - The data type of the attention mask tensor to create.
/// * `is_decoder` - Whether the attention mask is for a decoder model or not.
///
/// # Returns
///
/// The broadcastable attention mask tensor with shape `(batch_size, 1, 1, seq_len)`.
pub fn get_extended_attention_mask(
    attention_mask: &Tensor,
    dtype: DType,
    _is_decoder: bool,
) -> Result<Tensor> {
    let extended_attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
    let on_true = extended_attention_mask.zeros_like()?.to_dtype(dtype)?;
    let on_false = Tensor::new(f32::NEG_INFINITY, extended_attention_mask.device())?
        .broadcast_as(extended_attention_mask.shape())?
        .to_dtype(dtype)?;
    let extended_attention_mask = extended_attention_mask.where_cond(&on_true, &on_false)?;
    Ok(extended_attention_mask)
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
