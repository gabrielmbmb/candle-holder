use candle_core::{Tensor, D};
use candle_holder::Result;

/// Apply the repetition penalty to the logits. It will adjust the logits of tokens that have
/// already appeared in the sequence to reduce the probability of them being selected again.
///
/// # Arguments
///
/// * `logits` - The logits to adjust.
/// * `input_ids` - The input token ids.
/// * `repetition_penalty` - The value to modulate the logits by. Valid values are in the range of
///  `0.0` to `infinity`. A value of `1.0` means no penalty.
///
/// # Returns
///
/// The adjusted logits.
pub fn apply_repetition_penalty(
    logits: &Tensor,
    input_ids: &Tensor,
    repetition_penalty: f64,
) -> Result<Tensor> {
    // Create a mask that has 1s for tokens that have appeared and 0s elsewhere
    let mask = Tensor::zeros_like(logits)?.index_add(
        &input_ids,
        &Tensor::ones_like(&input_ids)?.to_dtype(logits.dtype())?,
        D::Minus1,
    )?;
    let adjusted_logits = (logits.broadcast_sub(&mask)? * (repetition_penalty - 1.0))?;
    Ok(logits.add(&adjusted_logits)?)
}
