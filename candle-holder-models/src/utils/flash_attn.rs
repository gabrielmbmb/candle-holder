use candle_core::Tensor;
use candle_holder::Result;

#[cfg(feature = "flash-attn")]
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    Ok(candle_flash_attn::flash_attn(
        q,
        k,
        v,
        softmax_scale,
        causal,
    )?)
}

#[cfg(not(feature = "flash-attn"))]
pub fn flash_attn(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _softmax_scale: f32,
    _causal: bool,
) -> Result<Tensor> {
    unimplemented!("compile with flash-attn")
}
