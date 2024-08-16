use candle_core::{Result, Tensor, D};

// https://github.com/EricLBuehler/mistral.rs/blob/e64a71a16e28f55f396490d5bc40cf1ca4187c13/mistralrs-core/src/ops.rs#L533
pub trait TopKLastDimOp {
    fn topk(&self, k: usize) -> Result<(Tensor, Tensor)>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, k: usize) -> Result<(Tensor, Tensor)> {
        let sorted_indices = self.arg_sort_last_dim(false)?;
        let k_indices = sorted_indices.narrow(D::Minus1, 0, k)?.contiguous()?;
        Ok((sorted_indices.gather(&k_indices, D::Minus1)?, k_indices))
    }
}

