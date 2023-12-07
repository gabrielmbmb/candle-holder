use anyhow::{Error, Result};
use candle_core::{Device, Tensor};
use tokenizers::Encoding;

pub trait TensorEncoding {
    fn get_ids_tensor(&self, device: &Device) -> Result<Tensor>;
    fn get_type_ids_tensor(&self, device: &Device) -> Result<Tensor>;
    fn get_attention_mask_tensor(&self, device: &Device) -> Result<Tensor>;
}

impl TensorEncoding for Encoding {
    fn get_ids_tensor(&self, device: &Device) -> Result<Tensor> {
        let ids = self.get_ids().to_vec();
        Tensor::new(ids, device)?.unsqueeze(0).map_err(Error::msg)
    }

    fn get_type_ids_tensor(&self, device: &Device) -> Result<Tensor> {
        let type_ids = self.get_type_ids().to_vec();
        Tensor::new(type_ids, device)?
            .unsqueeze(0)
            .map_err(Error::msg)
    }

    fn get_attention_mask_tensor(&self, device: &Device) -> Result<Tensor> {
        let attention_mask = self.get_attention_mask().to_vec();
        Tensor::new(attention_mask, device)?
            .unsqueeze(0)
            .map_err(Error::msg)
    }
}
