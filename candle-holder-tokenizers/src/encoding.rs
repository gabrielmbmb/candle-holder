use candle_core::{Device, Tensor};
use candle_holder::Result;
use tokenizers::Encoding;

/// A struct containing the encoding of several sequences.
#[derive(Debug)]
pub struct BatchEncoding {
    input_ids: Tensor,
    token_type_ids: Tensor,
    attention_mask: Tensor,
    encodings: Vec<Encoding>,
}

impl BatchEncoding {
    pub fn new(
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        encodings: Vec<Encoding>,
    ) -> Self {
        BatchEncoding {
            input_ids,
            token_type_ids,
            attention_mask,
            encodings,
        }
    }

    pub fn get_input_ids(&self) -> &Tensor {
        &self.input_ids
    }

    pub fn get_token_type_ids(&self) -> &Tensor {
        &self.token_type_ids
    }

    pub fn get_attention_mask(&self) -> &Tensor {
        &self.attention_mask
    }

    pub fn get_encodings(&self) -> &Vec<Encoding> {
        &self.encodings
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_ids = self.input_ids.to_device(device)?;
        self.token_type_ids = self.token_type_ids.to_device(device)?;
        self.attention_mask = self.attention_mask.to_device(device)?;
        Ok(())
    }
}
