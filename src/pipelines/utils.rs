use anyhow::{Error, Result};
use candle_core::{Device, Tensor};
use tokenizers::{EncodeInput, Encoding, Tokenizer};

pub fn get_encodings<'s, E>(
    inputs: Vec<E>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<(Tensor, Tensor, Vec<Encoding>)>
where
    E: Into<EncodeInput<'s>> + Send,
{
    let encodings = tokenizer.encode_batch(inputs, true).map_err(Error::msg)?;

    let mut input_ids: Vec<Vec<u32>> = Vec::new();
    let mut token_type_ids: Vec<Vec<u32>> = Vec::new();

    for encoding in &encodings {
        input_ids.push(encoding.get_ids().to_vec());
        token_type_ids.push(encoding.get_type_ids().to_vec());
    }

    let input_ids = Tensor::new(input_ids, device)?;
    let token_type_ids = Tensor::new(token_type_ids, device)?;

    Ok((input_ids, token_type_ids, encodings))
}
