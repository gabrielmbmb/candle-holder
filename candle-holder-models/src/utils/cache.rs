use candle_core::{Tensor, D};
use candle_holder::Result;

/// Dynamic cache for storing key and value states of each attention layer of a model.
pub struct DynamicCache {
    /// Key states of each attention layer.
    key_states: Vec<Tensor>,
    /// Value states of each attention layer.
    value_states: Vec<Tensor>,
}

impl DynamicCache {
    /// Create a new dynamic cache.
    pub fn new() -> Self {
        Self {
            key_states: vec![],
            value_states: vec![],
        }
    }

    /// Update the key and value states of a layer.
    ///
    /// # Arguments
    ///
    /// * `key_states` - The key states of the layer.
    /// * `value_states` - The value states of the layer.
    /// * `layer_idx` - The index of the layer.
    ///
    /// # Returns
    ///
    /// The updated key and value states of the layer.
    pub fn update_key_value_states(
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
                Tensor::cat(&[&self.key_states[layer_idx], &key_states], 2)?;
            self.value_states[layer_idx] =
                Tensor::cat(&[&self.value_states[layer_idx], &value_states], 2)?;
        }

        Ok((
            self.key_states[layer_idx].clone(),
            self.value_states[layer_idx].clone(),
        ))
    }

    /// Gets the sequence length of the cached states.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - The index of the layer. If not provided, the first layer is used.
    ///
    /// # Returns
    ///
    /// The sequence length of the cached states.
    pub fn get_seq_length(&self, layer_idx: Option<usize>) -> Result<u32> {
        let idx = layer_idx.unwrap_or(0);

        if self.key_states.len() <= idx {
            return Ok(0);
        }

        Ok(self.key_states[idx].dims4()?.2 as u32)
    }
}
