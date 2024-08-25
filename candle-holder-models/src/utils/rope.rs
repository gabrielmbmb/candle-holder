use std::f64::consts::PI;

use candle_core::{Device, Tensor};
use candle_holder::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeType {
    Default,
    Linear,
    Dynamic,
    Yarn,
    LongRope,
    Llama3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub rope_type: RopeType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beta_fast: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beta_slow: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_factor: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_factor: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low_freq_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high_freq_factor: Option<f64>,
}

impl Default for RopeScaling {
    fn default() -> Self {
        Self {
            rope_type: RopeType::Default,
            factor: None,
            original_max_position_embeddings: None,
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
            short_factor: None,
            long_factor: None,
            low_freq_factor: None,
            high_freq_factor: None,
        }
    }
}

impl RopeScaling {
    /// Compute the RoPE parameters based on the RoPE type.
    ///
    /// # Returns
    ///
    /// A tuple containing the inverse frequency tensor and the optional scaling factor that can be
    /// applied after the RoPE tensor is computed.
    pub fn compute_rope_parameters(
        &self,
        dim: usize,
        base: f32,
        device: &Device,
    ) -> Result<(Tensor, Option<f64>)> {
        match self.rope_type {
            RopeType::Default => self.compute_default_parameters(dim, base, device),
            RopeType::Linear => self.compute_linear_parameters(dim, base, device),
            RopeType::Dynamic => self.compute_dynamic_parameters(dim, base, device),
            RopeType::Yarn => self.compute_yarn_parameters(dim, base, device),
            RopeType::LongRope => self.compute_long_rope_parameters(dim, base, device),
            RopeType::Llama3 => self.compute_llama3_parameters(dim, base, device),
        }
    }

    fn compute_default_parameters(
        &self,
        dim: usize,
        base: f32,
        device: &Device,
    ) -> Result<(Tensor, Option<f64>)> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        Ok((inv_freq, None))
    }
    fn compute_linear_parameters(
        &self,
        _dim: usize,
        _base: f32,
        _device: &Device,
    ) -> Result<(Tensor, Option<f64>)> {
        unimplemented!("Computing linear RoPE parameters is not implemented")
    }
    fn compute_dynamic_parameters(
        &self,
        _dim: usize,
        _base: f32,
        _device: &Device,
    ) -> Result<(Tensor, Option<f64>)> {
        unimplemented!("Computing dynamic RoPE parameters is not implemented")
    }
    fn compute_yarn_parameters(
        &self,
        _dim: usize,
        _base: f32,
        _device: &Device,
    ) -> Result<(Tensor, Option<f64>)> {
        unimplemented!("Computing yarn RoPE parameters is not implemented")
    }
    fn compute_long_rope_parameters(
        &self,
        _dim: usize,
        _base: f32,
        _device: &Device,
    ) -> Result<(Tensor, Option<f64>)> {
        unimplemented!("Computing long RoPE parameters is not implemented")
    }

    fn compute_llama3_parameters(
        &self,
        dim: usize,
        base: f32,
        device: &Device,
    ) -> Result<(Tensor, Option<f64>)> {
        let factor = self
            .factor
            .ok_or(Error::MissingRopeParam("factor".to_string()))?;
        let low_freq_factor = self
            .low_freq_factor
            .ok_or(Error::MissingRopeParam("low_freq_factor".to_string()))?;
        let high_freq_factor = self
            .high_freq_factor
            .ok_or(Error::MissingRopeParam("high_freq_factor".to_string()))?;
        let original_max_position_embeddings =
            self.original_max_position_embeddings
                .ok_or(Error::MissingRopeParam(
                    "original_max_position_embeddings".to_string(),
                ))? as f64;

        let (inv_freqs, scaling_factor) = self.compute_default_parameters(dim, base, device)?;
        let low_freq_wavelen = original_max_position_embeddings / low_freq_factor;
        let inv_freqs_scaled = (&inv_freqs / factor)?;
        let high_freq_wavelen = original_max_position_embeddings / high_freq_factor;
        let wavelen = ((2.0 * PI) * &inv_freqs)?;
        let inv_freq_llama = wavelen
            .gt(low_freq_wavelen)?
            .where_cond(&inv_freqs_scaled, &inv_freqs)?;
        let smooth_factor = (((original_max_position_embeddings / &wavelen)? - low_freq_factor)?
            / (high_freq_factor - low_freq_factor))?;
        let smoothed_inv_freq = ((1f64 - &smooth_factor)?.mul(&inv_freq_llama)?
            / (factor + &smooth_factor)?.mul(&inv_freq_llama)?)?;
        let is_medium_freq = wavelen
            .ge(high_freq_wavelen)?
            .mul(&wavelen.le(low_freq_wavelen)?)?;
        let inv_freq_llama = is_medium_freq.where_cond(&smoothed_inv_freq, &inv_freq_llama)?;
        Ok((inv_freq_llama, scaling_factor))
    }
}
