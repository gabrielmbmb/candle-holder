use candle_core::{Tensor, D};
use candle_holder::Result;
use candle_nn::ops::softmax_last_dim;

use crate::config::GenerationConfig;

/// An enum containing the available sampling strategies for selecting the next token id of a
/// sequence using the outputs logits of an auto-regressive model.
pub enum SamplingStrategy {
    Greedy,
    TopK(usize),
    TopP(f32),
    TopKTopP { k: usize, p: f32 },
    Multinomial,
}

pub struct SamplingConfig {
    strategy: SamplingStrategy,
    temperature: Option<f64>,
}

impl SamplingConfig {
    pub fn new(strategy: SamplingStrategy, temperature: Option<f64>) -> SamplingConfig {
        Self {
            strategy,
            temperature,
        }
    }

    pub fn sample(&self, logits: &Tensor) -> Result<Tensor> {
        let scaled_logits = self.apply_temperature(logits)?;

        Ok(match &self.strategy {
            SamplingStrategy::Greedy => scaled_logits.argmax(D::Minus1)?,
            SamplingStrategy::TopK(k) => self.top_k_sample(&scaled_logits, *k)?,
            SamplingStrategy::TopP(p) => self.top_p_sample(&scaled_logits, *p)?,
            SamplingStrategy::TopKTopP { k, p } => {
                self.top_k_top_p_sample(&scaled_logits, *k, *p)?
            }
            SamplingStrategy::Multinomial => logits.clone(),
        })
    }

    fn apply_temperature(&self, logits: &Tensor) -> Result<Tensor> {
        let scaled_logits = if let Some(temperature) = self.temperature {
            let logits = (logits / temperature)?;
            softmax_last_dim(&logits)?
        } else {
            logits.clone()
        };
        Ok(scaled_logits)
    }

    fn top_k_sample(&self, logits: &Tensor, k: usize) -> Result<Tensor> {
        unimplemented!("")
    }

    fn top_p_sample(&self, logits: &Tensor, p: f32) -> Result<Tensor> {
        println!("logits {}", logits);
        let top_probabilities_indices = logits.arg_sort_last_dim(false)?;
        println!("top prob indices {}", top_probabilities_indices);
        let top_probs = logits.gather(&top_probabilities_indices, D::Minus1)?;
        println!("top probs {}", top_probs);
        let cumulative_probs = top_probs.cumsum(D::Minus1)?;
        println!("cumulative probs {}", cumulative_probs);
        Ok(logits.clone())
    }

    fn top_k_top_p_sample(&self, logits: &Tensor, k: usize, p: f32) -> Result<Tensor> {
        unimplemented!("")
    }

    fn multinomial_sample(&self, probs: &Tensor) -> Result<Tensor> {
        unimplemented!("")
    }
}

impl From<GenerationConfig> for SamplingConfig {
    fn from(generation_config: GenerationConfig) -> Self {
        if !generation_config.get_do_sample() {
            return Self {
                strategy: SamplingStrategy::Greedy,
                temperature: None,
            };
        }

        let temperature = generation_config.get_temperature();
        let top_k = generation_config.get_top_k();
        let top_p = generation_config.get_top_p();
        let strategy = match (top_k, top_p) {
            (Some(k), None) => SamplingStrategy::TopK(k),
            (None, Some(p)) => SamplingStrategy::TopP(p),
            (Some(k), Some(p)) => SamplingStrategy::TopKTopP { k, p },
            (None, None) => SamplingStrategy::Greedy,
        };

        Self {
            strategy,
            temperature: Some(temperature),
        }
    }
}
