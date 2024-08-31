use candle_core::{DType, Tensor, D};
use candle_holder::{Error, Result};
use candle_nn::ops::softmax_last_dim;
use rand::{distributions::WeightedIndex, prelude::Distribution, SeedableRng};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::generation::config::GenerationConfig;

/// An enum containing the available sampling strategies for selecting the next token id of a
/// sequence using the outputs logits of an auto-regressive model.
#[derive(Debug)]
pub enum SamplingStrategy {
    /// Greedy sampling selects the token with the highest probability.
    Greedy,
    /// Top-k sampling selects the token from the top `k` most probable tokens.
    TopK(usize),
    /// Top-p sampling selects the token from the smallest set of tokens whose cumulative
    /// probability exceeds the threshold `p`.
    TopP(f32),
    TopKTopP {
        k: usize,
        p: f32,
    },
}

/// A struct for sampling the next token id from the logits of an auto-regressive model.
#[derive(Debug)]
pub struct LogitSampler {
    /// The sampling strategy to use.
    strategy: SamplingStrategy,
    /// The temperature to apply to the logits before sampling.
    temperature: Option<f64>,
    /// The random number generator used in the multinomial sampling.
    rng: rand::rngs::StdRng,
}

impl LogitSampler {
    pub fn new(
        strategy: SamplingStrategy,
        temperature: Option<f64>,
        seed: Option<u64>,
    ) -> LogitSampler {
        Self {
            strategy,
            temperature,
            rng: Self::build_rng_from_seed(seed),
        }
    }

    /// Create a new `LogitSampler` from a `GenerationConfig` and an optional seed.
    ///
    /// # Arguments
    ///
    /// * `generation_config` - The generation configuration to use.
    /// * `seed` - The seed to use for the random number generator.
    ///
    /// # Returns
    ///
    /// A new `LogitSampler` instance.
    pub fn from_generation_config(generation_config: &GenerationConfig, seed: Option<u64>) -> Self {
        let rng = Self::build_rng_from_seed(seed);

        if !generation_config.get_do_sample() {
            return Self {
                strategy: SamplingStrategy::Greedy,
                temperature: None,
                rng,
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
            rng,
        }
    }

    fn build_rng_from_seed(seed: Option<u64>) -> rand::rngs::StdRng {
        let seed = seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_nanos() as u64
        });
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    /// Sample the next token id from the logits.
    ///
    /// # Arguments
    ///
    /// * `logits` - The logits to sample from.
    ///
    /// # Returns
    ///
    /// The ID of the sampled token.
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        if self.temperature.is_none() {
            return Ok(logits.argmax(D::Minus1)?.to_scalar()?);
        }

        let probs = self
            .apply_temperature(logits)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        Ok(match &self.strategy {
            SamplingStrategy::TopK(k) => self.top_k_sample(&probs, *k)?,
            SamplingStrategy::TopP(p) => self.top_p_sample(&probs, *p)?,
            SamplingStrategy::TopKTopP { k, p } => self.top_k_top_p_sample(&probs, *k, *p)?,
            SamplingStrategy::Greedy => unreachable!(),
        })
    }

    /// Applies the temperature scaling to the logits and returns the probabilities.
    ///
    /// The temperature scaling is applied as follows:
    ///    - Divide the logits by the temperature
    ///    - Apply the softmax function to the scaled logits
    ///
    /// # Arguments
    ///
    /// * `logits` - The logits to apply the temperature scaling to.
    ///
    /// # Returns
    ///
    /// A tensor containing the probabilities obtained by applying the temperature scaling to the
    /// logits.
    fn apply_temperature(&self, logits: &Tensor) -> Result<Tensor> {
        let temperature = self.temperature.expect("Temperature should be set");
        let scaled_logits = (logits / temperature)?;
        Ok(softmax_last_dim(&scaled_logits)?)
    }

    /// Sample the next token using Top-k sampling. It takes the top `k` probabilities and samples
    /// from them using a multinomial distribution.
    ///
    /// # Arguments
    ///
    /// * `probs` - The probabilities of the tokens.
    /// * `k` - The number of top probabilities to consider.
    ///
    /// # Returns
    ///
    /// The index of the sampled token.
    fn top_k_sample(&mut self, probs: &[f32], k: usize) -> Result<u32> {
        if probs.len() <= k {
            return self.sample_multinomial(probs);
        }
        let sorted_probs = sort_probs(probs);
        let top_k_probs: Vec<f32> = sorted_probs[..k].iter().map(|&(_, prob)| prob).collect();
        let sampled_index = self.sample_multinomial(&top_k_probs)?;
        Ok(sorted_probs[sampled_index as usize].0 as u32)
    }

    /// Sample the next token using Top-p sampling also known as nucleus sampling. It takes the top
    /// probabilities that sum equal or greater than `p` and samples from them using a multinomial
    /// distribution.
    ///
    /// # Arguments
    ///
    /// * `probs` - The probabilities of the tokens.
    /// * `p` - The cumulative probability threshold.
    ///
    /// # Returns
    ///
    /// The index of the sampled token.
    fn top_p_sample(&mut self, probs: &[f32], p: f32) -> Result<u32> {
        let sorted_probs = sort_probs(probs);

        // Compute the cumulative probabilities and get the index where `cumsum >= top_p` ensuring
        // that we include at least one token.
        let mut cumsum = 0.0;
        let cutoff_index = sorted_probs
            .iter()
            .take_while(|&&(_, prob)| {
                cumsum += prob;
                cumsum < p
            })
            .count()
            .max(1);

        // Extract probabilities for sampling
        let top_p_probs: Vec<f32> = sorted_probs[..cutoff_index]
            .iter()
            .map(|&(_, prob)| prob)
            .collect();
        let sampled_index = self.sample_multinomial(&top_p_probs)?;
        Ok(sorted_probs[sampled_index as usize].0 as u32)
    }

    /// Sample the next token using Top-k and Top-p sampling. It first applies top-k filtering to
    /// the probabilities and then applies top-p filtering on the top-k probabilities. Finally, it
    /// samples from the filtered probabilities using a multinomial distribution.
    ///
    /// # Arguments
    ///
    /// * `probs` - The probabilities of the tokens.
    /// * `k` - The number of top probabilities to consider.
    /// * `p` - The cumulative probability threshold.
    ///
    /// # Returns
    ///
    /// The index of the sampled token.
    fn top_k_top_p_sample(&mut self, probs: &[f32], k: usize, p: f32) -> Result<u32> {
        // Apply top-k filtering
        let sorted_probs = sort_probs(probs);
        let top_k_probs: Vec<(usize, f32)> = sorted_probs.into_iter().take(k).collect();

        // Apply top-p filtering on the top-k probabilities
        let mut cumsum = 0.0;
        let cutoff_index = top_k_probs
            .iter()
            .take_while(|&&(_, prob)| {
                cumsum += prob;
                cumsum < p
            })
            .count()
            .max(1); // Ensure we include at least one token

        // Extract probabilities for sampling
        let final_probs: Vec<f32> = top_k_probs[..cutoff_index]
            .iter()
            .map(|&(_, prob)| prob)
            .collect();
        let sampled_index = self.sample_multinomial(&final_probs)?;
        Ok(top_k_probs[sampled_index as usize].0 as u32)
    }

    /// Sample a token from the multinomial distribution defined by the probabilities.
    ///
    /// # Arguments
    ///
    /// * `probs` - The probabilities of the tokens.
    ///
    /// # Returns
    ///
    /// The index of the sampled token.
    fn sample_multinomial(&mut self, probs: &[f32]) -> Result<u32> {
        let dist = WeightedIndex::new(probs).map_err(Error::wrap)?;
        let sampled_token = dist.sample(&mut self.rng) as u32;
        Ok(sampled_token)
    }
}

/// Sort the probabilities in descending order and return the sorted probabilities along with their
/// indices.
///
/// # Arguments
///
/// * `probs` - The probabilities to sort.
///
/// # Returns
///
/// A vector of tuples containing the index and the probability of each token, sorted in descending
/// order.
fn sort_probs(probs: &[f32]) -> Vec<(usize, f32)> {
    let mut sorted_probs: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &prob)| (i, prob))
        .collect();
    sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sorted_probs
}
