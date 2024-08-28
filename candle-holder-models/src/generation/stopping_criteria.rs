use candle_holder::{Error, Result};
use candle_holder_tokenizers::Tokenizer;

use super::GenerationConfig;

/// Trait for stopping criteria used during generation.
pub trait StoppingCriteria {
    fn should_stop(&self, input_ids: &[u32]) -> Result<bool>;
}

/// Stopping criteria that stops generation when the EOS token is generated.
pub struct EosTokenStoppingCriteria {
    /// The ID of the EOS tokens.
    eos_token: Vec<u32>,
}

impl EosTokenStoppingCriteria {
    /// Creates a new `EosTokenStoppingCriteria` with the provided EOS token.
    ///
    /// # Arguments
    ///
    /// * `eos_token` - The IDs of the EOS tokens.
    ///
    /// # Returns
    ///
    /// A new `EosTokenStoppingCriteria`.
    pub fn new(eos_token: Vec<u32>) -> Self {
        Self { eos_token }
    }
}

impl StoppingCriteria for EosTokenStoppingCriteria {
    fn should_stop(&self, input_ids: &[u32]) -> Result<bool> {
        if let Some(last_token) = input_ids.last() {
            Ok(self.eos_token.contains(last_token))
        } else {
            Ok(false)
        }
    }
}

/// Stopping criteria that stops generation when a stop string is generated.
pub struct StopStringStoppingCriteria<'a> {
    /// The stop strings to check for.
    stop_strings: Vec<String>,
    /// The tokenizer to use to decode the input token IDs.
    tokenizer: Option<&'a Box<dyn Tokenizer>>,
}

impl<'a> StopStringStoppingCriteria<'a> {
    /// Creates a new `StopStringStoppingCriteria` with the provided stop strings.
    ///
    /// # Arguments
    ///
    /// * `stop_strings` - The stop strings to check for.
    /// * `tokenizer` - The tokenizer to use to decode the input token IDs.
    ///
    /// # Returns
    ///
    /// A new `StopStringStoppingCriteria`.
    pub fn new(stop_strings: Vec<String>, tokenizer: Option<&'a Box<dyn Tokenizer>>) -> Self {
        Self {
            stop_strings,
            tokenizer,
        }
    }
}

impl StoppingCriteria for StopStringStoppingCriteria<'_> {
    fn should_stop(&self, input_ids: &[u32]) -> Result<bool> {
        if let Some(tokenizer) = self.tokenizer {
            let input_str = tokenizer.decode(input_ids, true)?;
            Ok(self
                .stop_strings
                .iter()
                .any(|stop_string| input_str.contains(stop_string)))
        } else {
            Ok(false)
        }
    }
}

pub struct StoppingCriteriaApplier<'a> {
    stopping_criteria: Vec<Box<dyn StoppingCriteria + 'a>>,
}

impl<'a> StoppingCriteriaApplier<'a> {
    pub fn new(stopping_criteria: Vec<Box<dyn StoppingCriteria>>) -> Self {
        Self { stopping_criteria }
    }

    /// Creates a new `StoppingCriteriaApplier` from the provided `GenerationConfig` and provided
    /// additional stopping criterias.
    ///
    /// # Arguments
    ///
    /// * `configuration` - The generation configuration.
    /// * `stopping_criteria` - Optional additional stopping criteria.
    ///
    /// # Returns
    ///
    /// A new `StoppingCriteriaApplier`.
    pub fn from_configuration(
        configuration: &GenerationConfig,
        stopping_criteria: Option<Vec<Box<dyn StoppingCriteria>>>,
        tokenizer: Option<&'a Box<dyn Tokenizer>>,
    ) -> Result<Self> {
        let mut stopping_criteria = stopping_criteria.unwrap_or_else(|| vec![]);

        if let Some(eos_token_id) = configuration.get_eos_token_id() {
            let eos_token_criteria = EosTokenStoppingCriteria::new(eos_token_id.clone());
            stopping_criteria.push(Box::new(eos_token_criteria));
        }

        if let Some(stop_strings) = configuration.get_stop_strings() {
            if tokenizer.is_none() {
                return Err(Error::MissingGenerateParam(
                    "tokenizer".to_string(),
                    " `tokenizer` must be provided in `generate` method if `stop_strings` are provided in the generation configuration.".to_string(),
                ));
            }

            let stop_string_criteria =
                StopStringStoppingCriteria::new(stop_strings.clone(), tokenizer);
            stopping_criteria.push(Box::new(stop_string_criteria));
        }

        Ok(Self { stopping_criteria })
    }

    /// Checks if generation should stop based on the provided input token IDs.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - The input token IDs.
    ///
    /// # Returns
    ///
    /// Whether generation should stop.
    pub fn should_stop(&self, input_ids: &[u32]) -> Result<bool> {
        for criteria in &self.stopping_criteria {
            if criteria.should_stop(input_ids)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
