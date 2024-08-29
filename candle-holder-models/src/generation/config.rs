use candle_holder::utils::serde::deserialize_single_or_vec;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EarlyStopping {
    Bool(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopStrings {
    Single(String),
    Multiple(Vec<String>),
}

/// The generation configuration of a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// The maximum number of tokens the sequence can have. Corresponds to the number of input
    /// tokens + `max_new_tokens` that will be generated. Its effect is overriden if
    /// `max_new_tokens` is set. The default is `20`.
    #[serde(default)]
    pub max_length: usize,
    /// The maximun number of tokens to generate.
    pub max_new_tokens: Option<usize>,
    /// The minimun number of tokens the sequence to be generated should have. Corresponds to the
    /// number of input tokens + `min_new_tokens` that will be generated.
    pub min_length: Option<usize>,
    /// A string or a list of strings that should terminate the generation if the model outputs
    /// them.
    #[serde(deserialize_with = "deserialize_single_or_vec")]
    pub stop_strings: Option<Vec<String>>,
    /// Whether or not to use sampling. If `false`, then greedy decoding will be used. The default
    /// is `false`.
    #[serde(default)]
    pub do_sample: bool,
    /// Whether the model should use the past key/values attentions to speed up decoding.
    #[serde(default)]
    pub use_cache: bool,
    /// The value used to modulate the next token probabilities. The default is `1.0`.
    #[serde(default)]
    pub temperature: f64,
    /// The number of highest probability vocabulary tokens to keep for top-k-filtering. The
    /// default is `50`.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// The cumulative probability threshold for nucleus sampling. The default is `1.0`.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// The value to modulate the logits of tokens that have already appeared in the sequence to
    /// reduce the probability of them being selected again. Valid values are in the range of
    /// `0.0` to `infinity`. A value of `1.0` means no penalty. The default is `1.0`.
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    /// The ID of the PAD token.
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    /// The ID of the BOS token.
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    /// The IDs of the EOS tokens.
    #[serde(default, deserialize_with = "deserialize_single_or_vec")]
    pub eos_token_id: Option<Vec<u32>>,
}

impl GenerationConfig {
    pub fn get_max_length(&self) -> usize {
        self.max_length
    }

    pub fn get_max_new_tokens(&self) -> Option<usize> {
        self.max_new_tokens
    }

    pub fn get_min_length(&self) -> Option<usize> {
        self.min_length
    }

    pub fn get_stop_strings(&self) -> Option<&Vec<String>> {
        self.stop_strings.as_ref()
    }

    pub fn get_do_sample(&self) -> bool {
        self.do_sample
    }

    pub fn get_use_cache(&self) -> bool {
        self.use_cache
    }

    pub fn get_temperature(&self) -> f64 {
        self.temperature
    }

    pub fn get_top_k(&self) -> Option<usize> {
        self.top_k
    }

    pub fn get_top_p(&self) -> Option<f32> {
        self.top_p
    }

    pub fn get_repetition_penalty(&self) -> Option<f64> {
        self.repetition_penalty
    }

    pub fn get_pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    pub fn get_bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn get_eos_token_id(&self) -> Option<&Vec<u32>> {
        self.eos_token_id.as_ref()
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 20,
            max_new_tokens: None,
            min_length: None,
            stop_strings: None,
            do_sample: false,
            use_cache: true,
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(1.0),
            repetition_penalty: Some(1.0),
            pad_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
        }
    }
}
