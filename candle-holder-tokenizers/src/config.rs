use std::collections::HashMap;

use candle_holder::{Error, Result};
use serde::{Deserialize, Serialize};
use tokenizers::AddedToken;

/// Tokenizer configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub add_prefix_space: Option<bool>,
    pub added_tokens_decoder: Option<HashMap<u32, AddedToken>>,
    pub bos_token: Option<String>,
    pub clean_up_tokenization_spaces: Option<bool>,
    pub cls_token: Option<String>,
    pub eos_token: Option<String>,
    pub errors: Option<String>,
    pub do_basic_tokenize: Option<bool>,
    pub do_lower_case: Option<bool>,
    pub mask_token: Option<String>,
    pub model_max_length: Option<usize>,
    pub never_split: Option<Vec<String>>,
    pub pad_token: Option<String>,
    pub sep_token: Option<String>,
    pub strip_accents: Option<bool>,
    pub tokenize_chinese_charts: Option<bool>,
    pub tokenizer_class: Option<String>,
    pub unk_token: Option<String>,
}

impl TokenizerConfig {
    /// Loads the tokenizer config from a `tokenizer_config.json` file.
    ///
    /// # Arguments
    ///
    /// * `file` - Path to the `tokenizer_config.json` file.
    ///
    /// # Returns
    ///
    /// The tokenizer configuration.
    pub fn from_file(file: std::path::PathBuf) -> Result<Self> {
        let config = std::fs::read_to_string(file)?;
        let tokenizer_config: TokenizerConfig =
            serde_json::from_str(&config).map_err(Error::wrap)?;
        Ok(tokenizer_config)
    }
}
