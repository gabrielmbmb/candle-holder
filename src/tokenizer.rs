use std::{collections::HashMap, fs};

use crate::{models::bert::BertTokenizer, FromPretrainedParameters};
use anyhow::{bail, Error, Result};
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use tokenizers::{Encoding, Tokenizer};

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

pub fn load_vocab(vocab_file_path: std::path::PathBuf) -> Result<HashMap<String, u32>> {
    let vocab = fs::read_to_string(vocab_file_path)?
        .lines()
        .enumerate()
        .fold(HashMap::<String, u32>::new(), |mut map, (idx, line)| {
            map.insert(line.to_string(), idx as u32);
            map
        });
    Ok(vocab)
}

const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const VOCAB_FILE: &str = "vocab.txt";
const MERGE_FILE: &str = "merges.txt";

lazy_static! {
    static ref MODEL_TYPE_TO_TOKENIZER_CLASS: HashMap<String, String> = {
        let mut map = HashMap::new();
        map.insert("bert".to_string(), "BertTokenizer".to_string());
        map
    };
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddedToken {
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub added_tokens_decoder: Option<HashMap<String, AddedToken>>,
    pub clean_up_tokenization_spaces: Option<bool>,
    pub cls_token: Option<String>,
    pub do_basic_tokenize: Option<bool>,
    pub do_lower_case: Option<bool>,
    pub mask_token: Option<String>,
    pub model_max_length: Option<u32>,
    pub never_split: Option<Vec<String>>,
    pub pad_token: Option<String>,
    pub sep_token: Option<String>,
    pub strip_accents: Option<bool>,
    pub tokenize_chinese_charts: Option<bool>,
    pub tokenizer_class: Option<String>,
    pub unk_token: Option<String>,
}

impl TokenizerConfig {
    pub fn from_file(file: std::path::PathBuf) -> Result<Self> {
        let config = std::fs::read_to_string(file)?;
        let tokenizer_config: TokenizerConfig =
            serde_json::from_str(&config).map_err(Error::msg)?;
        Ok(tokenizer_config)
    }
}

#[derive(Debug)]
pub struct TokenizerInfo {
    pub config: Option<TokenizerConfig>,
    pub model_config: Option<serde_json::Value>,
    pub tokenizer_file_path: Option<std::path::PathBuf>,
    pub vocab: Option<HashMap<String, u32>>,
}

impl TokenizerInfo {
    fn new(
        config: Option<TokenizerConfig>,
        model_config: Option<serde_json::Value>,
        tokenizer_file_path: Option<std::path::PathBuf>,
        vocab: Option<HashMap<String, u32>>,
    ) -> Self {
        TokenizerInfo {
            config,
            model_config,
            tokenizer_file_path,
            vocab,
        }
    }

    pub fn get_tokenizer(&self) -> Result<Tokenizer> {
        if let Some(tokenizer_file_path) = &self.tokenizer_file_path {
            return Ok(Tokenizer::from_file(tokenizer_file_path).map_err(Error::msg)?);
        }

        Err(Error::msg("Could not load tokenizer"))
    }

    pub fn get_tokenizer_class(&self) -> &str {
        if let Some(config) = &self.config {
            if let Some(tokenizer_class) = &config.tokenizer_class {
                return tokenizer_class;
            }
        }

        if let Some(model_config) = &self.model_config {
            if let Some(model_type) = model_config["model_type"].as_str() {
                return MODEL_TYPE_TO_TOKENIZER_CLASS.get(model_type).unwrap();
            }
        }

        ""
    }
}

pub fn from_pretrained<I: AsRef<str>>(
    repo_id: I,
    params: Option<FromPretrainedParameters>,
) -> Result<TokenizerInfo> {
    let params = params.unwrap_or_default();

    let repo = Repo::with_revision(
        repo_id.as_ref().to_string(),
        RepoType::Model,
        params.revision,
    );

    let api = Api::new()?;
    let api = api.repo(repo);

    let tokenizer_config = match api.get(TOKENIZER_CONFIG_FILE) {
        Ok(tokenizer_config_file) => {
            let tokenizer_config = TokenizerConfig::from_file(tokenizer_config_file)?;
            Some(tokenizer_config)
        }
        Err(_) => None,
    };

    // Used to determine the tokenizer class if the other methods fail
    let model_config: Option<serde_json::Value> = match api.get("config.json") {
        Ok(model_config_file_path) => {
            let model_config = fs::read_to_string(model_config_file_path)?;
            let model_config: serde_json::Value = serde_json::from_str(&model_config)?;
            Some(model_config)
        }
        Err(_) => None,
    };

    // Try to load `tokenizer.json` config
    let tokenizer_file_path: Option<std::path::PathBuf> = match api.get(TOKENIZER_FILE) {
        Ok(tokenizer_file_path) => Some(tokenizer_file_path),
        Err(_) => None,
    };

    // Try to load `vocab.txt`
    let vocab: Option<HashMap<String, u32>> = match api.get(VOCAB_FILE) {
        Ok(vocab_file_path) => load_vocab(vocab_file_path).ok(),
        Err(_) => None,
    };

    Ok(TokenizerInfo::new(
        tokenizer_config,
        model_config,
        tokenizer_file_path,
        vocab,
    ))
}

pub trait PreTrainedTokenizer {}

pub struct AutoTokenizer {}

macro_rules! impl_auto_tokenizer_from_pretrained_method {
    ($auto_tokenizer_struct:ident, $(($tokenizer_class:expr, $tokenizer_struct:ident)), *) => {
        impl $auto_tokenizer_struct {
            pub fn from_pretrained<S: AsRef<str>>(repo_id: S, params: Option<FromPretrainedParameters>) -> Result<Tokenizer> {
                let tokenizer_info = from_pretrained(repo_id, params)?;

                let tokenizer = match tokenizer_info.get_tokenizer_class() {
                    $(
                        $tokenizer_class => $tokenizer_struct::from_tokenizer_info(tokenizer_info),
                        _ => bail!(format!("Could not determine tokenizer class")),
                    )*
                };

                tokenizer
            }
        }
    };
}

impl_auto_tokenizer_from_pretrained_method!(AutoTokenizer, ("BertTokenizer", BertTokenizer));

// Implement `from_pretrained` method for each tokenizer
macro_rules! impl_tokenizer_from_pretrained_method {
    ($tokenizer_struct:ident) => {
        impl $tokenizer_struct {
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                params: Option<FromPretrainedParameters>,
            ) -> Result<Tokenizer> {
                let tokenizer_info = from_pretrained(repo_id, params)?;
                $tokenizer_struct::from_tokenizer_info(tokenizer_info)
            }
        }
    };
}

impl_tokenizer_from_pretrained_method!(BertTokenizer);
