use std::{collections::HashMap, fs};

use crate::{
    models::{bert::BertTokenizer, roberta::tokenizer::RobertaTokenizer},
    FromPretrainedParameters,
};
use anyhow::{bail, Error, Result};
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use tokenizers::{
    models::bpe::{Merges, Vocab},
    Encoding, Tokenizer,
};

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

const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const VOCAB_TXT_FILE: &str = "vocab.txt";
const VOCAB_JSON_FILE: &str = "vocab.json";
const MERGES_FILE: &str = "merges.txt";

lazy_static! {
    static ref MODEL_TYPE_TO_TOKENIZER_CLASS: HashMap<String, String> = {
        let mut map = HashMap::new();
        map.insert("bert".to_string(), "BertTokenizer".to_string());
        map.insert("roberta".to_string(), "RobertaTokenizer".to_string());
        map
    };
}

pub fn load_vocab_txt(vocab_txt_file_path: std::path::PathBuf) -> Result<Vocab> {
    let vocab = fs::read_to_string(vocab_txt_file_path)?
        .lines()
        .enumerate()
        .fold(HashMap::<String, u32>::new(), |mut map, (idx, line)| {
            map.insert(line.to_string(), idx as u32);
            map
        });
    Ok(vocab)
}

pub fn load_vocab_json(vocab_json_file_path: std::path::PathBuf) -> Result<Vocab> {
    let vocab = fs::read_to_string(vocab_json_file_path)?;
    let vocab: Vocab = serde_json::from_str(vocab.as_str())?;
    Ok(vocab)
}

pub fn load_merges(merges_file_path: std::path::PathBuf) -> Result<Merges> {
    let merges = fs::read_to_string(merges_file_path)?.lines().skip(1).fold(
        Vec::<(String, String)>::new(),
        |mut vec, line| {
            let line = line.to_string();
            let merge = line.split_once(' ').unwrap();
            let merge = (merge.0.to_string(), merge.1.to_string());
            vec.push(merge);
            vec
        },
    );
    Ok(merges)
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
    pub vocab: Option<Vocab>,
    pub merges: Option<Merges>,
}

impl TokenizerInfo {
    fn new(
        config: Option<TokenizerConfig>,
        model_config: Option<serde_json::Value>,
        tokenizer_file_path: Option<std::path::PathBuf>,
        vocab: Option<Vocab>,
        merges: Option<Merges>,
    ) -> Self {
        TokenizerInfo {
            config,
            model_config,
            tokenizer_file_path,
            vocab,
            merges,
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
                if let Some(tokenizer_class) = MODEL_TYPE_TO_TOKENIZER_CLASS.get(model_type) {
                    return tokenizer_class;
                }
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
    let model_config = match api.get("config.json") {
        Ok(model_config_file_path) => {
            let model_config = fs::read_to_string(model_config_file_path)?;
            let model_config: serde_json::Value = serde_json::from_str(&model_config)?;
            Some(model_config)
        }
        Err(_) => None,
    };

    // Try to load `tokenizer.json` config
    let tokenizer_file_path = match api.get(TOKENIZER_FILE) {
        Ok(tokenizer_file_path) => Some(tokenizer_file_path),
        Err(_) => None,
    };

    // Try to load `vocab.json` or `vocab.txt`
    let vocab = match api.get(VOCAB_JSON_FILE) {
        Ok(vocab_json_file_path) => load_vocab_json(vocab_json_file_path).ok(),
        Err(_) => match api.get(VOCAB_TXT_FILE) {
            Ok(vocab_txt_file_path) => load_vocab_txt(vocab_txt_file_path).ok(),
            Err(_) => None,
        },
    };

    // Try to load `merges.txt`
    let merges = match api.get(MERGES_FILE) {
        Ok(merges_file_path) => load_merges(merges_file_path).ok(),
        Err(_) => None,
    };

    Ok(TokenizerInfo::new(
        tokenizer_config,
        model_config,
        tokenizer_file_path,
        vocab,
        merges,
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
                    )*
                    _ => bail!(format!("Could not determine tokenizer class")),
                };

                tokenizer
            }
        }
    };
}

impl_auto_tokenizer_from_pretrained_method!(
    AutoTokenizer,
    ("BertTokenizer", BertTokenizer),
    ("RobertaTokenizer", RobertaTokenizer)
);

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
