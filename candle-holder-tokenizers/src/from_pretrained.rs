use std::{collections::HashMap, fs};

use candle_holder::{
    get_repo_api,
    utils::from_pretrained::{load_model_config, FromPretrainedParameters, MODEL_CONFIG_FILE},
    Result,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use lazy_static::lazy_static;
use serde::{Deserialize, Deserializer, Serialize};
use tokenizers::{
    models::bpe::{Merges, Vocab},
    AddedToken,
};

use crate::config::TokenizerConfig;

const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";
const VOCAB_TXT_FILE: &str = "vocab.txt";
const VOCAB_JSON_FILE: &str = "vocab.json";
const MERGES_FILE: &str = "merges.txt";
const SPECIAL_TOKENS_MAP_FILE: &str = "special_tokens_map.json";

lazy_static! {
    static ref MODEL_TYPE_TO_TOKENIZER_CLASS: HashMap<String, String> = {
        let mut map = HashMap::new();
        map.insert("bert".to_string(), "BertTokenizer".to_string());
        map.insert("llama".to_string(), "LlamaTokenizer".to_string());
        map.insert("roberta".to_string(), "RobertaTokenizer".to_string());
        map
    };
    static ref IGNORE_TOKENIZER_CLASSES: Vec<&'static str> =
        vec!["PreTrainedTokenizerFast", "GPT2Tokenizer"];
}

/// An enum representing the special tokens of a tokenizer.
pub enum SpecialTokenName {
    Cls,
    Mask,
    Pad,
    Sep,
    Bos,
    Eos,
    Unk,
}

/// A struct representing a `SpecialToken` with a default value. Used to deserialize special tokens
/// from JSON files when the token is only a string and it doesn't contain any additional
/// information such as `single_word`, `lstrip`, `rstrip`, `normalized`, or `special`.
#[derive(Debug, Serialize, Deserialize)]
struct AddedTokenWithDefaults {
    /// The added token.
    added_token: AddedToken,
}

fn deserialize_special_token<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<AddedTokenWithDefaults>, D::Error>
where
    D: Deserializer<'de>,
{
    Deserialize::deserialize(deserializer).map(|v| {
        if let serde_json::Value::String(s) = v {
            Some(AddedTokenWithDefaults {
                added_token: AddedToken {
                    content: s,
                    single_word: false,
                    lstrip: false,
                    rstrip: false,
                    normalized: false,
                    special: true,
                },
            })
        } else {
            serde_json::from_value(v).ok()
        }
    })
}

/// A struct representing the special tokens map of a tokenizer.
#[derive(Debug, Serialize, Deserialize)]
pub struct SpecialTokensMap {
    /// The CLS token.
    #[serde(deserialize_with = "deserialize_special_token", default)]
    cls_token: Option<AddedTokenWithDefaults>,
    /// The MASK token.
    #[serde(deserialize_with = "deserialize_special_token", default)]
    mask_token: Option<AddedTokenWithDefaults>,
    /// The PAD token.
    #[serde(deserialize_with = "deserialize_special_token", default)]
    pad_token: Option<AddedTokenWithDefaults>,
    /// The SEP token.
    #[serde(deserialize_with = "deserialize_special_token", default)]
    sep_token: Option<AddedTokenWithDefaults>,
    /// The BOS token.
    #[serde(deserialize_with = "deserialize_special_token", default)]
    bos_token: Option<AddedTokenWithDefaults>,
    /// The EOS token.
    #[serde(deserialize_with = "deserialize_special_token", default)]
    eos_token: Option<AddedTokenWithDefaults>,
    /// The UNK token.
    #[serde(deserialize_with = "deserialize_special_token", default)]
    unk_token: Option<AddedTokenWithDefaults>,
}

impl SpecialTokensMap {
    pub fn get_cls_token(&self) -> Option<&AddedToken> {
        self.cls_token.as_ref().map(|t| &t.added_token)
    }

    pub fn get_mask_token(&self) -> Option<&AddedToken> {
        self.mask_token.as_ref().map(|t| &t.added_token)
    }

    pub fn get_pad_token(&self) -> Option<&AddedToken> {
        self.pad_token.as_ref().map(|t| &t.added_token)
    }

    pub fn get_sep_token(&self) -> Option<&AddedToken> {
        self.sep_token.as_ref().map(|t| &t.added_token)
    }

    pub fn get_bos_token(&self) -> Option<&AddedToken> {
        self.bos_token.as_ref().map(|t| &t.added_token)
    }

    pub fn get_eos_token(&self) -> Option<&AddedToken> {
        self.eos_token.as_ref().map(|t| &t.added_token)
    }

    pub fn get_unk_token(&self) -> Option<&AddedToken> {
        self.unk_token.as_ref().map(|t| &t.added_token)
    }
}

/// A struct containing all the required information to load a tokenizer model.
#[derive(Debug)]
pub struct TokenizerInfo {
    /// The configuration of the tokenizer.
    pub config: Option<TokenizerConfig>,
    /// The configuration of the model.
    pub model_config: Option<serde_json::Value>,
    /// The path to the `tokenizer.json` file if it exists.
    pub tokenizer_file_path: Option<std::path::PathBuf>,
    /// The vocabulary of the tokenizer.
    pub vocab: Option<Vocab>,
    /// The merges of the tokenizer.
    pub merges: Option<Merges>,
    /// The special tokens of the tokenizer.
    pub special_tokens_map: Option<SpecialTokensMap>,
}

impl TokenizerInfo {
    fn new(
        config: Option<TokenizerConfig>,
        model_config: Option<serde_json::Value>,
        tokenizer_file_path: Option<std::path::PathBuf>,
        vocab: Option<Vocab>,
        merges: Option<Merges>,
        special_tokens_map: Option<SpecialTokensMap>,
    ) -> Self {
        TokenizerInfo {
            config,
            model_config,
            tokenizer_file_path,
            vocab,
            merges,
            special_tokens_map,
        }
    }

    pub fn get_config(&self) -> Option<&TokenizerConfig> {
        self.config.as_ref()
    }

    pub fn get_tokenizer_class(&self) -> &str {
        if let Some(config) = &self.config {
            if let Some(tokenizer_class) = &config.tokenizer_class {
                if !IGNORE_TOKENIZER_CLASSES.contains(&tokenizer_class.as_str()) {
                    return tokenizer_class;
                }
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

    /// Gets the `String` representation of the CLS token.
    pub fn get_cls_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Cls)
    }

    /// Gets the `String` representation of the MASK token.
    pub fn get_mask_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Mask)
    }

    /// Gets the `String` representation of the PAD token.
    pub fn get_pad_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Pad)
    }

    /// Gets the `String` representation of the SEP token.
    pub fn get_sep_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Sep)
    }

    /// Gets the `String` representation of the BOS token.
    pub fn get_bos_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Bos)
    }

    /// Gets the `String` representation of the EOS token.
    pub fn get_eos_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Eos)
    }

    /// Gets the `String` representation of the UNK token.
    pub fn get_unk_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Unk)
    }

    /// Gets the `String` representation of a special token. It will first try to get the token from
    /// the special tokens map and if it doesn't exist, it will try to get it from the config file.
    ///
    /// # Arguments
    ///
    /// - `special_token_name` - The name of the special token.
    ///
    /// # Returns
    ///
    /// The `String` representation of the special token.
    fn get_special_token(&self, special_token_name: SpecialTokenName) -> Option<String> {
        // First, try from special tokens map
        if let Some(special_tokens_map) = &self.special_tokens_map {
            match special_token_name {
                SpecialTokenName::Cls => {
                    if let Some(cls_token) = &special_tokens_map.cls_token {
                        return Some(cls_token.added_token.content.clone());
                    }
                }
                SpecialTokenName::Mask => {
                    if let Some(mask_token) = &special_tokens_map.mask_token {
                        return Some(mask_token.added_token.content.clone());
                    }
                }
                SpecialTokenName::Pad => {
                    if let Some(pad_token) = &special_tokens_map.pad_token {
                        return Some(pad_token.added_token.content.clone());
                    }
                }
                SpecialTokenName::Sep => {
                    if let Some(sep_token) = &special_tokens_map.sep_token {
                        return Some(sep_token.added_token.content.clone());
                    }
                }
                SpecialTokenName::Bos => {
                    if let Some(bos_token) = &special_tokens_map.bos_token {
                        return Some(bos_token.added_token.content.clone());
                    }
                }
                SpecialTokenName::Eos => {
                    if let Some(eos_token) = &special_tokens_map.eos_token {
                        return Some(eos_token.added_token.content.clone());
                    }
                }
                SpecialTokenName::Unk => {
                    if let Some(unk_token) = &special_tokens_map.unk_token {
                        return Some(unk_token.added_token.content.clone());
                    }
                }
            }
        }

        // Then, try from config file
        if let Some(config) = &self.config {
            match special_token_name {
                SpecialTokenName::Cls => {
                    if let Some(cls_token) = &config.cls_token {
                        return Some(cls_token.clone());
                    }
                }
                SpecialTokenName::Mask => {
                    if let Some(mask_token) = &config.mask_token {
                        return Some(mask_token.clone());
                    }
                }
                SpecialTokenName::Pad => {
                    if let Some(pad_token) = &config.pad_token {
                        return Some(pad_token.clone());
                    }
                }
                SpecialTokenName::Sep => {
                    if let Some(sep_token) = &config.sep_token {
                        return Some(sep_token.clone());
                    }
                }
                SpecialTokenName::Bos => {
                    if let Some(bos_token) = &config.bos_token {
                        return Some(bos_token.clone());
                    }
                }
                SpecialTokenName::Eos => {
                    if let Some(eos_token) = &config.eos_token {
                        return Some(eos_token.clone());
                    }
                }
                SpecialTokenName::Unk => {
                    if let Some(unk_token) = &config.unk_token {
                        return Some(unk_token.clone());
                    }
                }
            }
        }

        None
    }
}

/// Loads the vocabulary of the tokenizer model from a text file.
///
/// # Arguments
///
/// - `file_path` - The path to the vocabulary file.
///
/// # Returns
///
/// The vocabulary as a `HashMap` where the key is the token and the value is the index.
pub fn load_vocab_txt(file_path: std::path::PathBuf) -> Result<Vocab> {
    let vocab = fs::read_to_string(file_path)?.lines().enumerate().fold(
        HashMap::<String, u32>::new(),
        |mut map, (idx, line)| {
            map.insert(line.to_string(), idx as u32);
            map
        },
    );
    Ok(vocab)
}

/// Loads the vocabulary of the tokenizer model from a JSON file.
///
/// # Arguments
///
/// - `file_path` - The path to the vocabulary file.
///
/// # Returns
///
/// The vocabulary as a `HashMap` where the key is the token and the value is the index.
pub fn load_vocab_json(file_path: std::path::PathBuf) -> Result<Vocab> {
    let vocab = fs::read_to_string(file_path)?;
    let vocab: Vocab = serde_json::from_str(vocab.as_str())?;
    Ok(vocab)
}

/// Loads the merges of the tokenizer model from a text file.
///
/// # Arguments
///
/// - `file_path` - The path to the merges file.
///
/// # Returns
///
/// The merges as a `Vec` of tuples where the first element is the merge and the second element is
/// the new token.
pub fn load_merges(file_path: std::path::PathBuf) -> Result<Merges> {
    let merges = fs::read_to_string(file_path)?.lines().skip(1).fold(
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

/// Loads the special tokens map of the tokenizer model from a JSON file.
///
/// # Arguments
///
/// - `file_path` - The path to the special tokens map file.
///
/// # Returns
///
/// The special tokens map as a `SpecialTokensMap` struct.
pub fn load_special_tokens_map(file_path: std::path::PathBuf) -> Result<SpecialTokensMap> {
    let special_tokens_map = fs::read_to_string(file_path)?;
    let special_tokens_map: SpecialTokensMap = serde_json::from_str(&special_tokens_map)?;
    Ok(special_tokens_map)
}

/// Gets all the information and files needed to load a tokenizers from a Hugging Face Hub
/// repository.
///
/// # Arguments
///
/// * `repo_id` - The ID of the repository to load the tokenizer from.
/// * `params` - Optional parameters to specify the revision, user agent, and auth token.
///
/// # Returns
///
/// A `TokenizerInfo` struct containing all the information needed to load a tokenizer.
pub fn from_pretrained<I: AsRef<str>>(
    repo_id: I,
    params: Option<FromPretrainedParameters>,
) -> Result<TokenizerInfo> {
    let api = get_repo_api(repo_id.as_ref(), params)?;

    let tokenizer_config = match api.get(TOKENIZER_CONFIG_FILE) {
        Ok(tokenizer_config_file) => {
            let tokenizer_config = TokenizerConfig::from_file(tokenizer_config_file)?;
            Some(tokenizer_config)
        }
        Err(_) => None,
    };

    // Used to determine the tokenizer class if the other methods fail
    let model_config = match api.get(MODEL_CONFIG_FILE) {
        Ok(model_config_file_path) => load_model_config(model_config_file_path).ok(),
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

    // Try to load `special_tokens_map.json`
    let special_tokens_map = match api.get(SPECIAL_TOKENS_MAP_FILE) {
        Ok(special_tokens_map_file) => load_special_tokens_map(special_tokens_map_file).ok(),
        Err(_) => None,
    };

    Ok(TokenizerInfo::new(
        tokenizer_config,
        model_config,
        tokenizer_file_path,
        vocab,
        merges,
        special_tokens_map,
    ))
}
