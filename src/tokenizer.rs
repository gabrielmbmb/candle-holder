use std::{collections::HashMap, fs};

use crate::{
    models::bert::{BertTokenizer, BertTokenizerBuilder},
    FromPretrainedParameters,
};
use anyhow::{bail, Error, Result};
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use lazy_static::lazy_static;
use serde::{Deserialize, Deserializer, Serialize};
use tokenizers::{
    models::bpe::{Merges, Vocab},
    Encoding, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer as CoreTokenizer,
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
const SPECIAL_TOKENS_MAP_FILE: &str = "special_tokens_map.json";

lazy_static! {
    static ref MODEL_TYPE_TO_TOKENIZER_CLASS: HashMap<String, String> = {
        let mut map = HashMap::new();
        map.insert("bert".to_string(), "BertTokenizer".to_string());
        map.insert("llama".to_string(), "LlamaTokenizer".to_string());
        map.insert("roberta".to_string(), "RobertaTokenizer".to_string());
        map
    };
}

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

pub fn load_vocab_json(file_path: std::path::PathBuf) -> Result<Vocab> {
    let vocab = fs::read_to_string(file_path)?;
    let vocab: Vocab = serde_json::from_str(vocab.as_str())?;
    Ok(vocab)
}

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

#[derive(Debug, Serialize, Deserialize)]
struct SpecialToken {
    content: String,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    single_word: bool,
}

fn deserialize_special_token<'de, D>(deserializer: D) -> Result<Option<SpecialToken>, D::Error>
where
    D: Deserializer<'de>,
{
    Deserialize::deserialize(deserializer).map(|v| {
        if let serde_json::Value::String(s) = v {
            Some(SpecialToken {
                content: s,
                lstrip: false,
                normalized: false,
                rstrip: false,
                single_word: false,
            })
        } else {
            serde_json::from_value(v).ok()
        }
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpecialTokensMap {
    #[serde(deserialize_with = "deserialize_special_token", default)]
    cls_token: Option<SpecialToken>,
    #[serde(deserialize_with = "deserialize_special_token", default)]
    mask_token: Option<SpecialToken>,
    #[serde(deserialize_with = "deserialize_special_token", default)]
    pad_token: Option<SpecialToken>,
    #[serde(deserialize_with = "deserialize_special_token", default)]
    sep_token: Option<SpecialToken>,
    #[serde(deserialize_with = "deserialize_special_token", default)]
    bos_token: Option<SpecialToken>,
    #[serde(deserialize_with = "deserialize_special_token", default)]
    eos_token: Option<SpecialToken>,
    #[serde(deserialize_with = "deserialize_special_token", default)]
    unk_token: Option<SpecialToken>,
}

impl SpecialTokensMap {
    pub fn get_cls_token(&self) -> Option<String> {
        self.cls_token.as_ref().map(|t| t.content.clone())
    }

    pub fn get_mask_token(&self) -> Option<String> {
        self.mask_token.as_ref().map(|t| t.content.clone())
    }

    pub fn get_pad_token(&self) -> Option<String> {
        self.pad_token.as_ref().map(|t| t.content.clone())
    }

    pub fn get_sep_token(&self) -> Option<String> {
        self.sep_token.as_ref().map(|t| t.content.clone())
    }

    pub fn get_bos_token(&self) -> Option<String> {
        self.bos_token.as_ref().map(|t| t.content.clone())
    }

    pub fn get_eos_token(&self) -> Option<String> {
        self.eos_token.as_ref().map(|t| t.content.clone())
    }

    pub fn get_unk_token(&self) -> Option<String> {
        self.unk_token.as_ref().map(|t| t.content.clone())
    }
}

pub fn load_special_tokens_map(file_path: std::path::PathBuf) -> Result<SpecialTokensMap> {
    let special_tokens_map = fs::read_to_string(file_path)?;
    let special_tokens_map: SpecialTokensMap = serde_json::from_str(&special_tokens_map)?;
    Ok(special_tokens_map)
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
    pub add_prefix_space: Option<bool>,
    pub added_tokens_decoder: Option<HashMap<String, AddedToken>>,
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
    pub special_tokens_map: Option<SpecialTokensMap>,
}

pub enum SpecialTokenName {
    Cls,
    Mask,
    Pad,
    Sep,
    Bos,
    Eos,
    Unk,
}

impl TokenizerInfo {
    pub fn get_cls_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Cls)
    }

    pub fn get_mask_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Mask)
    }

    pub fn get_pad_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Pad)
    }

    pub fn get_sep_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Sep)
    }

    pub fn get_bos_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Bos)
    }

    pub fn get_eos_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Eos)
    }

    pub fn get_unk_token(&self) -> Option<String> {
        self.get_special_token(SpecialTokenName::Unk)
    }

    fn get_special_token(&self, special_token_name: SpecialTokenName) -> Option<String> {
        // First, try from special tokens map
        if let Some(special_tokens_map) = &self.special_tokens_map {
            match special_token_name {
                SpecialTokenName::Cls => {
                    if let Some(cls_token) = &special_tokens_map.cls_token {
                        return Some(cls_token.content.clone());
                    }
                }
                SpecialTokenName::Mask => {
                    if let Some(mask_token) = &special_tokens_map.mask_token {
                        return Some(mask_token.content.clone());
                    }
                }
                SpecialTokenName::Pad => {
                    if let Some(pad_token) = &special_tokens_map.pad_token {
                        return Some(pad_token.content.clone());
                    }
                }
                SpecialTokenName::Sep => {
                    if let Some(sep_token) = &special_tokens_map.sep_token {
                        return Some(sep_token.content.clone());
                    }
                }
                SpecialTokenName::Bos => {
                    if let Some(bos_token) = &special_tokens_map.bos_token {
                        return Some(bos_token.content.clone());
                    }
                }
                SpecialTokenName::Eos => {
                    if let Some(eos_token) = &special_tokens_map.eos_token {
                        return Some(eos_token.content.clone());
                    }
                }
                SpecialTokenName::Unk => {
                    if let Some(unk_token) = &special_tokens_map.unk_token {
                        return Some(unk_token.content.clone());
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

#[derive(Debug)]
pub enum Padding {
    Longest,
    MaxLength,
    Fixed(usize),
}

#[derive(Debug)]
pub struct BatchEncoding {
    input_ids: Tensor,
    token_type_ids: Tensor,
    encodings: Vec<Encoding>,
}

impl BatchEncoding {
    pub fn new(input_ids: Tensor, token_type_ids: Tensor, encodings: Vec<Encoding>) -> Self {
        BatchEncoding {
            input_ids,
            token_type_ids,
            encodings,
        }
    }

    pub fn get_input_ids(&self) -> &Tensor {
        &self.input_ids
    }

    pub fn get_token_type_ids(&self) -> &Tensor {
        &self.token_type_ids
    }

    pub fn get_encodings(&self) -> &Vec<Encoding> {
        &self.encodings
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        self.input_ids = self.input_ids.to_device(device)?;
        self.token_type_ids = self.token_type_ids.to_device(device)?;
        Ok(())
    }
}

pub trait Tokenizer {
    fn get_tokenizer(&self) -> &CoreTokenizer;

    fn get_tokenizer_mut(&mut self) -> &mut CoreTokenizer;

    fn get_tokenizer_with_padding(&mut self, padding: Padding) -> Result<&CoreTokenizer> {
        let pad_token = self
            .get_pad_token()
            .ok_or_else(|| {
                Error::msg(
                    "Cannot get tokenizer with padding config because it doesn't have a pad token.",
                )
            })?
            .to_string();

        let pad_id = self.get_pad_token_id().ok_or_else(|| {
            Error::msg(
                "Cannot get tokenizer with padding config because it doesn't have a pad token id.",
            )
        })?;

        let direction = self.get_padding_side();

        let strategy = match padding {
            Padding::Longest => PaddingStrategy::BatchLongest,
            Padding::MaxLength => PaddingStrategy::Fixed(self.get_max_length()),
            Padding::Fixed(length) => PaddingStrategy::Fixed(length),
        };

        let tokenizer = self.get_tokenizer_mut();

        tokenizer.with_padding(Some(PaddingParams {
            strategy,
            direction,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token,
        }));

        Ok(tokenizer)
    }

    fn get_padding_side(&self) -> PaddingDirection;
    fn get_max_length(&self) -> usize;
    fn get_bos_token(&self) -> Option<&str>;
    fn get_cls_token(&self) -> Option<&str>;
    fn get_eos_token(&self) -> Option<&str>;
    fn get_mask_token(&self) -> Option<&str>;
    fn get_pad_token(&self) -> Option<&str>;
    fn get_sep_token(&self) -> Option<&str>;
    fn get_unk_token(&self) -> Option<&str>;

    fn get_token_id(&self, token: &str) -> Option<u32> {
        self.get_tokenizer().token_to_id(token)
    }

    fn get_bos_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_bos_token()?)
    }

    fn get_cls_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_cls_token()?)
    }

    fn get_eos_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_eos_token()?)
    }

    fn get_mask_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_mask_token()?)
    }

    fn get_pad_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_pad_token()?)
    }

    fn get_sep_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_sep_token()?)
    }

    fn get_unk_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_unk_token()?)
    }

    fn encode(
        &mut self,
        inputs: Vec<String>,
        add_special_tokens: bool,
        padding: Option<Padding>,
    ) -> Result<BatchEncoding> {
        let tokenizer = match padding {
            Some(padding) => self.get_tokenizer_with_padding(padding)?,
            None => self.get_tokenizer(),
        };

        let encodings = tokenizer
            .encode_batch(inputs, add_special_tokens)
            .map_err(Error::msg)?;

        let mut input_ids: Vec<Vec<u32>> = Vec::new();
        let mut token_type_ids: Vec<Vec<u32>> = Vec::new();

        for encoding in &encodings {
            input_ids.push(encoding.get_ids().to_vec());
            token_type_ids.push(encoding.get_type_ids().to_vec());
        }

        let input_ids = Tensor::new(input_ids, &Device::Cpu)?;
        let token_type_ids = Tensor::new(token_type_ids, &Device::Cpu)?;

        Ok(BatchEncoding::new(input_ids, token_type_ids, encodings))
    }

    fn encode_sequence_pairs(
        &mut self,
        sequence_pairs: Vec<(String, String)>,
        add_special_tokens: bool,
        padding: Option<Padding>,
    ) -> Result<BatchEncoding> {
        let tokenizer = match padding {
            Some(padding) => self.get_tokenizer_with_padding(padding)?,
            None => self.get_tokenizer(),
        };

        let encodings = tokenizer
            .encode_batch(sequence_pairs, add_special_tokens)
            .map_err(Error::msg)?;

        let mut input_ids: Vec<Vec<u32>> = Vec::new();
        let mut token_type_ids: Vec<Vec<u32>> = Vec::new();

        for encoding in &encodings {
            input_ids.push(encoding.get_ids().to_vec());
            token_type_ids.push(encoding.get_type_ids().to_vec());
        }

        let input_ids = Tensor::new(input_ids, &Device::Cpu)?;
        let token_type_ids = Tensor::new(token_type_ids, &Device::Cpu)?;

        Ok(BatchEncoding::new(input_ids, token_type_ids, encodings))
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokenizer = self.get_tokenizer();
        tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(Error::msg)
    }
}

#[macro_export]
macro_rules! impl_tokenizer {
    ($tokenizer_type:ty, $padding_dir:expr) => {
        impl Tokenizer for $tokenizer_type {
            fn get_tokenizer(&self) -> &CoreTokenizer {
                &self.tokenizer
            }

            fn get_tokenizer_mut(&mut self) -> &mut CoreTokenizer {
                &mut self.tokenizer
            }

            fn get_padding_side(&self) -> PaddingDirection {
                $padding_dir
            }

            fn get_max_length(&self) -> usize {
                self.max_length
            }

            fn get_bos_token(&self) -> Option<&str> {
                self.bos_token.as_deref()
            }

            fn get_cls_token(&self) -> Option<&str> {
                Some(self.cls_token.as_str())
            }

            fn get_eos_token(&self) -> Option<&str> {
                self.eos_token.as_deref()
            }

            fn get_mask_token(&self) -> Option<&str> {
                Some(self.mask_token.as_str())
            }

            fn get_pad_token(&self) -> Option<&str> {
                Some(self.pad_token.as_str())
            }

            fn get_sep_token(&self) -> Option<&str> {
                Some(self.sep_token.as_str())
            }

            fn get_unk_token(&self) -> Option<&str> {
                Some(self.unk_token.as_str())
            }
        }
    };
}

pub trait TokenizerBuilder<T: Tokenizer> {
    fn new(tokenizer_info: TokenizerInfo) -> Self;
    fn get_tokenizer_info(&self) -> &TokenizerInfo;

    fn build_tokenizer(&mut self) -> Result<CoreTokenizer> {
        unimplemented!("build_tokenizer method not implemented")
    }

    fn build_with_tokenizer(&self, _tokenizer: CoreTokenizer) -> Result<T> {
        unimplemented!("with_tokenizer method not implemented")
    }

    fn build(&mut self) -> Result<T> {
        let tokenizer_info = self.get_tokenizer_info();

        // Try first to build from `tokenizer.json`
        if let Some(tokenizer_file_path) = &tokenizer_info.tokenizer_file_path {
            let tokenizer = CoreTokenizer::from_file(tokenizer_file_path).map_err(Error::msg)?;
            return self.build_with_tokenizer(tokenizer);
        }

        // Try to build from `vocab.txt`, `vocab.json` and `merges.txt`
        let tokenizer = self.build_tokenizer()?;
        self.build_with_tokenizer(tokenizer)
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

pub struct AutoTokenizer {}

#[macro_export]
macro_rules! impl_auto_tokenizer_from_pretrained_method {
    ($auto_tokenizer_struct:ident, $(($tokenizer_class:expr, $tokenizer_struct:ident, $tokenizer_builder_struct:ident)), *) => {
        impl $auto_tokenizer_struct {
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                params: Option<FromPretrainedParameters>
            ) -> Result<Box<dyn Tokenizer>> {
                let tokenizer_info = from_pretrained(repo_id, params)?;

                let tokenizer = match tokenizer_info.get_tokenizer_class() {
                    $(
                        $tokenizer_class => $tokenizer_builder_struct::new(tokenizer_info).build()?,
                    )*
                    _ => bail!(format!("Could not determine tokenizer class")),
                };

                Ok(Box::new(tokenizer))
            }
        }
    };
}

impl_auto_tokenizer_from_pretrained_method!(
    AutoTokenizer,
    ("BertTokenizer", BertTokenizer, BertTokenizerBuilder)
);

// Implement `from_pretrained` method for each tokenizer
#[macro_export]
macro_rules! impl_tokenizer_from_pretrained_method {
    ($tokenizer_struct:ident, $tokenizer_builder_struct:ident) => {
        impl $tokenizer_struct {
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                params: Option<FromPretrainedParameters>,
            ) -> Result<Box<dyn Tokenizer>> {
                let tokenizer_info = from_pretrained(repo_id, params)?;
                let tokenizer = $tokenizer_builder_struct::new(tokenizer_info).build()?;
                Ok(Box::new(tokenizer))
            }
        }
    };
}

impl_tokenizer_from_pretrained_method!(BertTokenizer, BertTokenizerBuilder);
