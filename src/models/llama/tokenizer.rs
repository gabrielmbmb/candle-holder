use crate::{
    impl_tokenizer,
    tokenizer::{Tokenizer, TokenizerBuilder, TokenizerInfo},
};
use anyhow::{Error, Result};
use tokenizers::{PaddingDirection, Tokenizer as CoreTokenizer};

const LLAMA_MAX_LENGTH: usize = 4096;
const LLAMA_BOS_TOKEN: &str = "<s>";
const LLAMA_EOS_TOKEN: &str = "</s>";

pub struct LlamaTokenizer {
    tokenizer: CoreTokenizer,
    max_length: usize,
    bos_token: Option<String>,
    cls_token: Option<String>,
    eos_token: Option<String>,
    mask_token: Option<String>,
    pad_token: Option<String>,
    sep_token: Option<String>,
    unk_token: Option<String>,
}

impl_tokenizer!(LlamaTokenizer, PaddingDirection::Right);

pub struct LlamaTokenizerBuilder {
    tokenizer_info: TokenizerInfo,
}

impl LlamaTokenizerBuilder {}

impl TokenizerBuilder<LlamaTokenizer> for LlamaTokenizerBuilder {
    fn new(tokenizer_info: TokenizerInfo) -> Self {
        LlamaTokenizerBuilder { tokenizer_info }
    }

    fn get_tokenizer_info(&self) -> &TokenizerInfo {
        &self.tokenizer_info
    }

    fn build_tokenizer(&mut self) -> Result<CoreTokenizer> {
        Err(Error::msg("LlamaTokenizer cannot be build from scratch"))
    }

    fn build_with_tokenizer(&self, tokenizer: CoreTokenizer) -> Result<LlamaTokenizer> {
        let max_length = self
            .tokenizer_info
            .config
            .as_ref()
            .and_then(|config| config.model_max_length)
            .unwrap_or(LLAMA_MAX_LENGTH);
        let bos_token = self
            .tokenizer_info
            .get_bos_token()
            .unwrap_or(LLAMA_BOS_TOKEN.to_string());
        let eos_token = self
            .tokenizer_info
            .get_eos_token()
            .unwrap_or(LLAMA_EOS_TOKEN.to_string());

        Ok(LlamaTokenizer {
            tokenizer,
            max_length,
            bos_token: Some(bos_token),
            cls_token: None,
            eos_token: Some(eos_token.clone()),
            mask_token: None,
            pad_token: Some(eos_token),
            sep_token: None,
            unk_token: None,
        })
    }
}
