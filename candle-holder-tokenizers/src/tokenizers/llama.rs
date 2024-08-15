use crate::{
    from_pretrained::TokenizerInfo,
    impl_tokenizer,
    tokenizer::{Tokenizer, TokenizerBuilder},
};
use candle_holder::{Error, Result};
use tokenizers::{PaddingDirection, Tokenizer as CoreTokenizer};

const LLAMA_MAX_LENGTH: usize = 4096;
const LLAMA_BOS_TOKEN: &str = "<s>";
const LLAMA_EOS_TOKEN: &str = "</s>";

/// Llama Tokenizer
#[derive(Debug)]
pub struct LlamaTokenizer {
    tokenizer: CoreTokenizer,
    padding_side: PaddingDirection,
    max_length: usize,
    bos_token: Option<String>,
    cls_token: Option<String>,
    eos_token: Option<String>,
    mask_token: Option<String>,
    pad_token: Option<String>,
    sep_token: Option<String>,
    unk_token: Option<String>,
}

impl_tokenizer!(LlamaTokenizer);

/// `LlamaTokenizer` builder
pub struct LlamaTokenizerBuilder {
    tokenizer_info: TokenizerInfo,
    padding_side: Option<PaddingDirection>,
}

impl LlamaTokenizerBuilder {}

impl TokenizerBuilder<LlamaTokenizer> for LlamaTokenizerBuilder {
    fn new(tokenizer_info: TokenizerInfo, padding_side: Option<PaddingDirection>) -> Self {
        LlamaTokenizerBuilder {
            tokenizer_info,
            padding_side,
        }
    }

    fn get_tokenizer_info(&self) -> &TokenizerInfo {
        &self.tokenizer_info
    }

    fn build_tokenizer(&mut self) -> Result<CoreTokenizer> {
        Err(Error::TokenizerBuildError(
            "LlamaTokenizer cannot be build from scratch".to_string(),
        ))
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
            padding_side: self.padding_side.unwrap_or(PaddingDirection::Right),
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
