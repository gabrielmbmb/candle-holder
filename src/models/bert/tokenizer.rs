use crate::impl_tokenizer;
use crate::tokenizer::{Tokenizer, TokenizerBuilder, TokenizerConfig, TokenizerInfo};
use anyhow::{Error, Result};
use tokenizers::models::bpe::Vocab;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::{
    decoders::wordpiece::WordPiece as WordPieceDecoder, pre_tokenizers::bert::BertPreTokenizer,
    processors::template::TemplateProcessing, PaddingDirection, Tokenizer as CoreTokenizer,
    TokenizerImpl,
};

const BERT_MAX_LENGTH: usize = 512;
const BERT_CLS_TOKEN: &str = "[CLS]";
const BERT_MASK_TOKEN: &str = "[MASK]";
const BERT_PAD_TOKEN: &str = "[PAD]";
const BERT_SEP_TOKEN: &str = "[SEP]";
const BERT_UNK_TOKEN: &str = "[UNK]";

pub struct BertTokenizer {
    tokenizer: CoreTokenizer,
    max_length: usize,
    bos_token: Option<String>,
    cls_token: String,
    eos_token: Option<String>,
    mask_token: String,
    pad_token: String,
    sep_token: String,
    unk_token: String,
}

impl_tokenizer!(BertTokenizer, PaddingDirection::Right);

pub struct BertTokenizerBuilder {
    tokenizer_info: TokenizerInfo,
}

impl BertTokenizerBuilder {
    fn build_normalizer(&self, config: &TokenizerConfig) -> BertNormalizer {
        BertNormalizer::from(config)
    }

    fn build_pre_tokenizer(&self) -> BertPreTokenizer {
        BertPreTokenizer {}
    }

    fn build_model(&self, vocab: Vocab, unk_token: String) -> Result<WordPiece> {
        WordPiece::builder()
            .vocab(vocab)
            .unk_token(unk_token)
            .build()
            .map_err(Error::msg)
    }

    fn build_post_processor(
        &self,
        sep_token: (String, u32),
        cls_token: (String, u32),
    ) -> Result<TemplateProcessing> {
        TemplateProcessing::builder()
            .try_single(format!("{} $A {}", cls_token.0, sep_token.0))
            .map_err(Error::msg)?
            .try_pair(format!(
                "{} $A:0 {} $B:1 {}:1",
                cls_token.0, sep_token.0, sep_token.0
            ))
            .map_err(Error::msg)?
            .special_tokens(vec![cls_token, sep_token])
            .build()
            .map_err(Error::msg)
    }

    fn build_decoder(&self) -> WordPieceDecoder {
        WordPieceDecoder::default()
    }
}

impl TokenizerBuilder<BertTokenizer> for BertTokenizerBuilder {
    fn new(tokenizer_info: TokenizerInfo) -> Self {
        BertTokenizerBuilder { tokenizer_info }
    }

    fn get_tokenizer_info(&self) -> &TokenizerInfo {
        &self.tokenizer_info
    }

    fn build_tokenizer(&mut self) -> Result<CoreTokenizer> {
        let vocab = self
            .tokenizer_info
            .vocab
            .take()
            .ok_or_else(|| Error::msg("Cannot build BertTokenizer without 'vocab.txt'."))?;
        let cls_token = self
            .tokenizer_info
            .get_cls_token()
            .unwrap_or(BERT_CLS_TOKEN.to_string());
        let sep_token = self
            .tokenizer_info
            .get_sep_token()
            .unwrap_or(BERT_SEP_TOKEN.to_string());
        let unk_token = self
            .tokenizer_info
            .get_unk_token()
            .unwrap_or(BERT_UNK_TOKEN.to_string());
        let cls_token_id = *vocab.get(&cls_token).unwrap_or(&101u32);
        let sep_token_id = *vocab.get(&sep_token).unwrap_or(&102u32);

        let mut tokenizer: TokenizerImpl<
            WordPiece,
            BertNormalizer,
            BertPreTokenizer,
            TemplateProcessing,
            WordPieceDecoder,
        > = TokenizerImpl::new(self.build_model(vocab, unk_token.clone())?);

        let tokenizer_config = self
            .tokenizer_info
            .config
            .as_ref()
            .ok_or_else(|| Error::msg("Cannot build BertTokenizer without configuration."))?;

        tokenizer
            .with_normalizer(self.build_normalizer(tokenizer_config))
            .with_pre_tokenizer(self.build_pre_tokenizer())
            .with_post_processor(self.build_post_processor(
                (sep_token.clone(), sep_token_id),
                (cls_token.clone(), cls_token_id),
            )?)
            .with_decoder(self.build_decoder());

        Ok(CoreTokenizer::from(tokenizer))
    }

    fn build_with_tokenizer(&self, tokenizer: CoreTokenizer) -> Result<BertTokenizer> {
        let max_length = self
            .tokenizer_info
            .config
            .as_ref()
            .and_then(|config| config.model_max_length)
            .unwrap_or(BERT_MAX_LENGTH);
        let cls_token = self
            .tokenizer_info
            .get_cls_token()
            .unwrap_or(BERT_CLS_TOKEN.to_string());
        let mask_token = self
            .tokenizer_info
            .get_mask_token()
            .unwrap_or(BERT_MASK_TOKEN.to_string());
        let pad_token = self
            .tokenizer_info
            .get_pad_token()
            .unwrap_or(BERT_PAD_TOKEN.to_string());
        let sep_token = self
            .tokenizer_info
            .get_sep_token()
            .unwrap_or(BERT_SEP_TOKEN.to_string());
        let unk_token = self
            .tokenizer_info
            .get_unk_token()
            .unwrap_or(BERT_UNK_TOKEN.to_string());

        Ok(BertTokenizer {
            tokenizer,
            max_length,
            bos_token: None,
            cls_token,
            eos_token: None,
            mask_token,
            pad_token,
            sep_token,
            unk_token,
        })
    }
}

impl From<&TokenizerConfig> for BertNormalizer {
    fn from(config: &TokenizerConfig) -> Self {
        let strip_accents = config.strip_accents;
        let lowercase = config.do_lower_case.unwrap_or(false);
        BertNormalizer::new(true, true, strip_accents, lowercase)
    }
}
