use candle_holder::{Error, Result};
use tokenizers::models::bpe::Vocab;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::{
    decoders::wordpiece::WordPiece as WordPieceDecoder, pre_tokenizers::bert::BertPreTokenizer,
    processors::template::TemplateProcessing, TokenizerImpl,
};

use crate::config::TokenizerConfig;
use crate::from_pretrained::TokenizerInfo;
use crate::impl_tokenizer;
use crate::tokenizer::TokenizerBuilder;

const BERT_MAX_LENGTH: usize = 512;
const BERT_CLS_TOKEN: &str = "[CLS]";
const BERT_MASK_TOKEN: &str = "[MASK]";
const BERT_PAD_TOKEN: &str = "[PAD]";
const BERT_SEP_TOKEN: &str = "[SEP]";
const BERT_UNK_TOKEN: &str = "[UNK]";

/// BertTokenizer
#[derive(Debug)]
pub struct BertTokenizer {
    tokenizer: CoreTokenizer,
    padding_side: PaddingSide,
    max_length: usize,
    bos_token: Option<String>,
    cls_token: Option<String>,
    eos_token: Option<String>,
    mask_token: Option<String>,
    pad_token: Option<String>,
    sep_token: Option<String>,
    unk_token: Option<String>,
    chat_template: Option<ChatTemplate>,
}

impl_tokenizer!(BertTokenizer);

/// `BerTokenizer` builder.
pub struct BertTokenizerBuilder {
    tokenizer_info: TokenizerInfo,
    padding_side: Option<PaddingSide>,
}

impl BertTokenizerBuilder {
    fn build_normalizer(&self, config: &TokenizerConfig) -> BertNormalizer {
        BertNormalizer::new(
            config.clean_up_tokenization_spaces.unwrap_or(true),
            config.tokenize_chinese_charts.unwrap_or(true),
            config.strip_accents,
            config.do_lower_case.unwrap_or(true),
        )
    }

    fn build_pre_tokenizer(&self) -> BertPreTokenizer {
        BertPreTokenizer {}
    }

    fn build_model(&self, vocab: Vocab, unk_token: String) -> Result<WordPiece> {
        WordPiece::builder()
            .vocab(vocab)
            .unk_token(unk_token)
            .continuing_subword_prefix("##".to_string())
            .max_input_chars_per_word(100)
            .build()
            .map_err(|e| Error::TokenizerBuildError(e.to_string()))
    }

    fn build_post_processor(
        &self,
        sep_token: (String, u32),
        cls_token: (String, u32),
    ) -> Result<TemplateProcessing> {
        TemplateProcessing::builder()
            .try_single(format!("{} $A {}", cls_token.0, sep_token.0))
            .map_err(Error::TokenizerBuildError)?
            .try_pair(format!(
                "{} $A:0 {} $B:1 {}:1",
                cls_token.0, sep_token.0, sep_token.0
            ))
            .map_err(|e| Error::TokenizerBuildError(e.to_string()))?
            .special_tokens(vec![cls_token, sep_token])
            .build()
            .map_err(|e| Error::TokenizerBuildError(e.to_string()))
    }

    fn build_decoder(&self) -> WordPieceDecoder {
        WordPieceDecoder::new("##".to_string(), true)
    }
}

impl TokenizerBuilder<BertTokenizer> for BertTokenizerBuilder {
    fn new(tokenizer_info: TokenizerInfo, padding_side: Option<PaddingSide>) -> Self {
        BertTokenizerBuilder {
            tokenizer_info,
            padding_side,
        }
    }

    fn get_tokenizer_info(&self) -> &TokenizerInfo {
        &self.tokenizer_info
    }

    fn build_tokenizer(&mut self) -> Result<CoreTokenizer> {
        let vocab = self.tokenizer_info.vocab.take().ok_or_else(|| {
            Error::TokenizerBuildError(
                "Cannot build BertTokenizer without 'vocab.txt'.".to_string(),
            )
        })?;
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
            .ok_or_else(|| Error::TokenizerMissingConfig)?;

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
            padding_side: self.padding_side.clone().unwrap_or(PaddingSide::Right),
            max_length,
            bos_token: None,
            cls_token: Some(cls_token),
            eos_token: None,
            mask_token: Some(mask_token),
            pad_token: Some(pad_token),
            sep_token: Some(sep_token),
            unk_token: Some(unk_token),
            chat_template: None,
        })
    }
}
