use std::collections::HashMap;

use crate::tokenizer;
use anyhow::{Error, Result};
use tokenizers::normalizers::BertNormalizer;
use tokenizers::{
    decoders::wordpiece::WordPiece as WordPieceDecoder, pre_tokenizers::bert::BertPreTokenizer,
    processors::template::TemplateProcessing, TokenizerImpl,
};
use tokenizers::{PaddingParams, Tokenizer};

use tokenizers::models::wordpiece::WordPiece;

const BERT_UNK_TOKEN: &str = "[UNK]";
const BERT_SEP_TOKEN: &str = "[SEP]";
const BERT_PAD_TOKEN: &str = "[PAD]";
const BERT_CLS_TOKEN: &str = "[CLS]";

pub struct BertTokenizer {}

impl BertTokenizer {
    pub fn from_vocab(
        vocab: HashMap<String, u32>,
        config: tokenizer::TokenizerConfig,
    ) -> Tokenizer {
        let unk_token = config.unk_token.unwrap_or(BERT_UNK_TOKEN.to_string());
        let cls_token = config.cls_token.unwrap_or(BERT_CLS_TOKEN.to_string());
        let sep_token = config.sep_token.unwrap_or(BERT_SEP_TOKEN.to_string());

        let word_piece = WordPiece::builder()
            .vocab(vocab)
            .unk_token(unk_token)
            .build()
            .unwrap();
        let template_processing = TemplateProcessing::builder()
            .try_single(format!("{} $A {}", cls_token, sep_token))
            .unwrap()
            .try_pair(format!("{} $A {} $B:1 {}", cls_token, sep_token, sep_token))
            .unwrap()
            .special_tokens(vec![(cls_token, 101), (sep_token, 102)])
            .build()
            .unwrap();
        let pre_tokenizer = BertPreTokenizer {};
        let normalizer = BertNormalizer::default();
        let decoder = WordPieceDecoder::default();

        let mut tokenizer: TokenizerImpl<
            WordPiece,
            BertNormalizer,
            BertPreTokenizer,
            TemplateProcessing,
            WordPieceDecoder,
        > = TokenizerImpl::new(word_piece);

        tokenizer
            .with_normalizer(normalizer)
            .with_pre_tokenizer(pre_tokenizer)
            .with_post_processor(template_processing)
            .with_decoder(decoder)
            .with_padding(Some(PaddingParams::default()));

        Tokenizer::from(tokenizer)
    }

    pub fn from_tokenizer_info(tokenizer_info: tokenizer::TokenizerInfo) -> Result<Tokenizer> {
        if let Some(tokenizer) = tokenizer_info.tokenizer_file_path {
            return Tokenizer::from_file(tokenizer).map_err(Error::msg);
        }

        if let Some(vocab) = tokenizer_info.vocab {
            if let Some(config) = tokenizer_info.config {
                return Ok(Self::from_vocab(vocab, config));
            }
        }

        Err(Error::msg("Could not build BertTokenizer"))
    }
}
