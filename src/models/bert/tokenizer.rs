use std::collections::HashMap;

use crate::tokenizer;
use anyhow::{Error, Result};
use tokenizers::models::bpe::Vocab;
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

impl From<tokenizer::TokenizerConfig> for BertNormalizer {
    fn from(config: tokenizer::TokenizerConfig) -> Self {
        let strip_accents = config.strip_accents;
        let lowercase = config.do_lower_case.unwrap_or(false);
        BertNormalizer::new(true, true, strip_accents, lowercase)
    }
}

pub struct BertTokenizer {}

impl BertTokenizer {
    pub fn from_vocab(vocab: Vocab, config: tokenizer::TokenizerConfig) -> Tokenizer {
        let unk_token = config
            .unk_token
            .clone()
            .unwrap_or(BERT_UNK_TOKEN.to_string());
        let sep_token = config
            .sep_token
            .clone()
            .unwrap_or(BERT_SEP_TOKEN.to_string());
        let cls_token = config
            .cls_token
            .clone()
            .unwrap_or(BERT_CLS_TOKEN.to_string());

        let cls_token_id = *vocab.get(&cls_token).unwrap_or(&101u32);
        let sep_token_id = *vocab.get(&sep_token).unwrap_or(&102u32);

        let word_piece = WordPiece::builder()
            .vocab(vocab)
            .unk_token(unk_token)
            .build()
            .unwrap();
        let post_processor = TemplateProcessing::builder()
            .try_single(format!("{} $A {}", cls_token, sep_token))
            .unwrap()
            .try_pair(format!("{} $A {} $B:1 {}", cls_token, sep_token, sep_token))
            .unwrap()
            .special_tokens(vec![(cls_token, cls_token_id), (sep_token, sep_token_id)])
            .build()
            .unwrap();
        let pre_tokenizer = BertPreTokenizer {};
        let normalizer = BertNormalizer::from(config);
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
            .with_post_processor(post_processor)
            .with_decoder(decoder)
            // TODO: build padding from config
            .with_padding(Some(PaddingParams::default()));

        Tokenizer::from(tokenizer)
    }

    pub fn from_tokenizer_info(tokenizer_info: tokenizer::TokenizerInfo) -> Result<Tokenizer> {
        if let Some(tokenizer) = tokenizer_info.tokenizer_file_path {
            return Tokenizer::from_file(tokenizer).map_err(Error::msg);
        }

        let vocab = tokenizer_info
            .vocab
            .ok_or_else(|| Error::msg("Could not build BertTokenizer because vocab is missing"))?;
        let config = tokenizer_info.config.ok_or_else(|| {
            Error::msg("Could not build BertTokenizer because tokenizer config is missing")
        })?;

        Ok(BertTokenizer::from_vocab(vocab, config))
    }
}
