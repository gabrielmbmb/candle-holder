use anyhow::{Error, Result};
use tokenizers::{
    models::bpe::{Merges, Vocab, BPE},
    processors::{byte_level::ByteLevel, roberta::RobertaProcessing},
    NormalizerWrapper, Tokenizer as CoreTokenizer, TokenizerImpl,
};

use crate::tokenizer::{Tokenizer, TokenizerBuilder, TokenizerInfo};

const ROBERTA_BOS_TOKEN: &str = "<s>";
const ROBERTA_CLS_TOKEN: &str = "<s>";
const ROBERTA_EOS_TOKEN: &str = "</s>";
const ROBERTA_MASK_TOKEN: &str = "<mask>";
const ROBERTA_PAD_TOKEN: &str = "<pad>";
const ROBERTA_SEP_TOKEN: &str = "</s>";
const ROBERTA_UNK_TOKEN: &str = "<unk>";

pub struct RobertaTokenizer {
    tokenizer: CoreTokenizer,
    bos_token: String,
    cls_token: String,
    eos_token: String,
    mask_token: String,
    pad_token: String,
    sep_token: String,
    unk_token: String,
}

impl Tokenizer for RobertaTokenizer {
    fn get_tokenizer(&self) -> &CoreTokenizer {
        &self.tokenizer
    }
}

pub struct RobertaTokenizerBuilder {
    tokenizer_info: TokenizerInfo,
}

impl RobertaTokenizerBuilder {
    fn build_model(&self, vocab: Vocab, merges: Merges) -> Result<BPE> {
        BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(Error::msg)
    }

    fn build_pre_tokenizer(&self) -> ByteLevel {
        ByteLevel::new(false, true, true)
    }

    fn build_post_processor(
        &self,
        sep_token: (String, u32),
        cls_token: (String, u32),
    ) -> RobertaProcessing {
        RobertaProcessing::new(sep_token, cls_token)
            .trim_offsets(true)
            .add_prefix_space(false)
    }

    fn build_decoder(&self) -> ByteLevel {
        ByteLevel::new(true, true, true)
    }
}

impl TokenizerBuilder<RobertaTokenizer> for RobertaTokenizerBuilder {
    fn new(tokenizer_info: TokenizerInfo) -> Self {
        RobertaTokenizerBuilder { tokenizer_info }
    }

    fn get_tokenizer_info(&self) -> &TokenizerInfo {
        &self.tokenizer_info
    }

    fn build_tokenizer(&mut self) -> Result<CoreTokenizer> {
        let vocab = self
            .tokenizer_info
            .vocab
            .take()
            .ok_or_else(|| Error::msg("Cannot build RobertaTokenizer without 'vocab.json'."))?;
        let merges = self
            .tokenizer_info
            .merges
            .take()
            .ok_or_else(|| Error::msg("Cannot build RobertaTokenizer without 'merges.txt'."))?;
        let cls_token = self
            .tokenizer_info
            .get_cls_token()
            .unwrap_or(ROBERTA_CLS_TOKEN.to_string());
        let sep_token = self
            .tokenizer_info
            .get_sep_token()
            .unwrap_or(ROBERTA_SEP_TOKEN.to_string());
        let cls_token_id = *vocab.get(&cls_token).unwrap_or(&0);
        let sep_token_id = *vocab.get(&sep_token).unwrap_or(&2);

        let mut tokenizer: TokenizerImpl<
            BPE,
            NormalizerWrapper,
            ByteLevel,
            RobertaProcessing,
            ByteLevel,
        > = TokenizerImpl::new(self.build_model(vocab, merges)?);

        tokenizer
            .with_pre_tokenizer(self.build_pre_tokenizer())
            .with_post_processor(
                self.build_post_processor((sep_token, sep_token_id), (cls_token, cls_token_id)),
            )
            .with_decoder(self.build_decoder());

        Ok(CoreTokenizer::from(tokenizer))
    }

    fn build_with_tokenizer(&self, tokenizer: CoreTokenizer) -> Result<RobertaTokenizer> {
        let bos_token = self
            .tokenizer_info
            .get_bos_token()
            .unwrap_or(ROBERTA_BOS_TOKEN.to_string());
        let cls_token = self
            .tokenizer_info
            .get_cls_token()
            .unwrap_or(ROBERTA_CLS_TOKEN.to_string());
        let eos_token = self
            .tokenizer_info
            .get_eos_token()
            .unwrap_or(ROBERTA_EOS_TOKEN.to_string());
        let mask_token = self
            .tokenizer_info
            .get_mask_token()
            .unwrap_or(ROBERTA_MASK_TOKEN.to_string());
        let pad_token = self
            .tokenizer_info
            .get_pad_token()
            .unwrap_or(ROBERTA_PAD_TOKEN.to_string());
        let sep_token = self
            .tokenizer_info
            .get_sep_token()
            .unwrap_or(ROBERTA_SEP_TOKEN.to_string());
        let unk_token = self
            .tokenizer_info
            .get_unk_token()
            .unwrap_or(ROBERTA_UNK_TOKEN.to_string());

        Ok(RobertaTokenizer {
            tokenizer: CoreTokenizer::from(tokenizer),
            bos_token,
            cls_token,
            eos_token,
            mask_token,
            pad_token,
            sep_token,
            unk_token,
        })
    }
}
