use candle_holder::{Error, Result};
use tokenizers::{
    models::bpe::{Merges, Vocab, BPE},
    processors::{byte_level::ByteLevel, roberta::RobertaProcessing},
    NormalizerWrapper, PaddingDirection, Tokenizer as CoreTokenizer, TokenizerImpl,
};

use crate::{
    from_pretrained::TokenizerInfo,
    impl_tokenizer,
    tokenizer::{Tokenizer, TokenizerBuilder},
};

const ROBERTA_MAX_LENGTH: usize = 512;
const ROBERTA_BOS_TOKEN: &str = "<s>";
const ROBERTA_CLS_TOKEN: &str = "<s>";
const ROBERTA_EOS_TOKEN: &str = "</s>";
const ROBERTA_MASK_TOKEN: &str = "<mask>";
const ROBERTA_PAD_TOKEN: &str = "<pad>";
const ROBERTA_SEP_TOKEN: &str = "</s>";
const ROBERTA_UNK_TOKEN: &str = "<unk>";

/// Roberta Tokenizer
#[derive(Debug)]
pub struct RobertaTokenizer {
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

impl_tokenizer!(RobertaTokenizer, PaddingDirection::Right);

/// `RobertaTokenizer` builder
pub struct RobertaTokenizerBuilder {
    tokenizer_info: TokenizerInfo,
}

impl RobertaTokenizerBuilder {
    fn build_model(&self, vocab: Vocab, merges: Merges) -> Result<BPE> {
        BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .map_err(|e| Error::TokenizerBuildError(e.to_string()))
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
        let vocab = self.tokenizer_info.vocab.take().ok_or_else(|| {
            Error::TokenizerBuildError(
                "Cannot build RobertaTokenizer without 'vocab.json'.".to_string(),
            )
        })?;
        let merges = self.tokenizer_info.merges.take().ok_or_else(|| {
            Error::TokenizerBuildError(
                "Cannot build RobertaTokenizer without 'merges.txt'.".to_string(),
            )
        })?;
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
        let max_length = self
            .tokenizer_info
            .config
            .as_ref()
            .and_then(|config| config.model_max_length)
            .unwrap_or(ROBERTA_MAX_LENGTH);
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
            tokenizer,
            max_length,
            bos_token: Some(bos_token),
            cls_token: Some(cls_token),
            eos_token: Some(eos_token),
            mask_token: Some(mask_token),
            pad_token: Some(pad_token),
            sep_token: Some(sep_token),
            unk_token: Some(unk_token),
        })
    }
}
