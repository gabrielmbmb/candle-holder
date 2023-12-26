use anyhow::{Error, Result};
use tokenizers::{
    models::bpe::{Merges, Vocab, BPE},
    processors::{byte_level::ByteLevel, roberta::RobertaProcessing},
    NormalizerWrapper, Tokenizer, TokenizerImpl,
};

use crate::tokenizer;

const ROBERTA_BOS_TOKEN: &str = "<s>";
const ROBERTA_EOS_TOKEN: &str = "</s>";
const ROBERTA_SEP_TOKEN: &str = "</s>";
const ROBERTA_CLS_TOKEN: &str = "<s>";
const ROBERTA_UNK_TOKEN: &str = "<unk>";
const ROBERTA_PAD_TOKEN: &str = "<pad>";
const ROBERTA_MASK_TOKEN: &str = "<mask>";

pub struct RobertaTokenizer {}

impl RobertaTokenizer {
    pub fn from_vocab_and_merges(
        vocab: Vocab,
        merges: Merges,
        config: tokenizer::TokenizerConfig,
    ) -> Tokenizer {
        let cls_token = config.cls_token.unwrap_or(ROBERTA_CLS_TOKEN.to_string());
        let sep_token = config.sep_token.unwrap_or(ROBERTA_SEP_TOKEN.to_string());

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();
        let post_processor = RobertaProcessing::new((sep_token, 2), (cls_token, 0))
            .trim_offsets(true)
            .add_prefix_space(false);
        let pre_tokenizer = ByteLevel::new(false, true, true);
        let decoder = ByteLevel::new(true, true, true);
        let mut tokenizer: TokenizerImpl<
            BPE,
            NormalizerWrapper,
            ByteLevel,
            RobertaProcessing,
            ByteLevel,
        > = TokenizerImpl::new(bpe);

        tokenizer
            .with_pre_tokenizer(pre_tokenizer)
            .with_post_processor(post_processor)
            .with_decoder(decoder);

        Tokenizer::from(tokenizer)
    }

    pub fn from_tokenizer_info(tokenizer_info: tokenizer::TokenizerInfo) -> Result<Tokenizer> {
        if let Some(tokenizer_file_path) = tokenizer_info.tokenizer_file_path {
            return Tokenizer::from_file(tokenizer_file_path).map_err(Error::msg);
        }

        if let Some(vocab) = tokenizer_info.vocab {
            if let Some(merges) = tokenizer_info.merges {
                if let Some(config) = tokenizer_info.config {
                    return Ok(RobertaTokenizer::from_vocab_and_merges(
                        vocab, merges, config,
                    ));
                }
            }
        }

        Err(Error::msg("Could not build RobertaTokenizer"))
    }
}
