use candle_core::{DType, Device, Tensor};
use candle_holder::{Error, FromPretrainedParameters, Result};
use tokenizers::{
    AddedToken, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer as CoreTokenizer,
};

use crate::chat_template::{ChatTemplate, Message};
use crate::tokenizers::bert::{BertTokenizer, BertTokenizerBuilder};
use crate::tokenizers::llama::{LlamaTokenizer, LlamaTokenizerBuilder};
use crate::tokenizers::roberta::{RobertaTokenizer, RobertaTokenizerBuilder};
use crate::{
    encoding::BatchEncoding,
    from_pretrained::{from_pretrained, TokenizerInfo},
};

/// An enum containing the available padding configurations for the tokenizer.
#[derive(Debug, Clone)]
pub enum Padding {
    /// Pad to the longest sequence in the batch.
    Longest,
    /// Pad to the maximum length of the model.
    MaxLength,
    /// Pad to a specific length.
    Fixed(usize),
}

/// An enum containing the available sides in which a tokenizer can add the padding.
#[derive(Debug, Clone)]
pub enum PaddingSide {
    /// Add the padding tokens to the left.
    Left,
    /// Add the padding tokens to the right.
    Right,
}

impl Into<PaddingDirection> for PaddingSide {
    fn into(self) -> PaddingDirection {
        match self {
            PaddingSide::Left => PaddingDirection::Left,
            PaddingSide::Right => PaddingDirection::Right,
        }
    }
}

/// An struct containing the `Tokenizer` padding options.
#[derive(Debug, Clone)]
pub struct PaddingOptions {
    /// The kind of padding to be done.
    padding: Padding,
    /// The side where the padding tokens should be added.
    side: Option<PaddingSide>,
}

impl PaddingOptions {
    pub fn new(padding: Padding, side: Option<PaddingSide>) -> Self {
        Self { padding, side }
    }

    pub fn get_padding(&self) -> &Padding {
        &self.padding
    }

    pub fn get_padding_side(&self) -> Option<&PaddingSide> {
        self.side.as_ref()
    }
}

impl Default for PaddingOptions {
    fn default() -> Self {
        Self {
            padding: Padding::Longest,
            side: Some(PaddingSide::Right),
        }
    }
}

/// A thin wrapper around `tokenizers::Tokenizer` that provides additional functionality
/// for encoding and decoding sequences and to automatically load the tokenizer from a Hugging Face
/// Hub repository.
pub trait Tokenizer: std::fmt::Debug + Send + Sync {
    fn get_tokenizer(&self) -> &CoreTokenizer;

    // TODO: can we use `Arc` so we don't need to clone it?
    fn get_tokenizer_with_padding(&self, padding: PaddingOptions) -> Result<CoreTokenizer> {
        let pad_token = self
            .get_pad_token()
            .ok_or_else(|| Error::MissingSpecialToken("pad_token".to_string()))?
            .to_string();

        let pad_id = self
            .get_pad_token_id()
            .ok_or_else(|| Error::MissingSpecialTokenId("pad_token".to_string()))?;

        let strategy = match padding.get_padding() {
            Padding::Longest => PaddingStrategy::BatchLongest,
            Padding::MaxLength => PaddingStrategy::Fixed(self.get_max_length()),
            Padding::Fixed(length) => PaddingStrategy::Fixed(length.clone()),
        };

        let padding_side = padding
            .get_padding_side()
            .unwrap_or_else(|| self.get_padding_side())
            .clone();

        let mut tokenizer = self.get_tokenizer().clone();

        tokenizer.with_padding(Some(PaddingParams {
            strategy,
            direction: padding_side.into(),
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token,
        }));

        Ok(tokenizer)
    }

    fn get_padding_side(&self) -> &PaddingSide;
    fn get_max_length(&self) -> usize;
    fn get_bos_token(&self) -> Option<&str>;
    fn get_cls_token(&self) -> Option<&str>;
    fn get_eos_token(&self) -> Option<&str>;
    fn get_mask_token(&self) -> Option<&str>;
    fn get_pad_token(&self) -> Option<&str>;
    fn get_sep_token(&self) -> Option<&str>;
    fn get_unk_token(&self) -> Option<&str>;
    fn get_chat_template(&self) -> Option<&ChatTemplate>;

    /// Get the token ID of a given token.
    ///
    /// # Arguments
    ///
    /// * `token` - A string slice representing the token.
    ///
    /// # Returns
    ///
    /// The token ID if the token exists in the tokenizer, `None` otherwise.
    fn get_token_id(&self, token: &str) -> Option<u32> {
        self.get_tokenizer().token_to_id(token)
    }

    /// Get the token ID of the BOS token.
    fn get_bos_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_bos_token()?)
    }

    /// Get the token ID of the CLS token.
    fn get_cls_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_cls_token()?)
    }

    /// Get the token ID of the EOS token.
    fn get_eos_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_eos_token()?)
    }

    /// Get the token ID of the MASK token.
    fn get_mask_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_mask_token()?)
    }

    /// Get the token ID of the PAD token.
    fn get_pad_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_pad_token()?)
    }

    /// Get the token ID of the SEP token.
    fn get_sep_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_sep_token()?)
    }

    /// Get the token ID of the UNK token.
    fn get_unk_token_id(&self) -> Option<u32> {
        self.get_token_id(self.get_unk_token()?)
    }

    /// Encodes a list of sequences.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A list of sequences to encode.
    /// * `add_special_tokens` - A flag indicating if special tokens should be added.
    /// * `padding` - An optional padding configuration.
    ///
    /// # Returns
    ///
    /// A `BatchEncoding` containing the encoded sequences.
    fn encode(
        &self,
        inputs: Vec<String>,
        add_special_tokens: bool,
        padding: Option<PaddingOptions>,
    ) -> Result<BatchEncoding> {
        let tokenizer = match padding {
            Some(padding) => self.get_tokenizer_with_padding(padding)?,
            None => self.get_tokenizer().clone(),
        };

        let encodings = tokenizer
            .encode_batch(inputs, add_special_tokens)
            .map_err(Error::msg)?;

        let mut input_ids: Vec<Vec<u32>> = Vec::new();
        let mut token_type_ids: Vec<Vec<u32>> = Vec::new();
        let mut attention_mask: Vec<Vec<u32>> = Vec::new();

        for encoding in &encodings {
            input_ids.push(encoding.get_ids().to_vec());
            token_type_ids.push(encoding.get_type_ids().to_vec());
            attention_mask.push(encoding.get_attention_mask().to_vec());
        }

        let input_ids = Tensor::new(input_ids, &Device::Cpu)?;
        let token_type_ids = Tensor::new(token_type_ids, &Device::Cpu)?;
        let attention_mask = Tensor::new(attention_mask, &Device::Cpu)?.to_dtype(DType::U8)?;

        Ok(BatchEncoding::new(
            input_ids,
            token_type_ids,
            attention_mask,
            encodings,
        ))
    }

    /// Encodes a list of sequence pairs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A list of sequences to encode.
    /// * `add_special_tokens` - A flag indicating if special tokens should be added.
    /// * `padding` - An optional padding configuration.
    ///
    /// # Returns
    ///
    /// A `BatchEncoding` containing the encoded sequences.
    fn encode_sequence_pairs(
        &self,
        sequence_pairs: Vec<(String, String)>,
        add_special_tokens: bool,
        padding: Option<PaddingOptions>,
    ) -> Result<BatchEncoding> {
        let tokenizer = match padding {
            Some(padding) => self.get_tokenizer_with_padding(padding)?,
            None => self.get_tokenizer().clone(),
        };

        let encodings = tokenizer
            .encode_batch(sequence_pairs, add_special_tokens)
            .map_err(|e| Error::TokenizerEncodingError(e.to_string()))?;

        let mut input_ids: Vec<Vec<u32>> = Vec::new();
        let mut token_type_ids: Vec<Vec<u32>> = Vec::new();
        let mut attention_mask: Vec<Vec<u32>> = Vec::new();

        for encoding in &encodings {
            input_ids.push(encoding.get_ids().to_vec());
            token_type_ids.push(encoding.get_type_ids().to_vec());
            attention_mask.push(encoding.get_attention_mask().to_vec());
        }

        let input_ids = Tensor::new(input_ids, &Device::Cpu)?;
        let token_type_ids = Tensor::new(token_type_ids, &Device::Cpu)?;
        let attention_mask = Tensor::new(attention_mask, &Device::Cpu)?.to_dtype(DType::U8)?;

        Ok(BatchEncoding::new(
            input_ids,
            token_type_ids,
            attention_mask,
            encodings,
        ))
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokenizer = self.get_tokenizer();
        tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::TokenizerEncodingError(e.to_string()))
    }

    /// Applies the chat template to a list of messages i.e. using the model chat template and the
    /// provided messages it creates an input string for the model in the expected format.
    ///
    /// # Arguments
    ///
    /// * `messages` - A list of messages to apply the chat template.
    /// * `add_generation_prompt` - A flag indicating if the generation prompt should be added.
    ///
    /// # Returns
    ///
    /// The input string for the model in the expected format.
    fn apply_chat_template(
        &self,
        messages: Vec<Message>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let chat_template = self.get_chat_template().ok_or_else(|| {
            Error::MissingChatTemplate("Chat template not found in the tokenizer".to_string())
        })?;
        chat_template.apply(messages, add_generation_prompt)
    }

    /// Applies the chat template to a list of messages and encodes the result.
    ///
    /// # Arguments
    ///
    /// * `messages` - A list of messages to apply the chat template.
    ///
    /// # Returns
    ///
    /// A `BatchEncoding` containing the encoded sequences.
    fn apply_chat_template_and_encode(
        &self,
        messages: Vec<Message>,
        add_generation_prompt: bool,
    ) -> Result<BatchEncoding> {
        let chat_template = self.get_chat_template().ok_or_else(|| {
            Error::MissingChatTemplate("Chat template not found in the tokenizer".to_string())
        })?;
        let chat = chat_template.apply(messages, add_generation_prompt)?;
        self.encode(vec![chat], false, None)
    }
}

/// A macro that implements the `Tokenizer` trait for a given tokenizer type.
#[macro_export]
macro_rules! impl_tokenizer {
    ($tokenizer_type:ty) => {
        use crate::chat_template::ChatTemplate;
        use crate::tokenizer::{PaddingSide, Tokenizer};
        use tokenizers::Tokenizer as CoreTokenizer;

        impl Tokenizer for $tokenizer_type {
            fn get_tokenizer(&self) -> &CoreTokenizer {
                &self.tokenizer
            }

            fn get_padding_side(&self) -> &PaddingSide {
                &self.padding_side
            }

            fn get_max_length(&self) -> usize {
                self.max_length
            }

            fn get_bos_token(&self) -> Option<&str> {
                self.bos_token.as_deref()
            }

            fn get_cls_token(&self) -> Option<&str> {
                self.cls_token.as_deref()
            }

            fn get_eos_token(&self) -> Option<&str> {
                self.eos_token.as_deref()
            }

            fn get_mask_token(&self) -> Option<&str> {
                self.mask_token.as_deref()
            }

            fn get_pad_token(&self) -> Option<&str> {
                self.pad_token.as_deref()
            }

            fn get_sep_token(&self) -> Option<&str> {
                self.sep_token.as_deref()
            }

            fn get_unk_token(&self) -> Option<&str> {
                self.unk_token.as_deref()
            }

            fn get_chat_template(&self) -> Option<&ChatTemplate> {
                self.chat_template.as_ref()
            }
        }
    };
}

/// A trait that defines the methods required to build a `Tokenizer`.
pub trait TokenizerBuilder<T: Tokenizer> {
    fn new(tokenizer_info: TokenizerInfo, padding_side: Option<PaddingSide>) -> Self;
    fn get_tokenizer_info(&self) -> &TokenizerInfo;

    fn build_tokenizer(&mut self) -> Result<CoreTokenizer> {
        unimplemented!("build_tokenizer method not implemented")
    }

    fn build_with_tokenizer(&self, _tokenizer: CoreTokenizer) -> Result<T> {
        unimplemented!("with_tokenizer method not implemented")
    }

    fn build(&mut self) -> Result<T> {
        let tokenizer_info = self.get_tokenizer_info();

        let mut special_tokens: Vec<AddedToken> = Vec::new();
        if let Some(ref special_tokens_map) = tokenizer_info.special_tokens_map {
            if let Some(cls_token) = special_tokens_map.get_cls_token() {
                special_tokens.push(cls_token.clone());
            }

            if let Some(mask_token) = special_tokens_map.get_mask_token() {
                special_tokens.push(mask_token.clone());
            }

            if let Some(pad_token) = special_tokens_map.get_pad_token() {
                special_tokens.push(pad_token.clone());
            }

            if let Some(sep_token) = special_tokens_map.get_sep_token() {
                special_tokens.push(sep_token.clone());
            }

            if let Some(bos_token) = special_tokens_map.get_bos_token() {
                special_tokens.push(bos_token.clone());
            }

            if let Some(eos_token) = special_tokens_map.get_eos_token() {
                special_tokens.push(eos_token.clone());
            }

            if let Some(unk_token) = special_tokens_map.get_unk_token() {
                special_tokens.push(unk_token.clone());
            }
        }

        let mut added_tokens: Vec<AddedToken> = Vec::new();
        if let Some(config) = &tokenizer_info.config {
            if let Some(added_tokens_decoder) = &config.added_tokens_decoder {
                for added_token in added_tokens_decoder.values() {
                    added_tokens.push(added_token.clone());
                }
            }
        }

        // Try to build from `tokenizer.json`. Otherwise, build from `vocab.txt`, `vocab.json` and `merges.txt`
        let mut tokenizer = match &tokenizer_info.tokenizer_file_path {
            Some(tokenizer_file_path) => {
                CoreTokenizer::from_file(tokenizer_file_path).map_err(Error::msg)?
            }
            _ => self.build_tokenizer()?,
        };

        tokenizer.add_special_tokens(&special_tokens);
        tokenizer.add_tokens(&added_tokens);

        self.build_with_tokenizer(tokenizer)
    }
}

/// Allows to automatically load a tokenizer from a Hugging Face Hub repository.
#[derive(Debug)]
pub struct AutoTokenizer {}

/// Implement the `from_pretrained` method for the `AutoTokenizer` struct.
#[macro_export]
macro_rules! impl_auto_tokenizer_from_pretrained_method {
    ($auto_tokenizer_struct:ident, $(($tokenizer_class:expr, $tokenizer_struct:ident, $tokenizer_builder_struct:ident)), *) => {
        impl $auto_tokenizer_struct {
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                padding_side: Option<PaddingSide>,
                params: Option<FromPretrainedParameters>
            ) -> Result<Box<dyn Tokenizer>> {
                let tokenizer_info = from_pretrained(repo_id, params)?;
                let tokenizer_class = tokenizer_info.get_tokenizer_class();

                let tokenizer: Result<Box<dyn Tokenizer>> = match tokenizer_class {
                    $(
                        $tokenizer_class => {
                            $tokenizer_builder_struct::new(tokenizer_info, padding_side)
                                .build()
                                .map(|tokenizer| Box::new(tokenizer) as Box<dyn Tokenizer>)
                        }
                    )*
                    _ => Err(Error::TokenizerNotImplemented(tokenizer_class.to_string())),
                };

                tokenizer
            }
        }
    };
}

// Implement `from_pretrained` method for each tokenizer
#[macro_export]
macro_rules! impl_tokenizer_from_pretrained_method {
    ($tokenizer_struct:ident, $tokenizer_builder_struct:ident) => {
        impl $tokenizer_struct {
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                padding_side: Option<PaddingSide>,
                params: Option<FromPretrainedParameters>,
            ) -> Result<Box<dyn Tokenizer>> {
                let tokenizer_info = from_pretrained(repo_id, params)?;
                let tokenizer =
                    $tokenizer_builder_struct::new(tokenizer_info, padding_side).build()?;
                Ok(Box::new(tokenizer))
            }
        }
    };
}

impl_auto_tokenizer_from_pretrained_method!(
    AutoTokenizer,
    ("BertTokenizer", BertTokenizer, BertTokenizerBuilder),
    ("LlamaTokenizer", LlamaTokenizer, LlamaTokenizerBuilder),
    (
        "RobertaTokenizer",
        RobertaTokenizer,
        RobertaTokenizerBuilder
    )
);

// BERT
impl_tokenizer_from_pretrained_method!(BertTokenizer, BertTokenizerBuilder);

// Llama
impl_tokenizer_from_pretrained_method!(LlamaTokenizer, LlamaTokenizerBuilder);

//RoBERTa
impl_tokenizer_from_pretrained_method!(RobertaTokenizer, RobertaTokenizerBuilder);
