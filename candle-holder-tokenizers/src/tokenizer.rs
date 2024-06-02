use crate::tokenizers::bert::{BertTokenizer, BertTokenizerBuilder};
use crate::{
    encoding::BatchEncoding,
    from_pretrained::{from_pretrained, FromPretrainedParameters, TokenizerInfo},
};
use candle_core::{DType, Device, Tensor};
use candle_holder::{bail, Error, Result};
use tokenizers::{
    AddedToken, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer as CoreTokenizer,
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

/// A thin wrapper around `tokenizers::Tokenizer` that provides additional functionality
/// for encoding and decoding sequences and to automatically load the tokenizer from a Hugging Face
/// Hub repository.
pub trait Tokenizer: std::fmt::Debug {
    fn get_tokenizer(&self) -> &CoreTokenizer;

    fn get_tokenizer_mut(&mut self) -> &mut CoreTokenizer;

    fn get_tokenizer_with_padding(&mut self, padding: Padding) -> Result<&CoreTokenizer> {
        let pad_token = self
            .get_pad_token()
            .ok_or_else(|| Error::MissingSpecialToken("pad_token".to_string()))?
            .to_string();

        let pad_id = self
            .get_pad_token_id()
            .ok_or_else(|| Error::MissingSpecialTokenId("pad_token".to_string()))?;

        let direction = self.get_padding_side();

        let strategy = match padding {
            Padding::Longest => PaddingStrategy::BatchLongest,
            Padding::MaxLength => PaddingStrategy::Fixed(self.get_max_length()),
            Padding::Fixed(length) => PaddingStrategy::Fixed(length),
        };

        let tokenizer = self.get_tokenizer_mut();

        tokenizer.with_padding(Some(PaddingParams {
            strategy,
            direction,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token,
        }));

        Ok(tokenizer)
    }

    fn get_padding_side(&self) -> PaddingDirection;
    fn get_max_length(&self) -> usize;
    fn get_bos_token(&self) -> Option<&str>;
    fn get_cls_token(&self) -> Option<&str>;
    fn get_eos_token(&self) -> Option<&str>;
    fn get_mask_token(&self) -> Option<&str>;
    fn get_pad_token(&self) -> Option<&str>;
    fn get_sep_token(&self) -> Option<&str>;
    fn get_unk_token(&self) -> Option<&str>;

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
        &mut self,
        inputs: Vec<String>,
        add_special_tokens: bool,
        padding: Option<Padding>,
    ) -> Result<BatchEncoding> {
        let tokenizer = match padding {
            Some(padding) => self.get_tokenizer_with_padding(padding)?,
            None => self.get_tokenizer(),
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
        &mut self,
        sequence_pairs: Vec<(String, String)>,
        add_special_tokens: bool,
        padding: Option<Padding>,
    ) -> Result<BatchEncoding> {
        let tokenizer = match padding {
            Some(padding) => self.get_tokenizer_with_padding(padding)?,
            None => self.get_tokenizer(),
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
}

/// A macro that implements the `Tokenizer` trait for a given tokenizer type.
#[macro_export]
macro_rules! impl_tokenizer {
    ($tokenizer_type:ty, $padding_dir:expr) => {
        impl Tokenizer for $tokenizer_type {
            fn get_tokenizer(&self) -> &CoreTokenizer {
                &self.tokenizer
            }

            fn get_tokenizer_mut(&mut self) -> &mut CoreTokenizer {
                &mut self.tokenizer
            }

            fn get_padding_side(&self) -> PaddingDirection {
                $padding_dir
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
        }
    };
}

/// A trait that defines the methods required to build a `Tokenizer`.
pub trait TokenizerBuilder<T: Tokenizer> {
    fn new(tokenizer_info: TokenizerInfo) -> Self;
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
                params: Option<FromPretrainedParameters>
            ) -> Result<Box<dyn Tokenizer>> {
                let tokenizer_info = from_pretrained(repo_id, params)?;

                let tokenizer: Result<Box<dyn Tokenizer>> = match tokenizer_info.get_tokenizer_class() {
                    $(
                        $tokenizer_class => {
                            $tokenizer_builder_struct::new(tokenizer_info)
                                .build()
                                .map(|tokenizer| Box::new(tokenizer) as Box<dyn Tokenizer>)
                        }
                    )*
                    _ => bail!(format!("Could not determine tokenizer class")),
                };

                tokenizer
            }
        }
    };
}

impl_auto_tokenizer_from_pretrained_method!(
    AutoTokenizer,
    ("BertTokenizer", BertTokenizer, BertTokenizerBuilder)
);

// Implement `from_pretrained` method for each tokenizer
#[macro_export]
macro_rules! impl_tokenizer_from_pretrained_method {
    ($tokenizer_struct:ident, $tokenizer_builder_struct:ident) => {
        impl $tokenizer_struct {
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                params: Option<FromPretrainedParameters>,
            ) -> Result<Box<dyn Tokenizer>> {
                let tokenizer_info = from_pretrained(repo_id, params)?;
                let tokenizer = $tokenizer_builder_struct::new(tokenizer_info).build()?;
                Ok(Box::new(tokenizer))
            }
        }
    };
}

impl_tokenizer_from_pretrained_method!(BertTokenizer, BertTokenizerBuilder);
