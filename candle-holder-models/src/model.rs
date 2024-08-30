use candle_core::{DType, Device, Tensor};
use candle_holder::{bail, utils::from_pretrained::FromPretrainedParameters, Result};
use candle_holder_tokenizers::Tokenizer;
use candle_nn::VarBuilder;

use crate::{
    config::PretrainedConfig,
    from_pretrained::from_pretrained,
    generation::{config::GenerationConfig, generate::generate, StoppingCriteria, TokenStreamer},
    models::{
        bert::{
            BertForMaskedLM, BertForSequenceClassification, BertForTokenClassification, BertModel,
            BERT_DTYPE,
        },
        llama::modeling::{LlamaForCausalLM, LlamaModel, LLAMA_DTYPE},
    },
    utils::cache::DynamicCache,
};

/// Parameters for the `forward` method of a `PreTrainedModel`.
pub struct ForwardParams<'a> {
    pub input_ids: Option<&'a Tensor>,
    pub attention_mask: Option<&'a Tensor>,
    pub token_type_ids: Option<&'a Tensor>,
    pub position_ids: Option<&'a Tensor>,
    pub cache: Option<&'a mut DynamicCache>,
}

impl<'a> ForwardParams<'a> {
    pub fn new(
        input_ids: Option<&'a Tensor>,
        attention_mask: Option<&'a Tensor>,
        token_type_ids: Option<&'a Tensor>,
        position_ids: Option<&'a Tensor>,
        cache: Option<&'a mut DynamicCache>,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            cache,
        }
    }

    pub fn get_input_ids(&self) -> Option<&'a Tensor> {
        self.input_ids
    }

    pub fn get_attention_mask(&self) -> Option<&'a Tensor> {
        self.attention_mask
    }

    pub fn get_token_type_ids(&self) -> Option<&'a Tensor> {
        self.token_type_ids
    }

    pub fn get_position_ids(&self) -> Option<&'a Tensor> {
        self.position_ids
    }

    pub fn get_cache(&mut self) -> Option<&mut DynamicCache> {
        self.cache.as_deref_mut()
    }
}

impl<'a> Default for ForwardParams<'a> {
    fn default() -> Self {
        Self::new(None, None, None, None, None)
    }
}

#[cfg(feature = "tokenizers")]
impl<'a> From<&'a candle_holder_tokenizers::BatchEncoding> for ForwardParams<'a> {
    fn from(encodings: &'a candle_holder_tokenizers::BatchEncoding) -> Self {
        Self::new(
            Some(encodings.get_input_ids()),
            Some(encodings.get_attention_mask()),
            Some(encodings.get_token_type_ids()),
            None,
            None,
        )
    }
}

/// Parameters for the `generate` method of a `PreTrainedModel`.
pub struct GenerationParams<'a> {
    /// The generation configuration to use. If not provided, then the model's default generation
    /// configuration in `generation_config.json` will be used. If that is not available, then
    /// default generation configuration will be used. The default value is `None`.
    pub generation_config: Option<GenerationConfig>,
    /// The tokenizer to use for decoding the generated tokens. It's not extrictly required, but
    /// some stopping criteria may depend on it. The default value is `None`.
    pub tokenizer: Option<&'a Box<dyn Tokenizer>>,
    /// The list of stopping criteria to use that will determine when to stop generating tokens.
    /// The default value is `None`.
    pub stopping_criteria: Option<Vec<Box<dyn StoppingCriteria>>>,
    /// The token streamer which will receive the next tokens as they are being generated. The
    /// default value is `None`.
    pub token_streamer: Option<Box<dyn TokenStreamer<'a> + 'a>>,
    /// A seed that will be used in the sampling of the next token. The default value is `None`.
    pub seed: Option<u64>,
}

impl Default for GenerationParams<'_> {
    fn default() -> Self {
        Self {
            generation_config: None,
            tokenizer: None,
            stopping_criteria: None,
            token_streamer: None,
            seed: None,
        }
    }
}

/// Trait for a pre-trained model.
pub trait PreTrainedModel {
    /// Loads a model from a `VarBuilder` containing the model's parameters and a model
    /// configuration.
    ///
    /// # Arguments
    ///
    /// * `vb` - The `VarBuilder` containing the model's parameters.
    /// * `config` - The model configuration.
    ///
    /// # Returns
    ///
    /// The loaded model.
    fn load(vb: VarBuilder, config: serde_json::Value) -> Result<Self>
    where
        Self: Sized;

    /// Loads a model from a `VarBuilder` containing the model's parameters, a model configuration,
    /// and a generation configuration.
    ///
    /// # Arguments
    ///
    /// * `vb` - The `VarBuilder` containing the model's parameters.
    /// * `config` - The model configuration.
    /// * `generation_config` - The generation configuration.
    ///
    /// # Returns
    ///
    /// The loaded model.
    fn load_with_generation_config(
        _vb: VarBuilder,
        _config: serde_json::Value,
        _generation_config: Option<GenerationConfig>,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        unimplemented!("`load_with_generation_config` method not implemented for this model");
    }

    /// Returns the model's configuration.
    ///
    /// # Returns
    ///
    /// The model's configuration
    fn get_config(&self) -> &PretrainedConfig;

    /// Returns the model's generation configuration
    ///
    /// # Returns
    ///
    /// The model's generation configuration
    fn get_generation_config(&self) -> &GenerationConfig {
        unimplemented!("`get_generation_config` method not implemented for this model");
    }

    /// Runs the model forward pass.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameters for the forward pass.
    ///
    /// # Returns
    ///
    /// The model output
    fn forward(&self, params: ForwardParams) -> Result<Tensor>;

    /// Generates a completion of the input sequences.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - The input sequences.
    /// * `generation_config` - Optional generation configuration. If not provided, the model's
    ///  default generation configuration will be used (if available). Otherwise, default
    ///  generation configuration will be used.
    /// * `seed` - Optional seed for the random number generator.
    ///
    /// # Returns
    ///
    /// A vector containing vectors of token ids for each input sequence.
    fn generate<'a>(
        &self,
        input_ids: &Tensor,
        params: GenerationParams<'a>,
    ) -> Result<Vec<Vec<u32>>> {
        let generation_config = params
            .generation_config
            .unwrap_or_else(|| self.get_generation_config().clone());
        generate(
            self,
            input_ids,
            &generation_config,
            params.tokenizer,
            params.stopping_criteria,
            params.token_streamer,
            params.seed,
        )
    }
}

/// Implement `from_pretrained` method for a model struct.
#[macro_export]
macro_rules! impl_from_pretrained_method {
    ($model_struct:ident, $default_dtype:expr, $load_generation_config:expr) => {
        impl $model_struct {
            /// Loads a model from the Hugging Face Hub.
            ///
            /// # Arguments
            ///
            /// * `identifier` - The repository id of the model to load.
            /// * `device` - The device to run the model on.
            /// * `dtype` - The numeric type in which the model parameters should be loaded.
            /// * `params` - Optional parameters to specify the revision, user agent, and auth token.
            ///
            /// # Returns
            ///
            /// The loaded model.
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                device: &Device,
                dtype: Option<DType>,
                params: Option<FromPretrainedParameters>,
            ) -> Result<Self> {
                let model_info = from_pretrained(repo_id, params)?;
                let config = model_info
                    .get_config()
                    .expect("Model config not found. Cannot load the model.")
                    .clone();
                let dtype = dtype.unwrap_or($default_dtype);
                let vb = model_info.get_var_builder(dtype, device)?;
                if $load_generation_config {
                    Self::load_with_generation_config(
                        vb,
                        config,
                        model_info.get_generation_config().cloned(),
                    )
                } else {
                    Self::load(vb, config)
                }
            }
        }
    };
}

/// Implement `from_pretrained` method for the `AutoModel` struct.
#[macro_export]
macro_rules! impl_auto_model_from_pretrained_method {
    ($auto_model_struct:ident, $(($model_type:expr, $model_struct:ident, $default_dtype:expr, $load_generation_config:expr)), *) => {
        impl $auto_model_struct {
            /// Loads a model from the Hugging Face Hub.
            ///
            /// # Arguments
            ///
            /// * `identifier` - The repository id of the model to load.
            /// * `device` - The device to run the model on.
            /// * `dtype` - The numeric type in which the model parameters should be loaded.
            /// * `params` - Optional parameters to specify the revision, user agent, and auth token.
            ///
            /// # Returns
            ///
            /// The loaded model.
            pub fn from_pretrained<S: AsRef<str>>(
                repo_id: S,
                device: &Device,
                dtype: Option<DType>,
                params: Option<FromPretrainedParameters>,
            ) -> Result<Box<dyn PreTrainedModel>> {
                let model_info = from_pretrained(repo_id, params)?;
                let config = model_info
                    .get_config()
                    .expect("Model config not found. Cannot load the model.")
                    .clone();
                let model_type = config["model_type"].as_str().unwrap();

                let model: Result<Box<dyn PreTrainedModel>> = match model_type {
                    $(
                        $model_type => {
                            let dtype = dtype.unwrap_or($default_dtype);
                            let vb = model_info.get_var_builder(dtype, device)?;
                            if $load_generation_config {
                                Ok(Box::new($model_struct::load_with_generation_config(vb, config, model_info.get_generation_config().cloned())?))
                            } else {
                                Ok(Box::new($model_struct::load(vb, config)?))
                            }
                        },
                    )*
                    _ => bail!(format!("Model '{}' type not supported", model_type)),
                };

                model
            }
        }
    };
}

/// Allows to automatically load a `PreTrainedModel` from a Hugging Face Hub repository.
#[derive(Debug)]
pub struct AutoModel {}

impl_auto_model_from_pretrained_method!(
    AutoModel,
    ("bert", BertModel, BERT_DTYPE, false),
    ("llama", LlamaModel, LLAMA_DTYPE, false)
);

/// Alows to automatically load a `PreTrainedModel` for sequence classification from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForSequenceClassification {}

impl_auto_model_from_pretrained_method!(
    AutoModelForSequenceClassification,
    ("bert", BertForSequenceClassification, BERT_DTYPE, false)
);

/// Alows to automatically load a `PreTrainedModel` for token classification from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForTokenClassification {}

impl_auto_model_from_pretrained_method!(
    AutoModelForTokenClassification,
    ("bert", BertForTokenClassification, BERT_DTYPE, false)
);

/// Alows to automatically load a `PreTrainedModel` for masked language modeling from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForMaskedLM {}

impl_auto_model_from_pretrained_method!(
    AutoModelForMaskedLM,
    ("bert", BertForMaskedLM, BERT_DTYPE, false)
);

/// Alows to automatically load a `PreTrainedModel` for causal language modeling from a Hugging Face
/// Hub repository.
#[derive(Debug)]
pub struct AutoModelForCausalLM {}

impl_auto_model_from_pretrained_method!(
    AutoModelForCausalLM,
    ("llama", LlamaForCausalLM, LLAMA_DTYPE, true)
);

// Bert
impl_from_pretrained_method!(BertModel, BERT_DTYPE, false);
impl_from_pretrained_method!(BertForSequenceClassification, BERT_DTYPE, false);
impl_from_pretrained_method!(BertForTokenClassification, BERT_DTYPE, false);
impl_from_pretrained_method!(BertForMaskedLM, BERT_DTYPE, false);

// Llama
impl_from_pretrained_method!(LlamaModel, LLAMA_DTYPE, false);
impl_from_pretrained_method!(LlamaForCausalLM, LLAMA_DTYPE, true);
