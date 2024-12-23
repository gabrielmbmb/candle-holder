use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};
use candle_holder::{utils::from_pretrained::FromPretrainedParameters, Error, Result};
use candle_holder_tokenizers::Tokenizer;
use candle_nn::VarBuilder;

use crate::{
    config::PretrainedConfig,
    from_pretrained::from_pretrained,
    generation::{
        config::GenerationConfig,
        generate::{generate, GenerateOutput},
        StoppingCriteria, TokenStreamer,
    },
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
pub struct GenerationParams {
    /// The generation configuration to use. If not provided, then the model's default generation
    /// configuration in `generation_config.json` will be used. If that is not available, then
    /// default generation configuration will be used. The default value is `None`.
    pub generation_config: Option<GenerationConfig>,
    /// The tokenizer to use for decoding the generated tokens. It's not extrictly required, but
    /// some stopping criteria may depend on it. The default value is `None`.
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    /// The list of stopping criteria to use that will determine when to stop generating tokens.
    /// The default value is `None`.
    pub stopping_criteria: Option<Vec<Box<dyn StoppingCriteria>>>,
    /// The token streamer which will receive the next tokens as they are being generated. The
    /// default value is `None`.
    pub token_streamer: Option<Arc<Mutex<dyn TokenStreamer>>>,
    /// A seed that will be used in the sampling of the next token. The default value is `None`.
    pub seed: Option<u64>,
}

impl GenerationParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_generation_config(mut self, generation_config: GenerationConfig) -> Self {
        self.generation_config = Some(generation_config);
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    pub fn with_stopping_criteria(
        mut self,
        stopping_criteria: Vec<Box<dyn StoppingCriteria>>,
    ) -> Self {
        self.stopping_criteria = Some(stopping_criteria);
        self
    }

    pub fn with_token_streamer(mut self, token_streamer: Arc<Mutex<dyn TokenStreamer>>) -> Self {
        self.token_streamer = Some(token_streamer);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Default for GenerationParams {
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

/// The output of a model. It can contain the logits, the last hidden states, and the pooled output
/// depending on the kind of model.
#[non_exhaustive]
#[derive(Debug)]
pub struct ModelOutput {
    logits: Option<Tensor>,
    last_hidden_state: Option<Tensor>,
    pooled_output: Option<Tensor>,
}

impl ModelOutput {
    pub fn new(
        logits: Option<Tensor>,
        last_hidden_state: Option<Tensor>,
        pooled_output: Option<Tensor>,
    ) -> Self {
        Self {
            logits,
            last_hidden_state,
            pooled_output,
        }
    }

    pub fn get_logits(&self) -> Option<&Tensor> {
        self.logits.as_ref()
    }

    pub fn get_last_hidden_state(&self) -> Option<&Tensor> {
        self.last_hidden_state.as_ref()
    }

    pub fn get_pooled_output(&self) -> Option<&Tensor> {
        self.pooled_output.as_ref()
    }
}

/// Trait for a pre-trained model.
pub trait PreTrainedModel: std::fmt::Debug + Send + Sync {
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
    fn forward(&self, params: ForwardParams) -> Result<ModelOutput>;

    /// Generates a completion of the input sequences.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - The input sequences.
    /// * `params` - The generation parameters.
    ///
    /// # Returns
    ///
    /// A vector containing vectors of token ids for each input sequence.
    fn generate(
        &self,
        input_ids: &Tensor,
        params: GenerationParams,
    ) -> Result<Vec<GenerateOutput>> {
        let (mut generation_config, used_model_generation_config) = match params.generation_config {
            Some(config) => (config, false),
            None => (self.get_generation_config().clone(), true),
        };

        if !used_model_generation_config {
            if generation_config.get_bos_token_id().is_none() {
                generation_config.bos_token_id = self.get_generation_config().get_bos_token_id();
            }

            if generation_config.get_eos_token_id().is_none() {
                generation_config.eos_token_id =
                    self.get_generation_config().get_eos_token_id().cloned();
            }

            if generation_config.get_pad_token_id().is_none() {
                generation_config.pad_token_id = self.get_generation_config().get_pad_token_id();
            }
        }

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
                let mut dtype = dtype.unwrap_or(
                    serde_json::from_value::<PretrainedConfig>(config.clone())
                        .expect("Could not parse model config.")
                        .get_dtype()
                        .unwrap_or($default_dtype),
                );
                if dtype == DType::BF16 && (device.is_metal() || device.is_cpu()) {
                    dtype = DType::F16;
                }
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
                            let mut dtype = dtype.unwrap_or(
                                serde_json::from_value::<PretrainedConfig>(config.clone())
                                    .expect("Could not parse model config.")
                                    .get_dtype()
                                    .unwrap_or($default_dtype),
                            );
                            if dtype == DType::BF16 && (device.is_metal() || device.is_cpu()) {
                                dtype = DType::F16;
                            }
                            let vb = model_info.get_var_builder(dtype, device)?;
                            if $load_generation_config {
                                Ok(Box::new($model_struct::load_with_generation_config(vb, config, model_info.get_generation_config().cloned())?))
                            } else {
                                Ok(Box::new($model_struct::load(vb, config)?))
                            }
                        },
                    )*
                    _ => Err(Error::ModelNotImplemented(model_type.to_string()))
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
