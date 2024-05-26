use std::{collections::HashMap, fs, path::PathBuf};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

use crate::{config::GenerationConfig, error::Result};

pub const MODEL_CONFIG_FILE: &str = "config.json";
const MODEL_GENERATION_CONFIG_FILE: &str = "generation_config.json";

#[derive(Debug, Clone)]
pub struct FromPretrainedParameters {
    pub revision: String,
    pub user_agent: HashMap<String, String>,
    pub auth_token: Option<String>,
}

impl Default for FromPretrainedParameters {
    fn default() -> Self {
        Self {
            revision: "main".into(),
            user_agent: HashMap::new(),
            auth_token: None,
        }
    }
}

/// An struct holding all the information required to load a model from the Hugging Face Hub.
pub struct ModelInfo {
    /// The model configuration loaded from the `config.json` file.
    config: Option<serde_json::Value>,
    /// The generation configuration of the model loaded from the `generation_config.json` file.
    generation_config: Option<GenerationConfig>,
    weights_file_path: PathBuf,
    pub from_pth: bool,
}

impl ModelInfo {
    /// Loads the model weights from the provided paths into a `VarBuilder`.
    ///
    /// # Arguments
    ///
    /// - `dtype` - The data type of the model weights.
    /// - `device` - The device on which the model weights should be loaded.
    ///
    /// # Returns
    ///
    /// A `VarBuilder` containing the model weights.
    pub fn get_var_builder(&self, dtype: DType, device: &Device) -> Result<VarBuilder> {
        let vb = match self.from_pth {
            true => VarBuilder::from_pth(&self.weights_file_path, dtype, device)?,
            false => unsafe {
                VarBuilder::from_mmaped_safetensors(&[&self.weights_file_path], dtype, device)?
            },
        };
        Ok(vb)
    }

    /// Gets a reference to the model configuration.
    pub fn get_config(&self) -> Option<&serde_json::Value> {
        self.config.as_ref()
    }

    /// Gets a reference to the generation configuration of the model.
    pub fn get_generation_config(&self) -> Option<&GenerationConfig> {
        self.generation_config.as_ref()
    }
}

// TODO: function to convert old format to new format (gamma, beta) -> (weight, bias)
// TODO: detect prefix

/// Loads the model configuration from the provided file path.
///
/// # Arguments
///
/// * `file_path` - The path to the `config.json` file containing the model configuration.
///
/// # Returns
///
/// The loaded model configuration.
pub fn load_model_config(file_path: std::path::PathBuf) -> Result<serde_json::Value> {
    let model_config = fs::read_to_string(file_path)?;
    let model_config = serde_json::from_str(&model_config)?;
    Ok(model_config)
}

/// Loads the generation configuration of the model from the provided file path.
///
/// # Arguments
///
/// * `file_path` - The path of the `generation_config.json` file containing the generation
/// configuration of the model.
///
/// # Returns
///
/// The loaded generation configuration.
fn load_generation_config(file_path: std::path::PathBuf) -> Result<GenerationConfig> {
    let generation_config = fs::read_to_string(file_path)?;
    let generation_config = serde_json::from_str(&generation_config)?;
    Ok(generation_config)
}

/// Loads all the required configuration files for loading a model from the Hugging Face Hub.
///
/// # Arguments
///
/// * `repo_id`: The Hugging Face Hub model repository id.
/// * `params`:
///
/// # Returns
///
/// A `ModelInfo` struct containing all the information required to load the model.
pub fn from_pretrained<I: AsRef<str>>(
    repo_id: I,
    params: Option<FromPretrainedParameters>,
) -> Result<ModelInfo> {
    let params = params.unwrap_or_default();

    let repo = Repo::with_revision(
        repo_id.as_ref().to_string(),
        RepoType::Model,
        params.revision,
    );

    let mut builder = ApiBuilder::new();
    if let Some(token) = params.auth_token {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;
    let api = api.repo(repo);

    // Get the model configuration from `config.json`
    let config = match api.get(MODEL_CONFIG_FILE) {
        Ok(model_config_file_path) => load_model_config(model_config_file_path).ok(),
        Err(_) => None,
    };

    // Try to get the model generation config from `generation_config.json` file
    let generation_config = match api.get(MODEL_GENERATION_CONFIG_FILE) {
        Ok(generation_config_file_path) => load_generation_config(generation_config_file_path).ok(),
        Err(_) => None,
    };

    // Get the model weights
    let (weights_file_path, from_pth) = {
        if let Ok(weights_file_path) = api.get("model.safetensors") {
            (weights_file_path, false)
        } else {
            let weights_file_path = api.get("pytorch_model.bin")?;
            (weights_file_path, true)
        }
    };

    Ok(ModelInfo {
        config,
        generation_config: None,
        weights_file_path,
        from_pth,
    })
}
