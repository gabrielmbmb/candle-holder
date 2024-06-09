use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Repo, RepoType,
};
use serde::{Deserialize, Serialize};

use candle_holder::{
    utils::{load_model_config, FromPretrainedParameters, MODEL_CONFIG_FILE},
    Error, Result,
};

use crate::config::GenerationConfig;

const MODEL_GENERATION_CONFIG_FILE: &str = "generation_config.json";
const MODEL_SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";
const MODEL_SAFETENSORS_FILE: &str = "model.safetensors";
const MODEL_PYTORCH_FILE: &str = "pytorch_model.bin";

/// An struct holding all the information required to load a model from the Hugging Face Hub.
pub struct ModelInfo {
    /// The model configuration loaded from the `config.json` file.
    config: Option<serde_json::Value>,
    /// The generation configuration of the model loaded from the `generation_config.json` file.
    generation_config: Option<GenerationConfig>,
    /// The paths to the model weights files.
    weights_file_paths: Vec<PathBuf>,
    /// A flag indicating whether the model weights are stored in PyTorch format.
    from_pth: bool,
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
            true => VarBuilder::from_pth(&self.weights_file_paths[0], dtype, device)?,
            false => unsafe {
                VarBuilder::from_mmaped_safetensors(&self.weights_file_paths, dtype, device)?
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SafetensorsMetadata {
    total_size: usize,
}

/// Representation of the `model.safetensors.index.json` file which contains the metadata and
/// weight map of the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SafetensorsIndex {
    /// Metadata of the model.
    metadata: SafetensorsMetadata,
    /// A map of the model weights, where the key is the layer name and the value is the file path
    /// to the `safetensors` file containing the weights of that layer.
    weight_map: HashMap<String, String>,
}

impl SafetensorsIndex {
    /// Gets the list of `safetensors` files required to load the model.
    ///
    /// # Returns
    ///
    /// A list of `safetensors` files.
    fn get_safetensors_files(&self) -> Vec<String> {
        let mut files: Vec<String> = self
            .weight_map
            .values()
            .cloned()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        files.sort();
        files
    }
}

/// Loads the model weights from the Hugging Face Repository. It tries to first load the weights
/// from the `model.safetensors.index.json` file (and the corresponding `safetensors` files), then
/// from the `model.safetensors` file, and finally from the `pytorch_model.bin` file.
///
/// # Arguments
///
/// * `api` - The Hugging Face API repository.
///
/// # Returns
///
/// A tuple containing the paths to the model weights files and a flag indicating whether the
/// weights are stored in PyTorch format.
fn load_model_weights(api: &ApiRepo) -> Result<(Vec<PathBuf>, bool)> {
    if let Ok(model_safetensors_index_file_path) = api.get(MODEL_SAFETENSORS_INDEX_FILE) {
        let safetensors_index = fs::read_to_string(model_safetensors_index_file_path)?;
        let safetensors_index: SafetensorsIndex = serde_json::from_str(&safetensors_index)?;
        // Call `api.get to download the files
        let safetensors_files = safetensors_index
            .get_safetensors_files()
            .iter()
            .map(|file_name| api.get(&file_name).map_err(Error::wrap))
            .collect::<Result<Vec<_>>>()?;
        return Ok((safetensors_files, false));
    }

    if let Ok(model_safetensor_file_path) = api.get(MODEL_SAFETENSORS_FILE) {
        return Ok((vec![model_safetensor_file_path], false));
    }

    if let Ok(model_pytorch_file_path) = api.get(MODEL_PYTORCH_FILE) {
        return Ok((vec![model_pytorch_file_path], true));
    }

    Err(Error::ModelWeightsNotFound)
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

    // Load the model weights
    let (weights_file_paths, from_pth) = load_model_weights(&api)?;

    Ok(ModelInfo {
        config,
        generation_config,
        weights_file_paths,
        from_pth,
    })
}
