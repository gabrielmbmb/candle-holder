use crate::Result;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::PathBuf};

pub const MODEL_CONFIG_FILE: &str = "config.json";

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
