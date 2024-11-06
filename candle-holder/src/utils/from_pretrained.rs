use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};

use crate::{Error, Result};
use std::{collections::HashMap, fs};

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

/// Gets a [`ApiRepo`] instance from the provided repository ID using the provided parameters. It
/// will check if the repository exists.
///
/// # Arguments
///
/// * `repo_id` - The repository ID.
/// * `params` - The parameters to use when creating the API instance.
///
/// # Returns
///
/// The API instance.
pub fn get_repo_api(repo_id: &str, params: Option<FromPretrainedParameters>) -> Result<ApiRepo> {
    let params = params.unwrap_or_default();
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, params.revision);
    let api = Api::new()?.repo(repo);

    if api.info().is_err() && api.get(MODEL_CONFIG_FILE).is_err() {
        return Err(Error::RepositoryNotFound(repo_id.to_string()));
    }

    Ok(api)
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
