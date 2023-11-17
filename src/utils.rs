use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

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

pub struct ModelInfo {
    pub config_file_path: PathBuf,
    pub weights_file_path: PathBuf,
    pub safetensors: bool,
}

impl ModelInfo {
    pub fn vb(&self, dtype: DType, device: &Device) -> Result<VarBuilder> {
        let vb = VarBuilder::from_pth(&self.weights_file_path, dtype, device)?;
        Ok(vb)
    }

    pub fn config(&self) -> Result<serde_json::Value> {
        let config = std::fs::read_to_string(&self.config_file_path)?;
        let json_config: serde_json::Value = serde_json::from_str(&config)?;
        Ok(json_config)
    }
}

pub fn from_pretrained<S: AsRef<str>>(
    repo_id: S,
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

    let config_file_path = api.get("config.json")?;
    let (weights_file_path, safetensors) = { (api.get("pytorch_model.bin")?, false) };

    Ok(ModelInfo {
        config_file_path,
        weights_file_path,
        safetensors,
    })
}
