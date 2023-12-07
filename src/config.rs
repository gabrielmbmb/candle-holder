use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PretrainedConfig {
    id2label: Option<HashMap<String, String>>,
}

impl PretrainedConfig {
    pub fn num_labels(&self) -> usize {
        if let Some(id2label) = &self.id2label {
            return id2label.len();
        }
        0
    }
}
