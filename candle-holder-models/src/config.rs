use candle_core::DType;
use candle_holder::utils::serde::deserialize_single_or_vec;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

/// The type of problem the model was trained on.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum ProblemType {
    #[serde(rename = "regression")]
    Regression,
    #[serde(rename = "single_label_classification")]
    SingleLabelClassification,
    #[serde(rename = "multi_label_classification")]
    MultiLabelClassification,
    None,
}

impl Default for ProblemType {
    fn default() -> Self {
        Self::None
    }
}

/// The configuration of a pretrained model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PretrainedConfig {
    /// The type of problem the model was trained on.
    #[serde(default)]
    problem_type: ProblemType,
    /// A map of the label ids to their corresponding labels.
    #[serde(default, deserialize_with = "deserialize_id2label")]
    id2label: Option<HashMap<usize, String>>,
    /// The ID of the PAD token.
    #[serde(default)]
    pad_token_id: Option<u32>,
    /// The ID of the BOS token.
    #[serde(default)]
    bos_token_id: Option<u32>,
    /// The IDs of the EOS tokens.
    #[serde(default, deserialize_with = "deserialize_single_or_vec")]
    eos_token_id: Option<Vec<u32>>,
    #[serde(default)]
    torch_dtype: Option<String>,
}

fn deserialize_id2label<'de, D>(deserializer: D) -> Result<Option<HashMap<usize, String>>, D::Error>
where
    D: Deserializer<'de>,
{
    let map: Option<HashMap<String, String>> = Deserialize::deserialize(deserializer)?;
    Ok(map.map(|m| {
        m.into_iter()
            .map(|(k, v)| {
                let shot = k.parse::<usize>().unwrap();
                (shot, v)
            })
            .collect::<HashMap<usize, String>>()
    }))
}

impl PretrainedConfig {
    pub fn get_problem_type(&self) -> &ProblemType {
        &self.problem_type
    }

    pub fn get_id2label(&self) -> Option<&HashMap<usize, String>> {
        self.id2label.as_ref()
    }

    pub fn get_pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    pub fn get_bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn get_eos_token_id(&self) -> Option<&Vec<u32>> {
        self.eos_token_id.as_ref()
    }

    pub fn get_dtype(&self) -> Option<DType> {
        match &self.torch_dtype {
            Some(dtype) => match dtype.as_str() {
                "float16" => Some(DType::F16),
                "float32" => Some(DType::F32),
                "bfloat16" => Some(DType::BF16),
                _ => None,
            },
            None => None,
        }
    }
}

impl PretrainedConfig {
    /// Gets the number of labels the model was trained on.
    pub fn num_labels(&self) -> usize {
        if let Some(id2label) = &self.id2label {
            id2label.len()
        } else {
            0
        }
    }
}

impl Default for PretrainedConfig {
    fn default() -> Self {
        Self {
            problem_type: ProblemType::None,
            id2label: Some(HashMap::new()),
            pad_token_id: None,
            bos_token_id: None,
            eos_token_id: None,
            torch_dtype: None,
        }
    }
}
