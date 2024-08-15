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
    pub problem_type: ProblemType,
    /// A map of the label ids to their corresponding labels.
    #[serde(default, deserialize_with = "deserialize_id2label")]
    pub id2label: Option<HashMap<usize, String>>,
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
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum EarlyStopping {
    Bool(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum StopStrings {
    Single(String),
    Multiple(Vec<String>),
}

/// The generation configuration of a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// The maximum number of tokens the sequence can have. Corresponds to the number of input
    /// tokens + `max_new_tokens` that will be generated. Its effect is overriden if
    /// `max_new_tokens` is set. The default is `20`.
    #[serde(default)]
    max_length: usize,
    /// The maximun number of tokens to generate.
    max_new_tokens: Option<usize>,
    /// The minimun number of tokens the sequence to be generated should have. Corresponds to the
    /// number of input tokens + `min_new_tokens` that will be generated.
    min_length: Option<usize>,
    /// Controls the stopping condition for beam-based methods. It accepts the following values:
    ///
    /// - `True`: generation stops as soon as there are `num_beams` complete candidates.
    /// - `False`: an heuristic is applied to determine if it's very unlikely to find better
    /// candidates and the generation should stop.
    /// - `"never"`: beam search only stops when there cannot be better candidates.
    early_stopping: Option<EarlyStopping>,
    /// The maximum amount of time that computation should be allowed to run in seconds.
    max_time: Option<f64>,
    /// A string or a list of strings that should terminate the generation if the model outputs
    /// them.
    stop_strings: Option<StopStrings>,
    /// Whether or not to use sampling. If `false`, then greedy decoding will be used. The default
    /// is `false`.
    #[serde(default)]
    do_sample: bool,
    /// Number of beams for beam search. `1` means no beam search. The default is `1`.
    #[serde(default)]
    num_beams: usize,
    /// Number of groups to divide `num_beams`
    #[serde(default)]
    num_beam_groups: usize,
    penalty_alpha: Option<f64>,
    /// Whether the model should use the past key/values attentions to speed up decoding.
    #[serde(default)]
    use_cache: bool,
    #[serde(default)]
    temperature: f64,
}

impl GenerationConfig {
    pub fn get_max_length(&self) -> usize {
        self.max_length
    }

    pub fn get_max_new_tokens(&self) -> Option<usize> {
        self.max_new_tokens
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 20,
            max_new_tokens: None,
            min_length: None,
            early_stopping: None,
            max_time: None,
            stop_strings: None,
            do_sample: false,
            num_beams: 1,
            num_beam_groups: 1,
            penalty_alpha: None,
            use_cache: true,
            temperature: 1.0,
        }
    }
}
