use anyhow::{Error, Result};
use candle_core::{Device, IndexOp, Tensor, D};
use candle_nn::ops::softmax;
use dyn_fmt::AsStrFormatExt;
use tokenizers::{EncodeInput, Tokenizer};

use crate::{
    model::PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer,
    FromPretrainedParameters,
};

use super::utils::get_encodings;

const DEFAULT_HYPOTHESIS_TEMPLATE: &str = "This example is {}.";

pub struct ZeroShotClassificationOptions {
    pub hypothesis_template: String,
    pub multi_label: bool,
}

impl Default for ZeroShotClassificationOptions {
    fn default() -> Self {
        Self {
            hypothesis_template: DEFAULT_HYPOTHESIS_TEMPLATE.to_string(),
            multi_label: false,
        }
    }
}

pub struct ZeroShotClassificationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Tokenizer,
    device: Device,
    num_labels: usize,
    entailment_id: i8,
    contradiction_id: i8,
}

impl ZeroShotClassificationPipeline {
    pub fn new<S: AsRef<str> + Copy>(
        identifier: S,
        device: &Device,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let model = AutoModelForSequenceClassification::from_pretrained(
            identifier,
            device,
            params.clone(),
        )?;
        let tokenizer = AutoTokenizer::from_pretrained(identifier, params)?;
        let id2label = model.config().id2label.unwrap();
        let num_labels = id2label.len();
        let entailment_id = id2label
            .iter()
            .find(|(_, label)| label.starts_with("entail"))
            .map(|(id, _)| *id)
            .ok_or(-1)
            .unwrap() as i8;
        let contradiction_id = if entailment_id == 0 { -1 } else { 0 };
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            num_labels,
            entailment_id,
            contradiction_id,
        })
    }

    fn preprocess(
        &self,
        sentences: Vec<String>,
        candidate_labels: &[String],
        options: &ZeroShotClassificationOptions,
    ) -> Result<(Tensor, Tensor)> {
        let sequence_pairs = self.generate_sequence_pairs(
            sentences,
            candidate_labels,
            &options.hypothesis_template,
        )?;
        let (input_ids, token_type_ids, _) =
            get_encodings(sequence_pairs.clone(), &self.tokenizer, &self.device)?;
        Ok((input_ids, token_type_ids))
    }

    fn postprocess(
        &self,
        output: Tensor,
        num_sequences: usize,
        candidate_labels: &[String],
        multi_label: bool,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let output = output.reshape((num_sequences, candidate_labels.len(), self.num_labels))?;

        let scores = if candidate_labels.len() == 1 || multi_label {
            let entail_contr_logits = output.index_select(
                &Tensor::new(
                    &[self.contradiction_id as i64, self.entailment_id as i64],
                    &self.device,
                )?,
                D::Minus1,
            )?;
            softmax(&entail_contr_logits, D::Minus1)?.i((.., .., 1))?
        } else {
            let entail_logits = output.i((.., .., self.entailment_id as usize))?;
            softmax(&entail_logits, D::Minus1)?
        }
        .to_vec2::<f32>()?;

        let results: Vec<Vec<(String, f32)>> = scores
            .into_iter()
            .map(|s| {
                let mut label_and_score: Vec<(String, f32)> = s
                    .into_iter()
                    .enumerate()
                    .map(|(i, score)| (candidate_labels[i].clone(), score))
                    .collect();
                label_and_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                label_and_score
            })
            .collect();
        Ok(results)
    }

    fn generate_sequence_pairs(
        &self,
        sequences: Vec<String>,
        candidate_labels: &[String],
        hypothesis_template: &String,
    ) -> Result<Vec<(String, String)>> {
        if candidate_labels.is_empty() {
            return Err(Error::msg("At least one label has to be provided"));
        }

        if hypothesis_template.format(&[candidate_labels[0].clone()]) == *hypothesis_template {
            return Err(Error::msg(
                "The hypothesis template does not contain an empty {} label placeholder. \
                    This is required to inject the candidate label into the template.",
            ));
        }

        let hypotheses: Vec<String> = candidate_labels
            .iter()
            .map(|label| hypothesis_template.format([label]))
            .collect();

        let sequence_pairs: Vec<(String, String)> = sequences
            .iter()
            .flat_map(|sequence| {
                hypotheses
                    .iter()
                    .map(move |hypothesis| (sequence.clone(), hypothesis.clone()))
            })
            .collect();

        Ok(sequence_pairs)
    }

    pub fn run<I: AsRef<str>>(
        &self,
        input: I,
        candidate_labels: Vec<I>,
        options: Option<ZeroShotClassificationOptions>,
    ) -> Result<Vec<(String, f32)>> {
        let options = options.unwrap_or_default();
        let sentences = vec![input.as_ref().to_string()];
        let candidate_labels: Vec<String> = candidate_labels
            .into_iter()
            .map(|label| label.as_ref().to_string())
            .collect();
        let (input_ids, token_type_ids) =
            self.preprocess(sentences, &candidate_labels, &options)?;
        let output = self.model.forward(&input_ids, &token_type_ids)?;
        Ok(self.postprocess(output, 1, &candidate_labels, options.multi_label)?[0].clone())
    }

    pub fn run_batch<I: AsRef<str>>(
        &self,
        inputs: Vec<I>,
        candidate_labels: Vec<I>,
        options: Option<ZeroShotClassificationOptions>,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let options = options.unwrap_or_default();
        let mut sentences: Vec<String> = Vec::new();
        for input in inputs {
            let input_str = input.as_ref().to_string();
            sentences.push(input_str.clone());
        }
        let num_sequences = sentences.len();
        let candidate_labels: Vec<String> = candidate_labels
            .into_iter()
            .map(|label| label.as_ref().to_string())
            .collect();
        let (input_ids, token_type_ids) =
            self.preprocess(sentences, &candidate_labels, &options)?;
        let output = self.model.forward(&input_ids, &token_type_ids)?;
        self.postprocess(
            output,
            num_sequences,
            &candidate_labels,
            options.multi_label,
        )
    }
}
