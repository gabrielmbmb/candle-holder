use candle_core::{Device, IndexOp, Tensor, D};
use candle_holder::{Error, FromPretrainedParameters, Result};
use candle_holder_models::{AutoModelForSequenceClassification, ForwardParams, PreTrainedModel};
use candle_holder_tokenizers::{AutoTokenizer, BatchEncoding, Padding, Tokenizer};
use candle_nn::ops::softmax;
use dyn_fmt::AsStrFormatExt;

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

/// A pipeline for doing zero-shot classification.
pub struct ZeroShotClassificationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Box<dyn Tokenizer>,
    device: Device,
    num_labels: usize,
    entailment_id: i8,
    contradiction_id: i8,
}

impl ZeroShotClassificationPipeline {
    /// Creates a new `ZeroShotClassificationPipeline`.
    ///
    /// # Arguments
    ///
    /// * `identifier` - The repository id of the model to load.
    /// * `device` - The device to run the model on.
    /// * `params` - Optional parameters to specify the revision, user agent, and auth token.
    ///
    /// # Returns
    ///
    /// The `ZeroShotClassificationPipeline` instance.
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
        let id2label = model
            .config()
            .id2label
            .clone()
            .ok_or_else(|| Error::msg("id2label not found in model config"))?;
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
        &mut self,
        sentences: Vec<String>,
        candidate_labels: &[String],
        options: &ZeroShotClassificationOptions,
    ) -> Result<BatchEncoding> {
        let sequence_pairs = self.generate_sequence_pairs(
            sentences,
            candidate_labels,
            &options.hypothesis_template,
        )?;
        let mut encodings =
            self.tokenizer
                .encode_sequence_pairs(sequence_pairs, true, Some(Padding::Longest))?;
        encodings.to_device(&self.device)?;
        Ok(encodings)
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

    /// Classifies a single sequence.
    ///
    /// # Arguments
    ///
    /// * `input` - The input sequence to classify.
    /// * `candidate_labels` - The candidate labels to classify the input against.
    /// * `options` - Optional parameters for the pipeline..
    ///
    /// # Returns
    ///
    /// A list containing the predicted label and the confidence score.
    pub fn run<I: Into<String>, L: AsRef<str>>(
        &mut self,
        input: I,
        candidate_labels: Vec<L>,
        options: Option<ZeroShotClassificationOptions>,
    ) -> Result<Vec<(String, f32)>> {
        let options = options.unwrap_or_default();
        let inputs = vec![input.into()];
        let candidate_labels: Vec<String> = candidate_labels
            .into_iter()
            .map(|label| label.as_ref().to_string())
            .collect();
        let encodings = self.preprocess(inputs, &candidate_labels, &options)?;
        let output = self.model.forward(ForwardParams::from(&encodings))?;
        Ok(self.postprocess(output, 1, &candidate_labels, options.multi_label)?[0].clone())
    }

    /// Classifies a list of sequences.
    ///
    /// # Arguments
    ///
    /// * `input` - The input sequence to classify.
    /// * `candidate_labels` - The candidate labels to classify the input against.
    /// * `options` - Optional parameters for the pipeline..
    ///
    /// # Returns
    ///
    /// A list containing the predicted label and the confidence score for each sequence.
    pub fn run_batch<I: Into<String>, L: AsRef<str>>(
        &mut self,
        inputs: Vec<I>,
        candidate_labels: Vec<L>,
        options: Option<ZeroShotClassificationOptions>,
    ) -> Result<Vec<Vec<(String, f32)>>> {
        let options = options.unwrap_or_default();
        let inputs: Vec<String> = inputs.into_iter().map(|x| x.into()).collect();
        let num_sequences = inputs.len();
        let candidate_labels: Vec<String> = candidate_labels
            .into_iter()
            .map(|label| label.as_ref().to_string())
            .collect();
        let encodings = self.preprocess(inputs, &candidate_labels, &options)?;
        let output = self.model.forward(ForwardParams::from(&encodings))?;
        self.postprocess(
            output,
            num_sequences,
            &candidate_labels,
            options.multi_label,
        )
    }
}
