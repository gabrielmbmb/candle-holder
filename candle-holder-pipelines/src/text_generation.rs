use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_holder::{FromPretrainedParameters, Result};
use candle_holder_models::{
    generation::generate::GenerateOutput, AutoModelForCausalLM, GenerationParams, PreTrainedModel,
};
use candle_holder_tokenizers::{AutoTokenizer, BatchEncoding, Message, PaddingSide, Tokenizer};

#[derive(Debug)]
pub enum RunTextGenerationInput {
    Input(String),
    Messages(Vec<Message>),
}

impl From<&str> for RunTextGenerationInput {
    fn from(input: &str) -> Self {
        RunTextGenerationInput::Input(input.to_owned())
    }
}

impl From<String> for RunTextGenerationInput {
    fn from(input: String) -> Self {
        RunTextGenerationInput::Input(input)
    }
}

impl From<Vec<Message>> for RunTextGenerationInput {
    fn from(messages: Vec<Message>) -> Self {
        RunTextGenerationInput::Messages(messages)
    }
}

#[derive(Debug)]
pub enum RunBatchTextGenerationInput {
    Inputs(Vec<String>),
    Messages(Vec<Vec<Message>>),
}

impl From<Vec<String>> for RunBatchTextGenerationInput {
    fn from(inputs: Vec<String>) -> Self {
        RunBatchTextGenerationInput::Inputs(inputs)
    }
}

impl From<Vec<Vec<Message>>> for RunBatchTextGenerationInput {
    fn from(messages: Vec<Vec<Message>>) -> Self {
        RunBatchTextGenerationInput::Messages(messages)
    }
}

impl From<RunTextGenerationInput> for RunBatchTextGenerationInput {
    fn from(input: RunTextGenerationInput) -> Self {
        match input {
            RunTextGenerationInput::Input(input) => {
                RunBatchTextGenerationInput::Inputs(vec![input])
            }
            RunTextGenerationInput::Messages(messages) => {
                RunBatchTextGenerationInput::Messages(vec![messages])
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextGenerationPipelineOutput {
    text: Option<String>,
    messages: Option<Vec<Message>>,
}

impl TextGenerationPipelineOutput {
    pub fn get_text(&self) -> Option<String> {
        self.text.clone()
    }
    pub fn get_messages(&self) -> Option<Vec<Message>> {
        self.messages.clone()
    }
}

/// A pipeline for generating text with a causal language model.
pub struct TextGenerationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Arc<dyn Tokenizer>,
    device: Device,
}

impl TextGenerationPipeline {
    /// Creates a new `TextGenerationPipeline`.
    ///
    /// # Arguments
    ///
    /// * `identifier` - The repository id of the model to load.
    /// * `device` - The device to run the model on.
    /// * `params` - Optional parameters to specify the revision, user agent, and auth token.
    ///
    /// # Returns
    ///
    /// The `TextGenerationPipeline` instance.
    pub fn new<S: AsRef<str> + Copy>(
        identifier: S,
        device: &Device,
        dtype: Option<DType>,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let model =
            AutoModelForCausalLM::from_pretrained(identifier, device, dtype, params.clone())?;
        let tokenizer =
            AutoTokenizer::from_pretrained(identifier, Some(PaddingSide::Left), params)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    fn preprocess<I: Into<RunBatchTextGenerationInput>>(&self, inputs: I) -> Result<BatchEncoding> {
        let mut encodings = match inputs.into() {
            RunBatchTextGenerationInput::Inputs(inputs) => {
                self.tokenizer.encode(inputs.clone(), true, None)?
            }
            RunBatchTextGenerationInput::Messages(messages) => {
                let messages: Result<Vec<String>> = messages
                    .into_iter()
                    .map(|messages| self.tokenizer.apply_chat_template(messages, true))
                    .collect();
                let inputs = messages?;
                self.tokenizer.encode(inputs.clone(), false, None)?
            }
        };
        encodings.to_device(&self.device)?;
        Ok(encodings)
    }

    fn postprocess<I: Into<RunBatchTextGenerationInput>>(
        &self,
        inputs: I,
        input_ids: &Tensor,
        outputs: Vec<GenerateOutput>,
    ) -> Result<Vec<Vec<TextGenerationPipelineOutput>>> {
        let inputs = inputs.into();
        let inputs_prompt_lengths: Vec<usize> = input_ids
            .to_vec2::<u32>()?
            .into_iter()
            .map(|seq_input_ids| {
                self.tokenizer
                    .decode(&seq_input_ids[..], true)
                    .unwrap()
                    .len()
            })
            .collect();

        match inputs {
            RunBatchTextGenerationInput::Inputs(inputs) => inputs
                .into_iter()
                .zip(inputs_prompt_lengths.into_iter())
                .zip(outputs.into_iter())
                .map(|((_, input_prompt_length), output)| {
                    output
                        .get_sequences()
                        .iter()
                        .map(|sequence| {
                            let text = self.tokenizer.decode(sequence, true)?;
                            let generated = text.chars().skip(input_prompt_length).collect();
                            Ok(TextGenerationPipelineOutput {
                                text: Some(generated),
                                messages: None,
                            })
                        })
                        .collect()
                })
                .collect(),
            RunBatchTextGenerationInput::Messages(messages_batch) => messages_batch
                .into_iter()
                .zip(inputs_prompt_lengths.into_iter())
                .zip(outputs.into_iter())
                .map(|((messages, input_prompt_length), output)| {
                    output
                        .get_sequences()
                        .iter()
                        .map(|sequence| {
                            let text = self.tokenizer.decode(sequence, true)?;
                            let generated: String =
                                text.chars().skip(input_prompt_length).collect();
                            let mut messages = messages.clone();
                            messages.push(Message::assistant(generated));
                            Ok(TextGenerationPipelineOutput {
                                text: None,
                                messages: Some(messages),
                            })
                        })
                        .collect()
                })
                .collect(),
        }
    }

    /// Generates text from the given input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to generate text from.
    /// * `params` - Optional parameters to specify the generation configuration.
    ///
    /// # Returns
    ///
    /// The generated text.
    pub fn run<I: Into<RunTextGenerationInput> + Clone>(
        &self,
        input: I,
        params: Option<GenerationParams>,
    ) -> Result<Vec<TextGenerationPipelineOutput>> {
        let encodings = self.preprocess(input.clone().into())?;
        let outputs = self
            .model
            .generate(encodings.get_input_ids(), params.unwrap_or_default())?;
        Ok(self.postprocess(input.into(), encodings.get_input_ids(), outputs)?[0].clone())
    }

    /// Generates text from the given inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - The inputs to generate text from.
    /// * `params` - Optional parameters to specify the generation configuration.
    ///
    /// # Returns
    ///
    /// The generated texts.
    pub fn run_batch<I: Into<RunBatchTextGenerationInput> + Clone>(
        &self,
        inputs: I,
        params: Option<GenerationParams>,
    ) -> Result<Vec<Vec<TextGenerationPipelineOutput>>> {
        let encodings = self.preprocess(inputs.clone())?;
        let outputs = self
            .model
            .generate(encodings.get_input_ids(), params.unwrap_or_default())?;
        self.postprocess(inputs, encodings.get_input_ids(), outputs)
    }
}
