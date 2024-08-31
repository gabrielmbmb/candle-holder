use candle_core::{DType, Device};
use candle_holder::{FromPretrainedParameters, Result};
use candle_holder_models::{
    generation::generate::GenerateOutput, AutoModelForCausalLM, GenerationParams, PreTrainedModel,
};
use candle_holder_tokenizers::{AutoTokenizer, BatchEncoding, Message, PaddingSide, Tokenizer};

enum RunTextGenerationInput {
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

enum RunBatchTextGenerationInput {
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

/// A pipeline for generating text with a causal language model.
pub struct TextGenerationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Box<dyn Tokenizer>,
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

    fn preprocess<I: Into<RunBatchTextGenerationInput>>(
        &self,
        inputs: I,
    ) -> Result<(BatchEncoding, Vec<String>, bool)> {
        let (mut encodings, text_inputs, are_messages) = match inputs.into() {
            RunBatchTextGenerationInput::Inputs(inputs) => (
                self.tokenizer.encode(inputs.clone(), false, None)?,
                inputs,
                false,
            ),
            RunBatchTextGenerationInput::Messages(messages) => {
                let messages: Result<Vec<String>> = messages
                    .into_iter()
                    .map(|messages| self.tokenizer.apply_chat_template(messages))
                    .collect();
                let inputs = messages?;
                (
                    self.tokenizer.encode(inputs.clone(), false, None)?,
                    inputs,
                    true,
                )
            }
        };
        encodings.to_device(&self.device)?;
        Ok((encodings, text_inputs, are_messages))
    }

    fn postprocess(
        &self,
        outputs: Vec<GenerateOutput>,
        text_inputs: Vec<String>,
        are_messages: bool,
    ) -> Result<Vec<Vec<TextGenerationPipelineOutput>>> {
        let mut results = vec![];
        for (output, text_input) in outputs.iter().zip(text_inputs) {
            let mut outputs = vec![];
            let text_input_len = text_input.len();
            for sequence in output.get_sequences() {
                let text = self.tokenizer.decode(&sequence[..], true)?;
                println!("text input: {}", text_input);
                println!("text: {}", text);
                let generated: String = text.chars().skip(text_input_len).collect();
                println!("generated: {}", generated);
                outputs.push(TextGenerationPipelineOutput {
                    text: Some(generated),
                    messages: None,
                });
            }
            results.push(outputs);
        }
        Ok(results)
    }

    pub fn run<I: Into<RunTextGenerationInput>>(
        &self,
        input: I,
        params: Option<GenerationParams>,
    ) -> Result<Vec<TextGenerationPipelineOutput>> {
        let (encodings, text_inputs, are_messages) = self.preprocess(input.into())?;
        let outputs = self
            .model
            .generate(encodings.get_input_ids(), params.unwrap_or_default())?;
        Ok(self.postprocess(outputs, text_inputs, are_messages)?[0].clone())
    }

    // pub fn run_batch<I: Into<RunBatchTextGenerationInput>>(
    //     &mut self,
    //     inputs: I,
    //     params: Option<GenerationParams>,
    // ) -> Result<TextGenerationPipelineOutput> {
    //     let (encodings, are_messages) = self.preprocess(inputs)?;
    //     Ok(())
    // }
}
