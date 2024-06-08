use candle_core::Device;
use candle_holder::{FromPretrainedParameters, Result};
use candle_holder_models::{AutoModelForCausalLM, PreTrainedModel};
use candle_holder_tokenizers::{AutoTokenizer, BatchEncoding, Padding, Tokenizer};

pub struct TextGenerationPipeline {
    model: Box<dyn PreTrainedModel>,
    tokenizer: Box<dyn Tokenizer>,
    device: Device,
}

impl TextGenerationPipeline {
    pub fn new<S: AsRef<str> + Copy>(
        identifier: S,
        device: &Device,
        params: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let model = AutoModelForCausalLM::from_pretrained(identifier, device, params.clone())?;
        let tokenizer = AutoTokenizer::from_pretrained(identifier, params)?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    fn preprocess(&mut self, inputs: Vec<String>) -> Result<BatchEncoding> {
        let mut encodings = self
            .tokenizer
            .encode(inputs, true, Some(Padding::Longest))?;
        encodings.to_device(&self.device)?;
        Ok(encodings)
    }

    pub fn run<I: Into<String>>(&mut self, input: I) -> Result<()> {
        let encodings = self.preprocess(vec![input.into()])?;
        Ok(())
    }

    pub fn run_batch<I: Into<String>>(&mut self, inputs: Vec<I>) {}
}
