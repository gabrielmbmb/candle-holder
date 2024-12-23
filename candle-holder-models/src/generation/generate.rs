use std::sync::{Arc, Mutex};

use candle_core::{IndexOp, Tensor};
use candle_holder::{Error, Result};
use candle_holder_tokenizers::Tokenizer;

use super::{
    penalties::apply_repetition_penalty, sampling::LogitSampler,
    stopping_criteria::StoppingCriteriaApplier, GenerationConfig, StoppingCriteria, TokenStreamer,
};
use crate::{utils::cache::DynamicCache, ForwardParams, PreTrainedModel};

#[derive(Debug)]
pub struct GenerateOutput {
    sequences: Vec<Vec<u32>>,
}

impl GenerateOutput {
    pub fn get_sequences(&self) -> &Vec<Vec<u32>> {
        &self.sequences
    }
}

/// Generates a completion of the input sequences using the provided `model`.
///
/// # Arguments
///
/// * `model` - The pre-trained model to use for generation.
/// * `input_ids` - The input sequences.
/// * `generation_config` - The generation configuration.
/// * `seed` - Optional seed for random number generation.
///
/// # Returns
///
/// A vector containing vectors of token ids for each input sequence.
pub fn generate<M: PreTrainedModel + ?Sized>(
    model: &M,
    input_ids: &Tensor,
    generation_config: &GenerationConfig,
    tokenizer: Option<Arc<dyn Tokenizer>>,
    stopping_criteria: Option<Vec<Box<dyn StoppingCriteria>>>,
    mut token_streamer: Option<Arc<Mutex<dyn TokenStreamer>>>,
    seed: Option<u64>,
) -> Result<Vec<GenerateOutput>> {
    let num_return_sequences = generation_config.get_num_return_sequences().max(1);
    let mut input_ids = input_ids.repeat((num_return_sequences, 1))?;
    let mut output = input_ids.to_vec2::<u32>()?;
    let input_ids_dims = input_ids.dims2()?;

    if input_ids_dims.1.ge(&generation_config.get_max_length())
        && generation_config.get_max_new_tokens().is_none()
    {
        return Err(Error::GenerateParamValueError(
            "`max_length` must be greater than input sequence length.".to_string(),
        ));
    }

    // Calculate the number of max new tokens to be generated if `max_new_tokens` not provided
    let input_seq_len = input_ids.dims2()?.1;
    let max_new_tokens = generation_config
        .get_max_new_tokens()
        .unwrap_or_else(|| generation_config.get_max_length() - input_seq_len);

    // Create a KV cache to accelerate generation of next tokens
    let mut cache = if generation_config.get_use_cache() {
        Some(DynamicCache::new())
    } else {
        None
    };

    // TODO: refactor this into `LogitProcessor` trait
    let mut logit_sampler = LogitSampler::from_generation_config(generation_config, seed);

    // Initialize the stopping criteria applier that will be used to determine when to stop
    let stopping_criteria_applier = StoppingCriteriaApplier::from_configuration(
        generation_config,
        stopping_criteria,
        tokenizer,
    )?;

    let num_sequences = input_ids_dims.0;
    let mut active_sequences = num_sequences;

    stream_tokens(&mut token_streamer, &input_ids.to_vec2::<u32>()?)?;

    // Generation loop
    for _ in 0..max_new_tokens {
        if active_sequences == 0 {
            break;
        }

        let model_output = model.forward(ForwardParams {
            input_ids: Some(&input_ids),
            cache: cache.as_mut(),
            ..Default::default()
        })?;
        let logits = model_output.get_logits().unwrap();
        let dims = logits.dims3()?;

        let last_token_logits = logits.i((.., dims.1 - 1, ..))?;

        // Sample the next token for each sequence
        for i in 0..dims.0 {
            let mut seq_logits = last_token_logits.i((i, ..))?;

            // Apply penalties
            // TODO: refactor this into `LogitProcessor` trait
            if let Some(repetion_penalty) = generation_config.get_repetition_penalty() {
                seq_logits = apply_repetition_penalty(
                    &seq_logits,
                    &input_ids.i((i, ..))?,
                    repetion_penalty,
                )?;
            }

            // Sample next token
            let next_token_id = logit_sampler.sample(&seq_logits)?;

            // Update the sequences with the next token
            output[i].push(next_token_id);

            if stopping_criteria_applier.should_stop(&output[i])? {
                active_sequences -= 1;
            }
        }

        let sequences_last_tokens: Vec<Vec<u32>> = output
            .iter()
            .map(|inner_vec| inner_vec.last().map_or(Vec::new(), |&last| vec![last]))
            .collect();

        stream_tokens(&mut token_streamer, &sequences_last_tokens)?;

        // Build the next `input_ids` vectors with the last token of each sequence
        let sequences_last_tokens: Vec<u32> = sequences_last_tokens.into_iter().flatten().collect();
        input_ids = Tensor::new(&sequences_last_tokens[..], input_ids.device())?
            .reshape((num_sequences, 1))?;
    }

    stream_end(&mut token_streamer)?;

    Ok(create_outputs(output, num_return_sequences))
}

fn stream_tokens(
    token_streamer: &mut Option<Arc<Mutex<dyn TokenStreamer>>>,
    tokens: &[Vec<u32>],
) -> Result<()> {
    if let Some(streamer) = token_streamer {
        streamer.lock().unwrap().put(tokens)?;
    }
    Ok(())
}

fn stream_end(token_streamer: &mut Option<Arc<Mutex<dyn TokenStreamer>>>) -> Result<()> {
    if let Some(streamer) = token_streamer.as_mut() {
        streamer.lock().unwrap().end()?;
    }
    Ok(())
}

fn create_outputs(outputs: Vec<Vec<u32>>, num_return_sequences: usize) -> Vec<GenerateOutput> {
    let mut generate_outputs = Vec::new();
    for generated_sequences in outputs.chunks(num_return_sequences) {
        let sequences = generated_sequences.to_vec();
        let output = GenerateOutput { sequences };
        generate_outputs.push(output);
    }
    generate_outputs
}
