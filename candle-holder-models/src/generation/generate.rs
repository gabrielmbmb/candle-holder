use candle_core::{IndexOp, Tensor};
use candle_holder::{Error, Result};
use candle_holder_tokenizers::Tokenizer;

use super::{
    penalties::apply_repetition_penalty, sampling::LogitSampler,
    stopping_criteria::StoppingCriteriaApplier, GenerationConfig, StoppingCriteria, TokenStreamer,
};
use crate::{utils::cache::DynamicCache, ForwardParams, PreTrainedModel};

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
pub fn generate<'a, M: PreTrainedModel + ?Sized>(
    model: &M,
    input_ids: &Tensor,
    generation_config: &GenerationConfig,
    tokenizer: Option<&Box<dyn Tokenizer>>,
    stopping_criteria: Option<Vec<Box<dyn StoppingCriteria>>>,
    mut token_streamer: Option<Box<dyn TokenStreamer<'a> + 'a>>,
    seed: Option<u64>,
) -> Result<Vec<Vec<u32>>> {
    let mut output = input_ids.to_vec2::<u32>()?;
    let mut input_ids = input_ids.clone();
    let input_ids_dims = input_ids.dims2()?;

    // Calculate the number of max new tokens to be generated if `max_new_tokens` not provided
    let input_seq_len = input_ids.dims2()?.1;
    let max_new_tokens = generation_config
        .get_max_new_tokens()
        .unwrap_or_else(|| generation_config.get_max_length() - input_seq_len);

    let mut cache = if generation_config.get_use_cache() {
        Some(DynamicCache::new())
    } else {
        None
    };

    let stopping_criteria_applier = StoppingCriteriaApplier::from_configuration(
        generation_config,
        stopping_criteria,
        tokenizer,
    )?;

    let mut sampling_config = LogitSampler::from_generation_config(generation_config, seed);

    // TODO: update to try to get from generation config first before failing
    let eos_token_id = model
        .get_config()
        .get_eos_token_id()
        .ok_or_else(|| Error::MissingSpecialTokenId("eos_token_id".to_string()))?;

    // Initialize a vector to store the next token ids for each sequence
    let num_sequences = input_ids_dims.0;
    let mut active_sequences = num_sequences;

    // TODO: if `generation_config.num_return_sequences>1` then we need to expand the
    // `input_ids` tensor to have `num_return_sequences` times the number of sequences
    stream_tokens(&mut token_streamer, &input_ids.to_vec2::<u32>()?)?;

    // Generation loop
    for _ in 0..max_new_tokens {
        if active_sequences == 0 {
            break;
        }

        let logits = model.forward(ForwardParams {
            input_ids: Some(&input_ids),
            cache: cache.as_mut(),
            ..Default::default()
        })?;
        let dims = logits.dims3()?;

        let last_token_logits = logits.i((.., dims.1 - 1, ..))?;

        // Sample the next token for each sequence
        for i in 0..dims.0 {
            let mut seq_logits = last_token_logits.i((i, ..))?;

            // Apply penalties
            if let Some(repetion_penalty) = generation_config.get_repetition_penalty() {
                seq_logits = apply_repetition_penalty(
                    &seq_logits,
                    &input_ids.i((i, ..))?,
                    repetion_penalty,
                )?;
            }

            // Sample next token
            let next_token_id = sampling_config.sample(&seq_logits)?;

            // Update the sequences with the next token
            output[i].push(next_token_id);

            // TODO: check for other stop conditions
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
        input_ids = Tensor::new(&sequences_last_tokens[..], input_ids.device())?.unsqueeze(0)?;
    }

    stream_end(&mut token_streamer)?;

    Ok(output)
}

fn stream_tokens(
    token_streamer: &mut Option<Box<dyn TokenStreamer + '_>>,
    tokens: &[Vec<u32>],
) -> Result<()> {
    if let Some(streamer) = token_streamer.as_mut() {
        streamer.put(tokens)?;
    }
    Ok(())
}

fn stream_end(token_streamer: &mut Option<Box<dyn TokenStreamer + '_>>) -> Result<()> {
    if let Some(streamer) = token_streamer.as_mut() {
        streamer.end()?;
    }
    Ok(())
}
