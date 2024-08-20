use candle_core::{IndexOp, Tensor};
use candle_holder::{Error, Result};

use super::{sampling::LogitSampler, token_streamer::TokenStreamer};
use crate::{config::GenerationConfig, utils::cache::DynamicCache, ForwardParams, PreTrainedModel};

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
    generation_config: GenerationConfig,
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
    let mut sampling_config = LogitSampler::from_generation_config(generation_config, seed);
    // TODO: update to try to get from generation config first before failing
    let eos_token_id = model
        .get_config()
        .get_eos_token_id()
        .ok_or_else(|| Error::MissingSpecialTokenId("eos_token_id".to_string()))?;

    // Initialize a vector to store the next token ids for each sequence
    let num_sequences = input_ids_dims.0;
    let mut sequences_next_tokens: Vec<Vec<u32>> = vec![Vec::new(); num_sequences];
    let mut active_sequences = num_sequences;

    // TODO: if `generation_config.num_return_sequences>1` then we need to expand the
    // `input_ids` tensor to have `num_return_sequences` times the number of sequences
    stream_tokens(&mut token_streamer, input_ids.to_vec2::<u32>()?)?;

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

        // TODO: apply repeat penalty, frequency penalty

        // Sample the next token for each sequence
        for i in 0..dims.0 {
            let seq_logits = last_token_logits.i((i, ..))?;
            let next_token_id = sampling_config.sample(&seq_logits)?;
            sequences_next_tokens[i].push(next_token_id);

            // TODO: check for other stop conditions
            if next_token_id == eos_token_id {
                active_sequences -= 1;
            }
        }

        stream_tokens(
            &mut token_streamer,
            // Gather last token generated for each sequence
            sequences_next_tokens
                .iter()
                .map(|inner_vec| inner_vec.last().map_or(Vec::new(), |&last| vec![last]))
                .collect(),
        )?;

        // Build the next `input_ids` vectors with the last token of each sequence
        let sequences_last_tokens = sequences_next_tokens
            .iter()
            .map(|seq| seq.last().unwrap().clone())
            .collect::<Vec<_>>();
        input_ids = Tensor::new(&sequences_last_tokens[..], input_ids.device())?.unsqueeze(0)?;
    }

    // Append the generated sequences to the input sequences
    for (i, seq) in sequences_next_tokens.iter().enumerate() {
        output[i].extend(seq);
    }
    Ok(output)
}

fn stream_tokens(
    token_streamer: &mut Option<Box<dyn TokenStreamer + '_>>,
    tokens: Vec<Vec<u32>>,
) -> Result<()> {
    if let Some(streamer) = token_streamer.as_mut() {
        streamer.put(tokens)?;
    }
    Ok(())
}
