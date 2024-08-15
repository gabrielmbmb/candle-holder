use anyhow::{Error, Result};
use candle_core::{IndexOp, Tensor};
use candle_holder_examples::get_device_from_args;
use candle_holder_models::{
    model::ForwardParams, utils::cache::DynamicCache, LlamaForCausalLM, PreTrainedModel,
};
use candle_holder_tokenizers::LlamaTokenizer;

fn main() -> Result<()> {
    let device = get_device_from_args()?;
    println!("Device: {:?}", device);

    let tokenizer = LlamaTokenizer::from_pretrained("meta-llama/Meta-Llama-3-8B", None, None)?;
    let model =
        LlamaForCausalLM::from_pretrained("meta-llama/Meta-Llama-3-8B", &device, None, None)?;

    let mut encodings = tokenizer
        .encode(vec!["Once upon a time".to_string()], true, None)
        .map_err(Error::msg)?;

    encodings.to_device(&device)?;

    let mut cache = DynamicCache::new();

    let start = std::time::Instant::now();

    let tokens = &mut encodings.get_input_ids().to_vec2::<u32>()?[0];

    let mut input_ids = encodings.get_input_ids().clone();

    println!("input_ids {}", input_ids);

    for _index in 0..50 {
        // Predict next token
        let output = model.forward(ForwardParams {
            input_ids: Some(&input_ids),
            cache: Some(&mut cache),
            ..Default::default()
        })?;

        // Take logits of the last token
        let output = output.i((0, output.dims3()?.1 - 1, ..))?;

        // Temperature sampling
        //let output =
        //    output.broadcast_div(&Tensor::new(&[0.7], &Device::Cpu)?.to_dtype(DType::F16)?)?;
        // let output = softmax_last_dim(&output)?;
        let next_token_id = output.argmax(0)?.to_scalar()?;
        // println!("next token id {}", next_token_id);

        tokens.push(next_token_id);

        input_ids = Tensor::new(&[[next_token_id]], &device)?;
    }

    let text = tokenizer.decode(&tokens[..], true)?;
    println!("generated text: {}", text);

    println!("Took: {:?}", start.elapsed());

    Ok(())
}
