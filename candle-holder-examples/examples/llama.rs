use anyhow::{Error, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_holder_examples::get_device_from_args;
use candle_holder_models::{
    model::ForwardParams, utils::cache::DynamicCache, AutoModelForCausalLM, LlamaForCausalLM,
    PreTrainedModel,
};
use candle_holder_tokenizers::{BatchEncoding, LlamaTokenizer};

fn main() -> Result<()> {
    let device = get_device_from_args()?;
    println!("Device: {:?}", device);

    let mut tokenizer =
        LlamaTokenizer::from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", None)?;
    let model =
        LlamaForCausalLM::from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", &device, None)?;

    let mut encodings = tokenizer
        .encode(vec!["Hello Llama".to_string()], true, None)
        .map_err(Error::msg)?;

    encodings.to_device(&device)?;

    println!("input ids {}", encodings.get_input_ids());

    let mut cache = DynamicCache::new();

    let start = std::time::Instant::now();

    let mut tokens = &mut encodings.get_input_ids().to_vec2::<u32>()?[0];

    for index in 0..50 {
        // Predict next token
        let output = model
            .forward(ForwardParams {
                input_ids: Some(encodings.get_input_ids()),
                attention_mask: Some(encodings.get_attention_mask()),
                cache: Some(&mut cache),
                ..Default::default()
            })?
            .to_device(&Device::Cpu)?;

        // dirty argmax
        let next_token_id: u32 = output
            .arg_sort_last_dim(false)?
            .i((0, output.dims3()?.1 - 1, 0))?
            .to_scalar()?;
        tokens.push(next_token_id);
        let token = tokenizer.decode(&[next_token_id], true)?;
        println!("token_id: {}, token: {}", next_token_id, token);

        encodings = BatchEncoding::new(
            Tensor::new(&[[next_token_id]], &device)?,
            Tensor::new(&[[0u32]], &device)?,
            Tensor::new(&[[0u32]], &device)?,
            vec![],
        );
        encodings.to_device(&device)?;
    }

    println!("Took: {:?}", start.elapsed());

    Ok(())
}
