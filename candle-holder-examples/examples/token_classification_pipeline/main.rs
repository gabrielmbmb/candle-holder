use anyhow::Result;
use candle_core::Device;
use candle_holder_examples::get_device_from_args;
use candle_holder_pipelines::TokenClassificationPipeline;

fn main() -> Result<()> {
    let device = get_device_from_args()?;
    println!("Device: {:?}", device);

    let mut pipeline = TokenClassificationPipeline::new("dslim/bert-base-NER", &device, None)?;

    let entities = pipeline.run("My name is Gabriel and I live in Madrid", None)?;
    println!("`pipeline.run` results: {:?}", entities);

    let batch_entities = pipeline.run_batch(
        vec![
            "My name is Gabriel and I live in Madrid",
            "and I'll be traveling to Japan soon",
            "I would like to see the Tokyo Tower",
        ],
        None,
    )?;
    println!("`pipeline.run_batch` results: {:?}", batch_entities);

    Ok(())
}
