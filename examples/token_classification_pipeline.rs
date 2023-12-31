use anyhow::Result;
use candle_core::Device;
use candle_holder::TokenClassificationPipeline;

fn main() -> Result<()> {
    let pipeline = TokenClassificationPipeline::new("dslim/bert-base-NER", &Device::Cpu, None)?;

    let entities = pipeline.run("My name is Gabriel and I live in Madrid", None)?;
    println!("Single: {:?}", entities);

    let batch_entities =
        pipeline.run_batch(vec!["My name is Gabriel and I live in Madrid"], None)?;
    println!("Batch: {:?}", batch_entities);

    Ok(())
}
