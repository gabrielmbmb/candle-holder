use anyhow::Result;
use candle_core::Device;
use candle_holder::{AggregationStrategy, TokenClassificationOptions, TokenClassificationPipeline};

fn main() -> Result<()> {
    let pipeline = TokenClassificationPipeline::new("dslim/bert-base-NER", &Device::Cpu, None)?;

    let entities = pipeline.run(
        "My name is Gabriel and I live in Madrid",
        Some(TokenClassificationOptions {
            aggregation_strategy: AggregationStrategy::Average,
            ..Default::default()
        }),
    )?;
    println!("Single: {:?}", entities);

    let batch_entities = pipeline.run_batch(
        vec![
            "My name is Gabriel and I live in Madrid",
            "and I'll be traveling to Japan soon",
            "I would like to see the Tokyo Tower",
        ],
        Some(TokenClassificationOptions {
            aggregation_strategy: AggregationStrategy::Average,
            ..Default::default()
        }),
    )?;
    println!("Batch: {:?}", batch_entities);

    Ok(())
}
