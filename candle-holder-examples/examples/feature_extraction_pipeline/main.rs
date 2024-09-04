#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_holder_examples::get_device_from_args;
use candle_holder_pipelines::FeatureExtractionPipeline;

fn main() -> Result<()> {
    let device = get_device_from_args()?;
    println!("Device: {:?}", device);

    let pipeline = FeatureExtractionPipeline::new(
        "sentence-transformers/all-MiniLM-L6-v2",
        &device,
        None,
        None,
    )?;

    let results = pipeline.run("This is an example sentence", None)?;
    println!("`pipeline.run` results: {}", results);

    let results = pipeline.run_batch(
        vec!["This is an example sentence", "Each sentence is converted"],
        None,
    )?;
    println!("`pipeline.run_batch` results: {}", results);

    Ok(())
}
