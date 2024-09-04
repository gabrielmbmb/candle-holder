#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_holder_examples::get_device_from_args;
use candle_holder_pipelines::TextClassificationPipeline;

fn main() -> Result<()> {
    let device = get_device_from_args()?;
    println!("Device: {:?}", device);

    let pipeline = TextClassificationPipeline::new("gabrielmbmb/finbert", &device, None, None)?;

    let scores = pipeline.run("Stocks rallied and the British pound gained.", Some(1))?;

    println!("`pipeline.run` results: {:?}", scores);

    let results = pipeline.run_batch(
        vec![
            "Stocks rallied and the British pound gained.",
            "Google stock closes the market with 5 points down on the day",
        ],
        Some(1),
    )?;

    println!("`pipeline.run_batch` results: {:?}", results);

    Ok(())
}
