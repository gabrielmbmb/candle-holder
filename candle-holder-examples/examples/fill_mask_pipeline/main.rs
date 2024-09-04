#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_holder_examples::get_device_from_args;
use candle_holder_pipelines::FillMaskPipeline;

fn main() -> Result<()> {
    let device = get_device_from_args()?;
    println!("Device: {:?}", device);

    let pipeline = FillMaskPipeline::new("gabrielmbmb/bert-test", &device, None, None)?;

    let results = pipeline.run("Paris is the [MASK] of France", None)?;
    println!("`pipeline.run` results: {:?}", results);

    let results = pipeline.run_batch(
        vec![
            "Paris is the [MASK] of France",
            "Madrid is [MASK] capital of Madrid",
        ],
        None,
    )?;
    println!("`pipeline.run_batch` results: {:?}", results);

    Ok(())
}
