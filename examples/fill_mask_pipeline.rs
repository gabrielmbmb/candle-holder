use anyhow::Result;
use candle_core::Device;
use candle_holder::FillMaskPipeline;

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let mut pipeline = FillMaskPipeline::new("gabrielmbmb/bert-test", &device, None)?;

    let results = pipeline.run(
        "Paris is the [MASK] of France, and it's a [MASK] city",
        None,
    )?;
    println!("`pipeline.run` results: {:?}", results);

    let results = pipeline.run_batch(
        vec![
            "Paris is the [MASK] of France, and it's a [MASK] city",
            "Madrid is the [MASK] of Spain",
        ],
        None,
    )?;
    println!("`pipeline.run_batch` results: {:?}", results);

    Ok(())
}
