#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::Result;
use candle_holder_examples::get_device_from_args;
use candle_holder_pipelines::ZeroShotClassificationPipeline;

fn main() -> Result<()> {
    let device = get_device_from_args()?;
    println!("Device: {:?}", device);

    let pipeline = ZeroShotClassificationPipeline::new(
        "Recognai/bert-base-spanish-wwm-cased-xnli",
        &device,
        None,
        None,
    )?;

    let scores = pipeline.run(
        "El Ford Fiesta es de los mejores del mercado.",
        vec!["coches", "plantas", "eventos"],
        None,
    )?;

    println!("`pipeline.run` results: {:?}", scores);

    let results = pipeline.run_batch(
        vec![
            "El Ford Fiesta es de los mejores del mercado.",
            "Las margaritas son preciosas.",
            "El concierto fue un éxito.",
        ],
        vec!["coches", "plantas", "eventos"],
        None,
    )?;
    println!("`pipeline.run_batch` results: {:?}", results);

    Ok(())
}
