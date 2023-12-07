use candle_core::Device;
use candle_holder::TextClassificationPipeline;

fn main() {
    let pipeline = TextClassificationPipeline::new("gabrielmbmb/finbert", &Device::Cpu).unwrap();
    let scores = pipeline
        .forward(
            "Google announces new LLM Gemini to compete with GPT-4",
            true,
        )
        .unwrap();
    println!("{:?}", scores);
}
