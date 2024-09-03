mod routes;

use anyhow::{anyhow, Result};
use axum::{routing::get, Router};
use clap::Parser;
use serde::Serialize;
use std::str::FromStr;

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// The host to listen on.
    #[arg(long, default_value = "0.0.0.0:3000")]
    host: String,

    /// The Hugging Face repository id of the model to be loaded.
    #[arg(short, long)]
    model: String,

    /// The name of the pipeline to be served.
    #[arg(short, long)]
    pipeline: Pipeline,

    /// The device to run the pipeline on.
    #[arg(short, long, value_parser = parse_device, default_value = "cpu")]
    device: DeviceOption,
}

#[derive(Debug, Parser, Clone, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Pipeline {
    FeatureExtraction,
    FillMask,
    TextClassification,
    TextGeneration,
    TokenClassification,
    ZeroShotClassification,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum DeviceOption {
    Cpu,
    Metal,
    #[value(skip)]
    Cuda(usize),
}

impl FromStr for DeviceOption {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cpu" => Ok(DeviceOption::Cpu),
            "metal" => Ok(DeviceOption::Metal),
            s if s.starts_with("cuda:") => {
                let id = s.strip_prefix("cuda:").unwrap().parse::<usize>()?;
                Ok(DeviceOption::Cuda(id))
            }
            _ => Err(anyhow!("Invalid device option: {}", s)),
        }
    }
}

fn parse_device(s: &str) -> Result<DeviceOption, anyhow::Error> {
    DeviceOption::from_str(s)
}

#[tokio::main]
async fn main() {
    let args = Cli::parse();

    // Create a new router
    let app = Router::new().route("/", get(root));

    let listener = tokio::net::TcpListener::bind(args.host).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Hello, World!"
}
