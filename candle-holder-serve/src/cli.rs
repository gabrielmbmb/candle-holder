use anyhow::{anyhow, Result};
use candle_core::{DType, Device};
use clap::Parser;
use serde::Serialize;
use std::str::FromStr;

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
pub(crate) struct Cli {
    /// The host to listen on.
    #[arg(long, default_value = "0.0.0.0:8080")]
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

    /// The dtype to load the model weights with.
    #[arg(long)]
    dtype: Option<DTypeOption>,

    /// The number of workers to use for inference.
    #[arg(long, default_value = "1")]
    num_workers: usize,
}

impl Cli {
    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Get the [`candle_core::Device`] corresponding to the selected device option.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested device is not available.
    pub fn device(&self) -> Result<Device> {
        match self.device {
            DeviceOption::Cuda(device_id) if cfg!(feature = "cuda") => {
                Ok(Device::new_cuda(device_id)?)
            }
            DeviceOption::Metal if cfg!(feature = "metal") => Ok(Device::new_metal(0)?),
            DeviceOption::Cpu => Ok(Device::Cpu),
            _ => Err(anyhow!("Requested device is not available")),
        }
    }

    /// Get the [`candle_core::DType`] corresponding to the selected dtype option.
    pub fn dtype(&self) -> Option<DType> {
        self.dtype.as_ref().map(|dtype| match dtype {
            DTypeOption::Float16 => DType::F16,
            DTypeOption::BFloat16 => DType::BF16,
            DTypeOption::Float32 => DType::F32,
        })
    }
}

#[derive(Debug, Parser, Clone, Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Pipeline {
    FeatureExtraction,
    FillMask,
    TextClassification,
    TextGeneration,
    TokenClassification,
    ZeroShotClassification,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub(crate) enum DeviceOption {
    Cpu,
    Metal,
    #[value(skip)]
    Cuda(usize),
}

#[derive(Debug, Clone, clap::ValueEnum)]
#[clap(rename_all = "lowercase")]
pub(crate) enum DTypeOption {
    Float16,
    BFloat16,
    Float32,
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
