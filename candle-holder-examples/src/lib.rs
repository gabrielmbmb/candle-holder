use anyhow::{anyhow, Result};
use candle_core::Device;
use clap::Parser;
use std::str::FromStr;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[arg(long, value_parser = parse_device, default_value = "cpu")]
    pub device: DeviceOption,
}

impl Cli {
    pub fn get_device(&self) -> Result<Device> {
        match self.device {
            DeviceOption::Cuda(device_id) if cfg!(feature = "cuda") => {
                Ok(Device::new_cuda(device_id)?)
            }
            DeviceOption::Metal if cfg!(feature = "metal") => Ok(Device::new_metal(0)?),
            DeviceOption::Cpu => Ok(Device::Cpu),
            _ => Err(anyhow!("Requested device is not available")),
        }
    }
}

#[derive(Clone, Debug)]
pub enum DeviceOption {
    Cpu,
    Metal,
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

pub fn get_device_from_args() -> Result<Device> {
    let cli = Cli::parse();
    cli.get_device()
}
