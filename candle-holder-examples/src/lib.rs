use anyhow::Result;
use std::env;

use candle_core::Device;

pub enum DeviceOption {
    Cpu,
    Metal,
    Cuda(usize),
}

pub fn get_device(device: Option<DeviceOption>) -> Result<Device> {
    let device = match device {
        Some(DeviceOption::Cuda(device_id)) if cfg!(feature = "cuda") => {
            Device::new_cuda(device_id)?
        }
        Some(DeviceOption::Metal) if cfg!(feature = "metal") => Device::new_metal(0)?,
        _ => Device::Cpu,
    };

    Ok(device)
}

pub fn parse_device_option() -> Option<DeviceOption> {
    let args: Vec<String> = env::args().collect();

    // Expecting something like: --device cpu, --device metal, or --device cuda:<id>
    if args.len() > 2 && args[1] == "--device" {
        match args[2].as_str() {
            "metal" => Some(DeviceOption::Metal),
            cuda if cuda.starts_with("cuda:") => {
                let id_part = &cuda["cuda:".len()..];
                if let Ok(device_id) = id_part.parse::<usize>() {
                    Some(DeviceOption::Cuda(device_id))
                } else {
                    eprintln!("Error: Invalid CUDA device id: {}", id_part);
                    None
                }
            }
            _ => Some(DeviceOption::Cpu),
        }
    } else {
        Some(DeviceOption::Cpu)
    }
}

pub fn get_device_from_args() -> Result<Device> {
    get_device(parse_device_option())
}
