use candle_core::Device;
use candle_holder_models::AutoModel;

fn main() {
    let device = Device::new_metal(0).unwrap();
    let model =
        AutoModel::from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", &device, None).unwrap();
}
