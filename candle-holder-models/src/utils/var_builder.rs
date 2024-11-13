use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init};

/// A backend for retrieving tensors that ensures compatibility with old tensor naming conventions.
/// This backend is able to handle the following cases:
///
/// 1. The model prefix is missing from the tensor name.
/// 2. Tensors are named as `beta` and `gamma` instead of `weight` and `bias`.
///
/// This struct wraps a `SimpleBackend` implementation and provides an additional
/// `model_name` field to support model-specific tensor retrieval operations.
pub struct CompatibilityTensorRetrievalBackend {
    inner: Box<dyn SimpleBackend>,
    model_name: String,
}

impl CompatibilityTensorRetrievalBackend {
    pub fn new(inner: Box<dyn SimpleBackend>, model_name: String) -> Self {
        Self { inner, model_name }
    }

    /// Create a new `CompatibilityTensorRetrievalBackend` from a `PthTensors` instance that reads tensors from a `.pth` file.
    ///
    /// # Arguments
    ///
    /// * `p` - The path to the `.pth` file.
    /// * `model_name` - The name of the model.
    ///
    /// # Returns
    ///
    /// A `CompatibilityTensorRetrievalBackend
    pub fn from_pth<P: AsRef<std::path::Path>>(
        p: P,
        model_name: String,
    ) -> candle_core::Result<Self> {
        let pth = candle_core::pickle::PthTensors::new(p, None)?;
        Ok(Self {
            inner: Box::new(pth),
            model_name,
        })
    }

    /// Create a new `CompatibilityTensorRetrievalBackend` from a `MmapedSafetensors` instance that reads tensors from a `.safetensors` file.
    ///
    /// # Arguments
    ///
    /// * `paths` - A list of paths to the `.safetensors` files.
    /// * `model_name` - The name of the model.
    ///
    /// # Returns
    ///
    /// A `CompatibilityTensorRetrievalBackend
    pub unsafe fn from_mmaped_safetensors<P: AsRef<std::path::Path>>(
        paths: &[P],
        model_name: String,
    ) -> candle_core::Result<Self> {
        let tensors = candle_core::safetensors::MmapedSafetensors::multi(paths)?;
        Ok(Self {
            inner: Box::new(tensors),
            model_name,
        })
    }

    fn rename(&self, name: &str) -> String {
        // Check if the original name exists
        if self.inner.contains_tensor(name) {
            return name.to_string();
        }

        // Try removing the model name prefix
        let without_prefix = name.strip_prefix(&self.model_name).unwrap_or(name);

        // Function to replace weight/bias with beta/gamma
        let replace_weight_bias = |s: &str| s.replace("weight", "gamma").replace("bias", "beta");

        // Generate all possible combinations
        let possible_names = [
            without_prefix.to_string(),
            replace_weight_bias(name),
            replace_weight_bias(without_prefix),
        ];

        // Find the first name that exists in the tensor
        for possible_name in possible_names.iter() {
            if self.inner.contains_tensor(possible_name) {
                return possible_name.to_string();
            }
        }

        // If no matching tensor is found, return the original name
        name.to_string()
    }
}

impl SimpleBackend for CompatibilityTensorRetrievalBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let renamed = self.rename(name);
        self.inner.get(s, &renamed, h, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        let renamed = self.rename(name);
        self.inner.contains_tensor(&renamed)
    }
}
