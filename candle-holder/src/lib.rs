pub mod error;
pub mod utils;

pub use error::Error;
pub use utils::FromPretrainedParameters;

/// A type alias for `Result<T, Error>` for the `candle-holder` crate.
pub type Result<T> = std::result::Result<T, Error>;
