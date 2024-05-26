use hf_hub::api::sync::ApiError;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    // Load model errors
    #[error("Model weights not found in the repo.")]
    ModelWeightsNotFound,

    // `forward` method errors
    #[error("Forward param {0} cannot be `None`.")]
    MissingForwardParam(String),

    #[error("{0}")]
    Msg(String),

    // Wrapped errors from other crates
    #[error(transparent)]
    Wrapped(Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn wrap(e: impl std::error::Error + Send + Sync + 'static) -> Self {
        Error::Wrapped(Box::new(e))
    }
}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Msg(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Msg(e.to_string())
    }
}

impl From<ApiError> for Error {
    fn from(e: ApiError) -> Self {
        Error::Msg(e.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Msg(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! bail {
    ($msg:expr) => {
        return Err($crate::error::Error::Msg($msg.into()))
    };
}
