use backtrace::Backtrace;
use hf_hub::api::sync::ApiError;
use std::fmt;

#[derive(Debug)]
pub struct WrappedError {
    pub error: Box<dyn std::error::Error + Send + Sync>,
    pub backtrace: Backtrace,
}

impl fmt::Display for WrappedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for WrappedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

// `candle-holder` main error type
#[derive(thiserror::Error, Debug)]
pub enum Error {
    // -----------------------------------
    // From pretrained errors
    // -----------------------------------
    #[error("Repository '{0}' not found.")]
    RepositoryNotFound(String),
    #[error("Model '{0}' is not implemented. Create a new issue in 'https://github.com/gabrielmbmb/candle-holder' to request the implementation.")]
    ModelNotImplemented(String),
    #[error("Tokenizer '{0}' is not implemented. Create a new issue in 'https://github.com/gabrielmbmb/candle-holder' to request the implementation.")]
    TokenizerNotImplemented(String),

    // -----------------------------------
    // Load model errors
    // -----------------------------------
    #[error("Model weights not found in the repo.")]
    ModelWeightsNotFound,

    // -----------------------------------
    // Load tokenizer errors
    // -----------------------------------
    #[error("Tokenizer configuration is missing. Check the repository contains a `tokenizer_config.json` file.")]
    TokenizerMissingConfig,

    #[error("Tokenizer build error: {0}")]
    TokenizerBuildError(String),

    // -----------------------------------
    // Special tokens errors
    // -----------------------------------
    #[error("Missing the special token `{0}`.")]
    MissingSpecialToken(String),

    #[error("Missing the id of the special token `{0}`.")]
    MissingSpecialTokenId(String),

    // -----------------------------------
    // Tokenizer encoding errors
    // -----------------------------------
    #[error("Tokenizer encoding error: {0}.")]
    TokenizerEncodingError(String),

    // -----------------------------------
    // Chat template errors
    // -----------------------------------
    #[error("Tokenizer does not have a chat template.")]
    MissingChatTemplate(String),

    // -----------------------------------
    // `forward` method errors
    // -----------------------------------
    #[error("Forward param {0} cannot be `None`.")]
    MissingForwardParam(String),

    // -----------------------------------
    // `generate` method errors
    // -----------------------------------
    #[error("Generate param {0} cannot be `None`.{1}")]
    MissingGenerateParam(String, String),

    #[error("{0}")]
    GenerateParamValueError(String),

    #[error("{0}")]
    Msg(String),

    // -----------------------------------
    // RoPE errors
    // -----------------------------------
    #[error("RoPE param {0} cannot be `None`.")]
    MissingRopeParam(String),

    // Wrapped errors from other crates
    #[error(transparent)]
    Wrapped(#[from] WrappedError),
}

impl Error {
    pub fn wrap(e: impl std::error::Error + Send + Sync + 'static) -> Self {
        Error::Wrapped(WrappedError {
            error: Box::new(e),
            backtrace: Backtrace::new(),
        })
    }

    pub fn msg<T: std::fmt::Display>(msg: T) -> Self {
        Error::Msg(msg.to_string())
    }
}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::wrap(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::wrap(e)
    }
}

impl From<ApiError> for Error {
    fn from(e: ApiError) -> Self {
        Error::wrap(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::wrap(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[macro_export]
macro_rules! bail {
    ($msg:expr) => {
        return Err($crate::error::Error::msg($msg))
    };
}
