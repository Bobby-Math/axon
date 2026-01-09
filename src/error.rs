//! Error types for Axon
//!
//! Unified error types that can represent failures from any backend.

use std::fmt;

/// Result type for Axon operations
pub type Result<T> = std::result::Result<T, AxonError>;

/// Errors that can occur in Axon
#[derive(Debug)]
pub enum AxonError {
    /// Model configuration is invalid
    InvalidConfig(String),

    /// Model failed to load
    ModelLoadFailed(String),

    /// Inference request failed
    InferenceFailed(String),

    /// Backend process is not running
    BackendNotRunning,

    /// Health check failed
    Unhealthy(String),

    /// I/O error
    IoError(std::io::Error),

    /// HTTP client error
    HttpError(String),

    /// Timeout occurred
    Timeout(String),

    /// Backend-specific error
    BackendError(String),

    /// Unknown or uncategorized error
    Other(String),
}

impl fmt::Display for AxonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::ModelLoadFailed(msg) => write!(f, "Model load failed: {}", msg),
            Self::InferenceFailed(msg) => write!(f, "Inference failed: {}", msg),
            Self::BackendNotRunning => write!(f, "Backend process is not running"),
            Self::Unhealthy(msg) => write!(f, "Backend unhealthy: {}", msg),
            Self::IoError(err) => write!(f, "I/O error: {}", err),
            Self::HttpError(msg) => write!(f, "HTTP error: {}", msg),
            Self::Timeout(msg) => write!(f, "Timeout: {}", msg),
            Self::BackendError(msg) => write!(f, "Backend error: {}", msg),
            Self::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for AxonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AxonError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

impl From<reqwest::Error> for AxonError {
    fn from(err: reqwest::Error) -> Self {
        Self::HttpError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AxonError::InvalidConfig("model name is empty".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid configuration: model name is empty"
        );
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let axon_err: AxonError = io_err.into();
        assert!(matches!(axon_err, AxonError::IoError(_)));
    }
}
