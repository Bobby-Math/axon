//! vLLM process management
//!
//! Handles spawning, monitoring, and terminating vLLM server processes.

use crate::error::{AxonError, Result};
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::sleep;

use super::config::VllmConfig;

/// A running vLLM server process
pub struct VllmProcess {
    /// The child process ID
    pid: Option<u32>,
}

impl VllmProcess {
    /// Spawn a new vLLM server process
    pub async fn spawn(config: VllmConfig) -> Result<Self> {
        let mut cmd = Command::new("python");
        cmd.arg("-m")
            .arg("vllm.entrypoints.openai.api_server")
            .arg("--model")
            .arg(&config.model_name)
            .arg("--host")
            .arg(&config.host)
            .arg("--port")
            .arg(config.port.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add tensor parallelism if specified
        if let Some(tp) = config.tensor_parallel_size {
            cmd.arg("--tensor-parallel-size").arg(tp.to_string());
        }

        // Add max sequence length if specified
        if let Some(max_len) = config.max_sequence_length {
            cmd.arg("--max-model-len").arg(max_len.to_string());
        }

        // Add dtype if specified
        if let Some(dtype) = config.dtype {
            if dtype != "auto" {
                cmd.arg("--dtype").arg(dtype);
            }
        }

        // Spawn the process
        let child = cmd.spawn()
            .map_err(|e| AxonError::ModelLoadFailed(format!("Failed to spawn vLLM: {}", e)))?;

        Ok(Self {
            pid: Some(child.id()),
        })
    }

    /// Check if the process is still running
    pub fn is_running(&self) -> bool {
        if let Some(pid) = self.pid {
            // Try to send signal 0 to check if process exists
            unsafe {
                let result = libc::kill(pid as i32, 0);
                result == 0 || (result == -1 && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH))
            }
        } else {
            false
        }
    }

    /// Wait until vLLM is ready to serve requests
    pub async fn wait_until_ready(&self) -> Result<()> {
        let url = format!("http://{}:{}/health", self.host(), self.port());
        let client = reqwest::Client::new();

        for _ in 0..60 {
            sleep(Duration::from_secs(2)).await;

            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    return Ok(());
                }
                Ok(_) => continue,
                Err(_) => continue,
            }
        }

        Err(AxonError::ModelLoadFailed("vLLM did not become ready in time".into()))
    }

    /// Get the host the process is listening on
    fn host(&self) -> String {
        "127.0.0.1".to_string()
    }

    /// Get the port the process is listening on
    fn port(&self) -> u16 {
        8000
    }

    /// Terminate the vLLM process
    pub async fn terminate(self) -> Result<()> {
        if let Some(pid) = self.pid {
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }

            // Give process time to terminate gracefully
            sleep(Duration::from_secs(5)).await;

            // Force kill if still running
            if self.is_running() {
                unsafe {
                    libc::kill(pid as i32, libc::SIGKILL);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vllm_config() {
        let config = VllmConfig {
            model_name: "test-model".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8000,
            tensor_parallel_size: Some(1),
            max_sequence_length: Some(2048),
            dtype: Some("auto".to_string()),
        };

        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.port, 8000);
    }
}
