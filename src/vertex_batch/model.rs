use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::generation::GenerateContentRequest;

/// A single JSONL line for Vertex Gemini batch prediction input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexBatchPredictionRequestFileItem {
    /// The generateContent-compatible request payload.
    pub request: GenerateContentRequest,
}

/// Create-request payload for Vertex AI batch prediction jobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchPredictionJobRequest {
    /// User-visible job name.
    pub display_name: String,
    /// Model resource name, e.g. `publishers/google/models/gemini-2.5-flash`.
    pub model: String,
    /// Input configuration for the job.
    pub input_config: VertexBatchPredictionInputConfig,
    /// Output configuration for the job.
    pub output_config: VertexBatchPredictionOutputConfig,
}

/// Input configuration for a Vertex batch prediction job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchPredictionInputConfig {
    /// Input format. For GCS Gemini batch jobs this is typically `jsonl`.
    pub instances_format: String,
    /// GCS source file(s).
    pub gcs_source: VertexGcsSource,
}

/// Output configuration for a Vertex batch prediction job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchPredictionOutputConfig {
    /// Output format. For GCS Gemini batch jobs this is typically `jsonl`.
    pub predictions_format: String,
    /// GCS destination prefix.
    pub gcs_destination: VertexGcsDestination,
}

/// GCS input source.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexGcsSource {
    /// One or more source object URIs.
    pub uris: Vec<String>,
}

/// GCS output destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexGcsDestination {
    /// Output URI prefix such as `gs://bucket/prefix/`.
    pub output_uri_prefix: String,
}

/// Vertex batch prediction job resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchPredictionJob {
    /// Full resource name.
    pub name: String,
    /// User-visible display name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Model resource used by the job.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Current job state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// Job error if terminally failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<VertexBatchPredictionError>,
    /// Input configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_config: Option<VertexBatchPredictionInputConfig>,
    /// Output configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<VertexBatchPredictionOutputConfig>,
    /// Output info returned by Vertex.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_info: Option<VertexBatchPredictionOutputInfo>,
    /// Create time.
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub create_time: Option<OffsetDateTime>,
    /// Start time.
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub start_time: Option<OffsetDateTime>,
    /// End time.
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub end_time: Option<OffsetDateTime>,
    /// Update time.
    #[serde(default, with = "time::serde::rfc3339::option")]
    pub update_time: Option<OffsetDateTime>,
}

/// Terminal or partial job error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexBatchPredictionError {
    /// Numeric status code.
    pub code: i32,
    /// Human-readable status message.
    pub message: String,
    /// Optional typed details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// Output information returned by Vertex.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexBatchPredictionOutputInfo {
    /// GCS directory containing output files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gcs_output_directory: Option<String>,
    /// BigQuery output dataset if used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bigquery_output_dataset: Option<String>,
}
