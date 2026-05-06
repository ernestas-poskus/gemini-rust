//! Vertex AI batch prediction support for Gemini models.
//!
//! This module models the Vertex AI `batchPredictionJobs` workflow, which is
//! distinct from the Gemini API's inline `batchGenerateContent` surface.

pub mod builder;
pub mod handle;
pub mod model;

pub use builder::VertexBatchPredictionBuilder;
pub use handle::VertexBatchPredictionHandle;

use snafu::Snafu;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("client invocation error"))]
    Client { source: crate::client::Error },

    #[snafu(display("failed to serialize Vertex batch JSONL input"))]
    Serialize { source: serde_json::Error },

    #[snafu(display("missing required field: {field}"))]
    MissingField { field: &'static str },
}
