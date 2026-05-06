use std::sync::Arc;

use snafu::ResultExt;

use crate::client::GeminiClient;

use super::{model::VertexBatchPredictionJob, ClientSnafu, Error};

/// Handle for an existing Vertex AI batch prediction job.
#[derive(Clone)]
pub struct VertexBatchPredictionHandle {
    name: String,
    client: Arc<GeminiClient>,
}

impl VertexBatchPredictionHandle {
    pub(crate) fn new(name: String, client: Arc<GeminiClient>) -> Self {
        Self { name, client }
    }

    /// The full resource name of the batch prediction job.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Fetch the current state of the job.
    pub async fn status(&self) -> Result<VertexBatchPredictionJob, Error> {
        self.client
            .get_vertex_batch_prediction_job(&self.name)
            .await
            .context(ClientSnafu)
    }

    /// Cancel the job.
    pub async fn cancel(&self) -> Result<(), Error> {
        self.client
            .cancel_vertex_batch_prediction_job(&self.name)
            .await
            .context(ClientSnafu)
    }
}
