use std::sync::Arc;

use snafu::{OptionExt, ResultExt};

use crate::{client::GeminiClient, generation::GenerateContentRequest};

use super::{
    handle::VertexBatchPredictionHandle,
    model::{
        VertexBatchPredictionInputConfig, VertexBatchPredictionJobRequest,
        VertexBatchPredictionOutputConfig, VertexBatchPredictionRequestFileItem,
        VertexGcsDestination, VertexGcsSource,
    },
    ClientSnafu, Error, MissingFieldSnafu, SerializeSnafu,
};

/// Builder for Vertex AI Gemini batch prediction jobs.
#[derive(Clone)]
pub struct VertexBatchPredictionBuilder {
    client: Arc<GeminiClient>,
    display_name: String,
    project: Option<String>,
    location: Option<String>,
    model_resource: Option<String>,
    input_gcs_uri: Option<String>,
    output_gcs_uri_prefix: Option<String>,
    requests: Vec<GenerateContentRequest>,
}

impl VertexBatchPredictionBuilder {
    pub(crate) fn new(client: Arc<GeminiClient>) -> Self {
        Self {
            client,
            display_name: "RustVertexBatch".to_string(),
            project: None,
            location: None,
            model_resource: None,
            input_gcs_uri: None,
            output_gcs_uri_prefix: None,
            requests: Vec::new(),
        }
    }

    /// Sets the display name for the job.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = name.into();
        self
    }

    /// Sets the GCP project id used for the batch job resource path.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Sets the Vertex location used for the batch job resource path.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// Sets the full model resource name. If omitted, the builder derives
    /// `publishers/google/{model}` from the client model when possible.
    pub fn with_model_resource(mut self, model_resource: impl Into<String>) -> Self {
        self.model_resource = Some(model_resource.into());
        self
    }

    /// Sets the Cloud Storage JSONL input URI.
    pub fn with_gcs_input_uri(mut self, input_gcs_uri: impl Into<String>) -> Self {
        self.input_gcs_uri = Some(input_gcs_uri.into());
        self
    }

    /// Sets the Cloud Storage output URI prefix.
    pub fn with_gcs_output_uri_prefix(mut self, output_gcs_uri_prefix: impl Into<String>) -> Self {
        self.output_gcs_uri_prefix = Some(output_gcs_uri_prefix.into());
        self
    }

    /// Replaces the request set used to generate JSONL input lines.
    pub fn with_requests(mut self, requests: Vec<GenerateContentRequest>) -> Self {
        self.requests = requests;
        self
    }

    /// Adds a single request to the JSONL input set.
    pub fn with_request(mut self, request: GenerateContentRequest) -> Self {
        self.requests.push(request);
        self
    }

    /// Builds the JSONL content expected by Vertex Gemini batch prediction.
    pub fn build_jsonl(&self) -> Result<String, Error> {
        let mut jsonl = String::new();
        for request in &self.requests {
            let line = serde_json::to_string(&VertexBatchPredictionRequestFileItem {
                request: request.clone(),
            })
            .context(SerializeSnafu)?;
            jsonl.push_str(&line);
            jsonl.push('\n');
        }
        Ok(jsonl)
    }

    /// Builds the create-job request body.
    pub fn build(self) -> Result<VertexBatchPredictionJobRequest, Error> {
        let model = self
            .model_resource
            .unwrap_or_else(|| normalize_model_resource(self.client.model.as_str()));
        let input_gcs_uri = self.input_gcs_uri.context(MissingFieldSnafu {
            field: "input_gcs_uri",
        })?;
        let output_gcs_uri_prefix = self.output_gcs_uri_prefix.context(MissingFieldSnafu {
            field: "output_gcs_uri_prefix",
        })?;

        Ok(VertexBatchPredictionJobRequest {
            display_name: self.display_name,
            model,
            input_config: VertexBatchPredictionInputConfig {
                instances_format: "jsonl".to_string(),
                gcs_source: VertexGcsSource {
                    uris: vec![input_gcs_uri],
                },
            },
            output_config: VertexBatchPredictionOutputConfig {
                predictions_format: "jsonl".to_string(),
                gcs_destination: VertexGcsDestination {
                    output_uri_prefix: output_gcs_uri_prefix,
                },
            },
        })
    }

    /// Creates the Vertex batch prediction job and returns a handle.
    pub async fn execute(self) -> Result<VertexBatchPredictionHandle, Error> {
        let project = self
            .project
            .clone()
            .context(MissingFieldSnafu { field: "project" })?;
        let location = self
            .location
            .clone()
            .context(MissingFieldSnafu { field: "location" })?;
        let client = self.client.clone();
        let job = client
            .create_vertex_batch_prediction_job(self.build()?, &project, &location)
            .await
            .context(ClientSnafu)?;
        Ok(VertexBatchPredictionHandle::new(job.name, client))
    }
}

fn normalize_model_resource(model: &str) -> String {
    if model.starts_with("publishers/") || model.starts_with("projects/") {
        model.to_string()
    } else if model.starts_with("models/") {
        format!("publishers/google/{model}")
    } else {
        format!("publishers/google/models/{model}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gemini;

    #[test]
    fn normalize_model_resource_prefixes_google_models() {
        assert_eq!(
            normalize_model_resource("models/gemini-2.5-flash"),
            "publishers/google/models/gemini-2.5-flash"
        );
        assert_eq!(
            normalize_model_resource("publishers/google/models/gemini-2.5-flash"),
            "publishers/google/models/gemini-2.5-flash"
        );
    }

    #[test]
    fn build_jsonl_wraps_requests_in_request_field() {
        let gemini = Gemini::new("test-key").unwrap();
        let request = gemini.generate_content().with_user_message("hello").build();
        let jsonl = gemini
            .vertex_batch_prediction()
            .with_request(request)
            .build_jsonl()
            .unwrap();

        assert!(jsonl.contains("\"request\""));
        assert!(jsonl.contains("hello"));
    }

    #[test]
    fn build_request_uses_jsonl_and_gcs_fields() {
        let gemini = Gemini::new("test-key").unwrap();
        let request = gemini
            .vertex_batch_prediction()
            .with_project("project")
            .with_location("global")
            .with_gcs_input_uri("gs://bucket/input.jsonl")
            .with_gcs_output_uri_prefix("gs://bucket/output/")
            .build()
            .unwrap();

        assert_eq!(request.input_config.instances_format, "jsonl");
        assert_eq!(request.output_config.predictions_format, "jsonl");
        assert_eq!(
            request.input_config.gcs_source.uris[0],
            "gs://bucket/input.jsonl"
        );
        assert_eq!(
            request.output_config.gcs_destination.output_uri_prefix,
            "gs://bucket/output/"
        );
    }
}
