/// Anthropic Claude API client — built from scratch.
///
/// We're NOT using an SDK. You'll see every HTTP header, every JSON field,
/// every error case. This is how API clients actually work under the hood.
///
/// The Claude Messages API:
///   POST https://api.anthropic.com/v1/messages
///
/// Required headers:
///   x-api-key: <your key>          — authentication
///   anthropic-version: 2023-06-01  — API version (prevents breaking changes)
///   content-type: application/json — we're sending JSON
///
/// The response contains "content blocks" — usually one text block.
/// In streaming mode, these arrive as Server-Sent Events (SSE).
use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{CompletionRequest, CompletionResponse, LlmClient};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

/// The Anthropic client. Holds the HTTP client and auth credentials.
///
/// `reqwest::Client` internally uses connection pooling — creating one client
/// and reusing it across requests is much more efficient than creating a new
/// client per request (avoids TCP handshake + TLS negotiation overhead).
pub struct AnthropicClient {
    http: Client,
    api_key: String,
    model: String,
}

impl AnthropicClient {
    /// Create a new client.
    ///
    /// `model` is the model ID, e.g. "claude-sonnet-4-20250514".
    /// The API key comes from the environment — never hardcode secrets.
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            http: Client::new(),
            api_key,
            model,
        }
    }

    /// Create from environment variables.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let api_key =
            std::env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY not set")?;
        Ok(Self::new(api_key, model.into()))
    }
}

// --- API Request/Response types ---
// These mirror the Claude API's JSON schema exactly.
// We define them as Rust structs so serde can serialize/deserialize them.

#[derive(Serialize)]
struct ApiRequest {
    model: String,
    messages: Vec<ApiMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: String,
}

/// The API response.
///
/// `content` is an array because Claude can return multiple content blocks
/// (e.g., text + tool calls). For our purposes, we only care about text blocks.
///
/// `usage` tells us how many tokens were consumed. This matters because:
/// - Input tokens: cost scales with prompt length
/// - Output tokens: cost scales with response length AND determines compute budget
/// - CoT generates more output tokens — we need to track this to understand cost/benefit
#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

/// Error response from the API.
/// When something goes wrong, Claude returns structured error info.
#[derive(Deserialize)]
struct ApiError {
    error: ApiErrorDetail,
}

#[derive(Deserialize)]
struct ApiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

impl LlmClient for AnthropicClient {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Build the API request body
        let body = ApiRequest {
            model: self.model.clone(),
            messages: request
                .messages
                .iter()
                .map(|m| ApiMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect(),
            max_tokens: request.max_tokens,
            system: request.system,
            temperature: request.temperature,
        };

        // Send the HTTP request.
        //
        // Note the method chain:
        //   post(url)      — sets the HTTP method and URL
        //   header(k, v)   — adds request headers
        //   json(&body)    — serializes `body` to JSON and sets content-type
        //   send()         — actually sends the request (this is the async part)
        //   await?         — waits for the response, propagates errors with ?
        let response = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        // Check the HTTP status code.
        // 200 = success. Anything else = error.
        // Common errors:
        //   401 = invalid API key
        //   429 = rate limited (too many requests)
        //   500 = server error (retry with backoff)
        let status = response.status();
        let response_text = response
            .text()
            .await
            .context("Failed to read response body")?;

        if !status.is_success() {
            // Try to parse the structured error response
            if let Ok(api_error) = serde_json::from_str::<ApiError>(&response_text) {
                anyhow::bail!(
                    "Anthropic API error ({}): {} — {}",
                    status,
                    api_error.error.error_type,
                    api_error.error.message
                );
            }
            anyhow::bail!("Anthropic API error ({}): {}", status, response_text);
        }

        // Parse the successful response
        let api_response: ApiResponse = serde_json::from_str(&response_text)
            .context("Failed to parse Anthropic API response")?;

        // Extract text from content blocks.
        // The API can return multiple blocks, but for text completions
        // there's typically just one text block.
        let content = api_response
            .content
            .iter()
            .filter(|b| b.block_type == "text")
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("");

        Ok(CompletionResponse {
            content,
            input_tokens: api_response.usage.input_tokens,
            output_tokens: api_response.usage.output_tokens,
        })
    }
}
