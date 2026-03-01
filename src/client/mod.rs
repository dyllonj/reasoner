/// LLM Client abstraction.
///
/// Why a trait? Even though we're only using Claude right now, a trait lets us:
/// 1. Swap in a mock client for testing (no API calls needed)
/// 2. Add OpenAI/local models later without changing calling code
/// 3. Learn Rust's trait system — the foundation of polymorphism in Rust
///
/// In Rust, traits are like interfaces (Java/Go) but more powerful:
/// - They can have default implementations
/// - They support associated types
/// - They're the basis for generics (trait bounds)
/// - async-trait is needed because Rust doesn't natively support async fn in traits (yet)
pub mod anthropic;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// A message in a conversation. The Claude API uses a turn-based format:
/// - "user" messages are your prompts
/// - "assistant" messages are Claude's responses
/// - You alternate user/assistant to build multi-turn conversations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// Parameters for a completion request.
/// These map directly to the Claude API's request body fields.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// The conversation history
    pub messages: Vec<Message>,

    /// Optional system prompt — sets the model's behavior/persona.
    /// Separate from messages because it's not part of the conversation;
    /// it's instructions that persist across all turns.
    pub system: Option<String>,

    /// Maximum tokens to generate. A "token" is roughly 3/4 of a word.
    /// More tokens = more room for reasoning = better CoT results.
    /// But also more cost and latency.
    pub max_tokens: u32,

    /// Temperature: controls randomness in token selection.
    ///
    /// The math: after the model computes logits z_i for each token i,
    /// it applies softmax with temperature T:
    ///
    ///   P(token_i) = exp(z_i / T) / Σ_j exp(z_j / T)
    ///
    /// - T = 0: always pick the highest-probability token (greedy/deterministic)
    /// - T = 1: sample from the model's natural distribution
    /// - T > 1: flatten the distribution (more random)
    /// - T < 1: sharpen the distribution (more focused)
    ///
    /// For CoT: T=0 gives consistent results for benchmarking.
    /// For self-consistency (Phase 2): T=0.7-1.0 gives diverse reasoning paths.
    pub temperature: Option<f32>,
}

/// The response from a completion request.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// The generated text
    pub content: String,
    /// Token usage stats
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// The trait that all LLM clients must implement.
///
/// We use `async fn` here — this requires Rust 1.75+ (we have 1.93).
/// In older Rust, you'd need the `async-trait` crate which boxes the future.
/// The native version is zero-cost (no heap allocation for the future).
pub trait LlmClient: Send + Sync {
    /// Send a completion request and get the full response.
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse>> + Send;
}
