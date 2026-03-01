/// Core types for the reasoner.
///
/// These types represent the fundamental abstractions of chain-of-thought reasoning:
/// - A "thought" is a single step in a reasoning chain
/// - A "reasoning trace" is the complete chain of thoughts
/// - An "answer" is the final extracted result
///
/// Why separate Thought from the final Answer? Because in chain-of-thought,
/// the intermediate reasoning steps are just as important as the conclusion.
/// We need to inspect, score, and compare reasoning paths — not just answers.
use serde::{Deserialize, Serialize};

/// A single reasoning step. In the simplest case, this is just text.
/// Later (Phase 2), we'll extend this with scores, branch IDs, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    /// The text content of this reasoning step
    pub content: String,
    /// Which step number in the chain (0-indexed)
    pub step: usize,
}

/// A complete chain of reasoning from question to answer.
/// This is what "chain-of-thought" literally means — a linked sequence of thoughts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrace {
    /// The original question/problem
    pub question: String,
    /// The reasoning steps (the "chain")
    pub thoughts: Vec<Thought>,
    /// The raw text of the full response (before parsing)
    pub raw_response: String,
    /// The extracted final answer
    pub answer: Option<String>,
    /// How long the API call took
    pub latency_ms: u64,
    /// How many tokens were used (input + output)
    pub token_usage: TokenUsage,
}

/// Token usage from the API.
/// Important for understanding cost and the relationship between
/// "thinking tokens" and answer quality.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl TokenUsage {
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

/// The strategy used to generate a reasoning trace.
/// Each variant corresponds to a different prompting technique.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReasoningStrategy {
    /// No chain of thought — just ask for the answer directly.
    /// This is our baseline to measure how much CoT helps.
    Direct,

    /// Zero-shot CoT: append "Let's think step by step" to the prompt.
    /// No examples needed. The model has seen this pattern in training data
    /// and it activates more careful, sequential reasoning.
    ZeroShotCoT,

    /// Few-shot CoT: provide worked examples showing reasoning traces.
    /// More reliable because you control the format and depth of reasoning.
    FewShotCoT,
}

impl std::fmt::Display for ReasoningStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Direct => write!(f, "direct"),
            Self::ZeroShotCoT => write!(f, "zero-shot-cot"),
            Self::FewShotCoT => write!(f, "few-shot-cot"),
        }
    }
}
