// Package client provides an LLM client abstraction.
//
// Why an interface? Even though we're only using Claude right now, an interface lets us:
//   1. Swap in a mock client for testing (no API calls needed)
//   2. Add OpenAI/local models later without changing calling code
//   3. Learn Go's interface system — implicit satisfaction, no "implements" keyword
//
// Go interfaces are satisfied implicitly: if a type has the right methods,
// it implements the interface. No declaration needed. This is structural typing
// (vs nominal typing in Java/Rust). It makes testing trivially easy — just
// define a struct with a Complete method and you have a mock.
package client

import "context"

// Message represents a single message in a conversation.
// The Claude API uses a turn-based format:
//   - "user" messages are your prompts
//   - "assistant" messages are Claude's responses
//   - You alternate user/assistant to build multi-turn conversations
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// UserMsg creates a user message.
func UserMsg(content string) Message {
	return Message{Role: "user", Content: content}
}

// AssistantMsg creates an assistant message.
func AssistantMsg(content string) Message {
	return Message{Role: "assistant", Content: content}
}

// Request holds parameters for a completion request.
// These map directly to the Claude API's request body fields.
type Request struct {
	Messages []Message

	// System prompt — sets the model's behavior/persona.
	// Separate from messages because it's not part of the conversation;
	// it's instructions that persist across all turns.
	System string

	// MaxTokens is the maximum number of tokens to generate.
	// A "token" is roughly 3/4 of a word.
	// More tokens = more room for reasoning = better CoT results.
	// But also more cost and latency.
	MaxTokens int

	// Temperature controls randomness in token selection.
	//
	// The math: after the model computes logits z_i for each token i,
	// it applies softmax with temperature T:
	//
	//   P(token_i) = exp(z_i / T) / sum_j exp(z_j / T)
	//
	// - T → 0: always pick the highest-probability token (greedy/deterministic)
	// - T = 1: sample from the model's natural distribution
	// - T > 1: flatten the distribution (more random)
	// - T < 1: sharpen the distribution (more focused)
	//
	// For CoT: T=0 gives consistent results for benchmarking.
	// For self-consistency (Phase 2): T=0.7-1.0 gives diverse reasoning paths.
	Temperature *float64 // nil means use API default (1.0)
}

// Response holds the result of a completion request.
type Response struct {
	Content      string // The generated text
	InputTokens  int
	OutputTokens int
}

// LLM is the interface that all LLM clients must implement.
//
// We take a context.Context as the first parameter — this is idiomatic Go.
// Context carries cancellation signals and deadlines. If you cancel the context,
// the HTTP request gets cancelled too. This is how Go handles timeouts and
// graceful shutdown without async/await.
type LLM interface {
	Complete(ctx context.Context, req Request) (*Response, error)
}
