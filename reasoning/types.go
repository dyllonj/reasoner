// Package reasoning implements chain-of-thought reasoning strategies.
//
// Core types for the reasoner live here. These represent the fundamental
// abstractions of chain-of-thought reasoning:
//   - A "Thought" is a single step in a reasoning chain
//   - A "Trace" is the complete chain of thoughts
//   - An "Answer" is the final extracted result
//
// Why separate Thought from the final Answer? Because in chain-of-thought,
// the intermediate reasoning steps are just as important as the conclusion.
// We need to inspect, score, and compare reasoning paths — not just answers.
package reasoning

import "time"

// Thought represents a single reasoning step.
// In the simplest case, this is just text.
// Later (Phase 2), we'll extend this with scores, branch IDs, etc.
type Thought struct {
	Content string `json:"content"` // The text of this reasoning step
	Step    int    `json:"step"`    // Which step number in the chain (0-indexed)
}

// Trace is a complete chain of reasoning from question to answer.
// This is what "chain-of-thought" literally means — a linked sequence of thoughts.
type Trace struct {
	Question    string        `json:"question"`     // The original question/problem
	Thoughts    []Thought     `json:"thoughts"`     // The reasoning steps (the "chain")
	RawResponse string        `json:"raw_response"` // Full model response before parsing
	Answer      string        `json:"answer"`       // Extracted final answer (empty if extraction failed)
	Latency     time.Duration `json:"latency"`      // How long the API call took
	Usage       TokenUsage    `json:"usage"`         // Token consumption
	Strategy    Strategy      `json:"strategy"`     // Which prompting strategy produced this
}

// TokenUsage tracks token consumption from the API.
// Important for understanding cost and the relationship between
// "thinking tokens" and answer quality.
//
// Pricing math (as of 2026):
//   Claude Sonnet: ~$3/M input, ~$15/M output tokens
//   CoT generates 3-10x more output tokens than direct answering.
//   So CoT costs 3-10x more per question — is the accuracy gain worth it?
//   That's exactly what our benchmark will measure.
type TokenUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Total returns input + output tokens.
func (u TokenUsage) Total() int {
	return u.InputTokens + u.OutputTokens
}

// Strategy identifies which prompting technique produced a trace.
type Strategy string

const (
	// StrategyDirect asks for the answer with no chain of thought.
	// This is our baseline to measure how much CoT helps.
	StrategyDirect Strategy = "direct"

	// StrategyZeroShotCoT appends "Let's think step by step" to the prompt.
	// No examples needed. The model has seen this pattern in training data
	// and it activates more careful, sequential reasoning.
	StrategyZeroShotCoT Strategy = "zero-shot-cot"

	// StrategyFewShotCoT provides worked examples showing reasoning traces.
	// More reliable because you control the format and depth of reasoning.
	StrategyFewShotCoT Strategy = "few-shot-cot"
)
