package reasoning

import (
	"context"
	"regexp"
	"strings"
	"time"

	"github.com/dyllon/reasoner/client"
)

// Execute sends a prompt to the LLM and wraps the result in a Trace.
// This is the common plumbing — each strategy only differs in what
// messages and system prompt it constructs.
func Execute(
	ctx context.Context,
	llm client.LLM,
	question string,
	messages []client.Message,
	system string,
	maxTokens int,
	temperature *float64,
	strategy Strategy,
) (*Trace, error) {
	req := client.Request{
		Messages:    messages,
		System:      system,
		MaxTokens:   maxTokens,
		Temperature: temperature,
	}

	// Time the API call — latency matters for comparing strategies
	start := time.Now()
	resp, err := llm.Complete(ctx, req)
	if err != nil {
		return nil, err
	}
	latency := time.Since(start)

	// Parse the response into thoughts
	thoughts := ParseThoughts(resp.Content)

	// Extract the final answer
	answer := ExtractAnswer(resp.Content)

	return &Trace{
		Question:    question,
		Thoughts:    thoughts,
		RawResponse: resp.Content,
		Answer:      answer,
		Latency:     latency,
		Usage: TokenUsage{
			InputTokens:  resp.InputTokens,
			OutputTokens: resp.OutputTokens,
		},
		Strategy: strategy,
	}, nil
}

// ParseThoughts splits raw text into individual reasoning steps.
//
// We look for numbered steps ("1.", "Step 1:"), or paragraph breaks.
// This is heuristic — real production systems might ask the model to
// output structured JSON, but for learning, parsing natural language
// teaches you about the messiness of LLM output.
func ParseThoughts(text string) []Thought {
	var thoughts []Thought
	var current strings.Builder
	step := 0

	// Regex to detect numbered steps: "1.", "1)", "Step 1:", etc.
	numberedStep := regexp.MustCompile(`^\d+[\.\)]`)

	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)

		isNewStep := numberedStep.MatchString(trimmed)

		if isNewStep && current.Len() > 0 {
			thoughts = append(thoughts, Thought{
				Content: strings.TrimSpace(current.String()),
				Step:    step,
			})
			step++
			current.Reset()
			current.WriteString(trimmed)
		} else if trimmed == "" && current.Len() > 0 {
			// Paragraph break — might be a new thought
			thoughts = append(thoughts, Thought{
				Content: strings.TrimSpace(current.String()),
				Step:    step,
			})
			step++
			current.Reset()
		} else {
			if current.Len() > 0 {
				current.WriteString("\n")
			}
			current.WriteString(trimmed)
		}
	}

	// Don't forget the last thought
	if current.Len() > 0 {
		thoughts = append(thoughts, Thought{
			Content: strings.TrimSpace(current.String()),
			Step:    step,
		})
	}

	return thoughts
}

// ExtractAnswer extracts the final numerical answer from a reasoning trace.
//
// GSM8K answers follow the format "#### <number>" at the end.
// We also check for common patterns like "the answer is <number>".
//
// Why is answer extraction hard? Because LLMs are free-form text generators.
// They might say "11", "The answer is 11", "11 balls", "eleven", etc.
// Robust extraction is a real engineering challenge in evaluation systems.
func ExtractAnswer(text string) string {
	// Check for GSM8K format: #### <answer>
	if idx := strings.LastIndex(text, "####"); idx != -1 {
		answer := strings.TrimSpace(text[idx+4:])
		// Take everything up to the next newline
		if nl := strings.Index(answer, "\n"); nl != -1 {
			answer = answer[:nl]
		}
		answer = strings.TrimSpace(answer)
		if answer != "" {
			return answer
		}
	}

	// Check for "the answer is <X>" pattern (case insensitive)
	lower := strings.ToLower(text)
	patterns := []string{
		"the final answer is ",
		"therefore, the answer is ",
		"the answer is ",
	}
	for _, pattern := range patterns {
		if idx := strings.LastIndex(lower, pattern); idx != -1 {
			start := idx + len(pattern)
			answer := strings.TrimSpace(text[start:])
			// Take the first word/number
			fields := strings.FieldsFunc(answer, func(r rune) bool {
				return r == ' ' || r == '.' || r == ',' || r == '\n'
			})
			if len(fields) > 0 {
				return strings.TrimSpace(fields[0])
			}
		}
	}

	return ""
}
