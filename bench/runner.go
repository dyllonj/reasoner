package bench

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/dyllon/reasoner/client"
	"github.com/dyllon/reasoner/reasoning"
)

// StrategyFunc is a function that takes a question and returns a reasoning trace.
// Each reasoning strategy (direct, zero-shot, few-shot) has this signature.
//
// In Go, functions are first-class values — you can pass them around like
// any other variable. This lets us treat strategies as interchangeable
// without needing complex inheritance hierarchies.
//
// Compare this to Java where you'd need a Strategy interface + 3 classes.
// In Go: just a function.
type StrategyFunc func(ctx context.Context, llm client.LLM, question string) (*reasoning.Trace, error)

// Result holds the outcome of running one strategy on one problem.
type Result struct {
	Problem  Problem
	Trace    *reasoning.Trace
	Correct  bool
	Err      error
}

// BenchmarkResult holds aggregate results for one strategy across all problems.
type BenchmarkResult struct {
	Strategy      reasoning.Strategy
	Total         int
	Correct       int
	Errors        int
	Accuracy      float64       // Correct / (Total - Errors)
	AvgLatency    time.Duration
	TotalInput    int           // Total input tokens across all problems
	TotalOutput   int           // Total output tokens
	Results       []Result      // Per-problem results
}

// Run executes a reasoning strategy on a set of problems and collects results.
//
// Parameters:
//   - ctx: cancellation context — if the user hits Ctrl+C, all in-flight
//     API calls get cancelled via context propagation
//   - llm: the LLM client to use
//   - problems: the benchmark problems to solve
//   - strategy: which reasoning approach to use
//   - strategyName: label for display
//   - fn: the actual function that implements the strategy
//
// We run problems sequentially (not in parallel) to:
//   1. Avoid rate limiting from the API
//   2. Make results reproducible
//   3. Make it easy to see progress
//
// Phase 2 will add parallel execution with goroutines.
func Run(
	ctx context.Context,
	llm client.LLM,
	problems []Problem,
	strategyName reasoning.Strategy,
	fn StrategyFunc,
) *BenchmarkResult {
	br := &BenchmarkResult{
		Strategy: strategyName,
		Total:    len(problems),
	}

	var totalLatency time.Duration

	for i, p := range problems {
		// Check if context is cancelled (user hit Ctrl+C)
		// select is Go's way of checking multiple channels/conditions.
		// ctx.Done() returns a channel that closes when the context is cancelled.
		// The default case means "don't block if nothing is ready."
		select {
		case <-ctx.Done():
			fmt.Printf("\n  Cancelled after %d/%d problems\n", i, len(problems))
			break
		default:
		}

		fmt.Printf("  [%d/%d] %s...", i+1, len(problems), truncate(p.Question, 50))

		trace, err := fn(ctx, llm, p.Question)

		result := Result{Problem: p}

		if err != nil {
			result.Err = err
			br.Errors++
			fmt.Printf(" ERROR: %v\n", err)
		} else {
			result.Trace = trace
			result.Correct = CheckAnswer(trace.Answer, p.Answer)

			if result.Correct {
				br.Correct++
			}

			totalLatency += trace.Latency
			br.TotalInput += trace.Usage.InputTokens
			br.TotalOutput += trace.Usage.OutputTokens

			status := "WRONG"
			if result.Correct {
				status = "CORRECT"
			}
			fmt.Printf(" [%s] got=%q expected=%q (%s, %d tokens)\n",
				status, trace.Answer, p.Answer, trace.Latency.Round(time.Millisecond), trace.Usage.Total())
		}

		br.Results = append(br.Results, result)
	}

	// Calculate aggregate stats
	answered := br.Total - br.Errors
	if answered > 0 {
		br.Accuracy = float64(br.Correct) / float64(answered)
		br.AvgLatency = totalLatency / time.Duration(answered)
	}

	return br
}

// PrintSummary prints a formatted summary of benchmark results.
// This is what you'll look at to compare strategies.
func PrintSummary(results []*BenchmarkResult) {
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("BENCHMARK RESULTS")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("%-15s %8s %8s %8s %10s %8s %8s\n",
		"Strategy", "Correct", "Total", "Acc%", "Avg Lat", "In Tok", "Out Tok")
	fmt.Println(strings.Repeat("-", 70))

	for _, br := range results {
		fmt.Printf("%-15s %8d %8d %7.1f%% %10s %8d %8d\n",
			br.Strategy,
			br.Correct,
			br.Total-br.Errors,
			br.Accuracy*100,
			br.AvgLatency.Round(time.Millisecond),
			br.TotalInput,
			br.TotalOutput,
		)
	}
	fmt.Println(strings.Repeat("=", 70))

	// Show the key insight
	if len(results) >= 2 {
		base := results[0]
		best := results[0]
		for _, r := range results[1:] {
			if r.Accuracy > best.Accuracy {
				best = r
			}
		}
		if best.Accuracy > base.Accuracy {
			improvement := (best.Accuracy - base.Accuracy) * 100
			costRatio := 1.0
			if base.TotalOutput > 0 {
				costRatio = float64(best.TotalOutput) / float64(base.TotalOutput)
			}
			fmt.Printf("\nKey insight: %s improved accuracy by +%.1f%% over %s\n",
				best.Strategy, improvement, base.Strategy)
			fmt.Printf("Cost: %.1fx more output tokens (reasoning takes more tokens)\n", costRatio)
		}
	}
}

// truncate shortens a string to maxLen characters, adding "..." if truncated.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
