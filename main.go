// The reasoner CLI — entry point for running benchmarks and reasoning strategies.
//
// Usage:
//   go run . bench -data data/gsm8k/test.jsonl -n 10
//   go run . ask "What is 15 + 27?"
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/dyllon/reasoner/bench"
	"github.com/dyllon/reasoner/client"
	"github.com/dyllon/reasoner/reasoning"
)

const defaultModel = "claude-sonnet-4-20250514"

func main() {
	// Handle Ctrl+C gracefully.
	// signal.NotifyContext creates a context that gets cancelled when
	// the process receives SIGINT (Ctrl+C). This propagates through
	// all our API calls, cleanly stopping in-flight requests.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	// Go's flag package doesn't support subcommands natively,
	// so we handle the first arg manually and create flag sets per command.
	switch os.Args[1] {
	case "bench":
		runBench(ctx, os.Args[2:])
	case "ask":
		runAsk(ctx, os.Args[2:])
	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("Usage: reasoner <command> [options]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  bench    Run GSM8K benchmark comparing reasoning strategies")
	fmt.Println("  ask      Ask a single question with a specific strategy")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  reasoner bench -data data/gsm8k/test.jsonl -n 10")
	fmt.Println("  reasoner ask -strategy zero-shot-cot \"What is 15 + 27?\"")
}

// runBench runs the GSM8K benchmark comparing all three strategies.
func runBench(ctx context.Context, args []string) {
	// flag.NewFlagSet creates a separate set of flags for this subcommand.
	// This is how you implement "git commit -m" style subcommands in Go.
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	dataPath := fs.String("data", "data/gsm8k/test.jsonl", "Path to GSM8K JSONL file")
	n := fs.Int("n", 10, "Number of problems to run (0 = all)")
	model := fs.String("model", defaultModel, "Model ID")
	fs.Parse(args)

	// Create the Claude client
	llm, err := client.NewAnthropicFromEnv(*model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintf(os.Stderr, "Set ANTHROPIC_API_KEY environment variable first.\n")
		os.Exit(1)
	}

	// Load the benchmark data
	problems, err := bench.LoadGSM8K(*dataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading data: %v\n", err)
		fmt.Fprintf(os.Stderr, "Download GSM8K first: see data/gsm8k/README\n")
		os.Exit(1)
	}

	// Limit to n problems if requested
	if *n > 0 && *n < len(problems) {
		problems = problems[:*n]
	}

	fmt.Printf("Running benchmark on %d problems with model %s\n\n", len(problems), *model)

	// Run all three strategies and collect results.
	// We define them as a slice of structs to keep it DRY.
	strategies := []struct {
		name reasoning.Strategy
		fn   bench.StrategyFunc
	}{
		{reasoning.StrategyDirect, reasoning.Direct},
		{reasoning.StrategyZeroShotCoT, reasoning.ZeroShotCoT},
		{reasoning.StrategyFewShotCoT, reasoning.FewShotCoT},
	}

	var results []*bench.BenchmarkResult
	for _, s := range strategies {
		fmt.Printf("--- %s ---\n", s.name)
		result := bench.Run(ctx, llm, problems, s.name, s.fn)
		results = append(results, result)
		fmt.Println()
	}

	// Print the comparison table
	bench.PrintSummary(results)
}

// runAsk asks a single question with a chosen strategy.
// Useful for debugging and seeing the full reasoning trace.
func runAsk(ctx context.Context, args []string) {
	fs := flag.NewFlagSet("ask", flag.ExitOnError)
	strategy := fs.String("strategy", "zero-shot-cot", "Strategy: direct, zero-shot-cot, few-shot-cot")
	model := fs.String("model", defaultModel, "Model ID")
	fs.Parse(args)

	// The question is whatever's left after flags
	if fs.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Usage: reasoner ask [options] \"your question\"\n")
		os.Exit(1)
	}
	question := fs.Arg(0)

	llm, err := client.NewAnthropicFromEnv(*model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Pick the strategy function
	var fn bench.StrategyFunc
	switch *strategy {
	case "direct":
		fn = reasoning.Direct
	case "zero-shot-cot":
		fn = reasoning.ZeroShotCoT
	case "few-shot-cot":
		fn = reasoning.FewShotCoT
	default:
		fmt.Fprintf(os.Stderr, "Unknown strategy: %s\n", *strategy)
		fmt.Fprintf(os.Stderr, "Options: direct, zero-shot-cot, few-shot-cot\n")
		os.Exit(1)
	}

	trace, err := fn(ctx, llm, question)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Print the full reasoning trace
	fmt.Printf("Question: %s\n\n", trace.Question)
	fmt.Printf("Strategy: %s\n", trace.Strategy)
	fmt.Printf("Latency:  %s\n", trace.Latency.Round(1*time.Millisecond))
	fmt.Printf("Tokens:   %d in + %d out = %d total\n\n",
		trace.Usage.InputTokens, trace.Usage.OutputTokens, trace.Usage.Total())

	fmt.Println("--- Reasoning ---")
	fmt.Println(trace.RawResponse)
	fmt.Println()

	if trace.Answer != "" {
		fmt.Printf("Extracted answer: %s\n", trace.Answer)
	} else {
		fmt.Println("(Could not extract a final answer)")
	}
}
