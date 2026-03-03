// Package bench provides benchmarking tools for reasoning strategies.
//
// GSM8K (Grade School Math 8K) is the standard benchmark for testing
// chain-of-thought reasoning. It contains ~8,800 grade-school math
// word problems, each with a human-written solution and a final answer.
//
// Why GSM8K? Because math problems have VERIFIABLE answers.
// If the answer is 42, we can check if the model said 42.
// This is much harder with open-ended tasks like "write an essay"
// where evaluation is subjective. GSM8K gives us a clean signal:
// correct or incorrect, no ambiguity.
//
// Format (JSONL — one JSON object per line):
//
//	{"question": "Natalia sold...", "answer": "72"}
//
// The original dataset has full solutions, but for benchmarking
// we only need the question and the final numerical answer.
package bench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Problem is a single benchmark problem.
type Problem struct {
	Question string `json:"question"`
	Answer   string `json:"answer"` // The ground truth answer (a number)
}

// LoadGSM8K loads problems from a JSONL file.
//
// JSONL = JSON Lines: each line is a separate JSON object.
// This is the standard format for ML datasets because:
//   - You can stream it line-by-line (no need to load the whole file)
//   - You can easily count records (wc -l)
//   - You can split/shuffle with standard unix tools
//   - Appending new records is just appending a line
func LoadGSM8K(path string) ([]Problem, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", path, err)
	}
	defer f.Close()

	var problems []Problem
	scanner := bufio.NewScanner(f)

	// Increase the buffer size — some GSM8K entries with full solutions
	// can exceed the default 64KB line limit
	scanner.Buffer(make([]byte, 0, 256*1024), 256*1024)

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var p Problem
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNum, err)
		}

		// Normalize the answer — strip commas, whitespace, dollar signs
		p.Answer = NormalizeAnswer(p.Answer)
		problems = append(problems, p)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scanning: %w", err)
	}

	return problems, nil
}

// NormalizeAnswer cleans up an answer string for comparison.
//
// This is critical for fair evaluation. The model might output:
//
//	"42"      vs ground truth "42"      → match
//	"$42"     vs ground truth "42"      → should match
//	"42.0"    vs ground truth "42"      → should match
//	"42,000"  vs ground truth "42000"   → should match
//
// Without normalization, we'd undercount correct answers.
func NormalizeAnswer(s string) string {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, ",", "")  // Remove thousands separators
	s = strings.ReplaceAll(s, "$", "")  // Remove dollar signs
	s = strings.ReplaceAll(s, "%", "")  // Remove percent signs
	s = strings.TrimSuffix(s, ".0")     // 42.0 → 42
	s = strings.TrimSuffix(s, ".00")    // 42.00 → 42
	s = strings.TrimSpace(s)
	return s
}

// CheckAnswer compares a model's answer to the ground truth.
// Returns true if they match after normalization.
func CheckAnswer(modelAnswer, groundTruth string) bool {
	return NormalizeAnswer(modelAnswer) == NormalizeAnswer(groundTruth)
}
