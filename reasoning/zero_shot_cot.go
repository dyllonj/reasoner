package reasoning

import (
	"context"
	"fmt"

	"github.com/dyllon/reasoner/client"
)

// ZeroShotCoT implements zero-shot chain-of-thought prompting.
//
// The technique: append "Let's think step by step" to the prompt.
// That's it. No examples. No special formatting. Just those 6 words.
//
// Why does this work? Two reasons:
//
// 1. DISTRIBUTION SHIFT: LLMs are trained on internet text that includes
//    textbooks, tutorials, and worked examples. When you say "let's think
//    step by step", you're steering the model toward generating text that
//    looks like those worked examples — complete with intermediate calculations.
//
// 2. COMPUTATION SCALING: Each output token gets its own forward pass.
//    "Think step by step" makes the model generate intermediate tokens,
//    which means more forward passes, which means more total computation.
//    A model with L layers gets O(n * L * d^2) FLOPs for n output tokens,
//    vs O(L * d^2) for a single-token answer. More compute = harder
//    problems solvable.
//
// The original paper (Kojima et al., 2022 — "Large Language Models are
// Zero-Shot Reasoners") showed this simple trick improved accuracy on
// MultiArith from 17.7% to 78.7% with PaLM 540B. Just from adding
// "Let's think step by step."
func ZeroShotCoT(ctx context.Context, llm client.LLM, question string) (*Trace, error) {
	system := "You are a helpful math assistant. Think through problems step by step. " +
		"After your reasoning, give your final numerical answer on the last line in the format: #### <number>"

	// The key prompt engineering: we explicitly ask for step-by-step reasoning.
	// Note that we put the instruction AFTER the question — this matters because
	// attention is causal (each token only attends to previous tokens), so
	// the instruction to "think step by step" influences all subsequent generation.
	prompt := fmt.Sprintf("Question: %s\n\nLet's think step by step.", question)

	messages := []client.Message{
		client.UserMsg(prompt),
	}

	// More tokens than direct — we need room for reasoning steps.
	// Too few tokens and the model's reasoning gets truncated mid-thought.
	temp := 0.0
	return Execute(ctx, llm, question, messages, system, 1024, &temp, StrategyZeroShotCoT)
}
