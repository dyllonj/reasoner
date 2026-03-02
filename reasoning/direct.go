package reasoning

import (
	"context"
	"fmt"

	"github.com/dyllon/reasoner/client"
)

// Direct prompting — our baseline. No chain of thought.
//
// This just asks the model to answer the question directly.
// The model must compress ALL reasoning into a single forward pass
// through its transformer layers. For simple questions this works fine,
// but for multi-step math problems, accuracy drops significantly.
//
// Why include a baseline? Science requires controls. If we can't show
// that CoT actually improves over direct prompting on OUR benchmark
// with OUR evaluation, we can't claim CoT works — we'd just be
// hand-waving at the Wei et al. paper.
func Direct(ctx context.Context, llm client.LLM, question string) (*Trace, error) {
	system := "You are a helpful math assistant. Answer the question directly. " +
		"Give your final numerical answer on the last line in the format: #### <number>"

	messages := []client.Message{
		client.UserMsg(fmt.Sprintf("Question: %s", question)),
	}

	temp := 0.0 // Greedy decoding for consistent benchmarking
	return Execute(ctx, llm, question, messages, system, 256, &temp, StrategyDirect)
}
