package reasoning

import (
	"context"
	"fmt"

	"github.com/dyllon/reasoner/client"
)

// FewShotCoT implements few-shot chain-of-thought prompting.
//
// This is the technique from the original Wei et al. (2022) paper.
// Instead of just saying "think step by step", we provide EXAMPLES
// of complete reasoning traces. The model learns the format and depth
// of reasoning from these examples via in-context learning.
//
// Why is this better than zero-shot?
//
// 1. FORMAT CONTROL: You define exactly what a good reasoning trace
//    looks like — numbered steps, clear arithmetic, explicit intermediate
//    values. The model mimics this format.
//
// 2. DEPTH CALIBRATION: Your examples show how much detail to include.
//    Too little detail and the model skips steps (making errors).
//    Too much and it wastes tokens on obvious things.
//
// 3. IN-CONTEXT LEARNING: This is one of the most fascinating properties
//    of transformers. Without any gradient updates, the model learns to
//    perform a task just from examples in the prompt. Mechanistically,
//    the attention layers learn to "copy" the pattern of reasoning from
//    the examples to the new problem. This is possible because
//    attention computes similarity between the current token and ALL
//    previous tokens — including the exemplars.
//
// The tradeoff: few-shot uses more input tokens (= more cost) because
// the examples are included in every request. Our benchmark will measure
// whether the accuracy gain justifies the cost.

// exemplar is a question + worked solution used as a few-shot example.
type exemplar struct {
	question string
	solution string
}

// These exemplars are hand-crafted to demonstrate clear, step-by-step
// arithmetic reasoning. They're deliberately simple so the model
// learns the FORMAT of reasoning, not the CONTENT.
//
// In practice, selecting good exemplars is an art:
// - Too easy → model doesn't learn enough about multi-step reasoning
// - Too hard → model gets confused by the examples themselves
// - Too similar to test questions → inflated benchmark scores
// - Too different → model doesn't transfer the pattern
var mathExemplars = []exemplar{
	{
		question: "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
		solution: `1. We start with 15 trees in the grove.
2. After planting, there will be 21 trees.
3. To find how many were planted: 21 - 15 = 6 trees.
#### 6`,
	},
	{
		question: "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
		solution: `1. There are originally 3 cars.
2. 2 more cars arrive.
3. Total cars: 3 + 2 = 5 cars.
#### 5`,
	},
	{
		question: "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
		solution: `1. Leah had 32 chocolates.
2. Her sister had 42 chocolates.
3. Total chocolates: 32 + 42 = 74.
4. They ate 35 chocolates.
5. Remaining: 74 - 35 = 39 chocolates.
#### 39`,
	},
	{
		question: "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
		solution: `1. Jason started with 20 lollipops.
2. After giving some to Denny, Jason has 12.
3. Lollipops given to Denny: 20 - 12 = 8 lollipops.
#### 8`,
	},
}

func FewShotCoT(ctx context.Context, llm client.LLM, question string) (*Trace, error) {
	system := "You are a helpful math assistant. Solve math problems step by step, " +
		"showing your reasoning clearly. Give your final numerical answer on the last line " +
		"in the format: #### <number>"

	// Build the message sequence: alternating user/assistant turns with exemplars.
	//
	// This is how few-shot prompting works with chat models:
	//   User: "Question: <exemplar 1>"
	//   Assistant: "<exemplar 1 solution>"
	//   User: "Question: <exemplar 2>"
	//   Assistant: "<exemplar 2 solution>"
	//   ...
	//   User: "Question: <actual question>"
	//
	// The model sees these as a "conversation" where the assistant has been
	// solving math problems with detailed reasoning. It continues the pattern.
	var messages []client.Message
	for _, ex := range mathExemplars {
		messages = append(messages,
			client.UserMsg(fmt.Sprintf("Question: %s", ex.question)),
			client.AssistantMsg(ex.solution),
		)
	}

	// Now add the actual question
	messages = append(messages,
		client.UserMsg(fmt.Sprintf("Question: %s", question)),
	)

	temp := 0.0
	return Execute(ctx, llm, question, messages, system, 1024, &temp, StrategyFewShotCoT)
}
