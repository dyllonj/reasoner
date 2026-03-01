// Anthropic Claude API client — built from scratch using net/http.
//
// We're NOT using an SDK. You'll see every HTTP header, every JSON field,
// every error case. This is how API clients actually work under the hood.
//
// The Claude Messages API:
//   POST https://api.anthropic.com/v1/messages
//
// Required headers:
//   x-api-key: <your key>          — authentication
//   anthropic-version: 2023-06-01  — API version (prevents breaking changes)
//   content-type: application/json — we're sending JSON
//
// The response contains "content blocks" — usually one text block.
package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

const (
	apiURL     = "https://api.anthropic.com/v1/messages"
	apiVersion = "2023-06-01"
)

// Anthropic implements the LLM interface for the Claude API.
//
// http.Client is safe for concurrent use and internally pools connections.
// Creating one client and reusing it is much more efficient than creating
// a new one per request (avoids TCP handshake + TLS negotiation overhead).
type Anthropic struct {
	httpClient *http.Client
	apiKey     string
	model      string
}

// NewAnthropic creates a new Claude API client.
// model is the model ID, e.g. "claude-sonnet-4-20250514".
func NewAnthropic(apiKey, model string) *Anthropic {
	return &Anthropic{
		httpClient: &http.Client{},
		apiKey:     apiKey,
		model:      model,
	}
}

// NewAnthropicFromEnv creates a client using the ANTHROPIC_API_KEY env var.
func NewAnthropicFromEnv(model string) (*Anthropic, error) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("ANTHROPIC_API_KEY environment variable not set")
	}
	return NewAnthropic(key, model), nil
}

// --- API request/response types ---
// These mirror the Claude API's JSON schema exactly.
// We define them as Go structs so encoding/json can marshal/unmarshal them.
// Note: struct tags like `json:"model"` control the JSON field names.
// Go uses PascalCase for exported fields, but APIs use snake_case.

type apiRequest struct {
	Model       string       `json:"model"`
	Messages    []apiMessage `json:"messages"`
	MaxTokens   int          `json:"max_tokens"`
	System      string       `json:"system,omitempty"`      // omitempty: skip if empty
	Temperature *float64     `json:"temperature,omitempty"` // pointer so we can omit nil
}

type apiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// apiResponse is the success response from Claude.
//
// `Content` is an array because Claude can return multiple content blocks
// (e.g., text + tool calls). For our purposes, we only care about text blocks.
//
// `Usage` tells us how many tokens were consumed. This matters because:
//   - Input tokens: cost scales with prompt length
//   - Output tokens: cost scales with response length AND determines compute budget
//   - CoT generates more output tokens — we need to track cost/benefit
type apiResponse struct {
	Content []contentBlock `json:"content"`
	Usage   usage          `json:"usage"`
}

type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// apiError is the error response from Claude.
type apiError struct {
	Error apiErrorDetail `json:"error"`
}

type apiErrorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// Complete sends a completion request to the Claude API.
//
// The flow:
//  1. Marshal the request to JSON
//  2. Create an HTTP POST request with auth headers
//  3. Send it, read the response
//  4. Check for errors (4xx, 5xx)
//  5. Unmarshal the JSON response
//  6. Extract text from content blocks
func (a *Anthropic) Complete(ctx context.Context, req Request) (*Response, error) {
	// 1. Build the API request body
	body := apiRequest{
		Model:     a.model,
		MaxTokens: req.MaxTokens,
		System:    req.System,
		Temperature: req.Temperature,
	}
	for _, m := range req.Messages {
		body.Messages = append(body.Messages, apiMessage{
			Role:    m.Role,
			Content: m.Content,
		})
	}

	// Marshal to JSON bytes
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	// 2. Create the HTTP request.
	//
	// http.NewRequestWithContext attaches the context — if the caller cancels
	// the context (e.g., timeout or Ctrl+C), the HTTP request gets cancelled too.
	// This is Go's approach to cancellation: no async/await, just context propagation.
	httpReq, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	// Set required headers.
	// x-api-key: Anthropic's auth mechanism (not Bearer tokens like most APIs)
	// anthropic-version: pins the API behavior to a specific version
	// content-type: tells the server we're sending JSON
	httpReq.Header.Set("x-api-key", a.apiKey)
	httpReq.Header.Set("anthropic-version", apiVersion)
	httpReq.Header.Set("content-type", "application/json")

	// 3. Send the request
	httpResp, err := a.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("sending request: %w", err)
	}
	defer httpResp.Body.Close() // ALWAYS close the body to prevent connection leaks

	// Read the full response body
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response: %w", err)
	}

	// 4. Check for errors.
	// HTTP status codes:
	//   200 = success
	//   400 = bad request (malformed JSON, invalid params)
	//   401 = unauthorized (bad API key)
	//   429 = rate limited (too many requests — back off and retry)
	//   500 = server error (Anthropic's problem, retry with backoff)
	//   529 = overloaded (too much traffic, retry later)
	if httpResp.StatusCode != http.StatusOK {
		var apiErr apiError
		if json.Unmarshal(respBody, &apiErr) == nil {
			return nil, fmt.Errorf("anthropic API error (%d %s): %s — %s",
				httpResp.StatusCode, apiErr.Error.Type, apiErr.Error.Message, apiErr.Error.Type)
		}
		return nil, fmt.Errorf("anthropic API error (%d): %s",
			httpResp.StatusCode, string(respBody))
	}

	// 5. Unmarshal the success response
	var apiResp apiResponse
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		return nil, fmt.Errorf("unmarshaling response: %w", err)
	}

	// 6. Extract text from content blocks.
	// The API can return multiple blocks (text, tool_use, etc).
	// We concatenate all text blocks.
	var text string
	for _, block := range apiResp.Content {
		if block.Type == "text" {
			text += block.Text
		}
	}

	return &Response{
		Content:      text,
		InputTokens:  apiResp.Usage.InputTokens,
		OutputTokens: apiResp.Usage.OutputTokens,
	}, nil
}
