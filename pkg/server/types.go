package server

import (
	"errors"

	"github.com/bdqfork/go-llama.cpp/pkg/llm"
)

var (
	errUnableToLoadModel = errors.New("unable to load model")
	errInternalAppError  = errors.New("internal application error")
	errProcessingFailed  = errors.New("processing failed")
)

// Model ...
type Model struct {
	ID         string   `json:"id"`
	Object     string   `json:"object"`
	OwnedBy    string   `json:"owned_by"`
	Permission []string `json:"permission"`
}

// EmbeddingRequest ...
type EmbeddingRequest struct {
	Model string `json:"model" bind:"required"`
	Input any    `json:"input" bind:"required"`
	User  string `json:"user"`
}

// CompletionRequest ...
type CompletionRequest struct {
	Model            string          `json:"model" bind:"required"`
	Prompt           any             `json:"prompt"`
	Suffix           string          `json:"suffix"`
	MaxTokens        int             `json:"max_tokens"`
	Temperature      float32         `json:"temperature"`
	TopP             float32         `json:"top_p"`
	N                int             `json:"n"`
	Stream           bool            `json:"stream"`
	Logprobs         int             `json:"log_probs"`
	Echo             bool            `json:"echo"`
	Stop             any             `json:"stop"`
	PresencePenalty  float32         `json:"presence_penalty"`
	FrequencyPenalty float32         `json:"frequency_penalty"`
	BestOf           int             `json:"best_of"`
	LogitBias        map[int]float32 `json:"logit_bias"`
	User             string          `json:"user"`
}

// ChatCompletionRequest ...
type ChatCompletionRequest struct {
	Model            string                      `json:"model" bind:"required"`
	Messages         []llm.ChatCompletionMessage `json:"messages" bind:"required"`
	Temperature      float32                     `json:"temperature"`
	TopP             float32                     `json:"top_p"`
	N                int                         `json:"n"`
	Stream           bool                        `json:"stream"`
	Stop             any                         `json:"stop"`
	MaxTokens        int                         `json:"max_tokens"`
	PresencePenalty  float32                     `json:"presence_penalty"`
	FrequencyPenalty float32                     `json:"frequency_penalty"`
	LogitBias        map[int]float32             `json:"logit_bias"`
	User             string                      `json:"user"`
}
