package llm

import (
	"context"
	"io"

	"github.com/bdqfork/go-llama.cpp/pkg/model"
)

// LLM provides language related operations, based on model
type LLM interface {
	io.Closer

	// GetEmbedding returns embedding vector of inputs
	GetEmbedding(ctx context.Context, inputs []string) (*Embedding, error)

	Completion(ctx context.Context, prompt string, stops []string, suffix string, maxTokens int, echo bool, opts ...model.SampleOption) (*Completion, error)
	CompletionStream(ctx context.Context, prompt string, stops []string, maxTokens int, outChan chan *CompletionChunk, opts ...model.SampleOption) error
	// ChatCompletion returns completion for input
	ChatCompletion(ctx context.Context, id string, input []ChatCompletionMessage, stops []string, maxTokens int, opts ...model.SampleOption) (*ChatCompletion, error)
	// ChatCompletionStream returns completion for input via stream
	ChatCompletionStream(ctx context.Context, id string, input []ChatCompletionMessage, maxTokens int, stops []string, outChan chan *ChatCompletionChunk, opts ...model.SampleOption) error
}
