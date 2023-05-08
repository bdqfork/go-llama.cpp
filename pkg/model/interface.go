package model

import (
	"io"

	"github.com/bdqfork/go-llama.cpp/pkg/binding"
)

// Model defines basic model operations
type Model interface {
	io.Closer
	// Tokenize converts text to token array
	Tokenize(text string, addBos bool) ([]binding.Token, error)
	// Detokenize converts tokens to string
	Detokenize(tokens []binding.Token) string
	// Reset model context
	Reset()
	// Eval prompt tokens
	Eval(tokens []binding.Token) error
	// Sample a token
	Sample(options ...SampleOption) binding.Token
	// GetEmbedding returns current context embeddings
	GetEmbedding() ([]float32, error)
	// ContextSize returns context size
	ContextSize() int
	// Logits returns current context logits
	Logits() [][]float32
	// SaveSession stores all context into file
	SaveSession(filepath string) error
	// LoadSession loads all context from file
	LoadSession(filepath string, tokens []binding.Token) (int, error)
	// PrintTimings of eval
	PrintTimings()
	// ResetTimings of eval
	ResetTimings()
}
