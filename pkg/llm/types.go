package llm

const (
	// SystemRole system
	SystemRole = "system"
	// UserRole user
	UserRole = "user"
	// AssistantRole assistant
	AssistantRole = "assistant"
)

// EmbeddingUsage is token usage of embedding
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// EmbeddingData is embedding data
type EmbeddingData struct {
	Index     int       `json:"index"`
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
}

// Embedding is embedding result
type Embedding struct {
	Object string          `json:"object"`
	Model  string          `json:"model"`
	Data   []EmbeddingData `json:"data"`
	Usage  EmbeddingUsage  `json:"usage"`
}

// CompletionLogprobs is completion log probs for choice
type CompletionLogprobs struct {
	TextOffset    []int                `json:"text_offset"`
	TokenLogprobs []float32            `json:"token_logprobs"`
	Tokens        []string             `json:"tokens"`
	TopLogprobs   []map[string]float32 `json:"top_logprobs"`
}

// CompletionChoice is completion choice
type CompletionChoice struct {
	Text         string             `json:"text"`
	Index        int                `json:"index"`
	Logprobs     CompletionLogprobs `json:"logprobs"`
	FinishReason string             `json:"finish_reason"`
}

// CompletionUsage is token usage
type CompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CompletionChunk is chunk for stream
type CompletionChunk struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int                `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
}

// Completion is completion result
type Completion struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int                `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   CompletionUsage    `json:"usage"`
}

// ChatCompletionMessage is chat message
type ChatCompletionMessage struct {
	Role    string `json:"role" bind:"required"`
	Content string `json:"content" bind:"required"`
	User    string `json:"user"`
}

// ChatCompletionChoice is a choice of chat result
type ChatCompletionChoice struct {
	Index        int                   `json:"index"`
	Message      ChatCompletionMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

// ChatCompletion is chat result
type ChatCompletion struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int                    `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   CompletionUsage        `json:"usage"`
}

// ChatCompletionChunkDelta used for stream
type ChatCompletionChunkDelta struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionChunkChoice is a choice of chat result for stream
type ChatCompletionChunkChoice struct {
	Index        int                      `json:"index"`
	Delta        ChatCompletionChunkDelta `json:"delta"`
	FinishReason string                   `json:"finish_reason"`
}

// ChatCompletionChunk is chat result chunk for stream
type ChatCompletionChunk struct {
	ID      string                      `json:"id"`
	Model   string                      `json:"model"`
	Object  string                      `json:"object"`
	Created int                         `json:"created"`
	Choices []ChatCompletionChunkChoice `json:"choices"`
}
