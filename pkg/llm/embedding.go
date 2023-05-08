package llm

import (
	"context"
	"fmt"

	"k8s.io/klog/v2"
)

func (l *llm) GetEmbedding(ctx context.Context, inputs []string) (*Embedding, error) {
	l.locker.Lock()
	defer l.locker.Unlock()

	embeddingDatas := make([]EmbeddingData, 0)

	tokenNum := 0
	for i, input := range inputs {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			l.Model.Reset()

			tokens, err := l.Model.Tokenize(input, true)
			if err != nil {
				klog.Errorf("failed to tokenize, err: %v", err)
				return nil, err
			}

			tokenNum += len(tokens)
			if tokenNum > l.Model.ContextSize() {
				return nil, fmt.Errorf("input %s token exceeds context size limit %d", input, l.Model.ContextSize())
			}

			err = l.Model.Eval(tokens)
			if err != nil {
				klog.Errorf("failed to eval tokens, input: %s, err: %v", input, err)
				return nil, err
			}

			embedding, err := l.Model.GetEmbedding()
			if err != nil {
				return nil, err
			}
			embeddingDatas = append(embeddingDatas, EmbeddingData{
				Index:     i,
				Object:    "embedding",
				Embedding: embedding,
			})
		}
	}

	return &Embedding{
		Object: "list",
		Model:  l.modelConfig.Name,
		Data:   embeddingDatas,
		Usage: EmbeddingUsage{
			PromptTokens: tokenNum,
			TotalTokens:  tokenNum,
		},
	}, nil
}
