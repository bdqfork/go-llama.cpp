package llm

import (
	"context"
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/klog/v2"

	"github.com/bdqfork/go-llama.cpp/pkg/binding"
	"github.com/bdqfork/go-llama.cpp/pkg/model"
)

func (l *llm) Completion(ctx context.Context, prompt string, stops []string, suffix string, maxTokens int, echo bool, opts ...model.SampleOption) (*Completion, error) {
	l.locker.Lock()
	defer l.locker.Unlock()

	l.Reset()
	if l.modelConfig.Verbose {
		defer l.Model.PrintTimings()
	}

	if len(l.modelConfig.Stops) > 0 {
		stops = append(stops, l.modelConfig.Stops...)
	}

	promptTemplate := l.templates["completion"]
	input := prompt

	renderedPrompt, err := render(promptTemplate, input)
	if err != nil {
		klog.Errorf("failed to render prompt template: %v, input: %v, err: %v", promptTemplate, input, err)
		return nil, err
	}

	klog.V(3).Infof("rendered prompt: %v", renderedPrompt)

	promptTokens, err := l.Tokenize(renderedPrompt, true)
	if err != nil {
		klog.Errorf("failed to tokenize, prompt: %+v, err: %v", renderedPrompt, err)
		return nil, err
	}

	promptTokenNum := len(promptTokens)

	if promptTokenNum+maxTokens > *l.modelConfig.Context {
		return nil, fmt.Errorf("tokens exceeds max context size: %d", *l.modelConfig.Context)
	}

	tokenGenerator := l.generate(promptTokens, opts...)

	builder := strings.Builder{}

	finish := false
	finishReason := ""
	outTokenNum := 0

	for !finish {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			token, err := tokenGenerator()
			if err != nil {
				klog.Errorf("failed to generate, token: %v, err: %v", token, err)
				return nil, err
			}
			outTokenNum++
			out := l.Detokenize([]binding.Token{token})
			klog.V(4).Infof("got token %d, out: %v", token, out)

			builder.WriteString(out)
			if token == binding.TokenEos() {
				finish = true
				finishReason = "stop"
			} else if outTokenNum >= maxTokens || promptTokenNum+outTokenNum >= l.Model.ContextSize() {
				finish = true
				finishReason = "length"
			} else if ok, _ := matchAnyStop(builder.String(), stops); ok {
				finish = true
				finishReason = "stop"
			}
		}
	}

	text := builder.String()

	if echo {
		text = input + text
	}

	if suffix != "" {
		text += suffix
	}

	choice := CompletionChoice{Index: 0, Text: text, FinishReason: finishReason}

	completion := &Completion{}
	completion.Created = int(time.Now().Unix())
	completion.ID = string(uuid.NewUUID())
	completion.Model = l.modelConfig.Name
	completion.Object = "text_completion"
	completion.Choices = []CompletionChoice{choice}
	completion.Usage.PromptTokens = promptTokenNum
	completion.Usage.CompletionTokens = outTokenNum
	completion.Usage.TotalTokens = promptTokenNum + outTokenNum
	return completion, nil
}

func (l *llm) CompletionStream(ctx context.Context, prompt string, stops []string, maxTokens int, outChan chan *CompletionChunk, opts ...model.SampleOption) error {
	l.locker.Lock()
	defer l.locker.Unlock()

	defer close(outChan)

	if len(l.modelConfig.Stops) > 0 {
		stops = append(stops, l.modelConfig.Stops...)
	}

	promptTemplate := l.templates["completion"]
	input := prompt
	renderedPrompt, err := render(promptTemplate, input)
	if err != nil {
		klog.Errorf("failed to render prompt template: %v, input: %v, err: %v", promptTemplate, input, err)
		return err
	}

	klog.V(3).Infof("rendered prompt: %v", renderedPrompt)

	tokens, err := l.Tokenize(renderedPrompt, true)
	if err != nil {
		klog.Errorf("failed to tokenize, prompt: %+v, err: %v", renderedPrompt, err)
		return err
	}

	promptTokenNum := len(tokens)

	if promptTokenNum+maxTokens > *l.modelConfig.Context {
		return fmt.Errorf("tokens exceeds max context size: %d", *l.modelConfig.Context)
	}

	tokenGenerator := l.generate(tokens, opts...)

	l.Reset()
	if l.modelConfig.Verbose {
		defer l.Model.PrintTimings()
	}

	builder := strings.Builder{}

	id := string(uuid.NewUUID())
	created := int(time.Now().Unix())

	finish := false
	outTokenNum := 0
	for !finish {
		token, err := tokenGenerator()
		if err != nil {
			klog.Errorf("failed to generate, token: %v, err: %v", token, err)
			return err
		}
		outTokenNum++
		out := l.Detokenize([]binding.Token{token})

		klog.V(4).Infof("got token %d, out: %v", token, out)

		builder.WriteString(out)

		finishReason := ""
		if token == binding.TokenEos() {
			finish = true
			finishReason = "stop"
		} else if outTokenNum >= maxTokens || promptTokenNum+outTokenNum >= l.Model.ContextSize() {
			finish = true
			finishReason = "length"
		} else if ok, _ := matchAnyStop(builder.String(), stops); ok {
			finish = true
			finishReason = "stop"
		}

		choice := CompletionChoice{Index: 0, Text: out, FinishReason: finishReason}

		chunck := &CompletionChunk{}
		chunck.ID = id
		chunck.Created = created
		chunck.Model = l.modelConfig.Name
		chunck.Object = "text_completion"
		chunck.Choices = []CompletionChoice{choice}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case outChan <- chunck:
		}
	}
	return nil
}
