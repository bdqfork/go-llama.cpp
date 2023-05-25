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

func (l *llm) ChatCompletion(ctx context.Context, id string, input []ChatCompletionMessage, stops []string, maxTokens int, opts ...model.SampleOption) (*ChatCompletion, error) {
	l.locker.Lock()
	defer l.locker.Unlock()

	l.Reset()

	if l.modelConfig.Verbose {
		defer l.Model.PrintTimings()
	}

	if len(l.modelConfig.Stops) > 0 {
		stops = append(stops, l.modelConfig.Stops...)
	}

	promptTemplate := l.templates["chat"]

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

	if promptTokenNum >= *l.modelConfig.Context {
		return nil, fmt.Errorf("tokens exceeds max context size: %d", *l.modelConfig.Context)
	}

	sessionFilepath := l.generateSessionFilepath(id)

	matchNum := 0
	if l.modelConfig.Session.Enable {
		matchNum, err = l.Model.LoadSession(sessionFilepath, promptTokens)
		if err != nil {
			klog.Errorf("failed to load session, err: %v", err)
			return nil, err
		}
		klog.V(3).Infof("session %s state found, match token num: %d", id, matchNum)
		promptTokens = promptTokens[matchNum:]
		klog.V(3).Infof("current prompt tokens num: %d", len(promptTokens))
	}

	tokenGenerator := l.generate(promptTokens, opts...)

	outTokenNum := 0

	builder := strings.Builder{}

	finish := false
	finishReason := ""

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

			builder.Write([]byte(out))

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

	message := ChatCompletionMessage{
		Role:    AssistantRole,
		Content: text,
	}

	choice := ChatCompletionChoice{Index: 0, Message: message, FinishReason: finishReason}

	completion := &ChatCompletion{}
	completion.Created = int(time.Now().Unix())
	completion.ID = string(uuid.NewUUID())
	completion.Model = l.modelConfig.Name
	completion.Object = "chat.completion"
	completion.Choices = []ChatCompletionChoice{choice}
	completion.Usage.PromptTokens = promptTokenNum
	completion.Usage.CompletionTokens = outTokenNum
	completion.Usage.TotalTokens = promptTokenNum + outTokenNum

	similar := float32(float32(matchNum) / float32(outTokenNum+promptTokenNum))
	needSave := similar < l.modelConfig.Session.Threshold
	if needSave {
		err := l.Model.SaveSession(sessionFilepath)
		if err != nil {
			klog.Errorf("failed to save session, err: %v", err)
			return nil, err
		}
	}

	return completion, nil
}

func (l *llm) ChatCompletionStream(ctx context.Context, id string, input []ChatCompletionMessage, maxTokens int, stops []string, outChan chan *ChatCompletionChunk, opts ...model.SampleOption) error {
	l.locker.Lock()
	defer l.locker.Unlock()

	defer close(outChan)

	if len(l.modelConfig.Stops) > 0 {
		stops = append(stops, l.modelConfig.Stops...)
	}

	promptTemplate := l.templates["chat"]

	renderedPrompt, err := render(promptTemplate, input)
	if err != nil {
		klog.Errorf("failed to render prompt template: %v, input: %v, err: %v", promptTemplate, input, err)
		return err
	}

	klog.V(3).Infof("rendered prompt: %v", renderedPrompt)

	promptTokens, err := l.Tokenize(renderedPrompt, true)
	if err != nil {
		klog.Errorf("failed to tokenize, prompt: %+v, err: %v", renderedPrompt, err)
		return err
	}

	promptTokenNum := len(promptTokens)

	if promptTokenNum >= *l.modelConfig.Context {
		return fmt.Errorf("tokens exceeds max context size: %d", *l.modelConfig.Context)
	}

	l.Reset()
	if l.modelConfig.Verbose {
		defer l.Model.PrintTimings()
	}

	sessionFilepath := l.generateSessionFilepath(id)

	matchNum := 0
	if l.modelConfig.Session.Enable {
		matchNum, err = l.Model.LoadSession(sessionFilepath, promptTokens)
		if err != nil {
			klog.Errorf("failed to load session, err: %v", err)
			return err
		}
		klog.V(3).Infof("session %s state found, match token num: %d", id, matchNum)
		promptTokens = promptTokens[matchNum:]
		klog.V(3).Infof("current prompt tokens num: %d", len(promptTokens))
	}

	uid := string(uuid.NewUUID())
	created := int(time.Now().Unix())

	tokenGenerator := l.generate(promptTokens, opts...)

	outTokenNum := 0

	builder := strings.Builder{}
	finish := false
	for !finish {
		token, err := tokenGenerator()
		if err != nil {
			klog.Errorf("failed to generate, token: %v, err: %v", token, err)
			return err
		}

		outTokenNum++
		out := l.Detokenize([]binding.Token{token})
		klog.V(4).Infof("got token %d, out: %v", token, out)
		builder.Write([]byte(out))

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

		choice := ChatCompletionChunkChoice{Index: 0, Delta: ChatCompletionChunkDelta{
			Role:    AssistantRole,
			Content: out,
		}, FinishReason: finishReason}

		chunck := &ChatCompletionChunk{}
		chunck.ID = uid
		chunck.Created = created
		chunck.Model = l.modelConfig.Name
		chunck.Object = "chat.completion"
		chunck.Choices = []ChatCompletionChunkChoice{choice}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case outChan <- chunck:
		}
	}

	similar := float32(float32(matchNum) / float32(outTokenNum+promptTokenNum))
	needSave := similar < l.modelConfig.Session.Threshold
	if needSave {
		err := l.Model.SaveSession(sessionFilepath)
		if err != nil {
			klog.Errorf("failed to save session, err: %v", err)
			return err
		}
	}
	return nil
}
