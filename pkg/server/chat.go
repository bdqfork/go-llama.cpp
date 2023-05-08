package server

import (
	"context"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"k8s.io/klog/v2"

	"github.com/bdqfork/go-llama.cpp/pkg/binding"
	"github.com/bdqfork/go-llama.cpp/pkg/llm"
	"github.com/bdqfork/go-llama.cpp/pkg/model"
	"github.com/bdqfork/go-llama.cpp/pkg/util"
)

func (s *Server) chatCompletion(ctx *gin.Context) {
	req := &ChatCompletionRequest{MaxTokens: 16, Temperature: 1, TopP: 1, N: 1}
	if err := ctx.ShouldBind(req); err != nil {
		klog.Errorf("failed to bind completion request, err: %v", err)
		ctx.JSON(http.StatusInternalServerError, err.Error())
		return
	}

	klog.V(3).Infof("received chat completion request: %+v", req)
	startTime := time.Now()
	defer func() {
		klog.V(3).Infof("finished chat completion request: %+v, took: %v", req, time.Since(startTime))
	}()

	stops, err := util.ParseWords(req.Stop)
	if err != nil {
		ctx.JSON(http.StatusBadRequest, err.Error())
		return
	}

	l, err := s.ctx.LLM(req.Model)
	if err != nil {
		klog.Errorf("failed to load model, err: %v", err)
		ctx.JSON(http.StatusInternalServerError, errUnableToLoadModel)
		return
	}

	options := make([]model.SampleOption, 0)
	options = append(options, model.WithTopP(req.TopP), model.WithTemp(req.Temperature),
		model.WithFrequenctPenalty(req.FrequencyPenalty), model.WithPresencePenalty(req.PresencePenalty))

	if req.LogitBias != nil {
		logitBias := map[binding.Token]float32{}
		for k, v := range req.LogitBias {
			token := binding.Token(k)
			logitBias[token] = v
		}
		options = append(options, model.WithLogisBiasK(logitBias))
	}

	completionContext, cancel := context.WithCancel(context.Background())
	go func() {
		<-ctx.Writer.CloseNotify()
		cancel()
	}()

	modelConfig := s.ctx.Config.ModelConfigs[req.Model]
	input := make([]llm.ChatCompletionMessage, 0)
	for _, message := range req.Messages {
		role := message.Role
		if val, ok := modelConfig.Roles[message.Role]; ok {
			role = val
		}
		input = append(input, llm.ChatCompletionMessage{
			Role:    role,
			Content: message.Content,
			User:    message.User,
		})
	}

	if !req.Stream {
		completion, err := l.ChatCompletion(completionContext, req.User, input, stops, req.MaxTokens, options...)
		if err != nil {
			klog.Errorf("failed to chat completion: %v", err)
			ctx.JSON(http.StatusInternalServerError, errProcessingFailed)
			return
		}
		ctx.JSON(http.StatusOK, completion)
		return
	}

	chunkChan := make(chan *llm.ChatCompletionChunk)

	go func() {
		err = l.ChatCompletionStream(completionContext, req.User, input, req.MaxTokens, stops, chunkChan, options...)
		if err != nil {
			klog.Errorf("failed to chat completion stream: %v", err)
			ctx.JSON(http.StatusInternalServerError, errInternalAppError)
		}
	}()

	ctx.Stream(func(w io.Writer) bool {
		chunk, ok := <-chunkChan
		if !ok {
			ctx.SSEvent("message", "[DONE]")
			return false
		}
		ctx.SSEvent("message", chunk)
		return true
	})
}
