package server

import (
	"context"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"k8s.io/klog/v2"

	"github.com/bdqfork/go-llama.cpp/pkg/util"
)

func (s *Server) embedding(ctx *gin.Context) {
	req := &EmbeddingRequest{}
	if err := ctx.ShouldBind(req); err != nil {
		klog.Errorf("failed to bind embedding request, err: %v", err)
		ctx.JSON(http.StatusInternalServerError, err.Error())
		return
	}

	klog.V(3).Infof("received embedding request: %+v", req)
	startTime := time.Now()
	defer func() {
		klog.V(3).Infof("finished embedding request: %+v, took: %v", req, time.Since(startTime))
	}()

	inputs, err := util.ParseWords(req.Input)
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

	llmContext, cancel := context.WithCancel(context.Background())
	go func() {
		<-ctx.Writer.CloseNotify()
		cancel()
	}()

	embedding, err := l.GetEmbedding(llmContext, inputs)
	if err != nil {
		klog.Errorf("failed to get embedding, err: %v", err)
		ctx.JSON(http.StatusInternalServerError, errProcessingFailed)
		return
	}

	ctx.JSON(http.StatusOK, embedding)
}
