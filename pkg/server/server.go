package server

import (
	"fmt"

	"github.com/gin-gonic/gin"

	"github.com/bdqfork/go-llama.cpp/pkg/context"
)

// Server ...
type Server struct {
	ctx    *context.Context
	engine *gin.Engine
}

// New return a new Server instance
func New(ctx *context.Context) *Server {
	s := &Server{ctx: ctx}

	r := gin.Default()
	s.engine = r

	v1 := r.Group("/v1")
	v1.GET("/models", s.listModels)
	v1.GET("/models/:model", s.retreiveModel)
	v1.POST("/embeddings", s.embedding)
	v1.POST("/completions", s.completion)
	v1.POST("/chat/completions", s.chatCompletion)
	return s

}

// Run run the server
func (s *Server) Run() {
	address := fmt.Sprintf("%s:%d", s.ctx.Config.Host, s.ctx.Config.Port)
	s.engine.Run(address)
}
