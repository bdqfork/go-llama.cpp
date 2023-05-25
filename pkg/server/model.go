package server

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"k8s.io/klog/v2"
)

func (s *Server) listModels(ctx *gin.Context) {
	startTime := time.Now()
	klog.V(3).Infof("received list models request")
	defer func() {
		klog.V(3).Infof("finished list models request, took: %v", time.Since(startTime))
	}()

	models := make([]Model, 0)
	for _, model := range s.ctx.Config.ModelConfigs {
		models = append(models, Model{
			ID:         model.Name,
			Object:     "model",
			OwnedBy:    model.Owner,
			Permission: model.Permissions,
		})
	}
	rsp := map[string]interface{}{}
	rsp["data"] = models
	rsp["object"] = "list"
	ctx.JSON(http.StatusOK, rsp)
}

func (s *Server) retreiveModel(ctx *gin.Context) {
	modelID := ctx.Param("model")

	startTime := time.Now()
	klog.V(3).Infof("received retreive models request, model: %s", modelID)
	defer func() {
		klog.V(3).Infof("finished retreive models request, model: %s, took: %v", modelID, time.Since(startTime))
	}()

	model := s.ctx.Config.ModelConfigs[modelID]
	ctx.JSON(http.StatusOK, Model{
		ID:         model.Name,
		Object:     "model",
		OwnedBy:    model.Owner,
		Permission: model.Permissions,
	})
}
