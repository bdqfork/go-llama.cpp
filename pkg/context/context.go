package context

import (
	"sync"

	"k8s.io/klog/v2"

	"github.com/bdqfork/go-llama.cpp/pkg/config"
	"github.com/bdqfork/go-llama.cpp/pkg/llm"
	"github.com/bdqfork/go-llama.cpp/pkg/model"
)

// Context manages all runtime infomation, include configuration and model instances
type Context struct {
	Config *config.Config
	llms   map[string]llm.LLM
	locker sync.Mutex
}

// New return a context instance
func New(c *config.Config) *Context {
	ctx := &Context{Config: c}
	ctx.llms = make(map[string]llm.LLM)
	return ctx
}

// LLM return a LLM instance by model name
func (ctx *Context) LLM(name string) (llm.LLM, error) {
	ctx.locker.Lock()
	defer ctx.locker.Unlock()

	if val, ok := ctx.llms[name]; ok {
		return val, nil
	}

	modelConfig := ctx.Config.ModelConfigs[name]
	model, err := ctx.loadModel(modelConfig)
	if err != nil {
		return nil, err
	}
	l := llm.New(model, &modelConfig)
	ctx.llms[name] = l
	return l, nil
}

func (ctx *Context) loadModel(modelConfig config.ModelConfig) (model.Model, error) {
	modelOptions := make([]model.Option, 0)

	if modelConfig.Context != nil {
		modelOptions = append(modelOptions, model.WithCtxNum(*modelConfig.Context))
	}
	if modelConfig.Seed != nil {
		modelOptions = append(modelOptions, model.WithSeed(*modelConfig.Seed))
	}
	if modelConfig.F16KV != nil {
		modelOptions = append(modelOptions, model.WithF16KV(*modelConfig.F16KV))
	}

	modelOptions = append(modelOptions, model.WithLogitsAll(modelConfig.LogitsAll))
	modelOptions = append(modelOptions, model.WithVocabOnly(modelConfig.VocabOnly))

	if modelConfig.UseMMap != nil {
		modelOptions = append(modelOptions, model.WithUseMMap(*modelConfig.UseMMap))
	}
	if modelConfig.UseMLock != nil {
		modelOptions = append(modelOptions, model.WithUseMLock(*modelConfig.UseMLock))
	}

	modelOptions = append(modelOptions, model.WithEmbedding(modelConfig.Embedding))

	if modelConfig.Threads != nil {
		modelOptions = append(modelOptions, model.WithThreadNum(*modelConfig.Threads))
	}
	if modelConfig.Batch != nil {
		modelOptions = append(modelOptions, model.WithBatchNum(*modelConfig.Batch))
	}
	if modelConfig.LastNTokenSize != nil {
		modelOptions = append(modelOptions, model.WithLastNTokensSize(*modelConfig.LastNTokenSize))
	}

	if modelConfig.LoraBase != nil {
		modelOptions = append(modelOptions, model.WithLoraBase(*modelConfig.LoraBase))
	}
	if modelConfig.LoraPath != nil {
		modelOptions = append(modelOptions, model.WithLoraPath(*modelConfig.LoraPath))
	}

	if modelConfig.GPULayers != 0 {
		modelOptions = append(modelOptions, model.WithGPULayers(modelConfig.GPULayers))
	}

	modelOptions = append(modelOptions, model.WithVerbose(modelConfig.Verbose))

	model, err := model.New(modelConfig.Path, modelOptions...)
	if err != nil {
		klog.Errorf("failed to load model, config: %+v, err: %v", modelConfig, err)
		return nil, err
	}
	return model, nil
}

// Close all context, release all resources
func (ctx *Context) Close() error {
	for _, l := range ctx.llms {
		l.Close()
	}
	return nil
}
