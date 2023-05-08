package llm

import (
	"fmt"
	"os"
	"sync"
	"text/template"

	"k8s.io/klog/v2"

	"github.com/bdqfork/go-llama.cpp/pkg/binding"
	"github.com/bdqfork/go-llama.cpp/pkg/config"
	"github.com/bdqfork/go-llama.cpp/pkg/model"
)

type llm struct {
	model.Model
	locker sync.Mutex

	modelConfig *config.ModelConfig
	templates   map[string]*template.Template
}

// New returns a new LLM instance
func New(model model.Model, modelConfig *config.ModelConfig) LLM {
	templates := make(map[string]*template.Template)
	for k, v := range modelConfig.PromptTemplates {
		data, err := os.ReadFile(v)
		if err != nil {
			klog.Errorf("failed to read prompt template: %v, err: %v", v, err)
			panic(err)
		}
		templates[k] = template.Must(template.New(k).Parse(string(data)))
	}
	return &llm{Model: model, modelConfig: modelConfig, templates: templates}
}

func (l *llm) Close() error {
	return l.Model.Close()
}

func (l *llm) generate(tokens []binding.Token, opts ...model.SampleOption) func() (binding.Token, error) {
	next := func() (binding.Token, error) {
		err := l.Eval(tokens)
		if err != nil {
			return 0, err
		}
		token := l.Sample(opts...)
		tokens = []binding.Token{token}
		return token, nil
	}
	return next
}

func (l *llm) generateSessionFilepath(id string) string {
	filepath := fmt.Sprintf("%s/%s-%s.dat", l.modelConfig.Session.Path, l.modelConfig.Name, id)
	return filepath
}
