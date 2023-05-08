package config

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
	"k8s.io/klog/v2"
)

// Config include all runtime configuration
type Config struct {
	Debug     bool
	Host      string
	Port      int
	ModelPath string

	EnableSession bool
	SessionDir    string

	ModelConfigs map[string]ModelConfig
}

// ModelConfig is config for model
type ModelConfig struct {
	Name        string   `yaml:"name"`
	Owner       string   `yaml:"owner"`
	Permissions []string `yaml:"permissions"`

	Path           string  `yaml:"path"`
	Context        *int    `yaml:"context"`
	Seed           *int    `yaml:"seed"`
	F16KV          *bool   `yaml:"f16kv"`
	LogitsAll      bool    `yaml:"logitsAll"`
	VocabOnly      bool    `yaml:"vocabOnly"`
	UseMMap        *bool   `yaml:"useMMap"`
	UseMLock       *bool   `yaml:"useMLock"`
	Embedding      bool    `yaml:"embedding"`
	Threads        *int    `yaml:"threads"`
	Batch          *int    `yaml:"batch"`
	LastNTokenSize *int    `yaml:"lastNTokenSize"`
	LoraBase       *string `yaml:"loraBase"`
	LoraPath       *string `yaml:"loraPath"`
	Verbose        bool    `yaml:"verbose"`
	GPULayers      int     `yaml:"gpuLayers"`

	PromptTemplates map[string]string `yaml:"promptTemplates"`
	Roles           map[string]string `yaml:"roles"`

	Session SessionConfig `yaml:"session"`
}

// SessionConfig is config for session
type SessionConfig struct {
	Enable    bool    `yaml:"enable"`
	Path      string  `yaml:"path"`
	Threshold float32 `yaml:"threshold"`
}

// New returns a Config instance
func New() *Config {
	return &Config{
		ModelConfigs: map[string]ModelConfig{},
	}
}

// Init config flags
func (c *Config) Init(flag *flag.FlagSet) {
	flag.BoolVar(&c.Debug, "debug", true, "debug mode")
	flag.StringVar(&c.ModelPath, "model-path", "models", "where you store models")
	flag.StringVar(&c.Host, "host", "0.0.0.0", "server host address to listen")
	flag.IntVar(&c.Port, "port", 8000, "server port to listen")
}

// LoadModelConfigs from model path
func (c *Config) LoadModelConfigs() {
	if c.ModelPath == "" {
		klog.Fatalf("invalid empty model path")
	}

	files, err := os.ReadDir(c.ModelPath)
	if err != nil {
		klog.Fatalf("failed to read model dir, err: %v", err)
	}

	modelConfigs := map[string]ModelConfig{}
	for _, file := range files {
		if file.IsDir() {
			continue
		}
		if strings.HasSuffix(file.Name(), ".yaml") {
			data, err := os.ReadFile(fmt.Sprintf("%s/%s", c.ModelPath, file.Name()))
			if err != nil {
				klog.Fatalf("failed to read model file: %s, err: %v", file.Name(), err)
			}
			modelConfig := &ModelConfig{}
			err = yaml.Unmarshal(data, modelConfig)
			if err != nil {
				klog.Fatalf("failed to unmarshal data: %s, err: %v", string(data), err)
			}

			klog.Infof("loaded model config: %+v", modelConfig)

			modelConfig.Name = strings.TrimSpace(modelConfig.Name)
			if modelConfig.Name == "" {
				klog.Fatal("invalid empty model name")
			}
			modelConfigs[modelConfig.Name] = *modelConfig
		}
	}
	c.ModelConfigs = modelConfigs
}

// Print flags with its value
func (c *Config) Print() {
	flag.VisitAll(func(f *flag.Flag) {
		fmt.Printf("%s=%v\n", f.Name, f.Value)
	})
}
