package model

import "github.com/bdqfork/go-llama.cpp/pkg/binding"

// Params for model
type Params struct {
	binding.Params
	threadNum       int
	batchNum        int
	lastNTokensSize int
	loraBase        string
	loraPath        string
	verbose         bool
	logtisAll       bool
	embedding       bool
}

// Option for model creation and runtime
type Option func(*Params)

// WithCtxNum set context size
func WithCtxNum(ctxNum int) Option {
	return func(p *Params) {
		p.SetCtxNum(int32(ctxNum))
	}
}

// WithSeed set sample seed
func WithSeed(seed int) Option {
	return func(p *Params) {
		p.SetSeed(int32(seed))
	}
}

// WithF16KV enables f16 kv
func WithF16KV(f16KV bool) Option {
	return func(p *Params) {
		p.SetF16KV(f16KV)
	}
}

// WithLogitsAll enables logis all
func WithLogitsAll(logitsAll bool) Option {
	return func(p *Params) {
		p.SetLogitsAll(logitsAll)
		p.logtisAll = logitsAll
	}
}

// WithVocabOnly enables vocab only mode
func WithVocabOnly(vocabOnly bool) Option {
	return func(p *Params) {
		p.SetVocabOnly(vocabOnly)
	}
}

// WithUseMMap enables use mmap
func WithUseMMap(useMMap bool) Option {
	return func(p *Params) {
		p.SetUseMMap(useMMap)
	}
}

// WithUseMLock enables use mlock
func WithUseMLock(useMLock bool) Option {
	return func(p *Params) {
		p.SetUseMLock(useMLock)
	}
}

// WithEmbedding enables embedding operations
func WithEmbedding(enable bool) Option {
	return func(p *Params) {
		p.SetEmbedding(enable)
		p.embedding = enable
	}
}

// WithThreadNum set thread used by eval
func WithThreadNum(threadNum int) Option {
	return func(p *Params) {
		p.threadNum = threadNum
	}
}

// WithBatchNum set batch at eval
func WithBatchNum(batchNum int) Option {
	return func(p *Params) {
		p.batchNum = batchNum
	}
}

// WithLastNTokensSize set last token size used by sample
func WithLastNTokensSize(lastNTokensSize int) Option {
	return func(p *Params) {
		p.lastNTokensSize = lastNTokensSize
	}
}

// WithLoraBase set lora base path
func WithLoraBase(loraBase string) Option {
	return func(p *Params) {
		p.loraBase = loraBase
	}
}

// WithLoraPath set lora path
func WithLoraPath(loraPath string) Option {
	return func(p *Params) {
		p.loraPath = loraPath
	}
}

// WithVerbose enables verbose
func WithVerbose(verbose bool) Option {
	return func(p *Params) {
		p.verbose = verbose
	}
}

// WithGPULayers set gpu layers
func WithGPULayers(num int) Option {
	return func(p *Params) {
		p.Params.SetGPULayers(int32(num))
	}
}

type sampleParams struct {
	logisBias        map[binding.Token]float32
	topK             int
	topP             float32
	tfsZ             float32
	typicalP         float32
	temp             float32
	repeatPenalty    float32
	repeatLastN      int
	frequenctPenalty float32
	presencePenalty  float32
	mirostat         int
	mirostatTau      float32
	mirostatEta      float32
	penalizeNL       bool
}

// SampleOption for sample operation
type SampleOption func(*sampleParams)

// WithLogisBiasK set logis bias
func WithLogisBiasK(logisBias map[binding.Token]float32) SampleOption {
	return func(sp *sampleParams) {
		sp.logisBias = logisBias
	}
}

// WithTopK set top k
func WithTopK(topK int) SampleOption {
	return func(sp *sampleParams) {
		sp.topK = topK
	}
}

// WithTopP set top p
func WithTopP(topP float32) SampleOption {
	return func(sp *sampleParams) {
		sp.topP = topP
	}
}

// WithTfsZ set tfs z
func WithTfsZ(tfsZ float32) SampleOption {
	return func(sp *sampleParams) {
		sp.tfsZ = tfsZ
	}
}

// WithTypicalP set typical p
func WithTypicalP(typicalP float32) SampleOption {
	return func(sp *sampleParams) {
		sp.typicalP = typicalP
	}
}

// WithTemp set temperature
func WithTemp(temp float32) SampleOption {
	return func(sp *sampleParams) {
		sp.temp = temp
	}
}

// WithRepeatPenalty set repeat penalty
func WithRepeatPenalty(repeatPenalty float32) SampleOption {
	return func(sp *sampleParams) {
		sp.repeatPenalty = repeatPenalty
	}
}

// WithRepeatLastN set repeat last n tokens
func WithRepeatLastN(repeatLastN int) SampleOption {
	return func(sp *sampleParams) {
		sp.repeatLastN = repeatLastN
	}
}

// WithFrequenctPenalty set frequenct penalty
func WithFrequenctPenalty(frequenctPenalty float32) SampleOption {
	return func(sp *sampleParams) {
		sp.frequenctPenalty = frequenctPenalty
	}
}

// WithPresencePenalty set presence penalty
func WithPresencePenalty(presencePenalty float32) SampleOption {
	return func(sp *sampleParams) {
		sp.presencePenalty = presencePenalty
	}
}

// WithMirostat set mirostat version
func WithMirostat(mirostat int) SampleOption {
	return func(sp *sampleParams) {
		sp.mirostat = mirostat
	}
}

// WithMirostatTau set mirostat tau
func WithMirostatTau(mirostatTau float32) SampleOption {
	return func(sp *sampleParams) {
		sp.mirostatTau = mirostatTau
	}
}

// WithMirostatEta set mirostat eta
func WithMirostatEta(mirostatEta float32) SampleOption {
	return func(sp *sampleParams) {
		sp.mirostatEta = mirostatEta
	}
}

// WithPenalizeNL enables penalize nl
func WithPenalizeNL(penalizeNL bool) SampleOption {
	return func(sp *sampleParams) {
		sp.penalizeNL = penalizeNL
	}
}
