package model

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/bdqfork/go-llama.cpp/pkg/binding"
)

type model struct {
	modelPath string
	params    *Params

	lastNTokensData *TokenRingbuf
	tokensConsumed  int
	tokens          []binding.Token
	pastNum         int

	ctx *binding.Context
}

var _ Model = (*model)(nil)

// New returns a Model instance
func New(modelPath string, opts ...Option) (Model, error) {
	p := &Params{
		Params:          binding.ContextDefaultParams(),
		batchNum:        8,
		lastNTokensSize: 64,
		verbose:         true,
		threadNum:       runtime.NumCPU(),
	}
	for _, apply := range opts {
		apply(p)
	}

	l := &model{modelPath: modelPath}
	l.params = p

	l.lastNTokensData = NewTokenRingBuf(l.params.lastNTokensSize)
	l.tokens = make([]binding.Token, 0)

	binding.InitBackend()
	ctx, err := binding.InitFromFile(modelPath, p.Params)
	if err != nil {
		return nil, err
	}

	l.ctx = ctx

	if l.params.loraPath != "" {
		if err := l.ctx.ApplyLoraFromFile(l.params.loraPath, l.params.loraBase, int32(l.params.threadNum)); err != nil {
			return nil, err
		}
	}

	if l.params.verbose {
		fmt.Println(binding.PrintSystemInfo())
	}
	return l, nil
}

func (m *model) Close() error {
	if m.ctx != nil {
		m.ctx.Free()
	}
	m.ctx = nil
	return nil
}

func (m *model) Tokenize(text string, addBos bool) ([]binding.Token, error) {
	nCtx := m.ctx.CtxNum()
	tokens := make([]binding.Token, nCtx)
	result := m.ctx.Tokenize(text, tokens, nCtx, addBos)
	if result < 0 {
		return nil, fmt.Errorf("failed to tokenize: text=%s", text)
	}
	return tokens[:result], nil
}

func (m *model) Detokenize(tokens []binding.Token) string {
	builder := strings.Builder{}
	for _, token := range tokens {
		builder.WriteString(m.ctx.TokenToStr(token))
	}
	return builder.String()
}

func (m *model) Reset() {
	m.lastNTokensData = NewTokenRingBuf(m.params.lastNTokensSize)
	m.tokensConsumed = 0
	m.tokens = make([]binding.Token, 0)
	m.pastNum = 0
	m.ResetTimings()
}

func (m *model) Eval(tokens []binding.Token) error {
	nCtx := int(m.ctx.CtxNum())
	for i := 0; i < len(tokens); i += m.params.batchNum {
		limit := int(math.Min(float64(len(tokens)), float64(i+m.params.batchNum)))
		batch := tokens[i:limit]
		m.pastNum = int(math.Min(float64(nCtx-len(batch)), float64(m.tokensConsumed)))
		m.ctx.Eval(batch, int32(m.pastNum), int32(m.params.threadNum))
		m.tokens = append(m.tokens, batch...)
		m.lastNTokensData.Write(batch)
		m.tokensConsumed += len(batch)
	}
	return nil
}

func (m *model) ContextSize() int {
	return int(m.ctx.CtxNum())
}

func (m *model) Logits() [][]float32 {
	nVocab := m.ctx.VocabNum()
	cols := int(nVocab)
	rows := 1
	if m.params.logtisAll {
		rows = len(m.tokens)
	}
	logitsView := m.ctx.GetLogits(rows)
	logtis := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		logtis[i] = make([]float32, cols)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			logtis[i][j] = logitsView[i*cols+j]
		}
	}
	return logtis
}

func (m *model) GetEmbedding() ([]float32, error) {
	if !m.params.embedding {
		return nil, fmt.Errorf("embedding config is false, should be true to get embedding")
	}
	return m.ctx.GetEmbeddings(), nil
}

func (m *model) Sample(options ...SampleOption) binding.Token {
	op := &sampleParams{
		logisBias:        map[binding.Token]float32{},
		topK:             40,
		topP:             0.95,
		tfsZ:             1,
		typicalP:         1,
		temp:             0.8,
		repeatPenalty:    1.1,
		repeatLastN:      64,
		frequenctPenalty: 0,
		presencePenalty:  0,
		mirostat:         0,
		mirostatTau:      5,
		mirostatEta:      0.1,
	}

	for _, apply := range options {
		apply(op)
	}

	logitsAll := m.Logits()
	logits := logitsAll[len(logitsAll)-1]
	vocabNum := m.ctx.VocabNum()
	for k, v := range op.logisBias {
		logits[k] += v
	}

	candidatesP := binding.NewTokenDataArray(uint32(vocabNum), false)
	defer candidatesP.Free()
	candidates := candidatesP.Data()
	for i := 0; i < int(vocabNum); i++ {
		data := binding.TokenData{}
		data.SetID(int32(i))
		data.SetLogit(logits[i])
		data.SetP(0)
		candidates[i] = data
	}

	lastNTokensData := make([]binding.Token, m.params.lastNTokensSize)
	m.lastNTokensData.Read(lastNTokensData)

	lastNRepeat := int(math.Min(math.Min(float64(len(lastNTokensData)), float64(m.params.lastNTokensSize)), float64(m.ctx.CtxNum())))

	lastNtokens := lastNTokensData[len(lastNTokensData)-lastNRepeat:]
	m.ctx.SampleRepetitionPenalty(candidatesP, lastNtokens, op.repeatPenalty)
	m.ctx.SampleFrequencyAndPresencePenalties(candidatesP, lastNtokens, op.frequenctPenalty, op.presencePenalty)

	nlLogis := logits[binding.TokenNl()]
	if !op.penalizeNL {
		logits[binding.TokenNl()] = nlLogis
	}

	if op.temp < 0 {
		return m.ctx.SampleTokenGreedy(candidatesP)
	}

	if op.mirostat == 1 {
		mirostatMu := 2 * op.mirostatTau
		mirostatM := 100
		m.ctx.SampleTemperature(candidatesP, op.temp)
		return m.ctx.SampleTokenMirostat(candidatesP, op.mirostatTau, op.mirostatEta, mirostatM, &mirostatMu)
	}

	if op.mirostat == 2 {
		mirostatMu := 2 * op.mirostatTau
		m.ctx.SampleTemperature(candidatesP, op.temp)
		return m.ctx.SampleTokenMirostatV2(candidatesP, op.mirostatTau, op.mirostatEta, &mirostatMu)
	}

	m.ctx.SampleTopK(candidatesP, int32(op.topK), 1)
	m.ctx.SampleTailFree(candidatesP, op.tfsZ, 1)
	m.ctx.SampleTypical(candidatesP, op.typicalP, 1)
	m.ctx.SampleTopP(candidatesP, op.topP, 1)
	m.ctx.SampleTemperature(candidatesP, op.temp)
	return m.ctx.SampleToken(candidatesP)
}

func (m *model) SaveSession(filepath string) error {
	if m.params.verbose {
		stateTime := time.Now()
		fmt.Printf("start to save session, filepath: %s\n", filepath)
		defer func() {
			fmt.Printf("finished save session, filepath: %s, took: %v\n", filepath, time.Since(stateTime))
		}()
	}
	return m.ctx.SaveSessionFile(filepath, m.tokens)
}

func (m *model) LoadSession(filepath string, tokens []binding.Token) (int, error) {
	if m.params.verbose {
		stateTime := time.Now()
		fmt.Printf("start to load session, filepath: %s\n", filepath)
		defer func() {
			fmt.Printf("finished load session, filepath: %s, took: %v\n", filepath, time.Since(stateTime))
		}()
	}

	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		return 0, nil
	}

	m.Reset()

	sessionTokens, err := m.ctx.LoadSessionFile(filepath)
	if err != nil {
		return 0, err
	}

	min := int(math.Min(float64(len(tokens)), float64(len(sessionTokens))))
	index := 0
	for ; index < min-1; index++ {
		if sessionTokens[index] != tokens[index] {
			break
		}
	}

	m.pastNum = index
	m.tokensConsumed = index
	m.tokens = tokens[:index]
	m.lastNTokensData.Write(m.tokens)
	return index, nil
}

func (m *model) PrintTimings() {
	m.ctx.PrintTimings()
}

func (m *model) ResetTimings() {
	m.ctx.ResetTimings()
}
