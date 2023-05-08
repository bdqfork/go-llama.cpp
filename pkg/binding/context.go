package binding

import (
	"fmt"
	"unsafe"

	"github.com/bdqfork/go-llama.cpp/pkg/util"
)

/*
#include <llama.h>
#include <stdlib.h>
*/
import "C"

// Free ...
func (ctx *Context) Free() {
	C.llama_free((*C.struct_llama_context)(ctx))
}

// ApplyLoraFromFile ...
func (ctx *Context) ApplyLoraFromFile(loraPath, baseModelPath string, threadNum int32) error {
	cLoraPath := C.CString(loraPath)
	defer C.free(unsafe.Pointer(cLoraPath))

	cBaseModelPath := C.CString(baseModelPath)
	defer C.free(unsafe.Pointer(cBaseModelPath))

	if C.llama_apply_lora_from_file((*C.struct_llama_context)(ctx), cLoraPath, cBaseModelPath, (C.int)(threadNum)) != 0 {
		return fmt.Errorf("failed to apply lora from file, loraPath: %s, baseModelPath: %s, threadNum: %d", loraPath, baseModelPath, threadNum)
	}
	return nil
}

// GetKVCacheTokenCount ..
func (ctx *Context) GetKVCacheTokenCount() int32 {
	res := C.llama_get_kv_cache_token_count((*C.struct_llama_context)(ctx))
	return int32(res)
}

// SetRngSeed ...
func (ctx *Context) SetRngSeed(seed int32) {
	C.llama_set_rng_seed((*C.struct_llama_context)(ctx), C.int(seed))
}

// GetStateSize ...
func (ctx *Context) GetStateSize() uint32 {
	res := C.llama_get_state_size((*C.struct_llama_context)(ctx))
	return uint32(res)
}

// CopyStateData ...
func (ctx *Context) CopyStateData(dest []uint8) uint32 {
	res := C.llama_copy_state_data((*C.struct_llama_context)(ctx), (*C.uint8_t)(&dest[0]))
	return uint32(res)
}

// SetStateData ...
func (ctx *Context) SetStateData(src []uint8) uint32 {
	res := C.llama_set_state_data((*C.struct_llama_context)(ctx), (*C.uint8_t)(&src[0]))
	return uint32(res)
}

// LoadSessionFile ...
func (ctx *Context) LoadSessionFile(path string) ([]Token, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	tokens := make([]Token, ctx.CtxNum())
	n := len(tokens)
	countOut := new(uint32)
	ptr := unsafe.Pointer(countOut)
	res := C.llama_load_session_file((*C.struct_llama_context)(ctx), cPath, (*C.llama_token)(&tokens[0]), C.ulong(n), (*C.ulong)(ptr))
	if !res {
		return nil, fmt.Errorf("failed to load session file, path: %s", path)
	}
	return tokens[:*countOut], nil
}

// SaveSessionFile ...
func (ctx *Context) SaveSessionFile(path string, tokens []Token) error {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	n := len(tokens)
	res := C.llama_save_session_file((*C.struct_llama_context)(ctx), cPath, (*C.llama_token)(&tokens[0]), C.ulong(n))
	if !res {
		return fmt.Errorf("failed to save session file, path: %s", path)
	}
	return nil
}

// Eval ...
func (ctx *Context) Eval(tokens []Token, pastNum, threadNum int32) error {
	if C.llama_eval((*C.struct_llama_context)(ctx), (*C.llama_token)(&tokens[0]), C.int(len(tokens)), (C.int)(pastNum), (C.int)(threadNum)) == 1 {
		return fmt.Errorf("failed to eval, tokens: %+v", tokens)
	}
	return nil
}

// Tokenize ...
func (ctx *Context) Tokenize(text string, tokens []Token, maxTokens int32, addBos bool) int32 {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.llama_tokenize((*C.struct_llama_context)(ctx), cText, (*C.llama_token)(&tokens[0]), C.int(maxTokens), (C.bool)(addBos))
	return int32(result)
}

// VocabNum ...
func (ctx *Context) VocabNum() int32 {
	res := C.llama_n_vocab((*C.struct_llama_context)(ctx))
	return int32(res)
}

// CtxNum ...
func (ctx *Context) CtxNum() int32 {
	res := C.llama_n_ctx((*C.struct_llama_context)(ctx))
	return int32(res)
}

// EmbdNum ...
func (ctx *Context) EmbdNum() int32 {
	res := C.llama_n_embd((*C.struct_llama_context)(ctx))
	return int32(res)
}

// GetLogits ...
func (ctx *Context) GetLogits(tokenNum int) []float32 {
	res := C.llama_get_logits((*C.struct_llama_context)(ctx))
	ptr := unsafe.Pointer(res)

	result := make([]float32, 0)
	n := int(ctx.VocabNum()) * tokenNum
	util.GoSlice(unsafe.Pointer(&result), ptr, n, n)
	return result
}

// GetEmbeddings ...
func (ctx *Context) GetEmbeddings() []float32 {
	res := C.llama_get_logits((*C.struct_llama_context)(ctx))
	ptr := unsafe.Pointer(res)

	result := make([]float32, 0)
	n := int(ctx.EmbdNum())
	util.GoSlice(unsafe.Pointer(&result), ptr, n, n)
	return result
}

// TokenToStr ...
func (ctx *Context) TokenToStr(token Token) string {
	res := C.llama_token_to_str((*C.struct_llama_context)(ctx), (C.llama_token)(token))
	return C.GoString(res)
}

// SampleRepetitionPenalty ...
func (ctx *Context) SampleRepetitionPenalty(tokenDataArray *TokenDataArray, lastTokens []Token, penalty float32) {
	size := len(lastTokens)
	C.llama_sample_repetition_penalty((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(tokenDataArray), (*C.llama_token)(&lastTokens[0]), C.ulong(size), C.float(penalty))
}

// SampleFrequencyAndPresencePenalties ...
func (ctx *Context) SampleFrequencyAndPresencePenalties(tokenDataArray *TokenDataArray, lastTokens []Token, alphaFrequency, alphaPresence float32) {
	C.llama_sample_frequency_and_presence_penalties((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(tokenDataArray), (*C.llama_token)(&lastTokens[0]), C.ulong(len(lastTokens)), C.float(alphaFrequency), C.float(alphaPresence))
}

// SampleSoftmax ...
func (ctx *Context) SampleSoftmax(candidates *TokenDataArray) {
	C.llama_sample_softmax((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(candidates))
}

// SampleTopK ...
func (ctx *Context) SampleTopK(tokenDataArray *TokenDataArray, k int32, minKeep uint32) {
	C.llama_sample_top_k((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(tokenDataArray), C.int(k), C.ulong(minKeep))
}

// SampleTopP ...
func (ctx *Context) SampleTopP(tokenDataArray *TokenDataArray, p float32, minKeep uint32) {
	C.llama_sample_top_p((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(tokenDataArray), C.float(p), C.ulong(minKeep))
}

// SampleTailFree ...
func (ctx *Context) SampleTailFree(tokenDataArray *TokenDataArray, z float32, minKeep uint32) {
	C.llama_sample_tail_free((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(tokenDataArray), C.float(z), C.ulong(minKeep))
}

// SampleTypical ...
func (ctx *Context) SampleTypical(tokenDataArray *TokenDataArray, p float32, minKeep uint32) {
	C.llama_sample_typical((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(tokenDataArray), C.float(p), C.ulong(minKeep))
}

// SampleTemperature ...
func (ctx *Context) SampleTemperature(tokenDataArray *TokenDataArray, temp float32) {
	C.llama_sample_temperature((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(tokenDataArray), C.float(temp))
}

// SampleTokenMirostat sample a token via mirostat algorithm
func (ctx *Context) SampleTokenMirostat(tokenDataArray *TokenDataArray, tau, eta float32, m int, mu *float32) Token {
	res := C.llama_sample_token_mirostat((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(tokenDataArray), C.float(tau), C.float(eta), C.int(m), (*C.float)(unsafe.Pointer(mu)))
	return Token(res)
}

// SampleTokenMirostatV2 sample a token via mirostat v2 algorithm
func (ctx *Context) SampleTokenMirostatV2(tokenDataArray *TokenDataArray, tau, eta float32, mu *float32) Token {
	res := C.llama_sample_token_mirostat_v2((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(tokenDataArray), C.float(tau), C.float(eta), (*C.float)(unsafe.Pointer(mu)))
	return Token(res)
}

// SampleTokenGreedy sample a token via greedy algorithm
func (ctx *Context) SampleTokenGreedy(candidates *TokenDataArray) Token {
	res := C.llama_sample_token_greedy((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(candidates))
	return Token(res)
}

// SampleToken sample a token
func (ctx *Context) SampleToken(candidates *TokenDataArray) Token {
	res := C.llama_sample_token((*C.struct_llama_context)(ctx), (*C.struct_llama_token_data_array)(candidates))
	return Token(res)
}

// PrintTimings print the eval timings
func (ctx *Context) PrintTimings() {
	C.llama_print_timings((*C.struct_llama_context)(ctx))
}

// ResetTimings reset the eval timings
func (ctx *Context) ResetTimings() {
	C.llama_reset_timings((*C.struct_llama_context)(ctx))
}
