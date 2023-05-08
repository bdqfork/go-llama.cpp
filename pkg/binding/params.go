package binding

/*
#include <llama.h>
*/
import "C"

// SetCtxNum for eval operation
func (p *Params) SetCtxNum(num int32) {
	p.n_ctx = C.int(num)
}

// SetSeed for sample operation
func (p *Params) SetSeed(seed int32) {
	p.seed = C.int(seed)
}

// SetF16KV enables use f16 kv
func (p *Params) SetF16KV(enable bool) {
	p.f16_kv = C.bool(enable)
}

// SetLogitsAll enables get all logits
func (p *Params) SetLogitsAll(enable bool) {
	p.logits_all = C.bool(enable)
}

// SetVocabOnly enables vocab only mode
func (p *Params) SetVocabOnly(enable bool) {
	p.vocab_only = C.bool(enable)
}

// SetUseMMap enables use mmap when load model
func (p *Params) SetUseMMap(enable bool) {
	p.use_mmap = C.bool(enable)
}

// SetUseMLock enables use mmap when load model
func (p *Params) SetUseMLock(enable bool) {
	p.use_mlock = C.bool(enable)
}

// SetEmbedding enables embedding operation
func (p *Params) SetEmbedding(enable bool) {
	p.embedding = C.bool(enable)
}

// SetGPULayers offline model to gpu
func (p *Params) SetGPULayers(num int32) {
	p.n_gpu_layers = C.int(num)
}
