package binding

import (
	"reflect"
	"unsafe"
)

/*
#include <llama.h>
#include <stdlib.h>

llama_token_data_array* new_token_data_array(unsigned long size, bool sorted) {
	llama_token_data* data = (llama_token_data*)malloc(sizeof(llama_token_data)*size);
	llama_token_data_array* candidates_p = (llama_token_data_array*)malloc(sizeof(llama_token_data)*size);

	candidates_p->data = data;
	candidates_p->size = size;
	candidates_p->sorted = sorted;

	return candidates_p;
}
*/
import "C"

// SetID for token data
func (t *TokenData) SetID(id int32) {
	t.id = C.int(id)
}

// SetLogit for token data
func (t *TokenData) SetLogit(logit float32) {
	t.logit = C.float(logit)
}

// SetP for token data
func (t *TokenData) SetP(p float32) {
	t.p = C.float(p)
}

// NewTokenDataArray returns new TokenDataArray which allocate memory in C, should call Free() to release
func NewTokenDataArray(size uint32, sorted bool) *TokenDataArray {
	res := C.new_token_data_array(C.ulong(size), C.bool(sorted))
	return (*TokenDataArray)(res)
}

// Data returns TokenData array
func (t *TokenDataArray) Data() []TokenData {
	var data []TokenData
	var dataHdr = (*reflect.SliceHeader)(unsafe.Pointer(&data))
	dataHdr.Data = uintptr(unsafe.Pointer(t.data))
	dataHdr.Len = int(t.size)
	dataHdr.Cap = int(t.size)
	return data
}

// Free to release memory
func (t *TokenDataArray) Free() {
	C.free(unsafe.Pointer(t.data))
	C.free(unsafe.Pointer(t))
}
