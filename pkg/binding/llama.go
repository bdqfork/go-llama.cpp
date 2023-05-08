package binding

import (
	"fmt"
	"unsafe"
)

/*
#cgo CFLAGS: -I../../include
#cgo LDFLAGS: -lllama -lm -lstdc++
#cgo darwin LDFLAGS: -framework Accelerate
#include <llama.h>
#include <stdlib.h>
*/
import "C"

const (
	// FileVersion is version of llama file
	FileVersion = C.LLAMA_FILE_VERSION
	// FileMagic is magic number of llama file
	FileMagic = C.LLAMA_FILE_MAGIC
	// FileMagicUnversioned is unversioned magic number of llama file
	FileMagicUnversioned = C.LLAMA_FILE_MAGIC_UNVERSIONED
	// SessionMagic is magic number of session
	SessionMagic = C.LLAMA_SESSION_MAGIC
	// SessionVersion is version of session
	SessionVersion = C.LLAMA_SESSION_VERSION
)

type (
	// Context is llama runtime context
	Context C.struct_llama_context
	// Token is input for llama eval
	Token C.llama_token
	// TokenData is wrapper for Token with logits and p
	TokenData C.struct_llama_token_data
	// TokenDataArray is array warpper for TokenData
	TokenDataArray C.struct_llama_token_data_array
	// ProgressCallback is called on progress
	ProgressCallback C.llama_progress_callback
	// Params to config runtime context
	Params C.struct_llama_context_params
	// FType indicate model quantitze type
	FType C.enum_llama_ftype
)

const (
	// FTypeAllF32 f32
	FTypeAllF32 = C.LLAMA_FTYPE_ALL_F32
	// FtypeMostlyF16 f16
	FtypeMostlyF16 = C.LLAMA_FTYPE_MOSTLY_F16
	// FTypeMostlyQ4V0 q4.0
	FTypeMostlyQ4V0 = C.LLAMA_FTYPE_MOSTLY_Q4_0
	// FTypeMostlyQ4V1 q4.1
	FTypeMostlyQ4V1 = C.LLAMA_FTYPE_MOSTLY_Q4_1
	// FTypeMostlyQ4V1SomeF16 q4.1 f16
	FTypeMostlyQ4V1SomeF16 = C.LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16
	// FTypeMostlyQ8V0 q8.0
	FTypeMostlyQ8V0 = C.LLAMA_FTYPE_MOSTLY_Q8_0
	// FTypeMostlyQ5V0 q5.0
	FTypeMostlyQ5V0 = C.LLAMA_FTYPE_MOSTLY_Q5_0
	// FTypeMostlyQ5V1 q5.1
	FTypeMostlyQ5V1 = C.LLAMA_FTYPE_MOSTLY_Q5_1
)

// ContextDefaultParams returns default context params
func ContextDefaultParams() Params {
	params := C.llama_context_default_params()
	return Params(params)
}

// MMapSupported returns if support mmap
func MMapSupported() bool {
	res := C.llama_mmap_supported()
	return bool(res)
}

// MLockSupported returns if support mlock
func MLockSupported() bool {
	res := C.llama_mlock_supported()
	return bool(res)
}

// InitBackend to init llama and ggml backend
func InitBackend() {
	C.llama_init_backend()
}

// InitFromFile to get context from model.bin
func InitFromFile(path string, params Params) (*Context, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	if ctx := C.llama_init_from_file(cPath, (C.struct_llama_context_params)(params)); ctx != nil {
		return (*Context)(ctx), nil
	}
	return nil, fmt.Errorf("failed to init model context, path: %s", path)
}

// ModelQuantize to quantitze model
func ModelQuantize(input, output string, fType FType, threadNum int32) bool {
	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))

	cOutput := C.CString(output)
	defer C.free(unsafe.Pointer(cOutput))

	return C.llama_model_quantize(cInput, cOutput, (C.enum_llama_ftype)(fType), C.int(threadNum)) == 0
}

// TokenBos returns bos token
func TokenBos() Token {
	res := C.llama_token_bos()
	return Token(res)
}

// TokenEos returns eos token
func TokenEos() Token {
	res := C.llama_token_eos()
	return Token(res)
}

// TokenNl returns nl token
func TokenNl() Token {
	res := C.llama_token_nl()
	return Token(res)
}

// PrintSystemInfo returns system info of current runtime
func PrintSystemInfo() string {
	res := C.llama_print_system_info()
	return C.GoString(res)
}
