package util

import (
	"fmt"
	"reflect"
	"unsafe"
)

// GoSlice converts a pointer to slice
func GoSlice(slicePtr, data unsafe.Pointer, len, cap int) {
	slice := (*reflect.SliceHeader)(slicePtr)
	slice.Data = uintptr(data)
	slice.Len = len
	slice.Cap = cap
}

// ParseWords returns string array of input
func ParseWords(input any) ([]string, error) {
	if input == nil {
		return make([]string, 0), nil
	}

	res, ok := input.(string)
	if ok {
		return []string{res}, nil
	}

	words, ok := input.([]any)
	if !ok {
		return nil, fmt.Errorf("invalid input: %v", input)
	}

	results := make([]string, 0)
	for _, val := range words {
		res, ok := val.(string)
		if !ok {
			return nil, fmt.Errorf("invalid input: %v", input)
		}
		results = append(results, res)
	}
	return results, nil
}
