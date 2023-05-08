package model

import (
	"sync"

	"github.com/bdqfork/go-llama.cpp/pkg/binding"
)

// TokenRingbuf stores last n tokens
type TokenRingbuf struct {
	buf         []binding.Token
	start, size int
	sync.Locker
}

// NewTokenRingBuf return a TokenRingbuf instance
func NewTokenRingBuf(size int) *TokenRingbuf {
	return &TokenRingbuf{buf: make([]binding.Token, size)}
}

// Write tokens
func (r *TokenRingbuf) Write(b []binding.Token) {
	for len(b) > 0 {
		start := (r.start + r.size) % len(r.buf)
		n := copy(r.buf[start:], b)
		b = b[n:]

		if r.size >= len(r.buf) {
			if n <= len(r.buf) {
				r.start += n
				if r.start >= len(r.buf) {
					r.start = 0
				}
			} else {
				r.start = 0
			}
		}
		r.size += n
		if r.size > cap(r.buf) {
			r.size = cap(r.buf)
		}
	}
}

// Read tokens
func (r *TokenRingbuf) Read(b []binding.Token) int {
	read := 0
	size := r.size
	start := r.start
	for len(b) > 0 && size > 0 {
		end := start + size
		if end > len(r.buf) {
			end = len(r.buf)
		}
		n := copy(b, r.buf[start:end])
		size -= n
		read += n
		b = b[n:]

		start = (start + n) % len(r.buf)
	}
	return read
}

// Size returns the size of the ringbuf
func (r *TokenRingbuf) Size() int {
	return r.size
}
