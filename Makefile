INCLUDE_PATH := $(abspath .)
LIBRARY_PATH := $(abspath .)

LLAMA_CPP_VERSION := master-fa84c4b

fmt:
	revive --config revive.toml -formatter friendly github.com/bdqfork/go-llama.cpp pkg/...

clean:
	rm -rf libllama.so main include
	cd llama.cpp && make clean

llama.cpp:
	git clone -b ${LLAMA_CPP_VERSION} https://github.com/ggerganov/llama.cpp.git
	
libllama.so: clean llama.cpp
	mkdir include && cp llama.cpp/*.h include/
	cd llama.cpp && make libllama.so && mv libllama.so ../ && make clean

test: fmt libllama.so
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} go run main.go -v=4

run: fmt libllama.so
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} go run main.go -v=2

build: fmt libllama.so
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} go build -o main main.go

docker-build: build
	docker build -t github.com/bdqfork/go-llama:lastest .