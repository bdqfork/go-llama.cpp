This is a Go language binding for llama.cpp, which was implemented based on CGo. Provide the llama.cpp low level API and ChatGPT high level API.


### High level API
You can see `pkg/llm`, in which implemented ChatGPT completion and chat completion APIs. More details https://platform.openai.com/docs/api-reference/chat/create.

#### Usage
Using vicuna as an example.
1. First, download the vicuna model. You can use below commands to download.
```shell
wget -c https://huggingface.co/eachadea/ggml-vicuna-7b-1.1/resolve/main/ggml-vic7b-q4_2.bin
```
2. Then edit the file models/vicuna.yaml and modify its `path` to point to the location of the just downloaded model.
3. Run the command `make run`, you can see below. 
```shell
[GIN-debug] [WARNING] Running in "debug" mode. Switch to "release" mode in production.
 - using env:   export GIN_MODE=release
 - using code:  gin.SetMode(gin.ReleaseMode)

[GIN-debug] GET    /v1/models                --> github.com/bdqfork/go-llama.cpp/pkg/server.(*Server).listModels-fm (3 handlers)
[GIN-debug] GET    /v1/models/:model         --> github.com/bdqfork/go-llama.cpp/pkg/server.(*Server).retreiveModel-fm (3 handlers)
[GIN-debug] POST   /v1/completions           --> github.com/bdqfork/go-llama.cpp/pkg/server.(*Server).completion-fm (3 handlers)
[GIN-debug] POST   /v1/chat/completions      --> github.com/bdqfork/go-llama.cpp/pkg/server.(*Server).chatCompletion-fm (3 handlers)
[GIN-debug] [WARNING] You trusted all proxies, this is NOT safe. We recommend you to set a value.
Please check https://pkg.go.dev/github.com/gin-gonic/gin#readme-don-t-trust-all-proxies for details.
[GIN-debug] Listening and serving HTTP on 0.0.0.0:8000
```
4. Visit completion or chatcompletion API via http://localhost:8000.
```shell
curl --location 'http://localhost:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "vicuna",
    "messages": [
        {
            "role": "system",
            "content": "You are assistant for user. Have a chat with user, using markdown!"
        },
        {
            "role": "user",
            "content": "Hello!"
        }
    ],
    "max_tokens": 20,
    "presence_penalty": 1,
    "stream": false
}'
```
#### GPU
```shell
LLAMA_CUBLAS=1 make run
```