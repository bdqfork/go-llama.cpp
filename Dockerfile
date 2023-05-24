FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

COPY ./main /app/main
COPY ./libllama.so /app/libllama.so

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app

ENTRYPOINT ["/app/main"]