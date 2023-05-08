FROM golang:1.20 as builder

WORKDIR /app

COPY . .  

ENV LLAMA_OPENBLAS=1
ENV GOPROXY=https://goproxy.cn,direct

RUN sed -i s/deb.debian.org/mirrors.aliyun.com/g /etc/apt/sources.list && \
  apt update -y && \
  apt remove git -y && \
  apt install --no-install-recommends build-essential libopenblas-dev git -y && \
  go install github.com/mgechev/revive@master && \
  make build

FROM ubuntu

WORKDIR /app

COPY --from=builder /app/main /app/main
COPY --from=builder /app/libllama.so /app/libllama.so

RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && \
  apt update -y && \
  apt install --no-install-recommends libopenblas-dev -y && \
  rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app

ENTRYPOINT ["/app/main"]