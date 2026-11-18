# syntax=docker/dockerfile:1
# CUDA 13 cannot compile Volta. No GPU is needed on the image-build host.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build-base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake curl g++-14 git ninja-build pkg-config python3 \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
ARG BUILD_JOBS=2

FROM build-base AS llama-build
ARG LLAMA_REF=e95426f16982054c379ca7bdc8186996dd22f0fe
RUN mkdir /src && curl -fL --retry 3 \
    "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_REF}.tar.gz" \
    | tar -xz --strip-components=1 -C /src
RUN --mount=type=cache,id=qwen-v100-llama-${LLAMA_REF},target=/build \
    cmake -S /src -B /build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
    -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF \
    '-DGGML_CUDA_FA_QUANTS=f16-f16;q8_0-q8_0' \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_OPENSSL=ON \
    && cmake --build /build --parallel ${BUILD_JOBS} --target llama-server \
    && mkdir -p /out && cp /build/bin/llama-server /out/ \
    && printf 'llama.cpp=%s\n' "${LLAMA_REF}" > /out/llama-version.txt

FROM build-base AS ninfer-build
ARG NINFER_REF=56f7b091992fa9af85fecad646de954f6b459b77
RUN mkdir /src && curl -fL --retry 3 \
    "https://github.com/b-iurea/ninfer-v100-turbo/archive/${NINFER_REF}.tar.gz" \
    | tar -xz --strip-components=1 -C /src
RUN --mount=type=cache,id=qwen-v100-ninfer-${NINFER_REF},target=/build \
    cmake -S /src -B /build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
    -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel ${BUILD_JOBS} --target ninfer-serve \
    && mkdir -p /out && cp /build/apps/ninfer-serve /out/ \
    && printf 'ninfer-v100-turbo=%s\n' "${NINFER_REF}" > /out/ninfer-version.txt

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime-base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git libavcodec60 libavformat60 libavutil58 \
    libcurl4t64 libgomp1 libssl3t64 libstdc++6 libswscale7 \
    openssh-server python3 python3-venv ripgrep tmux tini unzip util-linux \
    && rm -rf /var/lib/apt/lists/*
# Use the baseline binary so the build machine's AVX2 support cannot affect Vast hosts.
ARG OPENCODE_VERSION=2.2.6
RUN curl -fL --retry 3 \
    "https://registry.npmjs.org/@opencode/cli-linux-x64-baseline/-/cli-linux-x64-baseline-${OPENCODE_VERSION}.tgz" \
    | tar -xz -C /tmp \
    && install -m 755 /tmp/package/bin/opencode /usr/local/bin/opencode \
    && rm -rf /tmp/package \
    && opencode --version
RUN python3 -m venv /opt/hf \
    && /opt/hf/bin/pip install --no-cache-dir 'huggingface_hub==1.1.2'
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH=/opt/hf/bin:/opt/qwen/scripts:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    DATA_DIR=/workspace MODEL_ID=qwen3.8-27b CONTEXT_SIZE=32768 \
    HOST=127.0.0.1 PORT=8080 SHELL=/bin/bash
COPY models.json opencode.json /opt/qwen/
COPY scripts/ /opt/qwen/scripts/
COPY fixtures/ /opt/qwen/fixtures/
RUN chmod +x /opt/qwen/scripts/* && mkdir -p /workspace /run/sshd
WORKDIR /workspace
STOPSIGNAL SIGTERM

FROM runtime-base AS runtime
COPY --from=llama-build /out/llama-server /usr/local/bin/
COPY --from=ninfer-build /out/ninfer-serve /usr/local/bin/
COPY --from=llama-build /out/llama-version.txt /opt/qwen/
COPY --from=ninfer-build /out/ninfer-version.txt /opt/qwen/
# Vast SSH mode replaces ENTRYPOINT: use the onstart command in README.md.
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/qwen/scripts/entrypoint.sh"]
CMD ["sleep", "infinity"]
