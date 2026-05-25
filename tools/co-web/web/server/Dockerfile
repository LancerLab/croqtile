FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    bison \
    flex \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/choreo

COPY . .

RUN make release

ENV CHOREO_BIN=/opt/choreo/build-release/choreo
ENV CHOREO_TARGET=cute
ENV CHOREO_ARCH=sm_90a
ENV CHOREO_SERVER_PORT=8081

EXPOSE 8081

CMD ["python3", "web/server/gpu_server.py"]
