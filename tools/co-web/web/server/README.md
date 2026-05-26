# Choreo GPU Execution Server

REST API server for compiling and running Choreo programs on an NVIDIA GPU.

## Quick Start

```bash
# Run directly (requires choreo binary and CUDA)
CHOREO_BIN=./choreo python3 web/server/gpu_server.py

# Or with Docker (requires nvidia-docker)
docker build -t choreo-gpu -f web/server/Dockerfile .
docker run --gpus all -p 8081:8081 choreo-gpu
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHOREO_BIN` | `./choreo` | Path to choreo compiler binary |
| `CHOREO_TARGET` | `cute` | Default compilation target |
| `CHOREO_ARCH` | `sm_90a` | Default GPU architecture |
| `CHOREO_SERVER_PORT` | `8081` | Server port |

## API Endpoints

### `GET /api/status`

Returns server health and GPU information.

```json
{
  "status": "ok",
  "gpus": [{"name": "NVIDIA H100", "memory": "80 GB", "driver": "535.86.10"}],
  "choreo": "available",
  "target": "cute",
  "arch": "sm_90a"
}
```

### `POST /api/compile`

Compile .co source and return generated CUDA/C++ code.

Request:
```json
{
  "source": "__co__ void kernel() { ... }",
  "target": "cute",
  "arch": "sm_90a",
  "flags": ""
}
```

### `POST /api/run`

Compile and execute .co source, return stdout/stderr.

Request:
```json
{
  "source": "__co__ void kernel() { ... } int main() { kernel(); }",
  "target": "cute",
  "arch": "sm_90a"
}
```

Response:
```json
{
  "success": true,
  "output": "Run Complete.\n",
  "errors": "",
  "returncode": 0,
  "phase": "execute"
}
```

## Integration with Web Playground

The web playground automatically detects the GPU server at `localhost:8081`.
When detected, the "Run on GPU" button becomes active. The web client can
also be configured to point to a remote GPU server by setting `gpuServerUrl`
in the browser console.
