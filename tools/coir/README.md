# CoIR -- Choreo Intermediate Representation Tooling

CoIR is an MLIR-based intermediate representation for Choreo programs. It provides
a structured, optimizable IR that captures GPU kernel choreography (DMA, MMA,
parallel structure, tensor operations) at a level amenable to analysis and
transformation.

## Tools

| Tool | Description |
|------|-------------|
| `coir-gen` | Translate Choreo `.co` source to CoIR MLIR |
| `coir-opt` | Run optimization/lowering passes on CoIR MLIR |
| `coir-codegen` | Emit C++ source code from (lowered) CoIR MLIR |

## Pipeline

```
.co source -> [coir-gen] -> CoIR MLIR -> [coir-opt] -> Lowered MLIR -> [coir-codegen] -> C++ source
```

## Build

CoIR requires LLVM/MLIR and is disabled by default:

```bash
# 1. Download LLVM/MLIR dependency
make setup-coir-deps

# 2. Build CoIR tools
make coir

# 3. Run tests
make coir-test
```

Or configure manually with CMake:

```bash
cmake -S . -B build -G Ninja \
  -DCHOREO_BUILD_COIR=ON \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build coir-gen coir-opt coir-codegen
```

## Usage

```bash
# Generate CoIR from a .co file
./build/tools/coir/coir-gen test.co > test.mlir

# Run optimization passes
./build/tools/coir/coir-opt --coir-classify-copies --coir-lower-mma --coir-lower-copy test.mlir -o lowered.mlir

# Emit C++ source
./build/tools/coir/coir-codegen lowered.mlir
```

## CoIR Passes

| Pass | Description |
|------|-------------|
| `--coir-classify-copies` | Lower generic `coir.data.copy` to specialized DMA/TMA/thread copy ops |
| `--coir-lower-mma` | Lower `coir.mma.*` ops to target-specific MMA instructions |
| `--coir-lower-copy` | Lower specialized copy ops to transfer/sync sequences |
| `--coir-emit-cuda` | Emit CUDA/C++ source from lowered CoIR IR |

## Dependencies

- LLVM/MLIR (pre-built, fetched via `make setup-coir-deps`)
- Choreo compiler libraries (built as part of normal `make build`)
