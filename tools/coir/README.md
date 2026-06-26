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

CoIR requires LLVM/MLIR. CMake downloads it automatically into
`extern/llvm-project/` on first build when it is not present.

```bash
# Build CoIR tools (auto-downloads LLVM/MLIR if missing)
make coir

# Run tests
make coir-test
```

Or configure manually with CMake:

```bash
cmake -S . -B build -G Ninja \
  -DCHOREO_BUILD_COIR=ON \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build co2ir coir-opt cocc
```

To disable the automatic download, pass `-DCOIR_AUTO_DOWNLOAD_LLVM=OFF`.

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

- LLVM/MLIR (pre-built, auto-downloaded by CMake on first build)
- Choreo compiler libraries (built as part of normal `make build`)
