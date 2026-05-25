# co-web -- Choreo Web Playground

`co-web` is a browser-based IDE for Choreo that compiles the Choreo compiler
itself to WebAssembly. Users can edit, compile, and interpret `.co` programs
entirely client-side without any server or hardware.

## Building

`co-web` requires the Emscripten SDK. The build system auto-downloads it if
not present:

```bash
make co-web
```

The output is placed in `tools/co-web/web/co-web.js` and `co-web.wasm`.

To disable building, pass `-DCHOREO_BUILD_CO_WEB=OFF` to CMake.

## Running

```bash
cd tools/co-web/web
python3 serve.py
# Open http://localhost:8080
```

## Features

- **Monaco Editor** with Choreo `.co` syntax highlighting
- **Compile** -- emit generated C++/CUDA source (emit-source mode)
- **Run (Interpreter)** -- execute via mock interpreter in-browser
- **Show AST** -- dump the parsed abstract syntax tree
- **Run on GPU** (Phase 2) -- send source to a remote GPU server

All compilation and interpretation runs inside the browser via WebAssembly.
No server is required for the core experience.

## Architecture

`co-web` uses Choreo as an SDK (same pattern as `co-mock`):

```
.co source --> [WASM: Preprocessor] --> [Parser] --> [Semantic Analysis] --> [Codegen/Interp]
                                                                              |
                                                                    MockCodeGen (IR text)
                                                                       or
                                                                    MockInterpreter (execution)
```

- **Frontend**: The full Choreo frontend (preprocessor, parser, type inference,
  normalization, semantic checks) compiled to WASM via Emscripten.
- **Backend**: Mock target for IR inspection; mock interpreter for execution.
- **Web UI**: Monaco editor + Web Worker loading the WASM module.

## Directory Layout

```
tools/co-web/
  CMakeLists.txt        # Emscripten WASM build configuration
  README.md             # This file
  wasm_api.cpp          # Embind API: compile(), mockRun(), dumpAST()
  web/
    index.html          # Main playground page
    editor.js           # Monaco editor + .co language definition
    choreo-worker.js    # Web Worker loading WASM module
    styles.css          # Dark theme UI
    serve.py            # Development HTTP server
    server/
      gpu_server.py     # Phase 2: REST API for GPU execution
      Dockerfile        # Docker image for GPU server
      README.md         # GPU server documentation
  tests/
    lit.cfg             # Lit test configuration
    basic_compile.co    # Basic compilation test
```

## GPU Server (Phase 2)

An optional GPU execution server lives at `web/server/gpu_server.py`. When
running, the web playground detects it at `localhost:8081` and enables the
"Run on GPU" button.

```bash
CHOREO_BIN=./choreo python3 tools/co-web/web/server/gpu_server.py
```
