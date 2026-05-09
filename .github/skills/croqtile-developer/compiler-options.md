### Section 4: Compiler Options Reference

`./choreo` is the compiler main entrypoint.

### Compilation mode options

| Option | Description |
|---|---|
| `-t <target>` | Set compilation target: `cute` (GPU) |
| `-gs` / `--generate-script` | Generate bash compile/execute script (main end-to-end method) |
| `-es` | Generate target source only (do not compile target code; inspect codegen output) |
| `-c` / `--compile` | Compile but do not link |
| `-o <file>` | Set output file (`-o -` outputs to stdout) |
| `-arch=<arch>` | Set GPU architecture (e.g., `sm_86`, `sm_90a`) |
| `-n` / `--remove-comments` | Remove comments (useful with FileCheck) |
| `-E` | Preprocess only (inspect macro-expanded `.co` code) |
| `-s` / `--no-codegen` | No codegen (frontend analysis only; often combined with debug flags) |

### Debug options by purpose

#### A. AST / program structure inspection

| Option | Description | Output Example |
|---|---|---|
| `-e` / `--dump-ast` | Dump AST tree right after parsing | Tree-form function/parameter/statement structure |
| `-pa=<PASS>` | Print AST **after** the specified pass | Shows AST transformed by that pass |
| `-pb=<PASS>` | Print AST **before** the specified pass | Compare before/after behavior |
| `-paa` / `--print-after-all` | Print AST after every pass | Trace full AST evolution (large output) |
| `-pba` / `--print-before-all` | Print AST before every pass | Same as above |
| `-pnt` / `--print-node-type` | Include node type annotations in AST dump | Shows `<{type}>` after nodes |

#### B. Type / symbol information

| Option | Description | Output Example |
|---|---|---|
| `-i` / `--infer-types` | Show type inference results (abort after INFER pass) | `Parameter: ::fn::lhs, Type: s32 mdspan<3> [2, 3, 64]` |
| `-l` / `--dump-symbol` | Dump full symbol table after LATENORM | `symbol: ::fn::var, type: f32 mdspan<2> [M, N]` |
| `-ds=<PASS>` | Dump symbol table after a specific pass | Same as `-l`, but at any pass point |

#### C. Value numbering debug

| Option | Description | Output Example |
|---|---|---|
| `-vn` / `--print-valno` | Trace full value-numbering process (abort after INFER) | `New VN #5: '#2,#3,#4'`; `Alias "::fn::lhs.span" -> #5` |
| `--debug-visit=valno` | Detailed per-node debug for VALNO pass | VN/MDS/UB state changes per AST node |

#### D. Pass-level debug

| Option | Description | Output Example |
|---|---|---|
| `-dv=<PASS>` / `--debug-visit=<PASS>` | Full node-level debug for one pass | State on every node enter/exit |
| `-tv=<PASS>` / `--trace-visit=<PASS>` | Trace only which nodes were visited by a pass | `AST::Program`, `AST::Parameter`, ... |
| `-d` / `--debug` | Enable debug for **all** passes (including parser bison debug) | Extremely verbose, starting from parser tokens |
| `-sp` / `--show-passes` | Show full pass pipeline | Prints pipeline pass list |
| `-sa=<PASS>` / `--stop-after=<PASS>` | Stop compilation after specified pass | No final output; runs until that pass only |
| `-dp=<PASS>` / `--disable-visit=<PASS>` | Disable one pass | Isolate issues introduced by a specific pass |

#### E. Feature-specific debug

| Option | Description |
|---|---|
| `-sr` / `--print-sym-replace` | Trace symbol replacement behavior |
| `-dvec` / `--debug-vectorize` | Debug loop vectorization |
| `-dd` / `--diag-dma` | Runtime DMA diagnostics (enabled in generated code) |
| `--dma-verbose` | Print detailed DMA runtime logs |
| `--liveness` | Analyze variable liveness |
| `--mem-reuse` | Analyze memory reuse |
| `-u` / `--visualize` | Visualize DMA data movement |
| `--save-temps` | Keep temporary compilation files |
| `-v` / `--verbose` | Show external programs invoked by compiler |
| `-vf` / `--verify` | Verify AST validity after every pass |

#### F. Other hidden options

| Option | Description |
|---|---|
| `-sfv` / `--simplify-fp-valno` | Simplify floating-point value numbering (experimental) |
| `-kt` / `--use-kernel-template` | Allow instantiation of C++ template functions |
| `-ht` / `--use-hetero-tileflow` | Implicit tileflow optimization across GPU memory tiers (experimental) |
| `-bn` / `--branch-norm` | Normalize if-else branches |
| `-ln` / `--loop-norm` | Normalize loops |
| `-nm` / `--no-vectorize` | Disable automatic vectorization |
| `-npp` / `--no-preprocess` | Skip preprocessor |
| `-adf` / `--analyze-device-functions` | Analyze device functions |
| `-fno-show-source-location` | Hide source locations |
| `-tos` / `--target-options` | Extra options passed to target compiler |
| `-fmax-local` | Max per-thread local memory size (bytes) |
| `-api=<mode>` | API mode: `cffi` or `sglang` |
