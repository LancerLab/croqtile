### Section 8: Debug Scenarios

Choose compiler flags from the failure pattern. Default to `-s` (no codegen) unless you need generated source. Trim huge logs with `| head -N` or `2>&1 | grep KEY`.

### Scenario 1: Syntax / parse errors

**Symptoms**: `syntax error`, `unexpected token`, parse-phase failure

```bash
./choreo -E file.co
./choreo -e file.co
./choreo -d -sa=sema file.co 2>&1 | head -100
```

### Scenario 2: Type inference issues

**Symptoms**: `type inconsistent`, `unable to apply to the types`, type mismatch

```bash
./choreo -i file.co
./choreo --debug-visit=infer -s file.co
./choreo -ds=infer -s file.co
./choreo -pnt -pa=infer -s file.co
```

### Scenario 3: Shape / span inference issues

**Symptoms**: `invalid span`, shape mismatch, `chunkat` out-of-bounds

```bash
./choreo -vn file.co
./choreo --debug-visit=valno -s file.co
./choreo -pa=sema -s file.co
```

### Scenario 4: Parallel decomposition issues

**Symptoms**: wrong parallel-by hierarchy, block/thread/group assignment anomalies

```bash
./choreo -pa=norm -s file.co
./choreo -pa=pbfill -s file.co
./choreo -paa -sa=pbfill file.co
```

### Scenario 5: DMA / data-movement issues

**Symptoms**: DMA semantic error, future/wait mismatch, `non-async future can not be waited`

```bash
./choreo --debug-visit=latenorm -sa=latenorm file.co
./choreo -ds=latenorm -s file.co
./choreo --debug-visit=check -s file.co
./choreo -u file.co
```

### Scenario 6: Memory reuse / liveness issues

**Symptoms**: abnormal allocation, buffer lifetime problems

```bash
./choreo --debug-visit=liveness -s file.co
./choreo --debug-visit=memreuse -s file.co
./choreo --debug-visit=memanlz -s file.co
```

### Scenario 7: Codegen issues

**Symptoms**: incorrect generated CUDA/C++, nvcc errors

```bash
./choreo -es -t cute file.co -o -
./choreo -es -t cute -arch=sm_86 file.co -o -
./choreo --save-temps -gs -t cute file.co -o build/out.cute.result
./choreo -pa=check -s file.co
```

### Scenario 8: Full pipeline tracing

**Symptoms**: unknown which pass breaks

```bash
./choreo -sp -s file.co
./choreo -paa -s file.co
./choreo -sa=sema file.co
./choreo -sa=valno file.co
./choreo -sa=infer file.co
./choreo -sa=latenorm file.co
./choreo -sa=check file.co
./choreo -dp=memreuse -s file.co
```

### Scenario 9: Vectorization issues

**Symptoms**: abnormal `foreach` vectorization

```bash
./choreo -dvec -s file.co
./choreo -nm -s file.co
```

### Scenario 10: Symbol replacement tracing

**Symptoms**: unexpected replacements after expansion or rewrites

```bash
./choreo -sr -s file.co
```

### Common fast debug commands

```bash
./choreo -e file.co
./choreo -i file.co
./choreo -vn file.co
./choreo -l -s file.co
./choreo -sp -s file.co
./choreo -pa=sema -s file.co
./choreo -pa=infer -pnt -s file.co
./choreo -paa -s file.co
./choreo --debug-visit=valno -s file.co
./choreo --debug-visit=infer -s file.co
./choreo -ds=infer -s file.co
./choreo --trace-visit=sema -s file.co
./choreo -es -t cute file.co -o -
./choreo -E file.co
```

### AI agent debug decision tree

When compiling and testing a `.co` file:

```
1. Validate ./choreo
   +- Invalid => make (rebuild)
   +- Valid   => continue

2. Inspect the .co file
   +- Contains "// RUN:"?
   |  +- Yes => run with lit.sh (or run the RUN line manually)
   |  +- No  => choose compile/codegen/run manually
   +- Contains "// REQUIRES: TARGET-GPU"?
   |  +- Yes => needs GPU
   |  +- No  => often frontend-only
   +- Content hints:
      +- has main() / CHECK: => likely end-to-end (-gs + --execute)
      +- only __co__ helpers  => codegen or library-style test

3. Target selection
   +- path hints cuda/gpu/cute => cute
   +- RUN: contains -t X      => use X
   +- still uncertain         => ask (Section 11)

4. During GPU execution
   +- nvidia-smi for status
   +- prefer idle GPU
   +- on OOM/device errors, re-check nvidia-smi before concluding
   +- use CUDA_VISIBLE_DEVICES=$FREE_GPU for GPU selection
```

When debugging the compiler by symptom:

```
syntax error / unexpected            => Scenario 1
type inconsistent / unable to apply    => Scenario 2
invalid span / chunkat                 => Scenario 3
parallel / block / thread              => Scenario 4
DMA / future / wait                    => Scenario 5
memory / buffer / reuse               => Scenario 6
nvcc or codegen errors                 => Scenario 7
unknown pass                           => Scenario 8
vectorize / foreach                    => Scenario 9
unexpected replacement / symbol      => Scenario 10
```

### CUDA debugging (generated executables)

Behavior (same as compiler runs): execute steps without unnecessary confirmation; capture failures; assume `cuda-gdb` / `cuda-memcheck` are allowed. If GPU 0 fails (OOM or no device), retry once with `CUDA_VISIBLE_DEVICES=1` when appropriate.

Regenerate the script if needed:

```bash
./choreo -gs -t cute -arch=sm_90a wip_code.co -o wip_code.cute.result
```

Build with device-friendly debug flags and keep the binary:

```bash
EXTRA_TARGET_CFLAGS="-g -G -O0 -lineinfo" bash wip_code.cute.result --compile-link
```

Locate the generated executable (the compile-link script prints its path). Then:

```bash
/usr/local/cuda/bin/cuda-gdb <path-printed-by-compile-link>
```

Memory and race checks:

```bash
/usr/local/cuda/bin/cuda-memcheck <path-printed-by-compile-link>
```

Notes:

- `-G` adds device debug info; `-O0` simplifies stepping.
- If tools are not on `PATH`, use full paths such as `/usr/local/cuda/bin/cuda-gdb`.
