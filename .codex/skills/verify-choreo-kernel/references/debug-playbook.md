# Verify Kernel Debug Playbook

## Compile fails

- If error includes `Compile Target 'xxx' is invalid`, run:
  - `./build/choreo --help-target`
- If error includes `unsupported architecture` or `group-4 ... not supported`, try:
  - `-t cute -arch=sm_90a` for SM90 benchmark kernels.
- If frontend/type issues are suspected, run:
  - `./build/choreo -i <file.co>`

## Runtime fails

- If log includes `CUDA failure`, first verify runtime environment:
  - `nvidia-smi`
- If log includes `values are not equal`, it is a correctness mismatch:
  - Re-run with `--skip-verify` to separate runtime stability from numerical checking.
  - Dump generated source:
    - `./build/choreo -es <file.co> -o -`
  - Re-check inferred types:
    - `./build/choreo -i <file.co>`

## Success criteria

- Exit code is `0`.
- Runtime output contains `Test Passed`.
