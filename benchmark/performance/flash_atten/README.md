# flash_atten benchmarks

Hopper flash attention: six comparison **variants** (FA3 + Choreo) and legacy suites in
[`.tmp/`](.tmp/).

- Full variant matrix: [VARIANTS.md](VARIANTS.md)
- Run all FA3 baselines: `bash scripts/run_all_fa3.sh --gpu 1` (default GPU is 1 if omitted)
- Per-variant: `bash <variant>/baselines/bench.sh --gpu 1` or `bash <variant>/compare_all.sh --gpu 1`

## Workload configuration

Each variant has a single `config.py` that defines `BENCHMARK_CONFIGS` -- the source of
truth for both Python baselines and Choreo kernels.

`scripts/gen_bench_configs.py` reads `config.py` and generates a C++ header
(`choreo/build/bench_configs.inc`) that Choreo `.co` files `#include`. This runs
automatically from `bench.sh` before compilation. To change workload shapes, edit
`config.py` only -- both backends pick it up.

## Choreo bench.sh options

```
--kernel PATH           Kernel .co file to build
--output PATH           Generated compile script path
--gpu ID                Bind execution to a specific GPU (default: auto-pick free)
--no-verify             Disable verification
--no-timing             Disable timing output
--profile [ITER]        Profile with ncu (--set full)
--force-compile         Force recompilation even if the output is up-to-date
--help                  Show this message
```

Compilation caching: `bench.sh` skips Choreo recompilation when the output script
is already newer than both the kernel `.co` source and `bench_configs.inc`.

Environment variables:

| Variable | Effect |
|----------|--------|
| `EXTRA_CHOREO_FLAGS` | Extra flags passed to the Choreo compiler |
| `EXTRA_NVCC_FLAGS` | Extra flags appended to `EXTRA_TARGET_CFLAGS` for nvcc |
| `EXTRA_NCU_FLAGS` | Extra flags passed to ncu during `--profile` |
