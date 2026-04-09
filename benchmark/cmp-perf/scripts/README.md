# Compile-Time Performance Benchmark Scripts

Evaluation scripts for the Choreo compiler's compile-time performance,
comparing static (concrete shape) vs dynamic (symbolic shape) compilation.

## Scripts

### Compile-Time Analysis
- `compile_time_bench.sh` - Full compile-time benchmark across all shapeinfer categories
- `compile_time_bench_perf.sh` - Compile-time benchmark for `benchmark/performance` files
- `cost_eval_symbolic_system.sh` - Compare static vs dynamic compilation + execution performance
- `test_cost_eval.sh` - Quick performance test with sample files

### Shape Resolution Analysis
- `shape_resolve_analysis.sh` - Analyze symbolic shape inference capabilities
- `test_shape_analysis.sh` - Quick shape analysis test

## Usage

```bash
# From repo root:

# Compile-time benchmarks (compilation-only, no GPU needed)
bash benchmark/cmp-perf/scripts/compile_time_bench.sh
bash benchmark/cmp-perf/scripts/compile_time_bench_perf.sh

# Full benchmarks (compilation + execution, needs GPU)
bash benchmark/cmp-perf/scripts/cost_eval_symbolic_system.sh
bash benchmark/cmp-perf/scripts/test_cost_eval.sh

# Shape analysis
bash benchmark/cmp-perf/scripts/shape_resolve_analysis.sh
bash benchmark/cmp-perf/scripts/test_shape_analysis.sh
```

Results are saved to `benchmark/cmp-perf/scripts/results/`.

## Performance Analysis Details

Compares static vs dynamic compilation modes:
- **Static**: `-D__STATIC_SHAPE__=1` (concrete dimensions via preprocessor flag)
- **Dynamic**: No flag (symbolic dimensions)

Uses `-tp` (time-passes) flag for per-pass phase timing analysis.

## Shape Analysis Details

Analyzes `choreo -i` output to categorize shape resolution:
- **Concrete**: `[32, 768]` - fully resolved dimensions
- **Symbolic**: `[32, ::seq_len, 768]` - symbolic dimensions
- **Partial**: `[32, ?, 768]` - unresolved dimensions

Reports success rates and resolution quality metrics.
