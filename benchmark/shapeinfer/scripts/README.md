# Benchmark Scripts

Evaluation scripts for the Choreo compiler's symbolic system.

## Scripts

### Performance Analysis
- `cost_eval_symbolic_system.sh` - Compare static vs dynamic compilation performance
- `test_cost_eval.sh` - Quick performance test with sample files

### Shape Resolution Analysis
- `shape_resolve_analysis.sh` - Analyze symbolic shape inference capabilities
- `test_shape_analysis.sh` - Quick shape analysis test

## Usage

```bash
# Performance benchmarks
./benchmark/scripts/cost_eval_symbolic_system.sh
./benchmark/scripts/test_cost_eval.sh

# Shape analysis
./benchmark/scripts/shape_resolve_analysis.sh
./benchmark/scripts/test_shape_analysis.sh
```

Results are saved to `benchmark/scripts/results/`.

## Performance Analysis Details

Compares static vs dynamic compilation modes:
- **Static**: `__STATIC_SHAPE__=1` (concrete dimensions)
- **Dynamic**: No flag (symbolic dimensions)

Measures compilation and execution time deltas with statistical analysis.

## Shape Analysis Details

Analyzes `choreo -i` output to categorize shape resolution:
- **Concrete**: `[32, 768]` - fully resolved dimensions
- **Symbolic**: `[32, ::seq_len, 768]` - symbolic dimensions
- **Partial**: `[32, ?, 768]` - unresolved dimensions

Reports success rates and resolution quality metrics.
