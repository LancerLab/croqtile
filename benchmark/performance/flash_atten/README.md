# flash_atten benchmarks

Hopper flash attention: six comparison **variants** (FA3 + Choreo) and legacy suites in
[`.tmp/`](.tmp/).

- Full variant matrix: [VARIANTS.md](VARIANTS.md)
- Run all FA3 baselines: `bash run_all_fa3.sh --gpu 1` (default GPU is 1 if omitted)
- Per-variant: `bash <variant>/baselines/bench.sh --gpu 1` or `bash <variant>/compare_all.sh --gpu 1`
