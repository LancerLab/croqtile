static mha_helper::BenchConfig kBenchConfigs[] = {
    {1, 16, 16384, 16384, true, "B=1 H=16 SEQ=16384 causal prefill"},
};
static constexpr int kBenchConfigCount = sizeof(kBenchConfigs) / sizeof(kBenchConfigs[0]);
