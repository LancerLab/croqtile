"""
Analyze SALA savings for each Triton configuration.
Pipeline buffers = stages * (BM*BK + BN*BK) * 2 bytes (f16)
Output buffer = BM * BN * 2 bytes
Overhead ~= 1 KB (barriers, alignment)
"""

configs = [
    # (BM, BN, BK, stages, measured_smem_kb)
    (128, 128, 64,  2,  99.34),
    (128, 128, 64,  3, 132.12),
    (128, 128, 64,  4, 164.90),
    (128, 256, 64,  2, 164.88),
    (128, 256, 64,  3, 214.04),
    (128, 128, 128, 2, 164.88),
    (128, 128, 128, 3, 230.42),
    ( 64, 128, 64,  2,  66.58),
    ( 64, 128, 64,  3,  91.16),
]

print(f"{'Config':<25} {'Meas':>8} {'Pipeline':>10} {'Output':>8} {'SALA':>8} {'Save':>6} {'Save%':>6}")
print("-" * 75)

for bm, bn, bk, stages, measured_kb in configs:
    lhs_bytes = bm * bk * 2
    rhs_bytes = bn * bk * 2
    pipeline_kb = stages * (lhs_bytes + rhs_bytes) / 1024
    output_kb = bm * bn * 2 / 1024

    # SALA: output overlaps with pipeline (max of both)
    # But pipeline has stages slots; output only needs 1 slot space
    # Since all pipeline slots are freed before output is used,
    # output can overlap with ALL pipeline slots
    sala_kb = max(pipeline_kb, output_kb) + 1  # 1 KB overhead approx
    # But actually, with separate LHS/RHS pipeline arrays:
    # LHS pipeline = stages * BM * BK * 2
    # RHS pipeline = stages * BN * BK * 2
    # Output can overlap with LHS pipeline (or RHS)
    lhs_pipeline_kb = stages * lhs_bytes / 1024
    rhs_pipeline_kb = stages * rhs_bytes / 1024

    # Best overlap: output shares space with one pipeline array
    # SALA total = max(lhs_pipeline, output) + rhs_pipeline + overhead
    # or         = lhs_pipeline + max(rhs_pipeline, output) + overhead
    option1 = max(lhs_pipeline_kb, output_kb) + rhs_pipeline_kb
    option2 = lhs_pipeline_kb + max(rhs_pipeline_kb, output_kb)
    sala_total = min(option1, option2) + 1  # 1 KB overhead

    save_kb = measured_kb - sala_total
    save_pct = save_kb / measured_kb * 100

    config_str = f"{bm}x{bn} BK={bk} s={stages}"
    print(f"{config_str:<25} {measured_kb:>7.1f} {pipeline_kb:>9.0f} {output_kb:>7.0f} "
          f"{sala_total:>7.1f} {save_kb:>5.1f} {save_pct:>5.1f}%")
