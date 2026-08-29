"""
Compute occupancy changes from SALA savings.
H100 SM has 228 KB shared memory max per SM.
"""

MAX_SMEM_KB = 228  # H100

configs = [
    # (config, measured_kb, measured_occ_limit_smem, regs)
    ("128x128 BK=64 s=2",   99.34, 2, 156),
    ("128x128 BK=64 s=3",  132.12, 1, 158),
    ("128x128 BK=64 s=4",  164.90, 1, 161),
    ("128x256 BK=64 s=2",  164.88, 1, 158),
    ("128x256 BK=64 s=3",  214.04, 1, 159),
    ("128x128 BK=128 s=2", 164.88, 1, 157),
    ("128x128 BK=128 s=3", 230.42, 1, 159),
    ("64x128 BK=64 s=2",    66.58, 3, 90),
    ("64x128 BK=64 s=3",    91.16, 2, 90),
]

# SALA savings (from previous analysis)
sala_savings = {
    "128x128 BK=64 s=2":   65.0,
    "128x128 BK=64 s=3":   97.0,
    "128x128 BK=64 s=4":  129.0,
    "128x256 BK=64 s=2":   97.0,
    "128x256 BK=64 s=3":  145.0,
    "128x128 BK=128 s=2": 129.0,
    "128x128 BK=128 s=3": 193.0,
    "64x128 BK=64 s=2":    49.0,
    "64x128 BK=64 s=3":    73.0,
}

print(f"{'Config':<25} {'Base':>7} {'SALA':>7} {'Base Occ':>9} {'SALA Occ':>9} {'Change':>8}")
print("-" * 70)

for config, base_kb, base_occ, regs in configs:
    sala_kb = sala_savings[config]
    sala_occ = int(MAX_SMEM_KB / sala_kb)
    change = f"{base_occ}->{sala_occ}" if sala_occ > base_occ else f"{base_occ} (same)"

    print(f"{config:<25} {base_kb:>6.1f} {sala_kb:>6.1f} "
          f"{base_occ:>8d} {sala_occ:>8d}  {change}")
