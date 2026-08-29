#!/usr/bin/env python3
"""
CUTLASS SM90 Warp-Specialized Kernel SMEM Layout Analysis.

Analyzes CUTLASS's three warp-specialized kernel schedules to show:
1. How CUTLASS manages shared memory manually (union vs struct)
2. Where SALA would identify safe overlap in persistent kernels
3. Quantifies the SMEM overhead of conservative allocation

This is a static analysis based on CUTLASS's known template structure,
not runtime profiling. It produces data for the paper's cross-compiler
evaluation table.
"""

import math


def compute_mainloop_smem(M, N, K, stages, dtype_bytes=2):
    """Compute mainloop A+B shared memory for a given tile and pipeline depth."""
    smem_A = M * K * dtype_bytes * stages
    smem_B = K * N * dtype_bytes * stages
    return smem_A, smem_B


def compute_epilogue_smem(M, N, stages_c=1, stages_d=1, dtype_bytes=2,
                          reuse_c=False):
    """Compute epilogue C+D shared memory."""
    smem_C = M * N * dtype_bytes * stages_c
    if reuse_c:
        smem_D = 0  # D overlaps with C
    else:
        smem_D = M * N * dtype_bytes * stages_d
    return smem_C, smem_D


def compute_pipeline_barrier_smem(stages, per_stage_bytes=128):
    """Approximate barrier metadata per pipeline stage (mbarrier + padding)."""
    return stages * per_stage_bytes


def analyze_kernel_schedule(name, M, N, K, stages, dtype_bytes=2,
                            persistent=False, num_math_wgs=1,
                            epilogue_stages_c=1, epilogue_stages_d=1,
                            reuse_smem_c=False):
    """Analyze one CUTLASS kernel configuration."""

    smem_A, smem_B = compute_mainloop_smem(M, N, K, stages, dtype_bytes)
    mainloop_tensor = smem_A + smem_B

    smem_C, smem_D = compute_epilogue_smem(M, N, epilogue_stages_c,
                                            epilogue_stages_d, dtype_bytes,
                                            reuse_smem_c)
    epilogue_tensor = smem_C + smem_D

    pipeline_meta = compute_pipeline_barrier_smem(stages)
    epi_pipeline_meta = compute_pipeline_barrier_smem(epilogue_stages_c)

    if persistent:
        # struct: both mainloop and epilogue allocated simultaneously
        cutlass_total = mainloop_tensor + epilogue_tensor
        overlap_method = "struct (no overlap)"

        # SALA analysis: could we overlap?
        # In persistent kernels, the epilogue of tile N runs concurrently
        # with the mainloop of tile N+1. However, SALA can analyze the
        # signal ordering:
        #   - OrderedSequenceBarrier ensures mainloop load completes
        #     before epilogue load starts
        #   - Pipeline barriers ensure consumer MMA finishes before
        #     epilogue store
        #   - For ping-pong: MathWarpGroupOrderBarrier serializes
        #     MMA and epilogue between Consumer0/Consumer1
        #
        # Result: with careful HB analysis, some overlap is possible
        # when the epilogue output uses different SMEM than mainloop input

        # Conservative SALA: overlap epilogue C-load with unused mainloop
        # stages. In practice, this means the last stage of A/B can overlap
        # with the first stage of epilogue C.
        sala_overlap_possible = min(smem_C, smem_A + smem_B) // stages
        sala_total = cutlass_total - sala_overlap_possible

        # Aggressive SALA: if epilogue D uses stmatrix (register->smem),
        # its SMEM can fully overlap with mainloop A+B after mainloop tail
        if reuse_smem_c:
            aggressive_overlap = min(epilogue_tensor, mainloop_tensor)
        else:
            aggressive_overlap = min(smem_C, mainloop_tensor // stages)
        aggressive_sala_total = cutlass_total - aggressive_overlap

    else:
        # union: CUTLASS already overlaps mainloop and epilogue
        cutlass_total = max(mainloop_tensor, epilogue_tensor)
        overlap_method = "union (full overlap)"
        sala_total = cutlass_total  # SALA matches CUTLASS here
        aggressive_sala_total = cutlass_total
        sala_overlap_possible = 0

    return {
        "name": name,
        "tile": f"{M}x{N}x{K}",
        "stages": stages,
        "persistent": persistent,
        "num_math_wgs": num_math_wgs,
        "smem_A_kb": smem_A / 1024,
        "smem_B_kb": smem_B / 1024,
        "smem_C_kb": smem_C / 1024,
        "smem_D_kb": smem_D / 1024,
        "mainloop_tensor_kb": mainloop_tensor / 1024,
        "epilogue_tensor_kb": epilogue_tensor / 1024,
        "cutlass_total_kb": cutlass_total / 1024,
        "cutlass_method": overlap_method,
        "sala_total_kb": sala_total / 1024,
        "sala_aggressive_kb": aggressive_sala_total / 1024,
        "sala_overlap_kb": sala_overlap_possible / 1024,
        "reduction_vs_cutlass_pct": (
            (cutlass_total - sala_total) / cutlass_total * 100
            if cutlass_total > 0 else 0),
        "aggressive_reduction_pct": (
            (cutlass_total - aggressive_sala_total) / cutlass_total * 100
            if cutlass_total > 0 else 0),
        "pipeline_meta_kb": pipeline_meta / 1024,
    }


def print_report(results):
    print("=" * 80)
    print("CUTLASS SM90 Warp-Specialized Kernel SMEM Analysis")
    print("=" * 80)

    print(f"\n{'Schedule':<35} {'Tile':<12} {'Stg':>3} "
          f"{'Mainloop':>8} {'Epilogue':>8} {'CUTLASS':>8} {'Method':<20} "
          f"{'SALA':>8} {'Save%':>6}")
    print("-" * 120)

    for r in results:
        print(f"{r['name']:<35} {r['tile']:<12} {r['stages']:>3} "
              f"{r['mainloop_tensor_kb']:>7.0f}K {r['epilogue_tensor_kb']:>7.0f}K "
              f"{r['cutlass_total_kb']:>7.0f}K {r['cutlass_method']:<20} "
              f"{r['sala_total_kb']:>7.0f}K {r['reduction_vs_cutlass_pct']:>5.1f}%")

    print("\n" + "=" * 80)
    print("Detailed Analysis")
    print("=" * 80)

    for r in results:
        print(f"\n--- {r['name']} ---")
        print(f"  Tile: {r['tile']}, Stages: {r['stages']}, "
              f"Persistent: {r['persistent']}, Math WGs: {r['num_math_wgs']}")
        print(f"  Mainloop SMEM: A={r['smem_A_kb']:.0f}KB + B={r['smem_B_kb']:.0f}KB "
              f"= {r['mainloop_tensor_kb']:.0f}KB")
        print(f"  Epilogue SMEM: C={r['smem_C_kb']:.0f}KB + D={r['smem_D_kb']:.0f}KB "
              f"= {r['epilogue_tensor_kb']:.0f}KB")
        print(f"  CUTLASS allocation: {r['cutlass_total_kb']:.0f}KB ({r['cutlass_method']})")
        print(f"  SALA conservative: {r['sala_total_kb']:.0f}KB "
              f"(saves {r['reduction_vs_cutlass_pct']:.1f}%)")
        print(f"  SALA aggressive:   {r['sala_aggressive_kb']:.0f}KB "
              f"(saves {r['aggressive_reduction_pct']:.1f}%)")
        if r['persistent']:
            print(f"  SALA overlap budget: {r['sala_overlap_kb']:.0f}KB reclaimable "
                  f"via signal-aware analysis")
            print(f"  Pipeline barrier metadata: {r['pipeline_meta_kb']:.0f}KB")

    # Summary table for paper
    print("\n" + "=" * 80)
    print("LaTeX Table Data (for paper)")
    print("=" * 80)
    print()

    persistent_results = [r for r in results if r['persistent']]
    non_persistent_results = [r for r in results if not r['persistent']]

    if non_persistent_results:
        print("Non-persistent (CUTLASS already unions -- SALA matches):")
        for r in non_persistent_results:
            print(f"  {r['name']}: {r['cutlass_total_kb']:.0f}KB "
                  f"(CUTLASS = SALA, manual union)")

    if persistent_results:
        print("\nPersistent (CUTLASS uses struct -- SALA improves):")
        for r in persistent_results:
            print(f"  {r['name']}: CUTLASS={r['cutlass_total_kb']:.0f}KB "
                  f"-> SALA={r['sala_total_kb']:.0f}KB "
                  f"({r['reduction_vs_cutlass_pct']:.1f}% reduction)")
            print(f"    Aggressive: SALA={r['sala_aggressive_kb']:.0f}KB "
                  f"({r['aggressive_reduction_pct']:.1f}% reduction)")


def main():
    configs = [
        # Non-persistent: KernelTmaWarpSpecialized (1P + 1C)
        analyze_kernel_schedule(
            "WarpSpec (non-persistent, f16)",
            M=128, N=128, K=64, stages=4, dtype_bytes=2,
            persistent=False, num_math_wgs=1,
            epilogue_stages_c=1, epilogue_stages_d=1),

        # Persistent cooperative: KernelTmaWarpSpecializedCooperative (1P + 2C)
        analyze_kernel_schedule(
            "Cooperative persistent (f16)",
            M=128, N=128, K=64, stages=4, dtype_bytes=2,
            persistent=True, num_math_wgs=2,
            epilogue_stages_c=2, epilogue_stages_d=2),

        # Persistent ping-pong: KernelTmaWarpSpecializedPingpong (1P + 2C)
        analyze_kernel_schedule(
            "PingPong persistent (f16)",
            M=128, N=256, K=64, stages=4, dtype_bytes=2,
            persistent=True, num_math_wgs=2,
            epilogue_stages_c=2, epilogue_stages_d=2),

        # FP8 persistent cooperative
        analyze_kernel_schedule(
            "Cooperative persistent (fp8)",
            M=128, N=128, K=128, stages=4, dtype_bytes=1,
            persistent=True, num_math_wgs=2,
            epilogue_stages_c=2, epilogue_stages_d=2),

        # Large tile ping-pong
        analyze_kernel_schedule(
            "PingPong persistent large (f16)",
            M=256, N=128, K=64, stages=3, dtype_bytes=2,
            persistent=True, num_math_wgs=2,
            epilogue_stages_c=2, epilogue_stages_d=1,
            reuse_smem_c=True),

        # Non-persistent FP8
        analyze_kernel_schedule(
            "WarpSpec (non-persistent, fp8)",
            M=128, N=128, K=128, stages=5, dtype_bytes=1,
            persistent=False, num_math_wgs=1,
            epilogue_stages_c=1, epilogue_stages_d=1),

        # Cooperative with ReuseSmemC
        analyze_kernel_schedule(
            "Cooperative ReuseC (f16)",
            M=128, N=128, K=64, stages=4, dtype_bytes=2,
            persistent=True, num_math_wgs=2,
            epilogue_stages_c=2, epilogue_stages_d=2,
            reuse_smem_c=True),
    ]

    print_report(configs)


if __name__ == "__main__":
    main()
