/*
 * SALA CUTLASS Union Test - SharedStorage struct vs union analysis.
 * Measures sizes and runs correctness for cooperative warp-specialized GEMM.
 */

#include <cstdio>
#include <cstdlib>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA = cutlass::half_t;
using LayoutA  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16 / sizeof(ElementA);
using ElementB = cutlass::half_t;
using LayoutB  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16 / sizeof(ElementB);
using ElementC = cutlass::half_t;
using LayoutC  = cutlass::layout::ColumnMajor;
constexpr int AlignmentC = 16 / sizeof(ElementC);
using ElementD = cutlass::half_t;
using LayoutD  = LayoutC;
constexpr int AlignmentD = AlignmentC;
using ElementAccumulator = float;
using ElementCompute     = float;
using ClusterShape = Shape<_1, _1, _1>;
using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

// Build epilogue ONCE (it doesn't depend on stages)
template <typename TileShape_>
using BuildEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

// Build mainloop with explicit carveout
template <typename TileShape_, int Stages>
using BuildMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_, ClusterShape,
    cutlass::gemm::collective::StageCount<Stages>,
    KernelSchedule
>::CollectiveOp;

// Full kernel
template <typename TileShape_, int Stages>
using BuildGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    BuildMainloop<TileShape_, Stages>,
    BuildEpilogue<TileShape_>
>;

template <typename GK>
bool run_gemm(int M, int N, int K) {
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;
    using StrideA = typename GK::StrideA;
    using StrideB = typename GK::StrideB;
    using StrideC = typename GK::StrideC;
    using StrideD = typename GK::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    size_t elems_A = M * K, elems_B = K * N, elems_CD = M * N;
    cutlass::half_t *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, elems_A * 2);
    cudaMalloc(&d_B, elems_B * 2);
    cudaMalloc(&d_C, elems_CD * 2);
    cudaMalloc(&d_D, elems_CD * 2);

    auto* h_A = new cutlass::half_t[elems_A];
    auto* h_B = new cutlass::half_t[elems_B];
    srand(42);
    for (size_t i = 0; i < elems_A; i++) h_A[i] = cutlass::half_t(float(rand() % 10 - 5) / 10.0f);
    for (size_t i = 0; i < elems_B; i++) h_B[i] = cutlass::half_t(float(rand() % 10 - 5) / 10.0f);
    cudaMemcpy(d_A, h_A, elems_A * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, elems_B * 2, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, elems_CD * 2);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {d_A, stride_A, d_B, stride_B},
        {{1.0f, 0.0f}, d_C, stride_C, d_D, stride_D}
    };

    Gemm gemm;
    size_t ws_size = Gemm::get_workspace_size(args);
    void* d_ws = nullptr;
    if (ws_size > 0) cudaMalloc(&d_ws, ws_size);

    auto status = gemm.initialize(args, d_ws);
    if (status != cutlass::Status::kSuccess) {
        printf("  Init FAILED\n");
        delete[] h_A; delete[] h_B;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
        if (d_ws) cudaFree(d_ws);
        return false;
    }

    status = gemm();
    cudaError_t err = cudaDeviceSynchronize();
    bool ok = (status == cutlass::Status::kSuccess && err == cudaSuccess);

    auto* h_D = new cutlass::half_t[elems_CD];
    cudaMemcpy(h_D, d_D, elems_CD * 2, cudaMemcpyDeviceToHost);
    float sum = 0; int nonzero = 0;
    for (size_t i = 0; i < elems_CD; i++) {
        float v = float(h_D[i]); sum += v;
        if (v != 0.0f) nonzero++;
    }
    printf("  GEMM: sum=%.2f nonzero=%d/%zu %s\n", sum, nonzero, elems_CD,
           (ok && nonzero > 0) ? "PASS" : "FAIL");

    delete[] h_A; delete[] h_B; delete[] h_D;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    if (d_ws) cudaFree(d_ws);
    return ok && nonzero > 0;
}

template <typename TileShape_, int Stages>
void analyze_and_run(const char* label, int M, int N, int K) {
    using GK = BuildGemmKernel<TileShape_, Stages>;
    using SS = typename GK::SharedStorage;
    using ML = typename GK::CollectiveMainloop;
    using EP = typename GK::CollectiveEpilogue;

    printf("\n--- %s ---\n", label);

    size_t mainloop_ts = sizeof(typename ML::TensorStorage);
    size_t epilogue_ts = sizeof(typename EP::TensorStorage);
    size_t total_ss    = sizeof(SS);

    printf("  Mainloop TensorStorage = %.1f KB\n", mainloop_ts / 1024.0);
    printf("  Epilogue TensorStorage = %.1f KB\n", epilogue_ts / 1024.0);
    printf("  SharedStorage (struct) = %.1f KB\n", total_ss / 1024.0);

    // Union: max(mainloop, epilogue) instead of sum
    size_t max_ts = (mainloop_ts > epilogue_ts) ? mainloop_ts : epilogue_ts;
    size_t min_ts = (mainloop_ts < epilogue_ts) ? mainloop_ts : epilogue_ts;
    size_t savings = min_ts;  // union saves the smaller one
    size_t new_total = total_ss - savings;
    // Round up to 128-byte boundary
    new_total = (new_total + 127) & ~127;
    savings = total_ss - new_total;

    printf("  SharedStorage (union)  = %.1f KB (est)\n", new_total / 1024.0);
    printf("  Savings: %.1f KB (%.1f%%)\n", savings / 1024.0,
           100.0 * savings / total_ss);

    int smem_per_sm = 228 * 1024;
    int orig_ctas = smem_per_sm / (int)total_ss;
    int new_ctas  = smem_per_sm / (int)new_total;
    printf("  Occupancy: %d -> %d CTAs/SM", orig_ctas, new_ctas);
    if (new_ctas > orig_ctas) printf(" ***IMPROVED***");
    printf("\n");

    run_gemm<GK>(M, N, K);
}

int main() {
    printf("================================================\n");
    printf("SALA CUTLASS Cooperative SharedStorage Analysis\n");
    printf("H100 (SM90a), 228 KB SMEM/SM\n");
    printf("================================================\n");

    using T128 = Shape<_128, _128, _64>;
    using T256 = Shape<_128, _256, _64>;

    analyze_and_run<T128, 2>("128x128x64, 2-stage", 2048, 2048, 2048);
    analyze_and_run<T128, 3>("128x128x64, 3-stage", 2048, 2048, 2048);
    analyze_and_run<T128, 4>("128x128x64, 4-stage", 2048, 2048, 2048);
    analyze_and_run<T256, 2>("128x256x64, 2-stage", 2048, 2048, 2048);
    analyze_and_run<T256, 3>("128x256x64, 3-stage", 2048, 2048, 2048);

    printf("\nDone.\n");
    return 0;
}

#else
int main() { printf("SM90 not supported\n"); return 1; }
#endif
