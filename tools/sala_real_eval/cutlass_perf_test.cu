/*
 * SALA CUTLASS Performance Test - struct vs union TFLOPS comparison.
 * Measures wall-clock kernel time using CUDA events, computes TFLOPS.
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

template <typename TileShape_, int Stages>
using BuildGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    BuildMainloop<TileShape_, Stages>,
    BuildEpilogue<TileShape_>
>;

template <typename GK>
float benchmark_gemm(int M, int N, int K, int warmup, int iters) {
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GK>;
    using StrideA = typename GK::StrideA;
    using StrideB = typename GK::StrideB;
    using StrideC = typename GK::StrideC;
    using StrideD = typename GK::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    size_t elems_A = (size_t)M * K, elems_B = (size_t)K * N, elems_CD = (size_t)M * N;
    cutlass::half_t *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, elems_A * 2);
    cudaMalloc(&d_B, elems_B * 2);
    cudaMalloc(&d_C, elems_CD * 2);
    cudaMalloc(&d_D, elems_CD * 2);

    // Initialize with random data
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
        return -1.0f;
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        gemm();
    }
    cudaDeviceSynchronize();

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        gemm();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_A; delete[] h_B;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    if (d_ws) cudaFree(d_ws);

    return (float)tflops;
}

template <typename TileShape_, int Stages>
void bench_config(const char* label, int M, int N, int K) {
    using GK = BuildGemmKernel<TileShape_, Stages>;
    using SS = typename GK::SharedStorage;
    using ML = typename GK::CollectiveMainloop;
    using EP = typename GK::CollectiveEpilogue;

    size_t total_ss = sizeof(SS);
    int smem_per_sm = 228 * 1024;
    int ctas = smem_per_sm / (int)total_ss;

    printf("%-30s  SMEM=%.1f KB  CTAs=%d  ", label, total_ss / 1024.0, ctas);
    fflush(stdout);

    int warmup = 10;
    int iters = 50;

    // Multiple sizes
    int sizes[][3] = {
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {8192, 8192, 8192}
    };

    for (auto& sz : sizes) {
        float tflops = benchmark_gemm<GK>(sz[0], sz[1], sz[2], warmup, iters);
        printf("%dx%dx%d: %.1f TFLOPS  ", sz[0], sz[1], sz[2], tflops);
        fflush(stdout);
    }
    printf("\n");
}

int main() {
    printf("=============================================================\n");
    printf("SALA CUTLASS Performance: struct vs union\n");
    printf("H100 (SM90a), 228 KB SMEM/SM, F16 Cooperative\n");
    printf("=============================================================\n\n");

    using T128 = Shape<_128, _128, _64>;

    bench_config<T128, 2>("128x128x64, 2-stg", 0, 0, 0);
    bench_config<T128, 3>("128x128x64, 3-stg", 0, 0, 0);
    bench_config<T128, 4>("128x128x64, 4-stg", 0, 0, 0);

    printf("\nDone.\n");
    return 0;
}

#else
int main() { printf("SM90 not supported\n"); return 1; }
#endif
