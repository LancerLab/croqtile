/*
 * Measure actual sizeof(SharedStorage) for CUTLASS SM90 warp-specialized
 * GEMM kernels - real template instantiation, not hypothetical.
 *
 * Build:
 *   nvcc -std=c++17 -arch=sm_90a -O2 \
 *     -I extern/cutlass/include -I extern/cutlass/tools/util/include \
 *     tools/sala_real_eval/cutlass_smem_measure.cu \
 *     -o tools/sala_real_eval/cutlass_smem_measure
 */

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
#define CUTLASS_ARCH_MMA_SM90_SUPPORTED 1
#endif

#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

using namespace cute;

template <typename GemmKernel>
void report(const char* name) {
  using SS = typename GemmKernel::SharedStorage;
  using MainTS = typename GemmKernel::CollectiveMainloop::TensorStorage;
  using EpiTS = typename GemmKernel::CollectiveEpilogue::TensorStorage;

  size_t total = sizeof(SS);
  size_t mt = sizeof(MainTS);
  size_t et = sizeof(EpiTS);
  int occ = static_cast<int>(228 * 1024 / (total + 1024));

  printf("%-48s  %6zu B (%5.1f KB)  main=%5.1f  epi=%5.1f  occ=%d\n",
         name, total, total / 1024.0, mt / 1024.0, et / 1024.0, occ);
}

// Pattern: build epilogue FIRST, then mainloop with carveout

// --- Non-persistent: KernelTmaWarpSpecialized (union) ---
namespace nonpersist_128x128x64 {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::NoSmemWarpSpecialized
  >::CollectiveOp;

  using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename EpilogueOp::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

// --- Cooperative: KernelTmaWarpSpecializedCooperative (struct) ---
// Auto stages
namespace coop_128x128x64_auto {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename EpilogueOp::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

// Cooperative 2 stages
namespace coop_128x128x64_2s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCount<2>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

// Cooperative 3 stages
namespace coop_128x128x64_3s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

// Cooperative 4 stages
namespace coop_128x128x64_4s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

// --- Pingpong: KernelTmaWarpSpecializedPingpong (struct) ---
// Pingpong 2 stages - NoSmem epilogue
namespace pp_128x128x64_2s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::NoSmemWarpSpecialized
  >::CollectiveOp;

  using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCount<2>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

// Pingpong 3 stages
namespace pp_128x128x64_3s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::NoSmemWarpSpecialized
  >::CollectiveOp;

  using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCount<3>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

// Pingpong 4 stages
namespace pp_128x128x64_4s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::NoSmemWarpSpecialized
  >::CollectiveOp;

  using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

// Cooperative 128x256x64 2 stages
namespace coop_128x256x64_2s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _256, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    Shape<_128, _256, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCount<2>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

// Pingpong auto stages
namespace pp_128x128x64_auto {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::NoSmemWarpSpecialized
  >::CollectiveOp;

  using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename EpilogueOp::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

int main() {
  printf("=== CUTLASS SM90 Warp-Specialized SMEM (real sizeof) ===\n\n");
  printf("%-48s  %s\n",
         "Kernel",
         "SharedStorage       Mainloop  Epilogue  Occ");
  printf("%-48s  %s\n",
         "------------------------------------------------",
         "-----------------------------------------------");

  report<nonpersist_128x128x64::Kernel>(
    "NonPersist f16 128x128x64 auto (union)");

  printf("\n--- Cooperative (struct: no overlap) ---\n");
  report<coop_128x128x64_auto::Kernel>(
    "Coop f16 128x128x64 auto");
  report<coop_128x128x64_2s::Kernel>(
    "Coop f16 128x128x64 2s");
  report<coop_128x128x64_3s::Kernel>(
    "Coop f16 128x128x64 3s");
  report<coop_128x128x64_4s::Kernel>(
    "Coop f16 128x128x64 4s");
  report<coop_128x256x64_2s::Kernel>(
    "Coop f16 128x256x64 2s");

  printf("\n--- Pingpong (struct: no overlap) ---\n");
  report<pp_128x128x64_auto::Kernel>(
    "PP f16 128x128x64 auto");
  report<pp_128x128x64_2s::Kernel>(
    "PP f16 128x128x64 2s");
  report<pp_128x128x64_3s::Kernel>(
    "PP f16 128x128x64 3s");
  report<pp_128x128x64_4s::Kernel>(
    "PP f16 128x128x64 4s");

  printf("\n=== SALA potential savings ===\n");
  printf("For persistent kernels, SALA would overlap epilogue with\n");
  printf("mainloop buffers after signal-aware HB analysis proves safety.\n");
  printf("SALA savings = min(MainloopTensor, EpilogueTensor) bytes.\n");
  printf("228 KB = max SMEM on H100 SM90a.\n");
  printf("Occ = floor(228KB / (total + 1KB driver overhead)).\n");

  return 0;
}
