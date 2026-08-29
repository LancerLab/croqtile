/*
 * Measure CUTLASS SM90 kernels with TMA-based epilogues that use SMEM.
 * This shows the case where SALA would save shared memory.
 *
 * TmaWarpSpecialized / TmaWarpSpecializedCooperative epilogues load C
 * from GMEM to SMEM, apply the epilogue fusion, and store D from SMEM
 * to GMEM via TMA. This creates an epilogue SMEM buffer.
 *
 * Build:
 *   nvcc -std=c++17 -arch=sm_90a -O2 \
 *     -I extern/cutlass/include -I extern/cutlass/tools/util/include \
 *     tools/sala_real_eval/cutlass_smem_tma_epilogue.cu \
 *     -o tools/sala_real_eval/cutlass_smem_tma_epilogue
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

  // SALA savings: if mainloop and epilogue tensor storage can overlap
  size_t sala_total = total - (mt + et) + ((mt > et) ? mt : et);
  // But for non-persistent (union), CUTLASS already does this
  size_t savings = total - sala_total;
  float savings_pct = total > 0 ? (savings * 100.0f / total) : 0;
  int occ_sala = static_cast<int>(228 * 1024 / (sala_total + 1024));

  printf("%-44s %6zu (%5.1fK)  m=%5.1f e=%5.1f  occ=%d",
         name, total, total / 1024.0, mt / 1024.0, et / 1024.0, occ);
  if (savings > 0) {
    printf("  SALA=%5.1fK save=%4.1f%% occ=%d",
           sala_total / 1024.0, savings_pct, occ_sala);
  }
  printf("\n");
}

// --- Cooperative with TMA epilogue ---
// 2 stages
namespace coop_tma_2s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative
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

// 3 stages
namespace coop_tma_3s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative
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

// 4 stages
namespace coop_tma_4s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative
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

// Auto stages
namespace coop_tma_auto {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative
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

// --- Pingpong with TMA epilogue ---
namespace pp_tma_2s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized
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

namespace pp_tma_3s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecialized
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

// Cooperative 128x256 with TMA epi
namespace coop_tma_128x256_2s {
  using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _256, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::TmaWarpSpecializedCooperative
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

// Non-persistent with NoSmem (baseline - union, no epi SMEM)
namespace nonpersist_nosm {
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
    cutlass::gemm::KernelTmaWarpSpecialized
  >::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, MainloopOp, EpilogueOp>;
}

int main() {
  printf("=== CUTLASS SM90 SMEM: NoSmem vs TMA Epilogue (real sizeof) ===\n\n");
  printf("%-44s %s\n",
         "Kernel",
         "Total         main    epi   occ   [SALA savings]");
  printf("%-44s %s\n",
         "--------------------------------------------",
         "---------------------------------------------------");

  printf("\n--- Baselines (NoSmem epilogue, epi=0KB) ---\n");
  report<nonpersist_nosm::Kernel>(
    "NonPersist f16 128x128 4s NoSmemEpi");

  printf("\n--- Cooperative + TMA Epilogue (struct) ---\n");
  report<coop_tma_auto::Kernel>(
    "Coop+TmaEpi f16 128x128 auto");
  report<coop_tma_2s::Kernel>(
    "Coop+TmaEpi f16 128x128 2s");
  report<coop_tma_3s::Kernel>(
    "Coop+TmaEpi f16 128x128 3s");
  report<coop_tma_4s::Kernel>(
    "Coop+TmaEpi f16 128x128 4s");
  report<coop_tma_128x256_2s::Kernel>(
    "Coop+TmaEpi f16 128x256 2s");

  printf("\n--- Pingpong + TMA Epilogue (struct) ---\n");
  report<pp_tma_2s::Kernel>(
    "PP+TmaEpi f16 128x128 2s");
  report<pp_tma_3s::Kernel>(
    "PP+TmaEpi f16 128x128 3s");

  printf("\n");
  printf("Notes:\n");
  printf("  - 228 KB = max dynamic SMEM on H100 SM90a\n");
  printf("  - Occ = floor(228KB / (total + 1KB overhead))\n");
  printf("  - SALA savings = overlap mainloop+epilogue tensor storage\n");
  printf("  - NoSmemEpi: epilogue writes from registers, no SMEM used\n");
  printf("  - TmaEpi: epilogue uses SMEM for C load + D store via TMA\n");

  return 0;
}
