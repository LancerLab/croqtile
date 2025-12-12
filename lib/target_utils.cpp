#include "target_utils.hpp"
#include "ast.hpp"

using namespace Choreo;

std::once_flag PlDepthMap::init_flag;
std::unique_ptr<PlDepthMap> PlDepthMap::instance;

const PlDepthMap& PlDepthMap::Get() {
  std::call_once(init_flag,
                 []() { instance = std::make_unique<PlDepthMap>(); });
  return *instance;
}

static std::unordered_map<int, ParallelLevel> gcu5_levels = {
    {0, ParallelLevel::SEQ},
    {1, ParallelLevel::BLOCK},
    {2, ParallelLevel::GROUP},
    {3, ParallelLevel::THREAD},
};

static std::unordered_map<ParallelLevel, int> gcu5_depths = {
    {ParallelLevel::SEQ, 0},
    {ParallelLevel::BLOCK, 1},
    {ParallelLevel::GROUP, 2},
    {ParallelLevel::THREAD, 3},
};

static std::unordered_map<int, ParallelLevel> gcu4_levels = {
    {0, ParallelLevel::SEQ},
    {1, ParallelLevel::BLOCK},
    {2, ParallelLevel::GROUP},
    {3, ParallelLevel::THREAD},
};

static std::unordered_map<ParallelLevel, int> gcu4_depths = {
    {ParallelLevel::SEQ, 0},
    {ParallelLevel::BLOCK, 1},
    {ParallelLevel::GROUP, 2},
    {ParallelLevel::THREAD, 3},
};

static std::unordered_map<int, ParallelLevel> gcu3_levels = {
    {0, ParallelLevel::SEQ},
    {1, ParallelLevel::BLOCK},
    {2, ParallelLevel::THREAD},
};

static std::unordered_map<ParallelLevel, int> gcu3_depths = {
    {ParallelLevel::SEQ, 0},
    {ParallelLevel::BLOCK, 1},
    {ParallelLevel::THREAD, 2},
};

static std::unordered_map<int, ParallelLevel> gcu3_mpi_levels = {
    {0, ParallelLevel::SEQ},
    {1, ParallelLevel::TERM},
    {2, ParallelLevel::BLOCK},
    {3, ParallelLevel::THREAD},
};

static std::unordered_map<ParallelLevel, int> gcu3_mpi_depths = {
    {ParallelLevel::SEQ, 0},
    {ParallelLevel::TERM, 1},
    {ParallelLevel::BLOCK, 2},
    {ParallelLevel::THREAD, 3},
};

static std::unordered_map<int, ParallelLevel> gpu_simple_levels = {
    {0, ParallelLevel::SEQ},
    {1, ParallelLevel::BLOCK},
    {2, ParallelLevel::THREAD},
};

static std::unordered_map<ParallelLevel, int> gpu_simple_depths = {
    {ParallelLevel::SEQ, 0},
    {ParallelLevel::BLOCK, 1},
    {ParallelLevel::THREAD, 2},
};

static std::unordered_map<int, ParallelLevel> wmma_levels = {
    {0, ParallelLevel::SEQ},
    {1, ParallelLevel::BLOCK},
    {2, ParallelLevel::GROUP},
    {3, ParallelLevel::THREAD},
};

static std::unordered_map<ParallelLevel, int> wmma_depths = {
    {ParallelLevel::SEQ, 0},
    {ParallelLevel::BLOCK, 1},
    {ParallelLevel::GROUP, 2},
    {ParallelLevel::THREAD, 3},
};

static std::unordered_map<int, ParallelLevel> wgmma_levels = {
    {0, ParallelLevel::SEQ},     {1, ParallelLevel::BLOCK},
    {2, ParallelLevel::GROUPx4}, {3, ParallelLevel::GROUP},
    {4, ParallelLevel::THREAD},
};

static std::unordered_map<ParallelLevel, int> wgmma_depths = {
    {ParallelLevel::SEQ, 0},    {ParallelLevel::BLOCK, 1},
    {ParallelLevel::GROUP, 2},  {ParallelLevel::GROUPx4, 3},
    {ParallelLevel::THREAD, 4},
};

PlDepthMap::PlDepthMap() {
  if (CCtx().GetTarget() == CompileTarget::MPI) {
    if (CCtx().GetTarget() == CompileTarget::Topscc) {
      if (CCtx().GetArch() == TargetArch::GCU3) {
        to_levels = &gcu3_mpi_levels;
        to_depths = &gcu3_mpi_depths;
      } else
        choreo_unreachable("unsupported target.");
    } else
      choreo_unreachable("unsupported target.");
  } else if ((CCtx().GetTarget() == CompileTarget::Topscc) ||
             (CCtx().GetTarget() == CompileTarget::Factor)) {
    if (CCtx().GetArch() == TargetArch::GCU5) {
      to_levels = &gcu5_levels;
      to_depths = &gcu5_depths;
    } else if (CCtx().GetArch() == TargetArch::GCU4) {
      to_levels = &gcu4_levels;
      to_depths = &gcu4_depths;
    } else if (CCtx().GetArch() == TargetArch::GCU3 ||
               CCtx().GetArch() == TargetArch::GCU21 ||
               CCtx().GetArch() == TargetArch::GCU20) {
      to_levels = &gcu3_levels;
      to_depths = &gcu3_depths;
    } else
      choreo_unreachable("unsupported target.");
  } else if (CCtx().GetTarget() == CompileTarget::CUDA) {
    to_levels = &gpu_simple_levels;
    to_depths = &gpu_simple_depths;
  } else if (CCtx().GetTarget() == CompileTarget::Cute) {
#if 0
    if (CCtx().TargetSupportWGMMA()) {
      to_levels = &wgmma_levels;
      to_depths = &wgmma_depths;
    } else {
      to_levels = &wmma_levels;
      to_depths = &wmma_depths;
    }
#else
    to_levels = &wmma_levels;
    to_depths = &wmma_depths;
#endif
  } else
    choreo_unreachable("unsupported target.");

  // retrieve the maximums
  for (auto d : *to_levels)
    max_depth = (d.first > max_depth) ? d.first : max_depth;

  assert(to_levels->count(max_depth));
  max_level = (*to_levels)[max_depth];
}
