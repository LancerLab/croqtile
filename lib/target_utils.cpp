#include "target_utils.hpp"
#include "ast.hpp"

using namespace Choreo;

std::once_flag PlDepthMap::init_flag;
std::unique_ptr<PlDepthMap> PlDepthMap::instance;

const PlDepthMap& PlDepthMap::Get() {
  // the context must be valid
  std::call_once(init_flag,
                 []() { instance = std::make_unique<PlDepthMap>(); });
  return *instance;
}

PlDepthMap::PlDepthMap() {
  int li = 0;
  for (auto& pl : CCtx().TargetParallelLevels()) {
    to_levels.emplace(li, pl);
    to_depths.emplace(pl, li);
    ++li;
  }

  // retrieve the maximums
  for (auto d : to_levels)
    max_depth = (d.first > max_depth) ? d.first : max_depth;

  assert(to_levels.count(max_depth));
  max_level = to_levels[max_depth];
}
