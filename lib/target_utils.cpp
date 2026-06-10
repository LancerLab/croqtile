#include "target_utils.hpp"
#include "ast.hpp"
#include "context.hpp"

using namespace Choreo;

struct CompilationContext::PlDepthMapHolder {
  PlDepthMap map;
};

const PlDepthMap& PlDepthMap::Get() {
  auto& ctx = CCtx();
  if (!ctx.pl_depth_map_)
    ctx.pl_depth_map_ =
        std::make_shared<CompilationContext::PlDepthMapHolder>();
  return ctx.pl_depth_map_->map;
}

PlDepthMap::PlDepthMap() {
  int li = 0;
  for (auto& pl : CCtx().TargetParallelLevels()) {
    to_levels.emplace(li, pl);
    to_depths.emplace(pl, li);
    ++li;
  }

  for (auto d : to_levels)
    max_depth = (d.first > max_depth) ? d.first : max_depth;

  if (to_levels.count(max_depth)) max_level = to_levels[max_depth];
}
