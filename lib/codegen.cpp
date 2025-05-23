#include "codegen.hpp"

namespace Choreo {

Storage GCUDeviceParallelLevel(int pl_depth) {
  static std::unordered_map<int, Storage> levels = {
      {0, Storage::GLOBAL},
      {1, Storage::SHARED},
      {2, Storage::LOCAL},
      {3, Storage::SUB},
  };

  if (!levels.count(pl_depth) ||
      (pl_depth == 3 && (CCtx().GetArch() != TargetArch::GCU4)))
    return Storage::NONE;

  return levels[pl_depth];
}

int GCUDeviceParallelDepth(Storage l) {
  static std::unordered_map<Storage, int> levels = {
      {Storage::GLOBAL, 0},
      {Storage::SHARED, 1},
      {Storage::LOCAL, 2},
      {Storage::SUB, 3},
  };
  if (!levels.count(l) ||
      (l == Storage::SUB && (CCtx().GetArch() != TargetArch::GCU4)))
    return -1;

  return levels[l];
}

} // end namespace Choreo
