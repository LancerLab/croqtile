#ifndef __CHOREO_GCU_TARGET_HPP__
#define __CHOREO_GCU_TARGET_HPP__

#include "context.hpp"
#include "target.hpp"
#include "target_registry.hpp"

namespace Choreo {

class GCUTarget : public Target {
public:
  size_t GetMemCapacity(const Storage& sto, const ArchId& arch) const override {
    if (!IsArchSupported(arch))
      choreo_unreachable("unsupported architecture '" + arch + "'.");

    if (arch == "gcu200" || arch == "gcu210") {
      switch (sto) {
      case Storage::LOCAL: return 1008ull * 1024;             // 1008KB
      case Storage::SHARED: return 24ull * 1024 * 1024;       // 24MB
      case Storage::GLOBAL: return 4ull * 1024 * 1024 * 1024; // 4GB
      default: choreo_unreachable("Unsupported mem level.");
      }
    } else if (arch == "gcu300") {
      /*
      TODO:
      For GCU3, all is different with Scorpio (1 Die) in the link below
      Is S60G same with c035?
      L3 (global) is different with Dorado (3VG per Cluster) in
      http://wiki.enflame.cn/display/~james.zhu/Enflame+GCU+Programming+Model#EnflameGCUProgrammingModel-get_memory_space
      */
      switch (sto) {
      case Storage::LOCAL: {
        if (Name() == "factor")
          return 1.5 * 1024 * 1024; // 1.5MB
        else if (Name() == "topscc")
          return 1.5 * 1024 * 1024 - 512; // special case
        else
          choreo_unreachable("Unhandled target.");
      } break;
      case Storage::SHARED: {
        if (Name() == "factor")
          return 24ull * 1024 * 1024; // 24MB
        else if (Name() == "topscc")
          return 64ull * 1024 * 1024; // 64MB
        else
          choreo_unreachable("Unhandled target.");
      } break;
      case Storage::GLOBAL: return 40.75 * 1024 * 1024 * 1024; // 40.75GB
      default: choreo_unreachable("unsupported storage level.");
      }
    } else if (arch == "gcu400") {
      assert(Name() == "topscc");
      switch (sto) {
      case Storage::LOCAL: return 1.5 * 1024 * 1024 - 512; // todo: check this
      case Storage::SHARED: return 64ull * 1024 * 1024;    // todo: check this
      case Storage::GLOBAL:
        return 40.75 * 1024 * 1024 * 1024; // TODO: check this
      default: choreo_unreachable("unsupported storage level.");
      }
    } else if (arch == "gcu500") {
      assert(Name() == "topscc");
      switch (sto) {
      case Storage::LOCAL: return 4 * 1024 * 1024 - 512; // todo: check this
      case Storage::SHARED: return 256ull * 1024 * 1024; // todo: check this
      case Storage::GLOBAL:
        return 128ull * 1024 * 1024 * 1024; // todo: check this
      default: choreo_unreachable("unsupported storage level.");
      }
    }
    choreo_unreachable("unsupported target.");
    return 0;
  }
  const ArchId DefaultArch() const override { return "gcu300"; }
  size_t GetMemAlignment(const Storage& sto, const ArchId&) const override {
    switch (sto) {
    case Storage::LOCAL:
    case Storage::SHARED: return 512;
    default: choreo_unreachable("Unsupported mem level.");
    }
    return 0;
  }

  size_t GetMinGroupDim(const ArchId& arch) const override {
    auto arch_num = ArchNum(arch);
    if (arch_num < 400)
      choreo_unreachable("unsupported architecture '" + arch + "'.");
    return 1;
  }

  const std::vector<ParallelLevel>
  GetParallelLevels(const ArchId& arch) const override {
    auto arch_num = ArchNum(arch);
    if (arch_num < 400)
      return {ParallelLevel::SEQ, ParallelLevel::BLOCK, ParallelLevel::THREAD};
    else
      return {ParallelLevel::SEQ, ParallelLevel::BLOCK, ParallelLevel::GROUP,
              ParallelLevel::THREAD};
  }
};

} // end namespace Choreo

#endif // __CHOREO_GCU_TARGET_HPP__
