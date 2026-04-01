#ifndef __CHOREO_CODEGEN_COMMON_H__
#define __CHOREO_CODEGEN_COMMON_H__

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ast.hpp"
#include "options.hpp"
#include "target_utils.hpp"
#include "types.hpp"

extern Choreo::Option<bool> native_f16;
extern Choreo::Option<bool> native_bf16;
extern Choreo::Option<bool> use_pic;
extern Choreo::Option<bool> verbose;
extern Choreo::Option<std::string> target_options;
extern Choreo::Option<bool> use_hetero_tileflow;

namespace Choreo {

extern Option<bool> no_decay_spanview;
extern Option<bool> dma_opt;
extern Option<bool> dma_verbose;
extern Option<bool> tma_cluster_aware;
extern Option<bool> ptx_barrier;
extern Option<bool> use_stmatrix;
extern Option<bool> hoist_offset;
extern Option<bool> hoist_scale;

inline void VerboseDMA(std::ostringstream& os, const std::string& indent,
                       const std::string& from, const std::string& to,
                       const std::string action, const std::string& offset,
                       size_t offcnt, const std::string& suffix = "") {
  if (!dma_verbose) return;

  os << indent << "printf(\"" << from << "->" << to << ", " << action
     << " offset: {";
  for (size_t i = 0; i < offcnt; ++i) {
    if (i > 0) os << ", ";
    os << "%d";
  }
  os << "} " << suffix << "\\n\"";
  if (offcnt > 0) os << ", " << offset;
  os << ");\n";
}

inline const std::string LevelPred(ParallelLevel pl = ParallelLevel::BLOCK,
                                   int dim = -1) {
  switch (pl) {
  case ParallelLevel::BLOCK: return "if (__CHOREO_BLOCK_SINGLE__) ";
  case ParallelLevel::GROUP:
    if (dim == -1)
      return "if (__CHOREO_GROUP_SINGLE__(32)) ";
    else
      return "if (__CHOREO_GROUP_SINGLE__(" + std::to_string(dim) + ")) ";
  case ParallelLevel::THREAD: return ""; // no guard is required
  default: choreo_unreachable("unsupported storage.");
  }
  return "";
}

// buffer is flexible to be declared anywhere place. For example:
//
//   parallel p by 1, parallel q by 1 {
//     shared f32 [1] bs{1};
//     local f32 [1] bl;
//     shared event es[3];
//   }
//
// However, the initialization of 'bs' should be guarded implicitly to guarantee
// 'atomic' initialization. This is especially important for the shared event
// storage.

inline const std::string BufferInitPred(Storage s) {
  switch (s) {
  case Storage::SHARED: return LevelPred(ParallelLevel::BLOCK);
  case Storage::LOCAL:
    if (TargetHasLevel(ParallelLevel::GROUP))
      return LevelPred(ParallelLevel::GROUP);
    else
      return "";
  default: choreo_unreachable("unsupported storage.");
  }
  return "";
}

inline const char* EmitSync(Storage s) {
  switch (s) {
  case Storage::SHARED: return "__syncthreads()";
  case Storage::LOCAL: return "__syncsubthreads()";
  default:
    choreo_unreachable("unsupported storage location for the synchronization.");
  }
  return "";
}

std::string GetAbsPath(const std::filesystem::path& cwd,
                       const std::string& relative_path);

void PrintSubscriptions(std::ostream& os, const std::string& prefix,
                        const std::string& suffix, const ValueList& dims,
                        std::vector<size_t>& indices, size_t depth = 0);

inline void GenerateSubscriptions(std::ostream& os, const std::string& prefix,
                                  const std::string& suffix,
                                  const ValueList& dims) {
  std::vector<size_t> indices(dims.size());
  PrintSubscriptions(os, prefix, suffix, dims, indices);
}

class ScopedSymbolMap {
  using SymbolMap = std::unordered_map<std::string, std::string>;
  std::vector<SymbolMap> host_map;
  std::vector<SymbolMap> device_map;
  bool debug;

public:
  ScopedSymbolMap(bool d = false) : debug(d) {}
  void EnterScope() {
    host_map.push_back({});
    device_map.push_back({});
  }
  void LeaveScope() {
    host_map.pop_back();
    device_map.pop_back();
  }
  void MapHostSymbol(const std::string& csym, const std::string& name) {
    assert(!host_map.back().count(csym) && "symbol existed");
    if (debug)
      dbgs() << "[Host] Map symbol: " << csym << " -> " << name << "\n";
    host_map.back()[csym] = name;
  }
  void MapDeviceSymbol(const std::string& csym, const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    assert(!device_map.back().count(csym) && "symbol existed");
    if (debug)
      dbgs() << "[Device] Map symbol: " << csym << " -> " << name << "\n";
    device_map.back()[csym] = name;
  }
  void MapDeviceSymbolIfNotExist(const std::string& csym,
                                 const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    if (!device_map.back().count(csym)) {
      if (debug)
        dbgs() << "[Device] Map symbol: " << csym << " -> " << name << "\n";
      MapDeviceSymbol(csym, name);
    }
  }
  void DumpHostMap() const {
    dbgs()
        << "==================== Host Map Information ====================\n";
    dbgs() << std::setw(30) << std::left << "Symbol" << std::setw(50)
           << std::left << " -> Host Name" << "\n";
    dbgs()
        << "--------------------------------------------------------------\n";

    for (auto& table : host_map) {
      if (table.empty()) continue;
      for (const auto& entry : table) {
        dbgs() << std::setw(30) << std::left << entry.first << " -> "
               << entry.second << "\n";
      }
    }

    dbgs() << "================================================================"
           << "\n";
  }
  void DumpDeviceMap() const {
    dbgs()
        << "==================== Device Map Information ====================\n";
    dbgs() << std::setw(30) << std::left << "Symbol" << std::setw(50)
           << std::left << " -> Device Name" << "\n";
    dbgs()
        << "----------------------------------------------------------------\n";

    for (auto& table : device_map) {
      if (table.empty()) continue;

      for (const auto& entry : table) {
        dbgs() << std::setw(30) << std::left << entry.first << " -> "
               << entry.second << "\n";
      }
    }

    dbgs() << "================================================================"
           << "\n";
  }
  void RemapDeviceSymbol(const std::string& csym, const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    device_map.back()[csym] = name;
  }
  void RemapHostSymbol(const std::string& csym, const std::string& name) {
    assert(PrefixedWith(csym, "::") && "expect a scoped name.");
    host_map.back()[csym] = name;
  }
  const std::string HostName(const std::string& csym) const {
    for (auto mapit = host_map.rbegin(); mapit != host_map.rend(); ++mapit)
      if (mapit->count(csym)) return (*mapit).at(csym);
    return csym;
  }
  bool HasHostName(const std::string& csym) const {
    for (auto mapit = host_map.rbegin(); mapit != host_map.rend(); ++mapit)
      if (mapit->count(csym)) return true;
    return false;
  }
  const std::string DeviceName(const std::string& csym) const {
    for (auto mapit = device_map.rbegin(); mapit != device_map.rend(); ++mapit)
      if (mapit->count(csym)) return (*mapit).at(csym);
    return csym;
  }
  bool HasDeviceName(const std::string& csym) const {
    for (auto mapit = device_map.rbegin(); mapit != device_map.rend(); ++mapit)
      if (mapit->count(csym)) return true;
    return false;
  }
  const std::string DeviceNameOrNull(const std::string& csym) const {
    for (auto mapit = device_map.rbegin(); mapit != device_map.rend(); ++mapit)
      if (mapit->count(csym)) return (*mapit).at(csym);
    return "";
  }
};

struct UniqueNamer {
  static int& get(const char* name) {
    static std::unordered_map<std::string, int> counters;
    return counters[name];
  }
  static std::string gen(const char* name, const std::string& prefix) {
    return prefix + std::to_string(++get(name));
  }
};

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_COMMON_H__
