#ifndef __CHOREO_SYMBOL_INFO_HPP__
#define __CHOREO_SYMBOL_INFO_HPP__

// shared global context for a compilation process

#include <iostream>
#include <map>
#include <sstream>

namespace Choreo {

enum DMABufferKind {
  DOK_UNKNOWN,
  DOK_SYMBOL,
  DOK_CHUNK,
};

inline const char* STR(DMABufferKind dok) {
  switch (dok) {
  case DOK_UNKNOWN: return "UNKNOWN";
  case DOK_SYMBOL: return "SYMBOL";
  case DOK_CHUNK: return "CHUNK";
  default: choreo_unreachable("Unsupported operand kind.");
  }
  return "";
}

struct DMABufferInfo {
  std::string buffer; // buffer name, if explicitly named
  DMABufferKind from_kind = DOK_UNKNOWN;
  DMABufferKind to_kind = DOK_UNKNOWN;
};

// per-function(name) future-buffer info
using FutureBufferInfo = std::map<std::string, DMABufferInfo>;
using FBItemInfo = FutureBufferInfo::value_type;

inline const std::string STR(const FBItemInfo& fbi) {
  std::ostringstream oss;
  oss << "[Future] " << fbi.first << " (buffer: " << fbi.second.buffer << "), "
      << STR(fbi.second.from_kind) << " -> " << STR(fbi.second.to_kind);
  return oss.str();
}

inline const std::string STR(const FutureBufferInfo& fbi) {
  std::ostringstream oss;
  oss << "Future-Buffers:\n";
  for (auto& item : fbi) oss << STR(item) << "\n";
  return oss.str();
}

using FutureBufferMap = std::map<std::string, FutureBufferInfo>;

// per-compilation context
class CompilationContext {
  bool debug_symtab = false;
  FutureBufferMap fb_map;

public:
  FutureBufferInfo& GetFutureBufferInfo(const std::string fname) {
    return fb_map[fname];
  }

  bool DebugSymTab() const { return debug_symtab; }

  static CompilationContext& GetInstance() {
    static CompilationContext instance;
    return instance;
  }
};

inline CompilationContext& CCtx() { return CompilationContext::GetInstance(); }

} // end namespace Choreo
#endif //__CHOREO_SYMBOL_INFO_HPP__
