#ifndef __CHOREO_CODEGEN_COMMON_H__
#define __CHOREO_CODEGEN_COMMON_H__

#include "options.hpp"
#include "types.hpp"

namespace Choreo {

extern Option<bool> dma_verbose;

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

inline const char* SingleInstancePredicate(bool shared_in_block = true) {
  if (shared_in_block) return "__CHOREO_SINGLE_SHARED__";
  return "__CHOREO_SINGLE_LOCAL__";
}

inline const std::string ImplicitPred(Storage cur) {
  switch (cur) {
  case Storage::LOCAL: return "__CHOREO_SINGLE_LOCAL__";
  case Storage::SHARED: return "__CHOREO_SINGLE_SHARED__";
  default: choreo_unreachable("unsupported storage level.");
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

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_COMMON_H__
