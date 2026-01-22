#ifndef __CHOREO_CODEGEN_COMMON_H__
#define __CHOREO_CODEGEN_COMMON_H__

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

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_COMMON_H__
