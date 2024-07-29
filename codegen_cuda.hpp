#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "choreo_cuda_header.inc"
#include "codegen.hpp"
#include "cuda_script.inc"
#include "types.hpp"

namespace Choreo {

namespace CUDA {

inline constexpr const char *backpatch_filename =
    "__choreo_kernel_file_name_that_will_be_back_patched_soon_ok_enough_i_am_"
    "bored__";


inline static void ReplaceInString(std::string &str, const std::string &from,
                                   const std::string &to) {
  if (from.empty()) return;

  size_t startPos = 0;
  while ((startPos = str.find(from, startPos)) != std::string::npos) {
    str.replace(startPos, from.length(), to);
    startPos += to.length();  // In case 'to' contains 'from', like replacing
                              // 'x' with 'yx'
  }
}

inline static std::string create_unique_path() {
  // Get a high-resolution timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();

  // Convert timestamp to a more granular unit, like nanoseconds
  auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

  // Get the thread or process ID
  std::stringstream ss;
  ss << std::this_thread::get_id();
  std::string thread_id = ss.str();

  // Construct the path
  std::string path = "/tmp/" + std::to_string(nanoseconds) + "_" + thread_id;

  return path;
}

// stringify func for Choreo::Storage emited to CUDA
static inline std::string cuda_storage_str(Choreo::Storage s) {
  switch (s) {
    case Storage::LOCAL:
      return "";
    case Storage::SHARED:
      return "__shared__";
    case Storage::GLOBAL:
      return "__global__";
    case Storage::DEFAULT:
      return "";
    default:
      choreo_unreachable();
  }
}

static inline std::string HostTypeString(const Choreo::Type &ty,
                                         bool is_ret = false) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (auto sty = dyn_cast<SpannedType>(&ty)) {
    if (is_ret)  // return by value
      return "choreo::spanned_data<choreo::" + STR(sty->f_type) + ", " +
             std::to_string(sty->Dims()) + ">";
    else  // pass by reference
      return "const choreo::spanned_view<choreo::" + STR(sty->f_type) + ", " +
             std::to_string(sty->Dims()) + "> &";
  }
  choreo_unreachable("unsupported host function type.");
  return "";
}

static inline std::string cuda_type_str(Choreo::BaseType t) {
  switch (t) {
    case BaseType::F32:
      return "float";
      break;
    case BaseType::F16:
      return "half";
      break;
    case BaseType::BF16:
      return "__nv_bfloat16";
      break;
    case BaseType::U32:
      return "uint32_t";
      break;
    case BaseType::S32:
      return "int32_t";
      break;
    case BaseType::U16:
      return "uint16_t";
      break;
    case BaseType::S16:
      return "int16_t";
      break;
    case BaseType::U8:
      return "uint8_t";
      break;
    case BaseType::S8:
      return "int8_t";
      break;
    // should it be passed in?
    case BaseType::INT:
      return "int";
      break;
    case BaseType::BOOL:
      return "bool";
      break;
    default:
      choreo_unreachable("Type '" + STR(t) + "' is not supported.");
  }
}

void EmitCUDAValueList(const ValueList &vl, std::ostream &os) {
  auto print_variant = [&os](const ValueItem &vle) {
    if (vle.index() == 0)
      os << std::get<0>(vle);
    else
      os << std::get<1>(vle);
  };
  os << "{";
  if (!vl.empty()) {
    print_variant(vl[0]);
    for (unsigned i = 1; i < vl.size(); ++i) {
      os << ", ";
      print_variant(vl[i]);
    }
  }
  os << "}";
}

template <typename T>
T GetAt(ValueList vlist, int idx) {
  return *(std::get_if<T>(&vlist[idx]));
};

}  // end namespace CUDA

}  // end namespace Choreo
