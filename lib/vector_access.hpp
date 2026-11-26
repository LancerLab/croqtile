#ifndef __CHOREO_VECTOR_ACCESS_HPP__
#define __CHOREO_VECTOR_ACCESS_HPP__

namespace Choreo {

// Address pattern of an explicit-vector memory access.
enum class VectorAddressKind {
  Contiguous,
  Indirect,
};

inline const char* VectorAddressKindName(VectorAddressKind kind) {
  switch (kind) {
  case VectorAddressKind::Contiguous: return "contiguous";
  case VectorAddressKind::Indirect: return "indirect";
  }
  return "unknown";
}

} // namespace Choreo

#endif // __CHOREO_VECTOR_ACCESS_HPP__
