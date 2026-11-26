#ifndef __CHOREO_VECTOR_LAYOUT_HPP__
#define __CHOREO_VECTOR_LAYOUT_HPP__

#include <cstddef>
#include <limits>
#include <optional>

namespace Choreo {

enum class VectorWidthClass {
  Partial,
  Fixed1,
  Fixed2,
  Fixed4,
};

inline const char* VectorWidthClassName(VectorWidthClass width) {
  switch (width) {
  case VectorWidthClass::Partial: return "partial";
  case VectorWidthClass::Fixed1: return "fixed_1";
  case VectorWidthClass::Fixed2: return "fixed_2";
  case VectorWidthClass::Fixed4: return "fixed_4";
  }
  return "unknown";
}

// Vector register layout used by code generation. A logical vector may
// occupy part of one vector register, or exactly two/four vector registers.
struct VectorRegisterLayout {
  // Number of vector registers occupied by the logical vector.
  size_t vector_register_count;
  // Total element slots across those registers, including padding.
  size_t physical_elements;

  // Elements in one vector register.
  size_t ElementsPerRegister() const {
    return vector_register_count ? physical_elements / vector_register_count
                                 : 0;
  }

  static std::optional<VectorRegisterLayout> Get(size_t vector_register_bytes,
                                                 size_t element_bytes,
                                                 size_t logical_lanes) {
    if (!vector_register_bytes || !element_bytes || !logical_lanes ||
        vector_register_bytes % element_bytes != 0)
      return std::nullopt;
    size_t lanes = vector_register_bytes / element_bytes;
    if (logical_lanes <= lanes) return VectorRegisterLayout{1, lanes};
    if (logical_lanes == 2 * lanes) return VectorRegisterLayout{2, 2 * lanes};
    if (logical_lanes == 4 * lanes) return VectorRegisterLayout{4, 4 * lanes};
    return std::nullopt;
  }
};

struct VectorRegisterLayoutInfo {
  // Number of vector registers occupied by the logical vector.
  size_t vector_register_count = 0;
  // Total element slots across those registers, including padding.
  size_t physical_elements = 0;
  size_t element_bytes = 0;
  size_t logical_lanes = 0;
  size_t logical_bytes = 0;
  VectorWidthClass width_class = VectorWidthClass::Partial;

  // Elements in one vector register.
  size_t ElementsPerRegister() const {
    return vector_register_count ? physical_elements / vector_register_count
                                 : 0;
  }

  bool IsFullWidth() const { return width_class != VectorWidthClass::Partial; }
};

inline std::optional<VectorRegisterLayoutInfo>
GetVectorRegisterLayoutInfo(size_t vector_register_bytes, size_t element_bytes,
                            size_t logical_lanes) {
  if (!vector_register_bytes || !element_bytes || !logical_lanes ||
      vector_register_bytes % element_bytes != 0)
    return std::nullopt;

  const size_t base_lanes = vector_register_bytes / element_bytes;
  if (logical_lanes > std::numeric_limits<size_t>::max() / element_bytes)
    return std::nullopt;
  VectorRegisterLayoutInfo info;
  info.element_bytes = element_bytes;
  info.logical_lanes = logical_lanes;
  info.logical_bytes = logical_lanes * element_bytes;
  if (logical_lanes <= base_lanes) {
    info.vector_register_count = 1;
    info.physical_elements = base_lanes;
    info.width_class = logical_lanes * element_bytes == vector_register_bytes
                           ? VectorWidthClass::Fixed1
                           : VectorWidthClass::Partial;
    return info;
  }
  if (logical_lanes == 2 * base_lanes) {
    info.vector_register_count = 2;
    info.physical_elements = 2 * base_lanes;
    info.width_class = VectorWidthClass::Fixed2;
    return info;
  }
  if (logical_lanes == 4 * base_lanes) {
    info.vector_register_count = 4;
    info.physical_elements = 4 * base_lanes;
    info.width_class = VectorWidthClass::Fixed4;
    return info;
  }
  return std::nullopt;
}

inline bool IsFixedVectorWidth(size_t vector_register_bytes,
                               size_t logical_bytes) {
  return logical_bytes == vector_register_bytes ||
         logical_bytes == 2 * vector_register_bytes ||
         logical_bytes == 4 * vector_register_bytes;
}

} // namespace Choreo

#endif // __CHOREO_VECTOR_LAYOUT_HPP__
