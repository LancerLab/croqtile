#ifndef __CHOREO_GPU_WGMMA_PATTERN_HPP__
#define __CHOREO_GPU_WGMMA_PATTERN_HPP__

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "types.hpp"

namespace Choreo {
namespace Cute {

enum class WGMMAMajor { UNKNOWN, K, MN };
enum class WGMMAOperandMode { SS, RS };

struct WGMMAOperandPattern {
  bool is_register = false;
  bool is_shared = false;
  size_t registers_per_k_tile = 0;
  WGMMAMajor major = WGMMAMajor::UNKNOWN;
  std::vector<uint64_t> descriptor_offsets;
};

struct WGMMAPatternRequest {
  // Static instruction and operand-layout facts available after WGMMA shape
  // normalization. Patterns must match these facts rather than application or
  // source-level operation names.
  int arch = 0;
  BaseType a_type = BaseType::UNKNOWN;
  BaseType b_type = BaseType::UNKNOWN;
  BaseType accumulator_type = BaseType::UNKNOWN;
  int atom_m = 0;
  int atom_n = 0;
  int atom_k = 0;
  WGMMAOperandMode operand_mode = WGMMAOperandMode::SS;
  bool sparse = false;
  bool scaled_accumulator = false;
  size_t accumulator_registers = 0;
  size_t k_tiles = 1;
  WGMMAOperandPattern a;
  WGMMAOperandPattern b;
  std::optional<std::vector<int>> requested_k_order;
};

enum class WGMMASpecialization {
  NONE,
  BF16_M64N128K16_SS_K128,
  BF16_M64N128K16_RS_K128
};

struct WGMMALoweringPattern {
  WGMMASpecialization specialization = WGMMASpecialization::NONE;
  std::vector<int> k_order;
  bool supports_zero_first = false;

  bool IsSpecialized() const {
    return specialization != WGMMASpecialization::NONE;
  }
};

class WGMMAPatternRegistry {
public:
  // Selects a tuned physical K-tile issue order when the complete lowering
  // pattern is known. An explicit requested_k_order is never overridden.
  static WGMMALoweringPattern Select(const WGMMAPatternRequest& request);
};

std::optional<std::string> ValidateWGMMAKOrder(const std::vector<int>& order,
                                               size_t k_tiles);
WGMMAMajor ParseWGMMAMajor(const std::string& major);
std::string WGMMASpecializationName(WGMMASpecialization specialization);
std::string
EmitWGMMASpecializations(const std::set<WGMMASpecialization>& specializations);

} // namespace Cute
} // namespace Choreo

#endif // __CHOREO_GPU_WGMMA_PATTERN_HPP__
