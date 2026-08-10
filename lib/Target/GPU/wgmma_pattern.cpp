#include "wgmma_pattern.hpp"

#include <sstream>

namespace Choreo {
namespace Cute {
namespace {

const std::vector<int> kAscendingK128 = {0, 1, 2, 3, 4, 5, 6, 7};
// Physical RS-register / MN-major descriptor traversal produced by the SM90
// BF16 accumulator-to-A layout used for the K=128 PV group. This is a tuned
// lowering order, not a semantic requirement of MN-major WGMMA.
const std::vector<int> kRSBF16MNMajorK128 = {7, 6, 5, 4, 0, 1, 2, 3};
const std::vector<uint64_t> kSSK128DescriptorOffsets = {
    0x0, 0x2, 0x4, 0x6, 0x400, 0x402, 0x404, 0x406};
const std::vector<uint64_t> kRSMNK128DescriptorOffsets = {
    0x0, 0x80, 0x100, 0x180, 0x200, 0x280, 0x300, 0x380};

bool AllowsOrder(const WGMMAPatternRequest& request,
                 const std::vector<int>& preferred) {
  return !request.requested_k_order || *request.requested_k_order == preferred;
}

bool IsCommonK128Pattern(const WGMMAPatternRequest& request) {
  return request.arch == 90 && request.a_type == BaseType::BF16 &&
         request.b_type == BaseType::BF16 &&
         request.accumulator_type == BaseType::F32 && request.atom_m == 64 &&
         request.atom_n == 128 && request.atom_k == 16 && !request.sparse &&
         !request.scaled_accumulator && request.accumulator_registers == 64 &&
         request.k_tiles == 8;
}

std::string DRegisters(const std::string& base) {
  std::ostringstream os;
  for (int i = 0; i < 64; ++i) {
    if (i > 0) os << ", ";
    os << base << "[" << i << "]";
  }
  return os.str();
}

std::string PTXDRegisters() {
  std::ostringstream os;
  os << "{";
  for (int i = 0; i < 64; ++i) {
    if (i > 0) os << ", ";
    os << "%" << i;
  }
  os << "}";
  return os.str();
}

std::string EmitSSK128() {
  const auto name =
      WGMMASpecializationName(WGMMASpecialization::BF16_M64N128K16_SS_K128);
  const auto ptx_d = PTXDRegisters();
  std::ostringstream os;
  os << "\ntemplate <bool ZeroFirst = false>\n"
     << "__device__ static __forceinline__ void " << name
     << "(uint64_t a_desc, uint64_t b_desc, float* d) {\n";
  for (int i = 0; i < 8; ++i)
    os << "  uint64_t a" << i << " = a_desc + 0x" << std::hex
       << kSSK128DescriptorOffsets[i] << std::dec << ";\n";
  for (int i = 0; i < 8; ++i)
    os << "  uint64_t b" << i << " = b_desc + 0x" << std::hex
       << kSSK128DescriptorOffsets[i] << std::dec << ";\n";
  os << "  asm volatile(\n"
     << "      \"{\\n\\t\"\n"
     << "      \".reg .pred p0, p1;\\n\\t\"\n"
     << "      \"setp.eq.b32 p0, %80, 0;\\n\\t\"\n"
     << "      \"setp.ne.b32 p1, 1, 0;\\n\\t\"\n";
  for (int i = 0; i < 8; ++i) {
    os << "      \"wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.\"\n"
       << "      \"bf16 " << ptx_d << ", %" << 64 + i << ", %" << 72 + i << ", "
       << (i == 0 ? "p0" : "p1") << ", 1, 1, 0, 0;\\n\\t\"\n";
  }
  os << "      \"}\"\n"
     << "      : ";
  for (int i = 0; i < 64; ++i) {
    if (i > 0) os << ", ";
    os << "\"+f\"(d[" << i << "])";
  }
  os << "\n      : ";
  for (int i = 0; i < 8; ++i) {
    if (i > 0) os << ", ";
    os << "\"l\"(a" << i << ")";
  }
  for (int i = 0; i < 8; ++i) os << ", \"l\"(b" << i << ")";
  os << ", \"n\"(ZeroFirst ? 1 : 0));\n"
     << "}\n";
  return os.str();
}

std::string EmitRSK128() {
  const auto name =
      WGMMASpecializationName(WGMMASpecialization::BF16_M64N128K16_RS_K128);
  const auto d_regs = DRegisters("d");
  std::ostringstream os;
  os << "\n__device__ static __forceinline__ void " << name
     << "(const bf16* a, uint64_t b_desc, float* d) {\n"
     << "  using Op = cute::SM90::GMMA::MMA_64x128x16_F32BF16BF16_RS<\n"
     << "      cute::SM90::GMMA::Major::K, "
        "cute::SM90::GMMA::Major::MN>;\n";
  for (int k : kRSBF16MNMajorK128) {
    os << "  Op::fma(reinterpret_cast<const uint32_t*>(a + " << k
       << " * 8)[0],\n"
       << "          reinterpret_cast<const uint32_t*>(a + " << k
       << " * 8)[1],\n"
       << "          reinterpret_cast<const uint32_t*>(a + " << k
       << " * 8)[2],\n"
       << "          reinterpret_cast<const uint32_t*>(a + " << k
       << " * 8)[3],\n"
       << "          b_desc + 0x" << std::hex << kRSMNK128DescriptorOffsets[k]
       << std::dec << ", " << d_regs << ");\n";
  }
  os << "}\n";
  return os.str();
}

} // namespace

WGMMALoweringPattern
WGMMAPatternRegistry::Select(const WGMMAPatternRequest& request) {
  WGMMALoweringPattern result;
  if (request.requested_k_order) result.k_order = *request.requested_k_order;
  if (!IsCommonK128Pattern(request)) return result;

  if (request.operand_mode == WGMMAOperandMode::SS && request.a.is_shared &&
      request.b.is_shared && request.a.major == WGMMAMajor::K &&
      request.b.major == WGMMAMajor::K &&
      request.a.descriptor_offsets == kSSK128DescriptorOffsets &&
      request.b.descriptor_offsets == kSSK128DescriptorOffsets &&
      AllowsOrder(request, kAscendingK128)) {
    result.specialization = WGMMASpecialization::BF16_M64N128K16_SS_K128;
    result.k_order = kAscendingK128;
    result.supports_zero_first = true;
    return result;
  }

  if (request.operand_mode == WGMMAOperandMode::RS && request.a.is_register &&
      request.a.registers_per_k_tile == 8 && request.b.is_shared &&
      request.b.major == WGMMAMajor::MN &&
      request.b.descriptor_offsets == kRSMNK128DescriptorOffsets &&
      AllowsOrder(request, kRSBF16MNMajorK128)) {
    result.specialization = WGMMASpecialization::BF16_M64N128K16_RS_K128;
    result.k_order = kRSBF16MNMajorK128;
  }
  return result;
}

std::optional<std::string> ValidateWGMMAKOrder(const std::vector<int>& order,
                                               size_t k_tiles) {
  if (order.size() != k_tiles)
    return "must contain exactly one entry for each auto-split WGMMA K "
           "iteration (expected " +
           std::to_string(k_tiles) + ", got " + std::to_string(order.size()) +
           ")";

  std::vector<bool> seen(k_tiles, false);
  for (int tile : order) {
    if (tile < 0 || static_cast<size_t>(tile) >= k_tiles || seen[tile])
      return "must be a permutation of [0, " + std::to_string(k_tiles) + ")";
    seen[tile] = true;
  }
  return std::nullopt;
}

WGMMAMajor ParseWGMMAMajor(const std::string& major) {
  if (major.find("K_MAJOR") != std::string::npos) return WGMMAMajor::K;
  if (major.find("MN_MAJOR") != std::string::npos) return WGMMAMajor::MN;
  return WGMMAMajor::UNKNOWN;
}

std::string WGMMASpecializationName(WGMMASpecialization specialization) {
  switch (specialization) {
  case WGMMASpecialization::BF16_M64N128K16_SS_K128:
    return "__choreo_wgmma_group_bf16_m64n128k16_ss_k128";
  case WGMMASpecialization::BF16_M64N128K16_RS_K128:
    return "__choreo_wgmma_group_bf16_m64n128k16_rs_k128";
  case WGMMASpecialization::NONE: break;
  }
  return "";
}

std::string
EmitWGMMASpecializations(const std::set<WGMMASpecialization>& specializations) {
  std::ostringstream os;
  for (auto specialization : specializations) {
    switch (specialization) {
    case WGMMASpecialization::BF16_M64N128K16_SS_K128:
      os << EmitSSK128();
      break;
    case WGMMASpecialization::BF16_M64N128K16_RS_K128:
      os << EmitRSK128();
      break;
    case WGMMASpecialization::NONE: break;
    }
  }
  return os.str();
}

} // namespace Cute
} // namespace Choreo
