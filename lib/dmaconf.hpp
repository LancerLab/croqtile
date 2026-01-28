#ifndef __CHOREO_DMA_CONFIG__
#define __CHOREO_DMA_CONFIG__

#include "types.hpp"

namespace Choreo {

struct DMAConfig {
  virtual const std::string Name() const = 0;
  virtual void Print(std::ostream&) const = 0;

  __UDT_TYPE_INFO_BASE__(none);
};

struct CopyConfig final : public DMAConfig, public TypeIDProvider<CopyConfig> {
  const std::string Name() const override { return "copy"; }
  void Print(std::ostream&) const override {};
  __UDT_TYPE_INFO__(DMAConfig, CopyConfig)
};

struct SliceConfig final : public DMAConfig,
                           public TypeIDProvider<SliceConfig> {
  const std::string Name() const override { return "slice"; }
  void Print(std::ostream&) const override {};
  __UDT_TYPE_INFO__(DMAConfig, SliceConfig)
};

namespace AST {
struct MultiValues;
struct Expr;
} // namespace AST

struct PadConfig final : public DMAConfig, public TypeIDProvider<PadConfig> {
  ptr<AST::MultiValues> pad_low;
  ptr<AST::MultiValues> pad_high;
  ptr<AST::MultiValues> pad_mid;

  // padding value
  ptr<AST::Expr> value;

  void SetPadValue(ptr<AST::Expr> val) { value = val; }
  ptr<AST::Expr> GetPadValue() const { return value; }

  const std::string Name() const override { return "pad"; }

  void Print(std::ostream& os) const override {
    os << "padding: low{" << PSTR(pad_low) << "}, high{" << PSTR(pad_high)
       << "}, mid{" << PSTR(pad_mid) << "}, value: ";
    os << PSTR(value);
  }

  __UDT_TYPE_INFO__(DMAConfig, PadConfig)
};

struct TransposeConfig final : public DMAConfig,
                               public TypeIDProvider<TransposeConfig> {
  std::vector<size_t> dim_values;
  const std::string Name() const override { return "transpose"; }
  void Print(std::ostream& os) const override {
    os << "transpose: dims{" << DelimitedString(dim_values) << "}";
  };
  __UDT_TYPE_INFO__(DMAConfig, TransposeConfig)
};

inline std::string STR(const DMAConfig& dc) {
  std::ostringstream oss;
  dc.Print(oss);
  return oss.str();
}

enum class SwizMode { NONE, B32, B64, B128 };

inline const std::string STR(SwizMode sm) {
  switch (sm) {
  case SwizMode::NONE: return "none";
  case SwizMode::B32: return "b32";
  case SwizMode::B64: return "b64";
  case SwizMode::B128: return "b128";
  default: choreo_unreachable("unsupported swizzle mode.");
  }
  return "";
}

} // end namespace Choreo

#endif // __CHOREO_DMA_CONFIG__
