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

struct PadConfig final : public DMAConfig, public TypeIDProvider<PadConfig> {
  std::vector<size_t> pad_low;
  std::vector<size_t> pad_high;
  std::vector<size_t> pad_mid;

  // may have different types of padding value
  using ValueType = std::variant<int, float>;
  ValueType value = 0;

  template <typename T>
  void SetPadValue(T val) {
    value = val;
  }

  template <typename T>
  T GetPadValue() const {
    assert(std::holds_alternative<T>(value));
    return std::get<T>(value);
  }

#if 0
#define DefineSetPadValue(type, ft)                                            \
  void SetPadValue(type val) {                                                 \
    value = {*(reinterpret_cast<uint32_t*>(&val)), ft};                        \
  }

  DefineSetPadValue(uint32_t, FundamentalType::U32);
  DefineSetPadValue(int32_t, FundamentalType::S32);
  DefineSetPadValue(uint16_t, FundamentalType::U16);
  DefineSetPadValue(int16_t, FundamentalType::S16);
  DefineSetPadValue(uint8_t, FundamentalType::U8);
  DefineSetPadValue(int8_t, FundamentalType::S8);
  DefineSetPadValue(float, FundamentalType::F32);

  DefineSetPadValue(f16, FundamentalType::F16);
  DefineSetPadValue(bf16, FundamentalType::BF16);
#endif

  const std::string Name() const override { return "pad"; }
  void Print(std::ostream& os) const override {
    os << "padding: low{" << DelimitedString(pad_low) << "}, high{"
       << DelimitedString(pad_high) << "}, mid{" << DelimitedString(pad_mid)
       << "}, value: ";
    if (std::holds_alternative<int>(value)) {
      os << std::get<int>(value);
    } else if (std::holds_alternative<float>(value)) {
      os << std::get<float>(value);
    } else {
      os << "unknown type";
    }
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

} // end namespace Choreo

#endif // __CHOREO_DMA_CONFIG__
