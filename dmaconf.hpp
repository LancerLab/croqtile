#ifndef __CHOREO_DMA_CONFIG__
#define __CHOREO_DMA_CONFIG__

#include "types.hpp"

namespace Choreo {

struct DMAConfig {
  virtual const std::string TypeNameString() = 0;
  virtual uint64_t RuntimeID() const { return 0xDEADBEEFULL; }
  static uint64_t TypeID() { return 0xDEADBEEFULL; }
  virtual const std::string Name() const = 0;
  virtual void Print(std::ostream &) const = 0;
};

struct CopyConfig final : public DMAConfig, public TypeIDProvider<CopyConfig> {
  const std::string Name() const { return "copy"; }
  void Print(std::ostream &) const override{};
  __UDT_TYPE_INFO__
};

struct SliceConfig final : public DMAConfig,
                           public TypeIDProvider<SliceConfig> {
  const std::string Name() const { return "slice"; }
  void Print(std::ostream &) const override{};
  __UDT_TYPE_INFO__
};

struct PadConfig final : public DMAConfig, public TypeIDProvider<PadConfig> {
  std::vector<size_t> pad_high;
  std::vector<size_t> pad_low;
  std::vector<size_t> pad_mid;

  struct PadValue {
    uint32_t v;
    FundamentalType t;
  } value;

#define DefineSetPadValue(type, ft) \
  void SetPadValue(type val) { value = {*((uint32_t *)&val), ft}; }

  DefineSetPadValue(uint32_t, FundamentalType::U32);
  DefineSetPadValue(int32_t, FundamentalType::S32);
  DefineSetPadValue(uint16_t, FundamentalType::U16);
  DefineSetPadValue(int16_t, FundamentalType::S16);
  DefineSetPadValue(uint8_t, FundamentalType::U8);
  DefineSetPadValue(int8_t, FundamentalType::S8);
  DefineSetPadValue(float, FundamentalType::F32);

  // TODO: involve scalar types
#if 0
  DefineSetPadValue(f16, FundamentalType::F16);
  DefineSetPadValue(bf16, FundamentalType::BF16);
#endif

  const std::string Name() const { return "pad"; }
  void Print(std::ostream &os) const override {
    os << "pad: high{" << DelimitedString(pad_high) << "}, low{"
       << DelimitedString(pad_high) << "}, mid{" << DelimitedString(pad_mid)
       << "}, value: " << value.v;
  }
  __UDT_TYPE_INFO__
};

inline std::string STR(const DMAConfig &dc) {
  std::ostringstream oss;
  dc.Print(oss);
  return oss.str();
}

}  // end namespace Choreo

#endif  // __CHOREO_DMA_CONFIG__
