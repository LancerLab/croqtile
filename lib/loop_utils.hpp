#ifndef __CHOREO_MASK_GEN_HPP__
#define __CHOREO_MASK_GEN_HPP__

#include "io.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <ostream>
#include <string>
#include <unordered_map>
namespace Choreo {
inline static const std::string NoLoopName() { return "no_loop"; }

static inline ValueItem UncomputableValueItem() {
  return sbe::sym("uncomputable");
}

// Diversity shape is classified into 3 kind.
// UNIFORM: all lanes have the same value, we use ValueItem to track its
// symbolic value. STRIDE: lanes have a constant stride value, e.g., 0,2,4,6.
// DIVERGENT: lanes have different values without a constant stride.
enum DiversityShapeKind { UNKNOWN = 0, UNIFORM, STRIDE, DIVERGENT };
struct DiversityShape {
  using Kind = DiversityShapeKind;
  Kind shape = UNKNOWN;
  ValueItem stride; // stride for STRIDE shape
  ValueItem value;  // value for UNIFORM shape

  DiversityShape() = default;
  DiversityShape(Kind k, ValueItem s = UncomputableValueItem(),
                 ValueItem v = UncomputableValueItem())
      : shape(k), stride(s), value(v) {
    if (shape == Kind::STRIDE) {
      // stride must be an integer and computable, otherwise it is DIVERGENT
      if (!VIIsInt(stride) || !stride->Computable()) {
        shape = Kind::DIVERGENT;
        stride = UncomputableValueItem();
        value = UncomputableValueItem();
      }
    }
    if (shape == Kind::UNIFORM) {
      // value must be computable, otherwise it is UNIFORM without value
      if (!value->Computable()) value = UncomputableValueItem();
    }
  }
  DiversityShape(const DiversityShape& other)
      : shape(other.shape), stride(other.stride), value(other.value) {}

  bool Uniform() const { return shape == Kind::UNIFORM; }

  bool Stride(int s = -1) const {
    if (shape != Kind::STRIDE) return false;

    return s == -1 || sbe::oc_eq(stride, sbe::nu(s));
  }

  bool Divergent() const { return shape == Kind::DIVERGENT; }

  bool Unknown() const { return shape == Kind::UNKNOWN; }

  bool Varying() const {
    return shape == Kind::STRIDE || shape == Kind::DIVERGENT;
  }

  bool ApprxEqual(const DiversityShape& other) const {
    if (shape != other.shape) return false;
    return true; // for DIVERGENT or UNKNOWN
  }

  DiversityShape& operator=(const DiversityShape& other) {
    shape = other.shape;
    stride = other.stride;
    value = other.value;
    return *this;
  }

  bool operator<(const DiversityShape& other) const {
    return shape < other.shape;
  }

  bool operator>(const DiversityShape& other) const {
    return shape > other.shape;
  }
};

// it stores info of masking
struct ScopedMaskInfo {
  // vector bool type of tops c++ needs explicitly specify the element type
  // like __vector bool int, so we need to track the element type of mask
  BaseType mask_element_type = BaseType::S32;
  int vector_width = 0;
  std::unordered_map<std::string, std::string> scoped_masks;
  std::unordered_set<std::string> all_true_masks;

  ScopedMaskInfo() {}
  ScopedMaskInfo(const ScopedMaskInfo& other)
      : mask_element_type(other.mask_element_type),
        scoped_masks(other.scoped_masks) {}

  bool NeedMask() const { return !scoped_masks.empty(); }
  BaseType GetMaskEType() { return mask_element_type; }
  void SetMaskEType(BaseType ele_type) { mask_element_type = ele_type; }

  int GetVectorWidth() { return vector_width; }
  void SetVectorWidth(int w) { vector_width = w; }

  std::string GetMaskInScope(const std::string& scope) {
    if (scoped_masks.find(scope) != scoped_masks.end())
      return scoped_masks[scope];
    return "";
  }

  void SetMaskInScope(const std::string& scope, const std::string& mask) {
    scoped_masks[scope] = mask;
  }

  bool IsMaskAlltrue(std::string mask) {
    return all_true_masks.find(mask) != all_true_masks.end();
  }

  void Dump() {
    for (auto& [k, v] : scoped_masks) { dbgs() << k << ", " << v << "\n"; }
  }
};

// Loop maps ForeachBlock(Normalized, only has one LoopRange)
struct Loop {
  std::string loop_name;
  std::string iv_name;
  ptr<Type> iv_type = nullptr;
  std::vector<ptr<Loop>> sub_loops;
  // scoped mask info, used to track the execution mask in this loop scope
  // and its child branch scope.
  ptr<ScopedMaskInfo> smi;
  int vector_width = 1;
  bool need_vectorize = false;
  bool can_vectorize = true;

  explicit Loop(std::string n, std::string i, ptr<Type> it,
                std::vector<ptr<Loop>> subs = {},
                ptr<ScopedMaskInfo> s = std::make_shared<ScopedMaskInfo>())
      : loop_name(n), iv_name(i), iv_type(it), sub_loops(subs), smi(s) {}

  std::string IVName() { return iv_name; }
  ptr<Type> GetIVType() { return iv_type; }
  void SetIVType(ptr<Type> ty) { iv_type = ty; }

  int GetVectorWidth() { return vector_width; }
  bool NeedVectorize() { return need_vectorize; }
  bool CanVectorize() { return can_vectorize; }

  bool operator==(const Loop& other) const {
    return loop_name == other.loop_name;
  }
  bool operator!=(const Loop& other) const { return !(*this == other); }
  bool HasLoop(const std::string& search_lname) const {
    for (const auto& sub_loop : sub_loops) {
      if (sub_loop->loop_name == search_lname ||
          sub_loop->HasLoop(search_lname))
        return true;
    }
    return false;
  }

  void dump(std::ostream& os, const std::string& prefix = "") const {
    os << prefix << loop_name << "\n";
    for (const auto& sub_loop : sub_loops) {
      sub_loop->dump(os, prefix + "  ");
    }
  }

  __UDT_TYPE_INFO_BASE__(Loop)
};

// scalar evolution expression
// it can be a constant/symbolic value, or an add recurrence expression
// add recurrence expression is in the form of {base, +, step} <loop>
// which means the value starts from base, and increases by step in each
// iteration of the loop. Different from LLVM, we do not need scev type like
// SCEVMUL or SCEVADD, we can directly use ValueItem to represent the value of
// the expression. Same as LLVM, we support multiple nesting add recurrence scev
// expr.
struct SCEV {
  enum SCEVType {
    Unknown,
    Val,
    AddRecExpr,
  };

  virtual SCEVType GetType() const = 0;
  virtual ~SCEV() = default;
  virtual std::string ToString() const = 0;
  virtual bool IsLoopInVariant(ptr<Loop>) const = 0;
  virtual ValueItem GetValue() const = 0;
  __UDT_TYPE_INFO_BASE__(SCEV)
};

struct SCEVVal : public SCEV {
  ValueItem value;
  ptr<Loop> loop = nullptr;
  SCEVVal(ValueItem v, ptr<Loop> l = nullptr) : value(v), loop(l) {}
  SCEVType GetType() const override { return Val; }
  std::string ToString() const override { return STR(value); }
  bool IsLoopInVariant(ptr<Loop> l) const override {
    assert(l && "loop cannot be null.");
    if (!loop) return true;
    return loop->HasLoop(l->loop_name);
  }
  ValueItem GetValue() const override { return value; }
  __UDT_TYPE_INFO__(SCEV, SCEVVal)
};

struct SCEVAddRecExpr : public SCEV {
  ptr<SCEV> base;
  ptr<SCEV> step;
  ptr<Loop> loop;
  ValueItem times = UncomputableValueItem(); // optional, for step * n
  SCEVAddRecExpr(ptr<SCEV> b, ptr<SCEV> s, ptr<Loop> l)
      : base(b), step(s), loop(l) {}
  SCEVType GetType() const override { return AddRecExpr; }
  std::string ToString() const override {
    std::ostringstream ss;
    ss << "{" << base->ToString() << ", +, " << step->ToString() << "} <"
       << loop->loop_name << ">";
    return ss.str();
  }
  bool IsLoopInVariant(ptr<Loop> l) const override {
    assert(l && loop && "loop cannot be null.");
    return loop->HasLoop(l->loop_name);
  }
  ValueItem GetValue() const override { return UncomputableValueItem(); }
  __UDT_TYPE_INFO__(SCEV, SCEVAddRecExpr)
};

inline ptr<SCEVAddRecExpr> MakeSCEVAddRecExpr(ptr<SCEV> base, ptr<SCEV> step,
                                              ptr<Loop> loop) {
  return std::make_shared<SCEVAddRecExpr>(base, step, loop);
}

inline ptr<SCEVVal> MakeSCEVVal(ValueItem v, ptr<Loop> loop = nullptr) {
  return std::make_shared<SCEVVal>(v, loop);
}

inline ptr<SCEVAddRecExpr> MakeSCEVAddRecExpr(ValueItem base, ValueItem step,
                                              ptr<Loop> loop) {
  return std::make_shared<SCEVAddRecExpr>(MakeSCEVVal(base), MakeSCEVVal(step),
                                          loop);
}

inline ptr<SCEVAddRecExpr> MakeSCEVAddRecExpr(ptr<SCEV> base, ValueItem step,
                                              ptr<Loop> loop) {
  return std::make_shared<SCEVAddRecExpr>(base, MakeSCEVVal(step), loop);
}

inline std::string STR(const ptr<SCEV>& scev) {
  if (!scev) return "UNKNOWN";
  return scev->ToString();
}

struct LoopInfo {
  std::unordered_map<std::string, std::string>
      iv2loop; // key: iv name, value: loop name
  std::unordered_map<std::string, ptr<Loop>>
      loops; // key: scope name of the loop

  std::string GetParentLoopName(const std::string& lname) const {
    auto removeLastLoop = [](const std::string& input) -> std::string {
      size_t lastPos = input.rfind("::");
      if (lastPos == std::string::npos) {
        return input; // No "::" found, return the original string
      }

      // Find the second-to-last "::" by searching up to the last found
      // position
      size_t secondLastPos = input.rfind("::", lastPos - 1);
      if (secondLastPos == std::string::npos) return input;
      return input.substr(0,
                          secondLastPos + 2); // Include the "::" in the result
    };

    if (lname == NoLoopName()) return "";
    std::string parent_loop_name = removeLastLoop(lname);
    while (parent_loop_name != "::") {
      if (loops.find(parent_loop_name) != loops.end()) {
        return parent_loop_name;
      }
      parent_loop_name = removeLastLoop(parent_loop_name);
    }
    return "";
  }

  std::string GetOutermostLoopName(const std::string& lname) const {
    auto parent_lname = lname;
    while (GetParentLoopName(parent_lname) != "") {
      parent_lname = GetParentLoopName(parent_lname);
    }

    return parent_lname;
  }

  bool IsInnermostLoop(const std::string& lname) const {
    return loops.find(lname) != loops.end() &&
           loops.at(lname)->sub_loops.empty();
  }

  ptr<Loop> GetLoop(const std::string& lname) const {
    auto it = loops.find(lname);
    if (it != loops.end()) { return it->second; }
    return nullptr;
  }

  void dump(std::ostream& os) {
    for (const auto& [loop_name, loop] : loops) { loop->dump(os); }
  }
};
} // end namespace Choreo
#endif // __CHOREO_MASK_GEN_HPP__