#ifndef __CHOREO_MASK_GEN_HPP__
#define __CHOREO_MASK_GEN_HPP__

#include "infra_utils.hpp"
#include "io.hpp"
#include "types.hpp"
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
  // vector bool type of some c++ needs explicitly specify the element type
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
private:
  std::string loop_name;
  ptr<Type> iv_type = nullptr;
  location loc;
  ptr<Loop> parent_loop = nullptr;
  std::vector<ptr<Loop>> sub_loops;
  ptr<ScopedMaskInfo> smi;

  // note: iv_sym is set once
  std::string iv_sym;

  int vector_factor = 1;
  bool has_vectorization_hint = false;
  bool need_vectorize = false;
  bool can_vectorize = false;
  BaseType data_type = BaseType::UNKNOWN;

public:
  explicit Loop(std::string n, ptr<Type> it, const location& l,
                ptr<Loop> p = nullptr, std::vector<ptr<Loop>> subs = {},
                ptr<ScopedMaskInfo> s = std::make_shared<ScopedMaskInfo>())
      : loop_name(n), iv_type(it), loc(l), parent_loop(p), sub_loops(subs),
        smi(s) {}

  std::string LoopName() { return loop_name; }
  ptr<Type> GetIVType() { return iv_type; }
  const location& LOC() { return loc; }
  std::string IVSym() { return iv_sym; }

  void SetIVSym(const std::string& sym) { iv_sym = sym; }
  void SetIVType(ptr<Type> ty) { iv_type = ty; }
  void SetLocation(const location& l) { loc = l; }

  ValueItem GetLoopCount() { return GetSingleUpperBound(iv_type); }

  ptr<ScopedMaskInfo> GetScopedMaskInfo() { return smi; }
  ptr<Loop> GetParentLoop() { return parent_loop; }
  std::vector<ptr<Loop>> GetSubLoops() { return sub_loops; }

  void SetParentLoop(ptr<Loop> p) { parent_loop = p; }
  void AddSubLoop(ptr<Loop> sub) { sub_loops.push_back(sub); }

  bool HasLoop(const std::string& search_lname) const {
    for (const auto& sub_loop : sub_loops) {
      if (sub_loop->loop_name == search_lname ||
          sub_loop->HasLoop(search_lname))
        return true;
    }
    return false;
  }

  int GetVectorFactor() { return vector_factor; }
  bool HasVectorizationHint() { return has_vectorization_hint; }
  bool NeedVectorize() { return need_vectorize; }
  bool CanVectorize() { return can_vectorize; }
  BaseType GetDataType() { return data_type; }

  void SetVectorFactor(int vf) { vector_factor = vf; }
  void SetHasVectorizationHint(bool has_hint) {
    has_vectorization_hint = has_hint;
  }
  void SetNeedVectorize(bool need) { need_vectorize = need; }
  void SetCanVectorize(bool can) { can_vectorize = can; }
  void SetDataType(BaseType dt) { data_type = dt; }

  bool operator==(const Loop& other) const {
    return loop_name == other.loop_name;
  }
  bool operator!=(const Loop& other) const { return !(*this == other); }

  void dump(std::ostream& os, const std::string& prefix = "",
            bool print_loc = false) const {
    os << prefix << loop_name;
    if (print_loc) os << ", " << loc;
    os << "\n";
    auto new_prefix = std::string(prefix.size() + 2, ' ');
    for (const auto& sub_loop : sub_loops) { sub_loop->dump(os, new_prefix); }
  }
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

protected:
  ptr<Loop> loop = nullptr;

public:
  SCEV() = default;
  explicit SCEV(ptr<Loop> l) : loop(l) {}
  virtual SCEVType GetType() const = 0;
  virtual ~SCEV() = default;
  virtual std::string ToString() const = 0;
  virtual bool IsLoopInVariant(ptr<Loop>) const = 0;
  virtual ValueItem GetValue() const = 0;
  ptr<Loop> GetLoop() const { return loop; }
  __UDT_TYPE_INFO_BASE__(SCEV)
};

struct SCEVVal : public SCEV {
private:
  ValueItem value;

public:
  SCEVVal(ValueItem v, ptr<Loop> l = nullptr) : SCEV(l), value(v) {}
  SCEVType GetType() const override { return Val; }
  std::string ToString() const override { return STR(value); }
  bool IsLoopInVariant(ptr<Loop> l) const override {
    assert(l && "loop cannot be null.");
    if (!loop) return true;
    return loop->HasLoop(l->LoopName());
  }
  ValueItem GetValue() const override { return value; }
  __UDT_TYPE_INFO__(SCEV, SCEVVal)
};

struct SCEVAddRecExpr : public SCEV {
private:
  ptr<SCEV> base = nullptr;
  ptr<SCEV> step = nullptr;

public:
  SCEVAddRecExpr(ptr<SCEV> b, ptr<SCEV> s, ptr<Loop> l)
      : SCEV(l), base(b), step(s) {}
  SCEVType GetType() const override { return AddRecExpr; }
  std::string ToString() const override {
    std::ostringstream ss;
    ss << "{" << base->ToString() << ", +, " << step->ToString() << "} <"
       << loop->LoopName() << ">";
    return ss.str();
  }
  bool IsLoopInVariant(ptr<Loop> l) const override {
    assert(l && loop && "loop cannot be null.");
    return loop->HasLoop(l->LoopName());
  }
  ptr<SCEV> GetBase() const { return base; }
  ptr<SCEV> GetStep() const { return step; }
  ValueItem GetStepVal() const { return dyn_cast<SCEVVal>(step)->GetValue(); }
  ValueItem GetStepValOfLoop(ptr<Loop> l) const {
    if (loop->LoopName() == l->LoopName()) {
      return dyn_cast<SCEVVal>(step)->GetValue();
    }
    if (auto base_ar = dyn_cast<SCEVAddRecExpr>(base)) {
      return base_ar->GetStepValOfLoop(l);
    }
    return UncomputableValueItem();
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
private:
  // note: iv_sym may be changed if ast-tree is changed, so it may happen that
  // one same induction variable may can not find its loop through iv2loop map
  // after some ast transformations.
  std::unordered_map<std::string, std::string>
      iv2loop; // key: iv sym, value: loop name
  std::unordered_map<std::string, ptr<Loop>>
      loops; // key: loop name, value: loop pointer
public:
  void AddLoop(ptr<Loop> loop) {
    iv2loop[loop->IVSym()] = loop->LoopName();
    loops[loop->LoopName()] = loop;
  }
  std::unordered_map<std::string, ptr<Loop>> GetAllLoops() const {
    return loops;
  }

  bool IsOuterMostLoop(const std::string& lname) const {
    auto loop = GetLoop(lname);
    if (loop && !loop->GetParentLoop()) return true;
    return false;
  }

  bool IsInnermostLoop(const std::string& lname) const {
    return loops.find(lname) != loops.end() &&
           loops.at(lname)->GetSubLoops().empty();
  }

  ptr<Loop> GetLoop(const std::string& lname) const {
    if (lname == "") return nullptr;
    auto it = loops.find(lname);
    if (it != loops.end()) { return it->second; }
    return nullptr;
  }

  ptr<Loop> GetLoopOfIV(const std::string& iv_sym) const {
    if (iv_sym == "") return nullptr;
    if (iv2loop.find(iv_sym) == iv2loop.end()) return nullptr;
    auto loop_name = iv2loop.at(iv_sym);
    return GetLoop(loop_name);
  }

  ptr<Loop> GetParentLoop(const std::string& lname) const {
    if (lname == "") return nullptr;
    auto loop = GetLoop(lname);
    if (loop) { return loop->GetParentLoop(); }
    return nullptr;
  }

  std::vector<ptr<Loop>> GetSubLoops(const std::string& lname) const {
    if (lname == "") return {};
    auto loop = GetLoop(lname);
    if (loop) { return loop->GetSubLoops(); }
    return {};
  }

  void dump(std::ostream& os) {
    size_t loop_idx = 0;
    os << "Total loops: " << loops.size() << "\n";
    for (const auto& [loop_name, loop] : loops) {

      loop->dump(os, "(" + std::to_string(++loop_idx) + ") ", true);
    }
    os << "\n";
  }
};
} // end namespace Choreo
#endif // __CHOREO_MASK_GEN_HPP__
