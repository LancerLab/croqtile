#ifndef CHOREO_CODEGEN_HPP_
#define CHOREO_CODEGEN_HPP_

#include <mutex>
#include <string>
#include <thread>

#include "memcheck.hpp"
#include "pbtree.hpp"
#include "target_utils.hpp"
#include "valbind.hpp"
#include "visitor.hpp"

namespace Choreo {

struct SymbolDetail {
  // information from choreo code
  std::string name;
  ptr<Type> type = nullptr;
  std::string rty_str;
  bool is_reference = false;
  int p_index = -1; // index of parameter in choreo function decl
  ParamAttr attr = ParamAttr::NONE;
  bool need_iv_prefix = false;

public:
  // information used for codegen
  std::string host_name;   // mapped host name
  std::string device_name; // mapped device name
  int d_index = -1;        // index of symbol in device function parameter list

  std::string h_name; // some target like factor requires host function names
  int h_index = -1;   // some target like factor requires host function indices

  SymbolDetail(const std::string& n, const ptr<Type>& t, bool ref = false,
               int index = -1, ParamAttr a = ParamAttr::NONE,
               const std::string& ret = "", bool iv = false)
      : name(n), type(t), rty_str(ret), is_reference(ref), p_index(index),
        attr(a), need_iv_prefix(iv) {}

  bool IsParameter() const { return p_index != -1; }
  bool IsReturn() const { return !rty_str.empty(); }
  bool IsReference() const { return is_reference; }
  void SetAsReturn(const std::string& t) {
    assert(!t.empty());
    rty_str = t;
  }
};

struct ParallelCounts {
  ValueItem z = sbe::nu(1);
  ValueItem y = sbe::nu(1);
  ValueItem x = sbe::nu(1);
  void Reset() {
    z = sbe::nu(1);
    y = sbe::nu(1);
    x = sbe::nu(1);
  }
  bool EqualsOne() const {
    return (x == sbe::nu(1)) && (y == sbe::nu(1)) && (z == sbe::nu(1));
  }
};

inline std::ostream& operator<<(std::ostream& os, ParallelCounts pc) {
  os << STR(pc.z) << ", " << STR(pc.y) << ", " << STR(pc.x);
  return os;
}

struct LaunchConfig {
  ParallelCounts block_count;
  ParallelCounts group4_count;
  ParallelCounts group_count;
  ParallelCounts thread_count;

  void SetBlockCount(const ValueList& dims) {
    block_count.Reset();
    switch (dims.size()) {
    case 3: block_count.z = dims[2]; [[fallthrough]];
    case 2: block_count.y = dims[1]; [[fallthrough]];
    case 1: block_count.x = dims[0]; break;
    default: choreo_unreachable("The number of dimensions is not supported.");
    }
  }

  void SetGroupCount(const ValueList& dims) {
    group_count.Reset();
    switch (dims.size()) {
    case 3: group_count.z = dims[2]; [[fallthrough]];
    case 2: group_count.y = dims[1]; [[fallthrough]];
    case 1: group_count.x = dims[0]; break;
    default: choreo_unreachable("The number of dimensions is not supported.");
    }
  }

  void SetGroupx4Count(const ValueList& dims) {
    group4_count.Reset();
    switch (dims.size()) {
    case 3: group4_count.z = dims[2]; [[fallthrough]];
    case 2: group4_count.y = dims[1]; [[fallthrough]];
    case 1: group4_count.x = dims[0]; break;
    default: choreo_unreachable("The number of dimensions is not supported.");
    }
  }

  void SetThreadCount(const ValueList& dims) {
    thread_count.Reset();
    switch (dims.size()) {
    case 3: thread_count.z = dims[2]; [[fallthrough]];
    case 2: thread_count.y = dims[1]; [[fallthrough]];
    case 1: thread_count.x = dims[0]; break;
    default: choreo_unreachable("The number of dimensions is not supported.");
    }
  }
};

struct ModuleTraits {
  bool has_tma = false;
};

struct FuncTrait {
  bool has_parallelby = false;
  bool multiple_parallelby = false;
  bool has_tma = false;
  bool has_async_dma = false;
};

struct MMAInfo {
  enum Fragment { FRAG_UNK, FRAG_A, FRAG_B, FRAG_C, FRAG_E };
  BaseType ty;
  ValueList shape;
  Fragment frag;
  AST::MMAOperation::ExecMethod method = AST::MMAOperation::ExecMethod::ROW_COL;
  bool operator==(MMAInfo i) {
    return ty == i.ty && IsValueListEqual(shape, i.shape) && frag == i.frag;
  }
  const ValueList& GetShape() const { return shape; }
};

inline std::ostream& operator<<(std::ostream& os, const MMAInfo& i) {
  os << "[mma_info] " << STR(i.ty) << " | " << STR(i.shape)
     << " | frag: " << (int)i.frag;
  return os;
}

struct TMADesc {
private:
  std::string tma_name;
  ptr<AST::ChunkAt> from; // shape for the global input/output
  ptr<AST::ChunkAt> to;
  std::string f_sym; // scoped symbol
  std::string t_sym;
  uint16_t idx;
  SwizMode swiz_mode = SwizMode::NONE; // Default to no swizzle
  ParallelLevel pb_level = ParallelLevel::BLOCK;
  AST::InThreadsBlock* in_thr_block =
      nullptr; // if the TMA is within an inthreads, record it here

public:
  TMADesc(const ptr<AST::ChunkAt>& f, const ptr<AST::ChunkAt>& t,
          const std::string& fs, const std::string& ts,
          SwizMode swizzle = SwizMode::NONE,
          ParallelLevel pb_lvl = ParallelLevel::BLOCK)
      : from(f), to(t), f_sym(fs), t_sym(ts), idx(index++), swiz_mode(swizzle),
        pb_level(pb_lvl) {
    assert(from && to);
    auto fty = GetSpannedType(from->GetType());
    auto tty = GetSpannedType(to->GetType());
    if (((fty->GetStorage() == Storage::GLOBAL ||
          fty->GetStorage() == Storage::DEFAULT) &&
         tty->GetStorage() == Storage::SHARED) ||
        ((tty->GetStorage() == Storage::GLOBAL ||
          tty->GetStorage() == Storage::DEFAULT) &&
         fty->GetStorage() == Storage::SHARED))
      return;
    choreo_unreachable("unsupported TMA storage: " + STR(fty->GetStorage()) +
                       " => " + STR(tty->GetStorage()));
  }

  bool IsLoad() const {
    auto tty = GetSpannedType(to->GetType());
    if (tty->GetStorage() == Storage::SHARED) return true;
    return false;
  }

  bool IsStore() const {
    auto fty = GetSpannedType(from->GetType());
    if (fty->GetStorage() == Storage::SHARED) return true;
    return false;
  }

  const ptr<AST::ChunkAt> GetFrom() const { return from; }
  const ptr<AST::ChunkAt> GetTo() const { return to; }
  const std::string GetFromSymbol() const { return f_sym; }
  const std::string GetToSymbol() const { return t_sym; }
  SwizMode GetSwizzleMode() const { return swiz_mode; }

  const std::string GetName() const {
    return "__choreo_tma_" + std::to_string(idx);
  }

  ParallelLevel GetPBLevel() const { return pb_level; }

  void SetInThreadsBlock(AST::InThreadsBlock* in) { in_thr_block = in; }
  AST::InThreadsBlock* GetInThreadsBlock() const { return in_thr_block; }

private:
  static int index;
};

using SymbolDetails = std::map<std::string, std::vector<SymbolDetail>>;
using LaunchDetails = std::map<std::string, std::vector<LaunchConfig>>;
using ReturnSymbols = std::map<std::string, std::string>;
using FunctionTraits = std::map<std::string, FuncTrait>;
using SharedFutures = std::map<std::string, std::set<std::string>>;
using LocalFutures = std::map<std::string, std::set<std::string>>;
using SymbolMMA = std::map<std::string, MMAInfo>;
using TMADescs = std::map<AST::ParallelBy*, std::vector<TMADesc>>;
using PBTreeInfo = std::map<std::string, PBTree>;

enum PassedOrDeclaredSymbolKind : int {
  PDSYM_NONE = 0,
  PDSYM_PARAMETERS_ONLY = 0x1, // only the parameters declared
  PDSYM_ALLOC_IN_DEVICE = 0x2, // must be allocated with a device storage
  PDSYM_NO_RETURN = 0x4,   // simply without symbols that is the return value
  PDSYM_RETURN_ONLY = 0x8, // only the symbol of return statement
  PDSYM_WITH_REFERENCE = 0x10, // with reference symbols
};

struct CodeGenInfo {
private:
  SymbolDetails all_syms;
  LaunchDetails launches;
  ReturnSymbols returns;
  ModuleTraits m_traits;
  FunctionTraits f_traits;
  SharedFutures shr_futs;
  LocalFutures loc_futs;
  SymbolMMA sym_mmas;
  TMADescs tma_descs;
  PBTreeInfo pb_tree;

  // TODO: maybe should add some vars here

public:
  const std::vector<SymbolDetail>&
  GetFunctionSymbols(const std::string& fname) const {
    return all_syms.at(fname);
  }
  std::vector<SymbolDetail>& GetFunctionSymbols(const std::string& fname) {
    return all_syms[fname];
  }

  const std::vector<LaunchConfig>&
  GetFunctionLaunches(const std::string& fname) const {
    return launches.at(fname);
  }

  std::vector<LaunchConfig>& GetFunctionLaunches(const std::string& fname) {
    return launches[fname];
  }

  const ModuleTraits& GetModuleTrait() const { return m_traits; }
  ModuleTraits& GetModuleTrait() { return m_traits; }

  const FuncTrait& GetFunctionTrait(const std::string& fname) const {
    return f_traits.at(fname);
  }
  FuncTrait& GetFunctionTrait(const std::string& fname) {
    return f_traits[fname];
  }

  const std::set<std::string>&
  GetFunctionSharedFutures(const std::string& fname) const {
    return shr_futs.at(fname);
  }
  std::set<std::string>& GetFunctionSharedFutures(const std::string& fname) {
    return shr_futs[fname];
  }

  const std::set<std::string>&
  GetFunctionLocalFutures(const std::string& fname) const {
    return loc_futs.at(fname);
  }
  std::set<std::string>& GetFunctionLocalFutures(const std::string& fname) {
    return loc_futs[fname];
  }

  const MMAInfo& GetSymbolMMA(const std::string& sym) const {
    return sym_mmas.at(sym);
  }
  MMAInfo& GetSymbolMMA(const std::string& sym) { return sym_mmas[sym]; }
  void AddSymbolMMA(const std::string& sym, const MMAInfo& i) {
    if (sym_mmas.count(sym)) {
      if (sym_mmas[sym] == i)
        return;
      else
        choreo_unreachable(
            "unable to infer a MMA symbol with different information:" +
            STR(sym_mmas[sym].shape) + " vs. " + STR(i.shape) + ".");
    } else
      sym_mmas.emplace(sym, i);
  }

  const std::vector<TMADesc> GetTMADesc(AST::ParallelBy* pb) const {
    if (tma_descs.count(pb))
      return tma_descs.at(pb);
    else
      return {};
  }
  const TMADescs& GetTMADescs() const { return tma_descs; }
  TMADescs& GetTMADescs() { return tma_descs; }

  const PBTree& GetPBTree(const std::string& fn) const {
    return pb_tree.at(fn);
  }
  PBTree& GetPBTree(const std::string& fn) { return pb_tree[fn]; }

  bool HasParallelBy(const std::string& fname) const {
    return GetFunctionTrait(fname).has_parallelby;
  }

  bool HasTMA() const { return GetModuleTrait().has_tma; }

  bool HasTMA(const std::string& fname) const {
    return GetFunctionTrait(fname).has_tma;
  }

  bool HasAsyncDMA(const std::string& fname) const {
    return GetFunctionTrait(fname).has_async_dma;
  }

  bool HasReturnSymbol(const std::string& fname) const {
    return returns.count(fname);
  }
  const std::string& GetReturnSymbol(const std::string& fname) const {
    return returns.at(fname);
  }
  bool IsReturnSymbol(const std::string& fname, const std::string& sym) const {
    if (HasReturnSymbol(fname)) return GetReturnSymbol(fname) == sym;
    return false;
  }
  void SetReturnSymbol(const std::string fname, const std::string& rs) {
    returns[fname] = rs;
  }

  void AddSymbolDetail(const std::string fname, const SymbolDetail& sd) {
    if (!all_syms.count(fname))
      all_syms[fname] = {sd};
    else
      all_syms[fname].emplace_back(sd);
  }

  void SetLaunchDetail(const std::string fname, const LaunchConfig& lc) {
    launches[fname].push_back(lc);
  }
  void SetLaunchDetails(const std::string fname,
                        const std::vector<LaunchConfig>& lcs) {
    launches[fname] = lcs;
  }

  // argument's index by its scoped name
  int GetArgumentIndex(const std::string& fname,
                       const std::string& pname) const {
    if (!PrefixedWith(pname, "::"))
      choreo_unreachable("Not a in-scope symbol name (" + pname + ").");

    for (auto& item : all_syms.at(fname))
      if (item.name == pname) return item.p_index;

    return -1;
  }

  size_t ParameterCount(const std::string& fname) const {
    size_t param_count = 0;
    for (auto& item : all_syms.at(fname))
      if (item.IsParameter()) param_count++;
    return param_count;
  }

  // Get symbols with global storage
  bool IsPassedOrDeclaredSymbols(const SymbolDetail& sd,
                                 int gsk = PDSYM_NONE) const {
    assert(!((gsk & PDSYM_RETURN_ONLY) && (gsk & PDSYM_NO_RETURN)) &&
           "return or not?");

    // no filter-outs
    if (gsk == PDSYM_NONE) return true;

    if (sd.IsReturn()) {
      if (gsk & PDSYM_NO_RETURN) return false;
      if (gsk & PDSYM_RETURN_ONLY) return true;
      if (gsk & PDSYM_ALLOC_IN_DEVICE) return true;
      // PDSYM_PARAMETERS_ONLY: pass-by: need further check
      // PDSYM_WITH_REFERENCE: pass-by: need further check
    }

    if (gsk & PDSYM_RETURN_ONLY) return false;

    // Parameters are passed to device. And host code should map and alloc the
    // device storage to shadow them.
    if (sd.IsParameter()) {
      if (gsk & PDSYM_PARAMETERS_ONLY) return true;
      // parameter should be mapped storage in device
      if (gsk & PDSYM_ALLOC_IN_DEVICE) return true;
      // PDSYM_NO_RETURN: pass-by
      // PDSYM_WITH_REFERENCE: pass-by
    }

    if (gsk & PDSYM_PARAMETERS_ONLY) return false;

    if (auto ety = dyn_cast<EventType>(sd.type)) {
      if ((ety->GetStorage() == Storage::GLOBAL) &&
          (gsk & PDSYM_ALLOC_IN_DEVICE))
        return true;
    }

    auto sty = dyn_cast<SpannedType>(sd.type);
    if (sty && ((sty->GetStorage() == Storage::GLOBAL) ||
                (sty->GetStorage() ==
                 Storage::DEFAULT /* default is mapped as global */))) {
      if (gsk & PDSYM_ALLOC_IN_DEVICE) {
        if (sd.IsReference()) {
          // reference should not be assigned
          if (gsk & PDSYM_WITH_REFERENCE) return true;
          return false;
        }
        return true;
      }
      // PDSYM_NO_RETURN: pass-by
    }

    if (CanYieldAnInteger(sd.type)) {
      if (sd.IsReference() && (gsk & PDSYM_WITH_REFERENCE)) return true;
      return false;
    }

    if (gsk & PDSYM_ALLOC_IN_DEVICE) return false;

    // no more filters
    return true;
  }

  // ranges
  FilterRange<SymbolDetail> GetParameters(const std::string& fname) {
    return FilterRange<SymbolDetail>(
        this->all_syms[fname], [this](const SymbolDetail& sd) {
          return this->IsPassedOrDeclaredSymbols(sd, PDSYM_PARAMETERS_ONLY);
        });
  }

  FilterRange<SymbolDetail> GetDevicePassIns(const std::string& fname) {
    return FilterRange<SymbolDetail>(
        this->all_syms[fname], [this](const SymbolDetail& sd) {
          return this->IsPassedOrDeclaredSymbols(sd, PDSYM_NO_RETURN |
                                                         PDSYM_ALLOC_IN_DEVICE |
                                                         PDSYM_WITH_REFERENCE);
        });
  }

  FilterRange<SymbolDetail> GetDeviceAllocatables(const std::string& fname) {
    return FilterRange<SymbolDetail>(
        this->all_syms[fname], [this](const SymbolDetail& sd) {
          return this->IsPassedOrDeclaredSymbols(sd, PDSYM_ALLOC_IN_DEVICE);
        });
  }

  FilterRange<SymbolDetail> GetDeviceAllIns(const std::string& fname) {
    return FilterRange<SymbolDetail>(
        this->all_syms[fname], [this](const SymbolDetail& sd) {
          return this->IsPassedOrDeclaredSymbols(sd, PDSYM_ALLOC_IN_DEVICE |
                                                         PDSYM_WITH_REFERENCE);
        });
  }

  // without return
  FilterRange<SymbolDetail> GetDeviceAllocIns(const std::string& fname) {
    return FilterRange<SymbolDetail>(
        this->all_syms[fname], [this](const SymbolDetail& sd) {
          return this->IsPassedOrDeclaredSymbols(sd, PDSYM_NO_RETURN |
                                                         PDSYM_ALLOC_IN_DEVICE);
        });
  }

  const SymbolDetail& GetReturnDetail(const std::string& fname) const {
    assert(all_syms.count(fname) != 0);
    assert(returns.count(fname) != 0);
    for (auto& item : all_syms.at(fname))
      if (item.name == returns.at(fname)) return item;
    return all_syms.at(fname).at(0);
  }

public:
  static CodeGenInfo& Get();
  static std::once_flag init_flag;
  static std::unique_ptr<CodeGenInfo> instance;
};

// Codegenerators for targets
struct CodeGenerator : public VisitorWithSymTab {
  // some default method for the derived classes that do not want to override.
  bool BeforeVisitImpl(AST::Node&) override { return true; }
  bool AfterVisitImpl(AST::Node&) override { return true; }

  CodeGenerator(const std::string& n)
      : VisitorWithSymTab(n), cgi(CodeGenInfo::Get()) {}

  virtual void TraceEachVisit(AST::Node& n, bool detail = false,
                              const std::string& m = "") const {
    if (!trace_visit) return;
    if (detail)
      dbgs() << m << STR(n) << "\n";
    else
      dbgs() << m << n.TypeNameString() << "\n";
  }

protected:
  CodeGenInfo& cgi;

protected:
  Storage FutureStorage(const std::string& n) const {
    assert(PrefixedWith(n, "::") && "requires a scoped name.");
    if (cgi.GetFunctionSharedFutures(fname).count(n))
      return Storage::SHARED;
    else if (cgi.GetFunctionLocalFutures(fname).count(n))
      return Storage::LOCAL;
    assert("illegal future.");
    return Storage::NONE;
  }
  bool IsBlockwiseFuture(const std::string& n) const {
    assert(PrefixedWith(n, "::") && "requires a scoped name.");
    return cgi.GetFunctionSharedFutures(fname).count(n);
  }
  bool IsGroupwiseFuture(const std::string& n) const {
    assert(PrefixedWith(n, "::") && "requires a scoped name.");
    return cgi.GetFunctionLocalFutures(fname).count(n);
  }
};

/////////////////////////////////////////////////////////////
///  Util functions shared between targets
/////////////////////////////////////////////////////////////

inline static std::string CreateUniquePath() {
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

inline static void ReplaceInString(std::string* pstr, const std::string& from,
                                   const std::string& to) {
  if (from.empty()) return;

  size_t startPos = 0;
  while ((startPos = pstr->find(from, startPos)) != std::string::npos) {
    pstr->replace(startPos, from.length(), to);
    startPos += to.length(); // In case 'to' contains 'from', like replacing
                             // 'x' with 'yx'
  }
}

static inline std::string
HostTypeStringify(const Choreo::Type& ty, bool is_ret = false,
                  [[maybe_unused]] bool is_ref = false) {
  std::string res;
  if (isa<VoidType>(&ty))
    res = "void";
  else if (isa<S8Type>(&ty))
    res = "char";
  else if (isa<U8Type>(&ty))
    res = "unsigned char";
  else if (isa<S16Type>(&ty))
    res = "short";
  else if (isa<U16Type>(&ty))
    res = "unsigned short";
  else if (isa<S32Type>(&ty))
    res = "int";
  else if (isa<U32Type>(&ty))
    res = "unsigned int";
  else if (isa<S64Type>(&ty))
    res = "long long";
  else if (isa<U64Type>(&ty))
    res = "unsigned long long";
  else if (isa<BooleanType>(&ty))
    res = "bool";
  else if (isa<FloatE4M3Type>(&ty))
    res = "choreo::f8_e4m3_t";
  else if (isa<FloatE5M2Type>(&ty))
    res = "choreo::f8_e5m2_t";
  else if (isa<FloatE2M3Type>(&ty))
    res = "choreo::f6_e2m3_t";
  else if (isa<FloatE3M2Type>(&ty))
    res = "choreo::f6_e3m2_t";
  else if (isa<FloatE2M1Type>(&ty))
    res = "choreo::f4_e2m1_t";
  else if (isa<F16Type>(&ty))
    res = "choreo::half";
  else if (isa<BF16Type>(&ty))
    res = "choreo::bfp16";
  else if (isa<F32Type>(&ty))
    res = "float";
  else if (isa<F64Type>(&ty))
    res = "double";
  else if (auto sty = dyn_cast<SpannedType>(&ty)) {
    if (is_ret) // return by value
      res = "choreo::spanned_data<choreo::" + STR(sty->e_type) + ", " +
            std::to_string(sty->Dims()) + ">";
    else // always pass in by reference
      res = "const choreo::spanned_view<choreo::" + STR(sty->e_type) + ", " +
            std::to_string(sty->Dims()) + "> &";
  } else if (auto bitt = dyn_cast<BoundedITupleType>(&ty)) {
    assert(bitt->Dims() == 1);
    (void)bitt;
    res = "int";
  } else if (isa<StreamType>(&ty))
    res = "cudaStream_t";
  else
    choreo_unreachable("unsupported host function type: " + STR(ty) + ".");

  if (isa<ScalarType>(&ty)) assert(!is_ref);
  // if (isa<ScalarType>(&ty) && is_ref) res += "&";

  return res;
}

static inline std::string KernelTypeStringify(const Choreo::BaseType& type) {
  switch (type) {
  case Choreo::BaseType::F32: return "float";
  case Choreo::BaseType::U32: return "unsigned int";
  case Choreo::BaseType::U16: return "uint16_t";
  case Choreo::BaseType::U8: return "uint8_t";
  case Choreo::BaseType::S32: return "int";
  case Choreo::BaseType::S16: return "int16_t";
  case Choreo::BaseType::S8: return "int8_t";
  case Choreo::BaseType::BOOL: return "bool";
  default:
    choreo_unreachable("unsupported kernel function type: " + STR(type) + ".");
  }
}

inline const std::string FineName(const std::string& input) {
  std::string result = input;

  // Replace all occurrences of '$' with '_'
  std::replace(result.begin(), result.end(), '$', '_');

  return result;
}

inline int MemLevel(Storage s) {
  switch (s) {
  case Storage::LOCAL: return 0;
  case Storage::SHARED: return 1;
  case Storage::GLOBAL:
  case Storage::DEFAULT: return 2;
  default: choreo_unreachable("Unexpected storage type."); return -1;
  }
  return -1;
}

namespace {

inline const std::string UnScopedExpr(const std::string& input) {
  // Regular expression to match scoped names
  std::regex scopedNameRegex(
      R"((::[a-zA-Z_][a-zA-Z0-9_]*|::\$[0-9]+)(::[a-zA-Z_][a-zA-Z0-9_]*|::\$[0-9]+)*)");

  // Output string
  std::string output;
  std::sregex_iterator it(input.begin(), input.end(), scopedNameRegex);
  std::sregex_iterator end;

  size_t lastPos = 0;

  // Iterate through all matches
  for (; it != end; ++it) {
    const std::smatch& match = *it;
    size_t matchPos = match.position();
    size_t matchLen = match.length();

    // Append the part of the input before the current match
    output += input.substr(lastPos, matchPos - lastPos);

    // Extract the last part of the scoped name
    std::string scopedName = match.str();
    size_t lastColon = scopedName.find_last_of("::");
    output += scopedName.substr(lastColon + 1);

    // Update the last processed position
    lastPos = matchPos + matchLen;
  }

  // Append the remaining part of the input after the last match
  output += input.substr(lastPos);

  return output;
}

// Remove all scope prefixes (e.g., ::foo::bar or ::$0) and return only the last
// symbol. Handles both normal identifiers and dynamic shape variables like $0.
// inline const std::string UnScopedExpr(const std::string& input) {
//   // Match ::identifier or ::$number
//   std::regex scopedNameRegex(R"((::[a-zA-Z_][a-zA-Z0-9_]*|::\$[0-9]+))");
//   std::sregex_iterator it(input.begin(), input.end(), scopedNameRegex);
//   std::sregex_iterator end;
//
//   std::string last;
//   for (; it != end; ++it) {
//     last = it->str();
//   }
//   if (!last.empty()) {
//     // Remove the leading '::'
//     return last.substr(2);
//   }
//   // If no match, return the original input
//   return input;
// }

inline const std::string UnScopedSizeExpr(const Type& ty) {
  return UnScopedExpr(SizeExprOf(ty, true));
}

inline static std::string UnScopedValueItemString(const ValueItem& input) {
  if (auto bo = dyn_cast<sbe::BinaryOperation>(input)) {
    return "(" + UnScopedValueItemString(bo->GetLeft()) + " " +
           STR(bo->GetOpCode()) + " " +
           UnScopedValueItemString(bo->GetRight()) + ")";
  } else if (auto name = VISym(input)) {
    size_t last_colon = name->find_last_of(":");
    return name->substr(last_colon + 1);
  } else if (auto iv = VIInt(input))
    return std::to_string(*iv);
  else
    choreo_unreachable("unexpected value item: " + STR(input) + ".");
}

inline const std::string UnScopedValueItem(const ValueItem& input) {
  return UnScopedValueItemString(input);
}

} // end anonymous namespace

} // end namespace Choreo

#endif // CHOREO_CODEGEN_HPP_
