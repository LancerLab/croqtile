#ifndef CHOREO_CODEGEN_HPP_
#define CHOREO_CODEGEN_HPP_

#include <string>
#include <thread>

#include "memcheck.hpp"
#include "valbind.hpp"
#include "visitor.hpp"

namespace Choreo {

extern Storage GCUDeviceParallelLevel(int);
extern int GCUDeviceParallelDepth(Storage);

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
        attr(a), need_iv_prefix(iv) {
    assert((!(IsParameter() && IsReference())) &&
           "Parameters are not references.");
  }

  bool IsParameter() const { return p_index != -1; }
  bool IsReturn() const { return !rty_str.empty(); }
  bool IsReference() const { return is_reference; }
  void SetAsReturn(const std::string& t) {
    assert(!t.empty());
    rty_str = t;
  }
};

struct LaunchConfig {
  ValueItem grid_dim_z = sbe::nu(1);
  ValueItem grid_dim_y = sbe::nu(1);
  ValueItem grid_dim_x = sbe::nu(1);
  ValueItem block_dim_z = sbe::nu(1);
  ValueItem block_dim_y = sbe::nu(1);
  ValueItem block_dim_x = sbe::nu(1);
  ValueItem warp_dim_x = sbe::nu(1);
  ValueItem warp_dim_y = sbe::nu(1);
  ValueItem warp_dim_z = sbe::nu(1);

  // reset the warp dimensions to 1
  void ResetWDims() {
    warp_dim_x = sbe::nu(1);
    warp_dim_y = sbe::nu(1);
    warp_dim_z = sbe::nu(1);
  }

  // reset the block dimensions to 1
  void ResetBDims() {
    block_dim_x = sbe::nu(1);
    block_dim_y = sbe::nu(1);
    block_dim_z = sbe::nu(1);
  }

  // reset the grid dimensions to 1
  void ResetGDims() {
    grid_dim_x = sbe::nu(1);
    grid_dim_y = sbe::nu(1);
    grid_dim_z = sbe::nu(1);
  }

  void SetWarpDims(const ValueList& dims) {
    ResetWDims();
    switch (dims.size()) {
    case 3: warp_dim_z = dims[2]; [[fallthrough]];
    case 2: warp_dim_y = dims[1]; [[fallthrough]];
    case 1: warp_dim_x = dims[0]; break;
    default: choreo_unreachable("The number of dimensions is not supported.");
    }
  }

  void SetBlockDims(const ValueList& dims) {
    ResetBDims();
    switch (dims.size()) {
    case 3: block_dim_z = dims[2]; [[fallthrough]];
    case 2: block_dim_y = dims[1]; [[fallthrough]];
    case 1: block_dim_x = dims[0]; break;
    default: choreo_unreachable("The number of dimensions is not supported.");
    }
  }

  void SetGridDims(const ValueList& dims) {
    ResetGDims();
    switch (dims.size()) {
    case 3: grid_dim_z = dims[2]; [[fallthrough]];
    case 2: grid_dim_y = dims[1]; [[fallthrough]];
    case 1: grid_dim_x = dims[0]; break;
    default: choreo_unreachable("The number of dimensions is not supported.");
    }
  }

  void OverwriteGDimsByBDims() {
    grid_dim_x = block_dim_x;
    grid_dim_y = block_dim_y;
    grid_dim_z = block_dim_z;
  };
};

struct OtherTrait {
  bool has_parallelby = false;
  bool multiple_parallelby = false;
};

using SymbolDetails = std::map<std::string, std::vector<SymbolDetail>>;
using LaunchDetails = std::map<std::string, std::vector<LaunchConfig>>;
using ReturnSymbols = std::map<std::string, std::string>;
using FunctionTraits = std::map<std::string, OtherTrait>;
using SharedFutures = std::map<std::string, std::set<std::string>>;
using LocalFutures = std::map<std::string, std::set<std::string>>;

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
  FunctionTraits traits;
  SharedFutures shr_futs;
  LocalFutures loc_futs;

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

  const OtherTrait& GetFunctionTrait(const std::string& fname) const {
    return traits.at(fname);
  }
  OtherTrait& GetFunctionTrait(const std::string& fname) {
    return traits[fname];
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

  bool HasParallelBy(const std::string& fname) const {
    return GetFunctionTrait(fname).has_parallelby;
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
};

// Codegenerators for targets
struct CodeGenerator : public VisitorWithSymTab {
  // some default method for the derived classes that do not want to override.
  bool BeforeVisitImpl(AST::Node&) override { return true; }
  bool AfterVisitImpl(AST::Node&) override { return true; }

  CodeGenerator(const std::string& n, const ptr<SymbolTable>& symtab)
      : VisitorWithSymTab(n, symtab) {
    if (symtab == nullptr)
      choreo_unreachable("symbol table must be initialized.");
  }

  virtual void TraceEachVisit(AST::Node& n, bool detail = false,
                              const std::string& m = "") const {
    if (!trace_visit) return;
    if (detail)
      dbgs() << m << STR(n) << "\n";
    else
      dbgs() << m << n.TypeNameString() << "\n";
  }

#if 0
protected:
  std::vector<std::string> ProbeEnclosedIVs(const std::string& iv,
                                            AST::ForeachBlock& n) {
    assert(PrefixedWith(iv, "::") && "requires IV name to be scoped.");

    std::vector<std::string> res;
    bool ignore = true;
    for (auto rng : n.GetRanges()) {
      for (auto iv_name :
           within_map.at(InScopeName(cast<AST::LoopRange>(rng)->IVName()))) {
        if (iv_name == iv) {
          ignore = false;
          continue;
        }
        if (ignore) continue;
        res.push_back(iv_name);
      }
    }

    if (ignore)
      choreo_unreachable("symbol '" + iv + "' is not found in " + STR(n) + ".");

    // now probe further to find any other foreach/inc
    std::stack<ptr<AST::MultiNodes>> worklist;
    worklist.push(n.stmts);

    while (!worklist.empty()) {
      auto mn = worklist.top();
      worklist.pop();

      for (auto node : mn->AllSubs()) {
        if (auto fb = dyn_cast<AST::ForeachBlock>(node)) {
          for (auto rng : fb->GetRanges()) {
            for (auto iv_name : within_map.at(
                     InScopeName(cast<AST::LoopRange>(rng)->IVName())))
              res.push_back(iv_name);
          }
          worklist.push(fb->stmts);
        } else if (auto ib = dyn_cast<AST::IncrementBlock>(node)) {
          for (auto iv : ib->GetIterationVars())
            res.push_back(cast<AST::Identifier>(iv)->name);
          worklist.push(ib->stmts);
        }
      }
    }
    return res;
  }
#endif
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

static inline std::string HostTypeStringify(const Choreo::Type& ty,
                                            bool is_ret = false) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (isa<Half8Type>(&ty))
    return "choreo::half8";
  else if (isa<HalfType>(&ty))
    return "choreo::half";
  else if (isa<BFP16Type>(&ty))
    return "choreo::bfp16";
  else if (isa<FloatType>(&ty))
    return "float";
  else if (isa<DoubleType>(&ty))
    return "double";
  else if (auto sty = dyn_cast<SpannedType>(&ty)) {
    if (is_ret) // return by value
      return "choreo::spanned_data<choreo::" + STR(sty->f_type) + ", " +
             std::to_string(sty->Dims()) + ">";
    else // pass in by reference
      return "const choreo::spanned_view<choreo::" + STR(sty->f_type) + ", " +
             std::to_string(sty->Dims()) + "> &";
  } else if (auto bitt = dyn_cast<BoundedITupleType>(&ty)) {
    assert(bitt->Dims() == 1);
    (void)bitt;
    return "int";
  } else
    choreo_unreachable("unsupported host function type: " + STR(ty) + ".");
  return "";
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
  case Choreo::BaseType::INT: return "int";
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
      R"((::[a-zA-Z_][a-zA-Z0-9_]*)(::[a-zA-Z_][a-zA-Z0-9_]*)*)");

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

inline const std::string UnScopedSizeExpr(const Type& ty) {
  return UnScopedExpr(SizeExprOf(ty, true));
}

inline static std::string UnScopedValueItemString(const ValueItem& input) {
  if (auto bo = dyn_cast<sbe::BinaryOperation>(input)) {
    return "(" + UnScopedValueItemString(bo->GetLeft()) + " " +
           STR(bo->GetOpCode()) + " " +
           UnScopedValueItemString(bo->GetRight()) + ")";
  } else if (auto name = VIStr(input)) {
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

inline int GetMaxParallelLevelFromNote(AST::ParallelBy& n) {
  auto pos = n.GetNote().find("mxl-");
  if (pos != std::string::npos)
    return std::stoi(n.GetNote().substr(pos + 4, pos + 5));
  return -1;
}

} // end anonymous namespace

} // end namespace Choreo

#endif // CHOREO_CODEGEN_HPP_
