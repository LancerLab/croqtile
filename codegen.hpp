#ifndef CHOREO_CODEGEN_HPP_
#define CHOREO_CODEGEN_HPP_

#include <string>
#include <thread>

#include "memcheck.hpp"
#include "valbind.hpp"
#include "visitor.hpp"

namespace Choreo {

// map a future to its associated buffer. The maps are indexed by the choreo
// function names
using FutureBufferMap =
    std::map<std::string, std::map<std::string, std::string>>;

struct SymbolDetail {
  // information from choreo code
  std::string name;
  ptr<Type> type = nullptr;
  bool is_return = false;
  bool is_reference = false;
  int p_index = -1; // index of parameter in choreo function decl

  // information used for codegen
  std::string host_name;   // mapped host name
  std::string device_name; // mapped device name
  int d_index = -1;        // index of symbol in device function parameter list

  std::string h_name; // some target like factor requires host function names
  int h_index = -1;   // some target like factor requires host function indices

  SymbolDetail(const std::string& n, const ptr<Type>& t, bool ref = false,
               int index = -1, bool ret = false)
      : name(n), type(t), is_return(ret), is_reference(ref), p_index(index) {
    assert((!(IsParameter() && IsReference())) &&
           "Parameters are not references.");
  }

  bool IsParameter() const { return p_index != -1; }
  bool IsReturn() const { return is_return; }
  bool IsReference() const { return is_reference; }
  void SetAsReturn() { is_return = true; }
};

struct LaunchConfig {
  size_t grid_dim_z = 1;
  size_t grid_dim_y = 1;
  size_t grid_dim_x = 1;
  size_t block_dim_z = 1;
  size_t block_dim_y = 1;
  size_t block_dim_x = 1;
};

using SymbolDetails = std::map<std::string, std::vector<SymbolDetail>>;
using LaunchDetails = std::map<std::string, LaunchConfig>;
using ReturnSymbols = std::map<std::string, std::string>;

enum PassedOrDeclaredSymbolKind : int {
  PDSYM_NONE = 0,
  PDSYM_PARAMETERS_ONLY = 0x1, // only the paramters declared
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

  size_t param_count = 0;

public:
  const std::vector<SymbolDetail>&
  GetFunctionSymbols(const std::string& fname) const {
    return all_syms.at(fname);
  }
  std::vector<SymbolDetail>& GetFunctionSymbols(const std::string& fname) {
    return all_syms[fname];
  }

  const LaunchConfig& GetFunctionLaunch(const std::string& fname) const {
    return launches.at(fname);
  }
  LaunchConfig& GetFunctionLaunch(const std::string& fname) {
    return launches[fname];
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

  void AddSymbolDetail(const std::string fname, const SymbolDetail& sd) {
    if (!all_syms.count(fname))
      all_syms[fname] = {sd};
    else
      all_syms[fname].emplace_back(sd);
  }

  void SetLaunchDetail(const std::string fname, const LaunchConfig& lc) {
    launches[fname] = lc;
  }

  void SetReturnSymbol(const std::string fname, const std::string& rs) {
    returns[fname] = rs;
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

  // without return
  FilterRange<SymbolDetail> GetDeviceAllocIns(const std::string& fname) {
    return FilterRange<SymbolDetail>(
        this->all_syms[fname], [this](const SymbolDetail& sd) {
          return this->IsPassedOrDeclaredSymbols(sd, PDSYM_NO_RETURN |
                                                         PDSYM_ALLOC_IN_DEVICE);
        });
  }

  SymbolDetail GetReturn(const std::string& fname) const {
    assert(all_syms.count(fname) != 0);
    assert(returns.count(fname) != 0);
    for (auto& item : all_syms.at(fname))
      if (item.name == returns.at(fname)) return item;
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
};

/////////////////////////////////////////////////////////////
///  Util functions shared between targets
/////////////////////////////////////////////////////////////

inline constexpr const char* backpatch_filename =
    "__choreo_kernel_file_name_that_will_be_back_patched_soon_ok_enough_i_am_"
    "bored__";

inline static std::string create_unique_path() {
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
  else if (auto sty = dyn_cast<SpannedType>(&ty)) {
    if (is_ret) // return by value
      return "choreo::spanned_data<choreo::" + STR(sty->f_type) + ", " +
             std::to_string(sty->Dims()) + ">";
    else // pass in by reference
      return "const choreo::spanned_view<choreo::" + STR(sty->f_type) + ", " +
             std::to_string(sty->Dims()) + "> &";
  }
  choreo_unreachable("unsupported host function type.");
  return "";
}

static inline std::string KernelTypeStringify(const Choreo::FundamentalType& type) {
  switch (type) {
    case Choreo::FundamentalType::F32: return "float";
    case Choreo::FundamentalType::U32: return "unsigned int";
    case Choreo::FundamentalType::U16: return "uint16_t";
    case Choreo::FundamentalType::U8: return "uint8_t";
    case Choreo::FundamentalType::S32: return "int";
    case Choreo::FundamentalType::S16: return "int16_t";
    case Choreo::FundamentalType::S8: return "int8_t";
    default:
      choreo_unreachable("unsupported kernel function type.");
  }
}

} // end namespace Choreo

#endif // CHOREO_CODEGEN_HPP_
