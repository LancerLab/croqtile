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
  std::string name;
  ptr<Type> type = nullptr;
  bool is_return = false;
  int p_index = -1; // index of parameter
};

inline bool IsParameter(const SymbolDetail& sd) { return sd.p_index != -1; }

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

struct CodeGenInfo {
  SymbolDetails storages;
  LaunchDetails launches;
  ReturnSymbols returns;

  // argument's index by its scoped name
  int GetArgumentIndex(const std::string& fname,
                       const std::string& pname) const {
    if (!PrefixedWith(pname, "::"))
      choreo_unreachable("Not a in-scope symbol name (" + pname + ").");

    for (auto& item : storages.at(fname))
      if (item.name == pname) return item.p_index;

    return -1;
  }

  std::vector<SymbolDetail> GetParameters(const std::string& fname) const {
    std::vector<SymbolDetail> res;
    for (auto& item : storages.at(fname)) {
      if (item.p_index != -1) res.push_back(item);
    }
    return res;
  }

  std::vector<SymbolDetail> GetGlobals(const std::string& fname,
                                       bool ignore_return = false) const {
    std::vector<SymbolDetail> res;
    for (auto& item : storages.at(fname)) {
      if (ignore_return && item.is_return) continue;

      if (item.p_index != -1) {
        res.push_back(item);
        continue;
      }

      if (auto sty = dyn_cast<SpannedType>(item.type))
        if ((sty->GetStorage() == Storage::GLOBAL) ||
            (sty->GetStorage() ==
             Storage::DEFAULT /* default is mapped as global */))
          res.push_back(item);
    }
    return res;
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

} // end namespace Choreo

#endif // CHOREO_CODEGEN_HPP_
