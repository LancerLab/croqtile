#ifndef __CHOREO_CODEGEN_HPP__
#define __CHOREO_CODEGEN_HPP__

#include <thread>

#include "valbind.hpp"
#include "visitor.hpp"
#include "MemUsageCheck.hpp"

namespace Choreo {

// utility macros define here
#define __TRACE_EACH_VISIT__(d)       \
  if (trace_visit) {                  \
    os << d.TypeNameString() << ": "; \
    os << "\n";                       \
  }


// Codegenerators for targets
struct CodeGenerator : public VisitorWithSymTab {
  std::ostream &os;

  // some default method for the derived classes that do not want to override.
  bool BeforeVisitImpl(AST::Node &) override { return true; }
  bool AfterVisitImpl(AST::Node &) override { return true; }

  CodeGenerator(std::ostream &o, const ptr<SymbolTable> &symtab)
      : VisitorWithSymTab(symtab), os(o) {
    if (symtab == nullptr)
      choreo_unreachable("symbol table must be initialized.");
  }
};


/////////////////////////////////////////////////////////////
///  Util functions shared between targets
/////////////////////////////////////////////////////////////

inline constexpr const char *backpatch_filename =
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

inline static void ReplaceInString(std::string &str, const std::string &from,
                                   const std::string &to) {
  if (from.empty()) return;

  size_t startPos = 0;
  while ((startPos = str.find(from, startPos)) != std::string::npos) {
    str.replace(startPos, from.length(), to);
    startPos += to.length();  // In case 'to' contains 'from', like replacing
                              // 'x' with 'yx'
  }
}


static inline std::string HostTypeString(const Choreo::Type &ty,
                                         bool is_ret = false) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (auto sty = dyn_cast<SpannedType>(&ty)) {
    if (is_ret)  // return by value
      return "choreo::spanned_data<choreo::" + STR(sty->f_type) + ", " +
             std::to_string(sty->Dims()) + ">";
    else  // pass by reference
      return "const choreo::spanned_view<choreo::" + STR(sty->f_type) + ", " +
             std::to_string(sty->Dims()) + "> &";
  }
  choreo_unreachable("unsupported host function type.");
  return "";
}


}  // end namespace Choreo

#endif  // __CHOREO_CODEGEN_HPP__
