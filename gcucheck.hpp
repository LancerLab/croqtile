#ifndef __CHOREO_GCU_CHECK_INFO_HPP__
#define __CHOREO_GCU_CHECK_INFO_HPP__

// This apply the GCU target specific check and information annotation

#include "ast.hpp"
#include "visitor.hpp"

namespace Choreo {

class WorkingList {
private:
  std::unordered_map<std::string, AST::DMA*> string_to_dma;

public:
  // Add a symbol to the symbol table
  // emittable = 'a'
  // type_symbol = 'a_type'
  // emitted = 'DRAMType(FloatType(32), {1, 2})'
  void AddDMA(AST::DMA& dma) { string_to_dma.emplace(dma.future, &dma); }

  // Retrieve typename of a symbol
  AST::DMA* GetDMA(const std::string& mnemonic) {
    if (string_to_dma.find(mnemonic) != string_to_dma.end())
      return string_to_dma.at(mnemonic);
    return nullptr;
  }

  void Reset() { string_to_dma.clear(); }
};

struct GCUCheck : public VisitorWithSymTab {
private:
  std::ostream& os;
  size_t error_count = 0;

  std::unordered_map<std::string, AST::Parameter*> cur_params;
  WorkingList workinglist;
  int parallel_level = 0;
  int max_parallel_level = 0;
  int local_level = 0;

private:
  bool BeforeVisitImpl(AST::Node& n) {
    TraceEachVisit(n, "(pre)");
    if (isa<AST::ChoreoFunction>(&n)) {
      local_level = 0;
      cur_params.clear();
    } else if (isa<AST::ParallelBy>(&n)) {
      parallel_level++;
      assert((parallel_level < 3) && "unexpected parallel level.");
      max_parallel_level = parallel_level;
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) {
    TraceEachVisit(n, "(post)");
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      auto pty = cast<BoundedITupleType>(GetSymbolType(pb->biv));
      pty->AppendNote(":" +
                      std::to_string(max_parallel_level - parallel_level));

      parallel_level--;
      assert(parallel_level >= 0 && "Unexpected parallel level");
      if (parallel_level == 0) max_parallel_level = 0;
    }
    return true;
  }

  void TraceEachVisit(AST::Node& n, std::string sup = "") {
    if (trace_visit) os << n.TypeNameString() << sup << "\n";
  }

public:
  GCUCheck(const ptr<SymbolTable> s_tab, std::ostream& o = std::cout)
      : VisitorWithSymTab("gcu", s_tab), os(o) {}
  ~GCUCheck() {}

  bool Visit(AST::MultiNodes& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::MultiValues& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::IntLiteral& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Boolean& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Expr& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::MultiDimSpans& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::NamedTypeDecl& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::NamedVariableDecl& n) {
    TraceEachVisit(n);
    auto ty = GetSymbolType(n.name_str);
    if (!isa<SpannedType>(ty)) return true;
    auto st = cast<SpannedType>(ty)->GetStorage();
    switch (st) {
    case Storage::GLOBAL:
      if (parallel_level != 0) {
        Error(n.LOC(), "global variable '" + n.name_str +
                           "` mustn't be declared inside parallel-by.");
        error_count++;
      }
      break;
    case Storage::SHARED:
      if (parallel_level != 1) {
        Error(n.LOC(),
              "shared variable '" + n.name_str +
                  "` must be declared inside single level of parallel-by.");
        error_count++;
      } else if (local_level == 1 && parallel_level == 2) {
        // if parallel_level == 1, allow
        // eg. parallel p by 6 { shared; local; }
        Error(n.LOC(), "shared variable '" + n.name_str +
                           "` mustn't be declared within the same level of "
                           "parallel-by as local variables.");
        error_count++;
      }
      break;
    case Storage::LOCAL:
      if (parallel_level == 0) {
        Error(n.LOC(), "local variable '" + n.name_str +
                           "` must be declared inside parallel-by.");
        error_count++;
      } else if (local_level != 0 && parallel_level != local_level) {
        Error(n.LOC(), "local variable '" + n.name_str +
                           "` must be declared inside a level of parallel-by "
                           "that is identical to other local variables.");
        error_count++;
      } else if (local_level == 0)
        local_level = parallel_level;
      break;
    default:
      Error(n.LOC(), "can not declare variable '" + n.name_str + "` as " +
                         STR(st) + " inside choreo function.");
      error_count++;
      break;
    }
    return true;
  }
  bool Visit(AST::IntTuple& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Assignment& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::IntIndex& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::DataType& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Identifier& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Parameter& n) {
    TraceEachVisit(n);
    if (n.sym) cur_params.emplace(InScopeName(n.sym->name), &n);
    return true;
  }
  bool Visit(AST::ParamList& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ParallelBy& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WhereBind& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WithIn& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WithBlock& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Memory& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::SpanAs& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::DMA& n) {
    TraceEachVisit(n);
    // The user does not have to explicitly claim a global memory that requires
    // direct copy from host to device. Here Choreo judge if a spanned memory is
    // shadowed from the data movement. Later, codegen handles such a shadow.
    if (!isa<AST::ChunkAt>(n.from)) return true;
    auto f_name = cast<AST::ChunkAt>(n.from)->RefSymbol();

    // remember all DMA for last chain check
    workinglist.AddDMA(n);
    // handle chained info, filling the DMA chain.
    if (n.chained) {
      auto _chain_from_ptr = workinglist.GetDMA(n.chain_from);
      assert(_chain_from_ptr != nullptr &&
             "after primitive chained to non-exist future id\n");
      _chain_from_ptr->chained = true;
      _chain_from_ptr->chain_to = n.future;
    }

    auto sty = GetSpannedType(GetSymbolType(f_name));

    // storage level must be specified
    if (sty->GetStorage() == Storage::NONE) return false;

    // not referencing the parameter
    if (!cur_params.count(InScopeName(f_name))) return false;

    auto annotate_by_storage = [this, &f_name](Storage st) {
      switch (st) {
      case Storage::GLOBAL:
      case Storage::SHARED:
      case Storage::LOCAL: {
        auto p = cur_params[InScopeName(f_name)];
        p->attr = ATT_SHADOW_TO_GLOBAL;
        break;
      }
      default: break;
      }
    };
    if (auto to = dyn_cast<AST::ChunkAt>(n.to))
      annotate_by_storage(
          cast<SpannedType>(GetSymbolType(to->RefSymbol()))->GetStorage());
    else if (auto m = dyn_cast<AST::Memory>(n.to))
      annotate_by_storage(m->st);

    return true;
  }
  bool Visit(AST::ChunkAt& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Wait& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Call& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Rotate& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Select& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Return& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::LoopRange& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ForeachBlock& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::FunctionDecl& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ChoreoFunction& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::CppSourceCode& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Program& n) {
    TraceEachVisit(n);
    return true;
  }

  bool HasError() {
    if (error_count)
      os << "Totally " << error_count << " errors have been detected.\n";
    return error_count != 0;
  }
};

} // end namespace Choreo

#endif // __CHOREO_GCU_CHECK_INFO_HPP__
