#include "mem_reuse.hpp"
#include "ast.hpp"
#include "aux.hpp"
#include "context.hpp"
#include "types.hpp"
#include "visitor.hpp"

using namespace Choreo;

bool MemReuse::BeforeVisitImpl(AST::Node& n) {
  if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    cur_func_name = cf->name;
    parallel_level = 0;
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    parallel_level++;
    max_parallel_level = std::max(parallel_level, max_parallel_level);
    if (parallel_level == 1) {
      size_t shared_spm_size = spm_size_map[cur_func_name].shared_spm_size;
      if (shared_spm_size == 0) return true;
      shared_spm_name = SymbolTable::GetAnonName();
      auto shared_spm =
          AST::Make<AST::NamedVariableDecl>(n.LOC(), shared_spm_name);
      assert(shared_spm_size > 0 &&
             "Shared scratch pad memory size is not set.");
      auto ssty = MakeSpannedType(
          BaseType::U8, Shape(1, Size_t2Int(shared_spm_size)), Storage::SHARED);
      shared_spm->SetType(ssty);
      shared_spm->AppendNote("spm,");
      pb->stmts->values.insert(pb->stmts->values.begin(), shared_spm);
      SSTab().DefineSymbol(shared_spm_name, ssty);
      VST_DEBUG(dbgs() << "Defined shared scratch pad memory: "
                       << PSTR(shared_spm) << ", type: " << PSTR(ssty)
                       << ".\n");
    } else if (parallel_level == 2) {
      size_t local_spm_size = spm_size_map[cur_func_name].local_spm_size;
      if (local_spm_size == 0) return true;
      local_spm_name = SymbolTable::GetAnonName();
      auto local_spm =
          AST::Make<AST::NamedVariableDecl>(n.LOC(), local_spm_name);
      assert(local_spm_size > 0 && "Local scratch pad memory size is not set.");
      auto lsty = MakeSpannedType(
          BaseType::U8, Shape(1, Size_t2Int(local_spm_size)), Storage::LOCAL);
      local_spm->SetType(lsty);
      local_spm->AppendNote("spm,");
      pb->stmts->values.insert(pb->stmts->values.begin(), local_spm);
      SSTab().DefineSymbol(local_spm_name, lsty);
      VST_DEBUG(dbgs() << "Defined local scratch pad memory: "
                       << PSTR(local_spm) << ", type: " << PSTR(lsty) << ".\n");
    } else if (parallel_level == 3) {
    } else
      choreo_unreachable("The parallel-by level " +
                         std::to_string(parallel_level) + " is not supported.");
  }
  return true;
}

bool MemReuse::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::ParallelBy>(&n)) {
    if (parallel_level == 1) max_parallel_level = 0;
    parallel_level--;
  }
  return true;
}

bool MemReuse::Visit(AST::MultiNodes& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::MultiValues& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::IntLiteral& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::FloatLiteral& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::StringLiteral& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Boolean& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Expr& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::MultiDimSpans& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);
  if (isa<AST::Select>(n.init_expr)) return true;
  if (n.note.find("spm") != std::string::npos) return true;
  auto ty = GetSymbolType(n.name_str);
  if (auto sty = dyn_cast<SpannedType>(ty)) {
    auto sto = sty->GetStorage();
    if (sto == Storage::LOCAL || sto == Storage::SHARED) ApplyMemOffset(n, sto);
  }
  return true;
}
bool MemReuse::Visit(AST::IntTuple& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Assignment& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::IntIndex& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::DataType& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Identifier& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Parameter& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::ParamList& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);
  if (parallel_level == 1) {
    // TODO: handle the case of parallel-by: 1
  } else if (parallel_level == 2) {
  }
  return true;
}
bool MemReuse::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::WithIn& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Memory& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::SpanAs& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::DMA& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::ChunkAt& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Wait& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Call& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Rotate& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Select& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Return& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::LoopRange& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::IncrementBlock& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::FunctionDecl& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);
  return true;
}
bool MemReuse::Visit(AST::Program& n) {
  TraceEachVisit(n);
  return true;
}

bool MemReuse::HasError() {
  if (error_count > 0) {
    dbgs() << "Totally " << error_count << " errors have been detected.\n";
    return true;
  }
  return false;
}
