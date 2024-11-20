#include "codegen_factor.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "choreo_header.inc"
#include "codegen.hpp"
#include "codegen_factor_types.hpp"
#include "factor_script.inc"
#include "types.hpp"

#ifndef __CHOREO_FACTOR_DIR__
#error "missing macro definition of __CHOREO_FACTOR_DIR__"
#endif

// utility macros define here
using namespace Choreo;
using namespace Choreo::Factor;

inline const std::string ValueSTR(const ValueItem& vi) {
  if (auto i = dyn_cast<int>(&vi))
    return "Value(" + std::to_string(*i) + ")";
  else
    return STR(vi);
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

bool FactorCodeGen::ContainsLoopVar(const std::string& iv) const {
  for (auto& loop_var : loop_vars)
    if (loop_var.count(iv)) return true;
  return false;
}

bool FactorCodeGen::BeforeVisitImpl(AST::Node& n) {
  TraceEachVisit(n);

  if (auto c = dyn_cast<AST::ChoreoFunction>(&n)) {
    ClearChoreoFunctionStates();
    fname = c->name;
    factor_fname = "__choreo_" + fname;

    // declare a factor function with proper name
    fs << R"(#include <vector>

#include "gcu/factor/factor.h"

using namespace factor;
)";
    fs << "void " << factor_fname << "() {\n";
    this->IncrementIndent();
    fs << indent;
    fs << "include_(\"" << backpatch_filename << "\");\n";
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level++;
  } else if (isa<AST::ForeachBlock>(&n)) {
    loop_vars.push_back({});
  }

  return 0;
}

// CLEAN
bool FactorCodeGen::AfterVisitImpl(AST::Node& n) {
  TraceEachVisit(n);
  if (isa<AST::Program>(&n)) {
    if (HasError()) return false; // do not generate code when error happens

    outs() << "\n# step 4: generate the host source\n";
    outs() << "host_src=" << host_filename << "\n";
    outs() << "echo \"#include \\\"\"${gcu_target_string}\"_lib" << factor_fname
           << ".h\\\"\" > ${host_src}\n";
    outs() << "cat <<'EOF' >> ${host_src}\n";
    outs() << hs.str() << "\nEOF\n\n";

    outs() << "\n# step 5: JIT compile and execute\n";
    outs() << "# TODO: enable workflow of AOT compilation\n";
    outs() << "factor_function=" << factor_fname << "\n";
    outs() << R"(
if command -v nvim &> /dev/null
then
  EDITOR=nvim
else
  EDITOR=less
fi

show_usage() {
    echo "    Usage: $0 | --execute           -> compile and execute choreo in factor
                    | --statistics        -> show Line Of Code (LOC) statistic compare between kernel code boosted w./w.o. Choreo
                    | --show-kernel       -> show the generated inner kernel code
                    | --show-tileflow     -> show the generated tileflow code scheduled by choreo
                    | --show-host         -> show the generated host side boilerplates
                    | --show-choreo       -> show the choreo source code"
    exit 1
}
)";
    outs() << R"(
if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then
)";
    outs() << "  export FACTOR_INSTALL="
           << STRINGIZE(__CHOREO_FACTOR_DIR__)
                        << "\n# JIT compile and execute\n";
    if (compile_with_dynshape) outs() << "VIEW_CONFIG=1 ENABLE_DYNSHAPE=1 ";
    outs() << build_path
           << "/factor_script.sh ${factor_src} ${factor_bin} ${host_src} "
              "${factor_function} "
              "${gcu_arch} ${gcu_resource}";
    outs() << R"script(
elif [ "$1" == "--statistics" ]; then
  echo ">>>> Line of Code without Choreo"
  wc -l ${factor_src} ${host_src} ${kernel_src}
  echo ">>>> Line of Code with Choreo"
  wc -l ~/choreo/demo/elementwise_add.co
  # grep -v '^ *//' ~/choreo/demo/elementwise_add.co | wc -l
elif [ "$1" == "--show-kernel" ]; then
  ${EDITOR} ${kernel_src}
elif [ "$1" == "--show-host" ]; then
  ${EDITOR} ${host_src}
elif [ "$1" == "--show-tileflow" ]; then
  ${EDITOR} ${factor_src}
elif [ "$1" == "--show-choreo" ]; then
  ${EDITOR} ~/choreo/demo/elementwise_add.co
else
  show_usage
fi
)script";

  } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    auto fty = cast<FunctionType>(f->GetType());
    fs << "}\n\n";

    fs << "MODULE_REGISTER(\"lib" << factor_fname << "\", " << factor_fname
       << ");"; // end the factor function definition

    OutputScript(fty);
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level--;
    if (parallel_level == 0) {
      this->DecrementIndent();
      fs << this->indent << "}); // end of choreo-factor kernel function\n";
    }
  } else if (auto f = dyn_cast<AST::ForeachBlock>(&n)) {
    // erase the loop variables
    assert(!loop_vars.empty());
    loop_vars.pop_back();

    const auto& range_nodes = f->getRangeNodes();
    for (int j = range_nodes->Count() - 1; j >= 0; --j) {
      auto name = cast<AST::LoopRange>(range_nodes->ValueAt(j))->IVName();
      int dec_by = 1;
      bool multiple_bounds = !cur_bounded_vars[name].empty();
      if (multiple_bounds) dec_by = cur_bounded_vars[name].top().size();
      for (int i = dec_by - 1; i >= 0; --i) {
        DecrementIndent();
        fs << indent << "}); // end of choreo-foreach block on '";
        if (multiple_bounds)
          fs << cur_bounded_vars[name].top()[i];
        else
          fs << name;
        fs << "'.\n";
      }
    }
  } else if (auto wb = dyn_cast<AST::WithBlock>(&n)) {
    for (auto wi : wb->withins->AllSubs()) {
      auto w = cast<AST::WithIn>(wi);
      if (w->with && w->with_matchers) {
        cur_bounded_vars[w->with->name].pop();
      }
    }
    fs << indent << "} // end of with-in: " << n.LOC() << "\n";
  }
  return 0;
}

bool FactorCodeGen::Visit(AST::MultiNodes&) { return true; }
bool FactorCodeGen::Visit(AST::MultiValues&) { return true; }
bool FactorCodeGen::Visit(AST::IntLiteral&) { return true; }
bool FactorCodeGen::Visit(AST::Boolean&) { return true; }
bool FactorCodeGen::Visit(AST::Expr&) { return true; }
bool FactorCodeGen::Visit(AST::MultiDimSpans&) { return true; }
bool FactorCodeGen::Visit(AST::NamedTypeDecl&) { return true; }

// handle stmts like:
//   f32 [a.span] g_buffer;
//   local f32[f1.span] l_buffer;
//
// ast like:
//   NamedVariableDecl
//   CLEAN
bool FactorCodeGen::Visit(AST::NamedVariableDecl& node) {
  TraceEachVisit(node);

  auto nty = NodeType(node);
  auto sym = node.name_str;

  if (auto s = dyn_cast<AST::Select>(node.init_expr)) {
    assert(!s->inDMA);
    size_t val_count = s->expr_list->Count();
    assert(val_count >= 2);
    fs << this->indent << "auto " << sym << " = ";
    for (size_t i = 0; i < val_count - 1; i++) {
      std::string select_factor_str;
      auto factor = cast<IntegerType>(s->select_factor->GetType());
      if (auto expr = factor->GetValidExpression())
        select_factor_str = STR(
            expr.value()); // use the expression simplified by value numbering
      else
        select_factor_str = ExprSTR(s->select_factor);

      fs << "select_(" << select_factor_str << "== " << i << ", "
         << ExprSTR(s->expr_list->ValueAt(i))
         << (i < val_count - 1 ? ", " : "");
    }
    fs << PSTR(s->expr_list->AllValues().back())
       << std::string(val_count - 1, ')') << ";\n";

    return true;
  }

  // TODO(albert): 'a.span' will be replace to the type-decl related to 'a'
  // TODO(albert): refine this function with TYPE_STR new API
  if (auto sty = dyn_cast<SpannedType>(nty)) {
    assert(isa<SpannedType>(GetSymbolType(sym)) && "Inconsistent types!");
    if (auto e = dyn_cast<AST::Expr>(node.init_expr);
        e && isa<AST::SpanAs>(e->GetR())) {
      assert(e->IsReference());
      auto sa = dyn_cast<AST::SpanAs>(e->GetR());
      int arg_idx = cgi->GetArgumentIndex(fname, InScopeName(sa->id->name));
      std::string buffer_name =
          arg_idx < 0 ? sa->id->name : "args[" + std::to_string(arg_idx) + "]";
      auto sty = dyn_cast<SpannedType>(node.GetType());
      assert(sty);
      std::string storage_type = stringify(sty->GetStorage());
      std::string base_type = stringify(Choreo::BaseType(sty->f_type));
      fs << indent << "auto " << sym << " = bitcast_(" << storage_type << "("
         << base_type << ", {";

      // TODO(wsj): has_dynamic is hard to decide. Maybe still need to implement
      // span_as as mdspan?
      bool all_intLiteral_shape = true;
      for (auto value : sa->list->AllValues()) {
        auto value_expr = dyn_cast<AST::Expr>(value);
        if (!(value_expr->IsReference() &&
              isa<AST::IntLiteral>(value_expr->GetReference()))) {
          all_intLiteral_shape = false;
          break;
        }
      }
      std::string orig_delimiter = sa->list->delimiter;
      sa->list->SetDelimiter(", ");
      if (all_intLiteral_shape) {
        fs << PSTR(sa->list) << "}), " << buffer_name << ");\n";
      } else {
        for (size_t i = 0; i < sa->list->Count(); ++i)
          fs << (i != 0 ? ", " : "") << "-1";
        fs << "}), " << buffer_name << ", {";
        for (size_t i = 0; i < sa->list->Count(); ++i) {
          auto value = sa->list->ValueAt(i);
          auto value_expr = dyn_cast<AST::Expr>(value);
          auto expr_str = PSTR(value_expr);
          for (auto& [id_name, _] : idnm_rts) {
            expr_str = RegexReplaceAll(expr_str, "\\b" + id_name + "\\b",
                                       named_dim_ref_prefix + id_name);
          }
          fs << (i != 0 ? ", " : "") << expr_str;
        }
        fs << "});\n";
        sa->list->SetDelimiter(orig_delimiter);
      }
    } else if (factor_symbols.Exists(InScopeName(sym))) {
      // factor weird behavior: only the output needs alloc
      if (MemLevel(sty->GetStorage()) < 2 ||
          cgi->IsReturnSymbol(fname, InScopeName(sym))) {
        if (factor_symbols.GetTypeName(InScopeName(sym)) == "SRAMType")
          fs << indent << "auto " << sym << " = alloc_("
             << factor_symbols.GetTypeName(InScopeName(sym))
             << ").shared_(SharedType::kBlockShared);\n";
        else
          fs << indent << "auto " << sym << " = alloc_("
             << factor_symbols.GetTypeName(InScopeName(sym)) << ");\n";
      }
    } else {
      std::string storage_type = stringify(sty->GetStorage());
      std::string base_type = stringify(sty->ElementType());
      std::ostringstream _os, _os_shared;
      _os << "auto " << sym << " = alloc_(" << storage_type << "(" << base_type
          << "," << ReplaceRuntimeNames(LSTR(sty->GetShape()), "", false) << ")"
          << ");\n";
      _os_shared << "auto " << sym << " = alloc_(" << storage_type << "("
                 << base_type << ","
                 << ReplaceRuntimeNames(LSTR(sty->GetShape()), "", false) << ")"
                 << ").shared_(SharedType::kBlockShared);\n";
      if (storage_type == "DRAMType")
        fs << indent << _os.str();
      else if (storage_type == "SRAMType") {
        alloc_in_fs << "    " << _os_shared.str();
      } else
        alloc_in_fs << "    " << _os.str();

      if (node.init_value) {
        // generate "memset_()" action when span-initializer exists
        fs << indent << "auto " << sym << "_init = alloc_dma_("
           << ((storage_type == "L1Type") ? "SDMAType()" : "CDMAType()")
           << ");\n";

        // generate "memset_()" action to initiate each alloc_memory with value
        // 0
        fs << indent << "memset_(" << sym << "_init, " << sym << ", 0);\n";
      }
    }
  } else if (isa<IntegerType>(nty) || isa<ITupleType>(nty)) {
    // simply ignore the generation of integers since valno has propagate the
    // values on the use sites
  } else {
    choreo_unreachable("non-spanned (" + PSTR(nty) + ") is not yet supported.");
    // TODO(albert): handle anon case
    fs << this->indent;
    fs << "auto " << node.name_str << " = alloc_(?";
    fs << ");\n";
  }

  return true;
}

bool FactorCodeGen::Visit(AST::IntTuple&) { return true; }

bool FactorCodeGen::Visit(AST::Assignment& node) {
  if (auto sa = dyn_cast<AST::SpanAs>(node.value)) {
    int arg_idx = factor_symbols.GetSymbolIndex(sa->id->name);
    std::string buffer_name = sa->id->name;
    if (isa<FutureType>(GetSymbolType(sa->id->name))) {
      assert(fut_buf->at(fname).count(sa->id->name));
      buffer_name = fut_buf->at(fname).at(sa->id->name);
    }
    if (arg_idx >= 0) buffer_name = "args[" + std::to_string(arg_idx) + "]";
    auto sty = cast<SpannedType>(node.GetType());
    fs << indent << "auto " << node.name << " = bitcast_("
       << stringify(sty->GetStorage()) << "(" << stringify(sty->ElementType())
       << ", {";

    // TODO(wsj): has_dynamic is hard to decide. Maybe still need to implement
    // span_as as mdspan?
    bool all_intLiteral_shape = true;
    for (auto value : sa->list->AllValues()) {
      auto value_expr = dyn_cast<AST::Expr>(value);
      if (!(value_expr->IsReference() &&
            isa<AST::IntLiteral>(value_expr->GetReference()))) {
        all_intLiteral_shape = false;
        break;
      }
    }
    std::string orig_delimiter = sa->list->delimiter;
    sa->list->SetDelimiter(", ");
    if (all_intLiteral_shape) {
      fs << PSTR(sa->list) << "}), " << buffer_name << ");\n";
    } else {
      for (size_t i = 0; i < sa->list->Count(); ++i)
        fs << (i != 0 ? ", " : "") << "-1";
      fs << "}), " << buffer_name << ", {";
      for (size_t i = 0; i < sa->list->Count(); ++i) {
        auto value = sa->list->ValueAt(i);
        auto value_expr = dyn_cast<AST::Expr>(value);
        auto expr_str = PSTR(value_expr);
        for (auto& [id_name, _] : idnm_rts) {
          expr_str = RegexReplaceAll(expr_str, "\\b" + id_name + "\\b",
                                     named_dim_ref_prefix + id_name);
        }
        fs << (i != 0 ? ", " : "") << expr_str;
      }
      fs << "});\n";
      sa->list->SetDelimiter(orig_delimiter);
    }
  } else if (isa<BoundedType>(NodeType(node)) ||
             isa<SpannedType>(NodeType(node)) ||
             isa<FutureType>(NodeType(node))) {
    fs << indent << "auto " << node.name << " = " << ExprSTR(node.value)
       << ";\n";
  }

  return true;
}
bool FactorCodeGen::Visit(AST::IntIndex&) { return true; }
bool FactorCodeGen::Visit(AST::DataType&) { return true; }

bool FactorCodeGen::Visit(AST::Identifier& n) {
  TraceEachVisit(n);
  (void)n;
  return true;
}

bool FactorCodeGen::Visit(AST::Parameter& p) {
  TraceEachVisit(p);
  (void)p;
  return true;
}

bool FactorCodeGen::Visit(AST::ParamList& pl) {
  TraceEachVisit(pl);
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::ParallelBy& by) {
  TraceEachVisit(by);
  if (parallel_level > 1) { return true; }

  fs << this->indent << "Dim3 grid_dim("
     << cgi->GetFunctionLaunch(fname).grid_dim_x << ");\n";
  fs << this->indent << "Dim3 block_dim("
     << cgi->GetFunctionLaunch(fname).block_dim_x << ");\n";

  // [Factor host] LaunchKernel statement:
  // symbols which are passed to the device are listed as launch parameters
  {
    std::ostringstream launch;
    launch << this->indent << "auto ts = launch_kernel_(\"" << factor_fname
           << "_parallel\", grid_dim, block_dim, args.back(), {";
    size_t index = 0;
    for (auto& item : GetFactorDeviceInParams()) {
      assert(item.d_index == (int)index);
      launch << ((index++ > 0) ? ", " : "");
      if (!item.h_name.empty())
        launch << item.h_name;
      else
        launch << item.device_name;
    }
    launch << "}, {"
           << ((void_return) ? "" : UnScopedName(cgi->GetReturnSymbol(fname)))
           << "});\n";

    if (debug_visit)
      VST_DEBUG(dbgs() << "[Factor Host] Launch Kernel:\n" << launch.str());
    fs << launch.str();
  }

  // [Factor-host] Return statement
  {
    std::ostringstream ret;
    ret << this->indent << "return std::vector<Value>{"
        << ((void_return) ? "" : UnScopedName(cgi->GetReturnSymbol(fname)))
        << "};\n";
    if (debug_visit)
      VST_DEBUG(dbgs() << "[Factor Host] Return:\n" << ret.str());
    fs << ret.str();
  }

  this->DecrementIndent();
  fs << this->indent
     << "}, true); // end of choreo-factor dataflow program\n\n";

  // [Factor Device] Function declaration
  {
    std::ostringstream dfun;
    {
      dfun << this->indent << "D(func_)(\"" << factor_fname << "_parallel\", ";

      // input arguments of factor device function
      dfun << "{";
      size_t index = 0;
      for (auto& item : GetFactorDeviceInParams())
        dfun << ((index++ > 0) ? ", " : "") << UnScopedName(item.name)
             << "_type";
      dfun << "}, ";

      // output argument
      dfun << "{" << ((void_return) ? "" : "output_type") << "},";

      // fixed parameter list
      dfun << " [&](auto args, auto results)";

      if (debug_visit)
        VST_DEBUG(dbgs() << "[Factor Device] Function Declaration:\n"
                         << dfun.str() << "\n");

      fs << dfun.str();
    }
  }

  fs << " {\n";
  this->IncrementIndent();

  // [Factor Device] A fixed pattern: name the parameters and handle dynamic
  // shapes
  {
    std::ostringstream drefs;
    // Generate references to global symbols
    for (auto& item : GetFactorDeviceInParams()) {
      drefs << this->indent << "auto & " << UnScopedName(item.name)
            << " = args[" << item.d_index << "];\n";
    }

    // generate a reference name of the output
    if (!void_return) {
      auto name = (cgi->HasReturnSymbol(fname))
                      ? UnScopedName(cgi->GetReturnSymbol(fname))
                      : "output";
      drefs << indent << "auto & " << name << " = results[0];\n";
    }

    // fixed, thread/block ids
    drefs << this->indent << "auto thread_id = thread_id_();\n";
    drefs << this->indent << "auto block_id = block_id_();\n";

    // dynamic-shape alias reference
    for (auto& [id_name, sym_name] : idnm_rts) {
      drefs << indent << "auto " << named_dim_ref_prefix << id_name << " = "
            << ReplaceFactorDynDimName(sym_name) << ";\n";
    }

    if (debug_visit)
      VST_DEBUG(dbgs() << "[Factor Device] Reference Symbols:\n"
                       << drefs.str() << "\n");

    fs << drefs.str();
  }

  // record the position since some codes requires declaration in function scope
  alloc_pos = fs.str().size();
  alloc_indent = indent;

  return true;
}

bool FactorCodeGen::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);
  // establish the binding
  auto lid = cast<AST::Identifier>(n.lhs);
  auto rid = cast<AST::Identifier>(n.rhs);
  bind_info.AddBind(SSTab().ScopedName(lid->name),
                    SSTab().ScopedName(rid->name));

  // also adds the value binding for the with-matchers
  if (!cur_bounded_vars[lid->name].empty()) {
    assert(!cur_bounded_vars[rid->name].empty());
    auto& lbvs = cur_bounded_vars[lid->name].top();
    auto& rbvs = cur_bounded_vars[rid->name].top();
    assert(lbvs.size() == rbvs.size());

    for (size_t i = 0; i < lbvs.size(); ++i) {
      bind_info.AddBind(SSTab().ScopedName(lbvs[i]),
                        SSTab().ScopedName(rbvs[i]));
    }
  }
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::WithIn& n) {
  TraceEachVisit(n);
  assert(n.with_matchers && "expect matcher to be exist.");

  // make with-in scopes be isolated
  fs << indent << "{ // start of with-in: " << n.LOC() << "\n";

  // associate with to the matcher.
  if (n.with && n.with_matchers) {
    std::vector<std::string> matchers;
    for (auto mn : n.with_matchers->AllValues()) {
      matchers.push_back(cast<AST::Identifier>(mn)->name);
    }
    cur_bounded_vars[n.with->name].push(matchers);
  }

  for (auto mn : n.with_matchers->AllValues()) {
    auto mname = cast<AST::Identifier>(mn)->name;
    fs << indent << "var_ " << mname << "(IntType(32));\n";
    fs << indent << mname << " = 0;\n";
  }

  return true;
};

bool FactorCodeGen::Visit(AST::WithBlock&) { return true; }

bool FactorCodeGen::Visit(AST::Memory& n) {
  TraceEachVisit(n);
  (void)n;
  return true;
}

bool FactorCodeGen::Visit(AST::SpanAs&) { return true; }

bool FactorCodeGen::Visit(AST::DMA& d) {
  TraceEachVisit(d);

  if (auto ph = dyn_cast<PlaceHolderType>(NodeType(d))) {
    assert(ph->Category() == TypeCategory::FUTURE);
    // TODO: optimize when it should be SDMA
    auto fty = cast<FutureType>(GetSymbolType(d.future));
    auto gcu_dma = "CDMA";
    if (GetSpannedType(fty)->GetStorage() == Storage::LOCAL) gcu_dma = "SDMA";
    alloc_in_fs << alloc_indent << "auto " << d.future << " = alloc_dma_("
                << gcu_dma << "Type());\n";
    return true;
  }

  // handle .to  in AST::Memory
  assert(isa<AST::ChunkAt>(d.from) && "Unexpected type for DMA's source.");
  assert(isa<AST::ChunkAt>(d.to) && "Unexpected type for DMA's destination.");

  auto ty = dyn_cast<FutureType>(d.GetType());
  assert(ty && "Invalid type of DMA statement!");

  // cook a valid future name
  auto future_name = d.future;
  if (future_name.empty()) {
    static size_t future_count = 0;
    future_name = "__choreo_anon_fut__" + std::to_string(future_count++);
  }

  auto dst_buffer_name = cast<AST::ChunkAt>(d.to)->RefSymbol();
  auto src_buffer_name = cast<AST::ChunkAt>(d.from)->RefSymbol();

  auto sty = GetSpannedType(NodeType(*d.from)); // source spanned type
  auto tty = GetSpannedType(NodeType(*d.to));   // dest spanned type

  int src_level = MemLevel(sty->GetStorage());
  int dst_level = MemLevel(tty->GetStorage());

  auto GenerateOffsetString = [this](AST::Node& n) {
    auto sty = GetSpannedType(NodeType(n));
    auto shape = sty->GetShape();
    size_t rank = sty->Dims();

    auto ca = cast<AST::ChunkAt>(&n);
    if (!ca->positions) {
      // symbol only, the offset is a multi-dim-zeros
      return "{" + DelimitedString(std::vector<size_t>(rank, 0)) + "}";
    }

    std::ostringstream offss;
    size_t dim_cursor = 0;
    for (auto& bv : ca->positions->AllValues()) {
      // It could either be identifier or a 'getith' expr
      if (auto id = dyn_cast<AST::Identifier>(bv)) {
        auto bvn = id->name;
        auto ty = cast<BoundedType>(NodeType(*id));
        // iterate over single bounded variables
        for (size_t it_idx = 0; it_idx < ty->Dims(); ++it_idx) {
          std::string name;
          if (within_map.count(bvn)) // with-matcher existed
            name = within_map[bvn][it_idx];
          else
            name = bvn;
          auto iv_str = ExprSTR(AST::Make<AST::Identifier>(id->LOC(), name));
          offss << "Value(" << RSTR(shape.ValueAt(dim_cursor)) << ")*"
                << iv_str;
          if (++dim_cursor < rank) offss << ",";
        }
      } else if (auto gi_exp = dyn_cast<AST::Expr>(bv)) {
        auto id = cast<AST::Expr>(gi_exp->GetL())->GetSymbol();
        auto ty = cast<BoundedType>(NodeType(*id));
        assert((ty->Dims() == 1) &&
               "Bounded ituple has not been supported yet.");
        assert((within_map.count(id->name) == 0) &&
               "Bounded ituple has not been supported yet.");
        auto iv_str = ExprSTR(bv);
        offss << "Value(" << RSTR(shape.ValueAt(dim_cursor)) << ")*" << iv_str;
        if (++dim_cursor < rank) offss << ",";
      } else
        choreo_unreachable("unsupported chunkat expressions.");
    }
    return "{" + offss.str() + "}";
  };

  // factor_symbols.Print(fs);
  int arg_idx = factor_symbols.GetSymbolIndex(src_buffer_name);
  src_buffer_name =
      arg_idx < 0 ? src_buffer_name : "args[" + std::to_string(arg_idx) + "]";

  // decide the dma allocation type
  auto DMATypeString = [](int src_lvl, int dst_lvl) {
    if ((src_lvl == 2 && dst_lvl == 2) || (src_lvl == 2 && dst_lvl == 1) ||
        (src_lvl == 1 && dst_lvl == 2) || (src_lvl == 1 && dst_lvl == 1))
      return "CDMAType";
    else
      return "SDMAType";
  };

  // buffer the allocation in another stream
  // if use pipeline-mode, make all cdma with shared_ annotation
  if (d.GetNote() != "use-fut") {
    if (d.chained == true && ((d.chain_to != "" && src_level > dst_level) ||
                              (d.chain_from != "" && src_level < dst_level)))
      alloc_in_fs << alloc_indent << "auto " << future_name << " = alloc_dma_("
                  << DMATypeString(src_level, dst_level) << "()).shared_();\n";
    else
      alloc_in_fs << alloc_indent << "auto " << future_name << " = alloc_dma_("
                  << DMATypeString(src_level, dst_level) << "());\n";
  }

  // decide the dma operation
  std::string dma_op = "";
  if (src_level >= dst_level)
    dma_op.append("async_load_");
  else
    dma_op.append("async_store_");

  ptr<AST::Node> chunkat_node = nullptr;

  if (isa<AST::Memory>(d.to) || isa<AST::Select>(d.to))
    chunkat_node = d.from;
  else if (auto c = cast<AST::ChunkAt>(d.to)) {
    if (!c->positions) // xxx.chunkat() => identifier
      chunkat_node = d.from;
    else
      chunkat_node = d.to;
  } else
    choreo_unreachable("factor: unsupported chunkat.");

  fs << indent << dma_op << "(" << future_name << ", " << src_buffer_name
     << ", " << dst_buffer_name << ", " << GenerateOffsetString(*chunkat_node);

  if (auto pcfg = dyn_cast<PadConfig>(d.config)) {
    std::vector<size_t> layout(sty->Dims());
    std::iota(layout.begin(), layout.end(), 0); // no transpose
    fs << ", {" << DelimitedString(layout) << "}, {"
       << DelimitedString(pcfg->pad_low) << "}, {"
       << DelimitedString(pcfg->pad_high) << "}, {"
       << DelimitedString(pcfg->pad_mid) << "}, " << pcfg->value.v;
  } else if (auto tcfg = dyn_cast<TransposeConfig>(d.config)) {
    auto& layout = tcfg->dim_values;
    fs << ", {" << DelimitedString(layout) << "}";
  }

  if (d.chained == false) {
    fs << ");\n";
    // synchornized dma must be waited
    if (!ty->IsAsync()) fs << indent << "wait_dma_(" << future_name << ");\n";
  } else {
    assert(ty->IsAsync() &&
           "Notifying DMA only apply to async primitives in factor lang.");
    if (d.chain_from != "") {
      if (src_level >= dst_level)
        fs << ").wait_on_(" << d.chain_from << ");\n";
      else
        fs << ").multi_wait_on_(" << d.chain_from << ");\n";
    }

    if (d.chain_to != "") {
      if (src_level >= dst_level)
        fs << ").multi_notify_(" << d.chain_to << ");\n";
      else
        fs << ").notify_(" << d.chain_to << ");\n";
    }
  }

  return true;
}

bool FactorCodeGen::Visit(AST::ChunkAt&) { return true; }

bool FactorCodeGen::Visit(AST::Wait& w) {
  TraceEachVisit(w);
  auto dmas = w.targets;
  assert(dmas && "Invalid wait target!");

  for (auto dma : dmas->AllValues()) {
    fs << this->indent << "wait_dma_(" << AST::STR(*dma) << ");\n";
  }

  return true;
}

bool FactorCodeGen::Visit(AST::Call& c) {
  TraceEachVisit(c);

  assert(c.arguments && "Invalid kernel call args!");

  if (!use_kernel_template)
    fs << this->indent << "call_(\"" << STR(*c.function) << "\", {";
  else
    fs << this->indent << "call_(\"" << STR(*c.function) << "_template_wrapper" << "\", {";
  size_t arg_num = c.arguments->Count();
  for (size_t index = 0; index < arg_num; ++index) {
    auto arg = c.arguments->ValueAt(index);
    fs << ExprSTR(arg)
       << ((isa<SpannedType>(NodeType(*arg))) ? ".addr_()" : "");
    if (index < arg_num - 1) fs << ",";
  }
  fs << "});\n";

  if (use_kernel_template) {
    // handle kernel template wrapper
    ks << "extern \"C\" void " << STR(*c.function) << "_template_wrapper(";
    for (size_t index = 0; index < arg_num; ++index) {
      auto arg = c.arguments->ValueAt(index);
      ks << KernelTypeStringify(cast<SpannedType>(arg->GetType())->f_type) << "* ";
      ks << "arg" << index;
      if (index < arg_num - 1) ks << ", ";
    }

    ks << ") {\n";
    ks << "  " << STR(*c.function);
    // c.template_params->SetDelimiter(", ");
    ks << "<" << STR(*c.template_params) << ">";
    ks << "(";
    bool need_delimiter = false;
    for (size_t index = 0; index < arg_num; ++index) {
      if (need_delimiter)
        ks << ", ";
      need_delimiter = true;
      ks << "arg" << index;
    }
    ks << ");\n";
    ks << "}\n";
  }

  return true;
}

bool FactorCodeGen::Visit(AST::Rotate& n) {
  TraceEachVisit(n);

  return true;
}

bool FactorCodeGen::Visit(AST::Select& c) {
  TraceEachVisit(c);
  assert(!c.inDMA);
  return true;
}

bool FactorCodeGen::Visit(AST::Return& returnNode) {
  TraceEachVisit(returnNode);
  return true;
}

bool FactorCodeGen::Visit(AST::LoopRange& n) {
  TraceEachVisit(n);
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::ForeachBlock& forNode) {
  TraceEachVisit(forNode);
  // auto ty = this->GetSymbolType("l2_tile");
  // ty->Print(os);
  // auto l2_tile_idx = itervars->ValueAt(0);
  // auto l1_tile_idx = itervars->ValueAt(1);
  //
  auto ranges = forNode.getRanges();
  for (size_t idx = 0; idx != ranges.size(); ++idx) {
    // TODO(albert): support non-unit stride in loop
    std::ostringstream _os;
    auto loop_range = cast<AST::LoopRange>(ranges[idx]);
    auto iv_name = loop_range->IVName();

    // get the lower/upper and stride for spanned iter var
    auto iv_type = this->GetSymbolType(iv_name);
    auto iv_sizes = cast<BoundedITupleType>(iv_type)->GetSizes();

    // NOTES: foreach block ranges between [0, UB),
    // it always use one integer indicating the UB
    // we can certainly use idx=0 directly

    // synthesise the emitting string
    /*
    A: with index={m,n} in [1,2] { foreach m {} }
    B: with index in [2] { foreach index {} }
    if (iv_type->Dims() == 1 && cur_bounded_vars[iv_name].empty()): A
    if (iv_type->Dims() == 1 && !cur_bounded_vars[iv_name].empty()):  B
    */
    if (iv_type->Dims() == 1 && cur_bounded_vars[iv_name].empty()) {
      fs << this->indent << "for_(" << iv_name;
      if (IsValidBound(loop_range->lbound))
        fs << " + (" << loop_range->lbound << ")";
      fs << ", " << ReplaceFactorDynDimName(STR(iv_sizes.ValueAt(0)));
      if (IsValidBound(loop_range->ubound))
        fs << " + (" << loop_range->ubound << ")";
      fs << ", ";
      if (IsValidStride(loop_range->stride))
        fs << loop_range->stride;
      else
        fs << 1;
      fs << ", [&](auto iv_" << iv_name << ") {\n";
      IncrementIndent();
      loop_vars.back().insert(iv_name);
      for (auto bind : bind_info.GetBinds(InScopeName(iv_name))) {
        auto bname = SSTab().UnScopedName(bind);
        loop_vars.back().insert(bname);
        fs << indent << "auto iv_" << SSTab().UnScopedName(bind) << " = iv_"
           << iv_name << ";\n";
      }
    } else {
      assert(!cur_bounded_vars[iv_name].empty() &&
             "can not find the bounded name.");
      assert((cur_bounded_vars[iv_name].top().size() == iv_sizes.Dims()) &&
             "can not find the bounded name.");
      size_t i = 0;
      for (auto name : cur_bounded_vars[iv_name].top()) {
        fs << this->indent << "for_(" << name;
        if (IsValidBound(loop_range->lbound))
          fs << " + (" << loop_range->lbound << ")";
        fs << ", " << ReplaceFactorDynDimName(STR(iv_sizes.ValueAt(i)));
        if (IsValidBound(loop_range->ubound))
          fs << " + (" << loop_range->ubound << ")";
        fs << ", ";
        if (IsValidStride(loop_range->stride))
          fs << loop_range->stride;
        else
          fs << 1;
        fs << ", [&](auto iv_" << name << ") {\n";
        std::string scoped_var = InScopeName(name);
        loop_vars.back().insert(name);
        IncrementIndent();
        for (auto bind : bind_info.GetBinds(InScopeName(name))) {
          auto bname = SSTab().UnScopedName(bind);
          loop_vars.back().insert(bname);
          if (bname != name)
            fs << indent << "auto iv_" << bname << " = iv_" << name << ";\n";
        }
        ++i;
      }
    }
  }
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::FunctionDecl& d) {
  TraceEachVisit(d);

  assert(d.name == fname && "incosistent in function names.");
  assert(isa<FunctionType>(d.GetType()) && "unexpected type.");

  auto& fty = *cast<FunctionType>(d.GetType());

  auto MapRuntimeShapeNames = [this](const ptr<SpannedType>& sty,
                                     const std::string& hp_name,
                                     size_t hp_index) {
    size_t dim_index = 0;
    for (auto vi : sty->GetShape().Value()) {
      if (auto vale = dyn_cast<ValueExpr>(&vi)) { // the dimension is symbolic
        assert(PrefixedWith(*vale, "::" + fname + "::") &&
               "unexpected symbol name.");

        auto dim_name = hp_name + ".shape()[" + std::to_string(dim_index) + "]";
        if (dims_info.count(*vale) == 0)
          dims_info[*vale] = {dim_name, hp_index, dim_index};

        idnm_rts.emplace(FineName(UnScopedName(*vale)), *vale);
      }
      dim_index++;
    }
  };

  // Go through all the symbols appears in factor host function, do:
  //
  //  - decide the host parameter names,
  //  - map the runtime shape dimensions to the real host code expression
  //  - decide the factor-host parameter names,
  //  - decide the factor-host parameter indices,
  //
  size_t host_pindex = 0;
  for (auto& item : GetFactorHostInParams()) {
    if (item.IsParameter()) {
      assert((int)host_pindex == item.p_index);
      item.host_name = GenHostParamName();
      if (auto sty = dyn_cast<SpannedType>(item.type))
        MapRuntimeShapeNames(sty, item.host_name, host_pindex);
    } else
      item.host_name = UnScopedName(item.name);

    item.h_name = "args[" + std::to_string(host_pindex) + "]";
    item.h_index = host_pindex;
    host_pindex++;
  }

  // Generate code for factor variable types
  if (debug_visit) VST_DEBUG(dbgs() << "[Factor Decls] Input Types:\n");

  // TODO: unify inputs/output handling

  size_t device_pindex = 0;
  for (auto& item : GetFactorDeviceInParams()) {
    // set the factor-device parameters indices and names
    item.d_index = device_pindex;
    item.device_name = UnScopedName(item.name);

    std::string type_name = UnScopedName(item.name) + "_type";
    std::string type_string;
    if (auto sty = dyn_cast<SpannedType>(item.type)) {
      // define spanned type
      type_string = "DRAMType(" + stringify(sty->ElementType()) + ", " +
                    ReplaceRuntimeNames(LSTR(sty->GetShape()), "", false) + ")";
      factor_symbols.AddSymbol(item.name, type_name, type_string);
    } else if (isa<ScalarType>(item.type)) {
      type_string = "DRAMType(" + stringify(item.type) + ", (1))";
      factor_symbols.AddSymbol(item.name, type_name, type_string);
    } else
      choreo_unreachable("unsupported type (" + PSTR(item.type) +
                         " for type declaration.");

    if (!type_string.empty()) {
      fs << indent << "auto " << type_name << " = " << type_string << ";\n";
      if (debug_visit)
        VST_DEBUG(dbgs() << indent << "auto " << type_name << " = "
                         << type_string << ";\n");
    }
    ++device_pindex;
  }

  if (debug_visit) VST_DEBUG(dbgs() << "[Factor Decls] Output Types:\n");

  std::ostringstream dss; // for the shape string
  if (isa<VoidType>(fty.out_ty)) {
    void_return = true;
    if (debug_visit) VST_DEBUG(dbgs() << "VOID\n");
  } else if (auto rty = dyn_cast<SpannedType>(fty.out_ty)) {
    auto name = cgi->GetReturnSymbol(fname);
    std::string type_name = "output_type";
    auto type_string = "DRAMType(" + stringify(rty->ElementType()) + ", " +
                       ReplaceRuntimeNames(LSTR(rty->GetShape()), "", false) +
                       ")";

    // handle dynamic-typed output when necessary. Generate code snippet like:
    //
    //   auto output_rt_dim0 = dim_(args[0], 1);
    //   auto output = alloc_({output_rt_dim0}, output_type);
    //
    const auto& dyn_dims = rty->GetShape().GetDynamicDims();
    if (!dyn_dims.empty()) {
      compile_with_dynshape = true;
      type_name.clear();
      for (auto& ddim : dyn_dims) {
        auto ddim_name =
            UnScopedName(name) + "_rt_dim" + std::to_string(ddim.first);
        dss << indent << "  auto " << ddim_name << " = "
            << ReplaceFactorDynDimName(ddim.second) << ";\n";
        if (type_name.size() == 0)
          type_name += ddim_name;
        else
          type_name += ", " + ddim_name;
      }
      type_name = "{" + type_name + "}, output_type";
    }

    fs << indent << "auto output_type = " << type_string << ";\n";
    factor_symbols.AddSymbol(name, type_name, type_string);

    if (debug_visit)
      VST_DEBUG(dbgs() << indent << "auto output_type = " << type_string
                       << ");\n");
  } else {
    auto name =
        (cgi->HasReturnSymbol(fname)) ? cgi->GetReturnSymbol(fname) : "output";
    auto type_name = "output_type";
    auto type_string =
        "DRAMType(" + stringify(TC2BT(fty.out_ty->Category())) + ", (1))";
    fs << indent << "auto " << type_name << " = " << type_string << ";\n";
    factor_symbols.AddSymbol(name, type_name, type_string);

    if (debug_visit)
      VST_DEBUG(dbgs() << indent << "auto " << type_name << " = " << type_string
                       << ";\n");
  }

  fs << "\n";

  // [Factor Host] Generate the factor host function decl
  {
    std::ostringstream hfs;
    hfs << this->indent << "// choreo-factor dataflow function\n";
    hfs << this->indent << "D(host_func_)(\"" << factor_fname << "\", {";

    for (auto& item : GetFactorHostInParams())
      hfs << UnScopedName(item.name) + "_type, ";

    hfs << "StreamType()}, [&](auto args)";

    if (debug_visit)
      VST_DEBUG(dbgs() << "[Factor Host] Entry Function Declaration:\n"
                       << hfs.str() << "\n");

    fs << hfs.str();
  }

  fs << " {\n";
  this->IncrementIndent();

  // [Factor Host] Generate reference symbols to parameters
  {
    std::ostringstream rfs;
    for (auto& item : GetFactorHostInParams())
      rfs << indent << "auto & " << UnScopedName(item.name) + " = "
          << item.h_name << ";\n";
    if (debug_visit)
      VST_DEBUG(dbgs() << "[Factor Host] Reference Symbols:\n"
                       << rfs.str() << "\n");
    fs << rfs.str();
  }

  // dynamic-shape alias reference
  for (auto& [id_name, sym_name] : idnm_rts) {
    dss << indent << "auto " << named_dim_ref_prefix << id_name << " = "
        << ReplaceFactorDynDimName(sym_name) << ";\n";
  }

  fs << dss.str(); // dynamic-shape specific

  return true;
}

bool FactorCodeGen::Visit(AST::ChoreoFunction&) { return true; }

bool FactorCodeGen::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);

  if (n.host)
    hs << n.GetCode();
  else
    ks << n.GetCode();

  return true;
}

bool FactorCodeGen::Visit(AST::Program&) { return true; }

void FactorCodeGen::EmitHostHead(std::ostream& os) {
  os <<
      R"(
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

// dependant on the topsruntime
#include "tops/tops_ext.h"
#include "tops/tops_runtime.h"

// choreo header
#include "choreo.h"

using namespace choreo;

namespace {

int64_t SizeOfRankedMemref(size_t rank) {
  return sizeof(topsMemref) +
         rank * sizeof(reinterpret_cast<topsMemref *>(0)->data[0]);
}

struct topsUnrankedMemref CreateUnrankedMemref(void *dev_mem, char *memref_raw,
                                               std::vector<int64_t> shape) {
  struct topsUnrankedMemref unranked_memref;
  unranked_memref.rank = shape.size();
  unranked_memref.ranked_memref = reinterpret_cast<topsMemref *>(memref_raw);

  // Set address
  uint64_t dev_addr = reinterpret_cast<uint64_t>(dev_mem);
  unranked_memref.ranked_memref->high_addr =
      reinterpret_cast<int32_t *>((dev_addr >> 32) & 0xFFFFFFFF);
  unranked_memref.ranked_memref->low_addr =
      reinterpret_cast<int32_t *>(dev_addr & 0xFFFFFFFF);

  // Set offset & shape
  unranked_memref.ranked_memref->offset = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    unranked_memref.ranked_memref->data[i] = shape[i];
  }

  return unranked_memref;
}

// Nasty data copy. Need optimization together with factor
template <typename T, int Rank>
static inline std::vector<uint8_t>
ToFactorData(const spanned_view<T, Rank> &v) {
  return std::vector<uint8_t>((const uint8_t *)(v.data()), v.bytes());
}

template <int N, typename T, typename U>
static inline spanned_data<T, N>
ToSpanned(const std::vector<U> &v, std::initializer_list<int> && shape) {
  return copy_as_spanned<N, T>((T*)v.data(), v.size() * sizeof(U), shape);
}

// must be true
//#define CHECK(a) choreo_assert((a), "", __FILE__, __LINE__)
#define CHECK(a) (a)

} // end anonymous namespace
)";
}

void FactorCodeGen::EmitHostFuncBody(std::ostream& os, const FunctionType& fty,
                                     const std::string& bin_filename) {
  auto rty = fty.out_ty;
  auto out_size_expr = SizeExprOf(*rty);
  // phase 1: create tops executable from a file
  os << " {\n";

  EmitRuntimeCheck(os);
  EmitRuntimeMemUsageCheck(os);

  os << R"(
  std::vector<char> binary;
  // Read bin file and store to a vector
)";
  os << "  std::ifstream ifs(\"" << bin_filename << "\", std::ios::binary);";
  os << R"(
  std::copy(std::istreambuf_iterator<char>(ifs),
            std::istreambuf_iterator<char>(), std::back_inserter(binary));
  ifs.close();

  // Create stream
  topsStream_t stream = nullptr;
  CHECK(topsStreamCreate(&stream));

)";

  // phase 2: allocate device memory and copy
  std::vector<std::string> device_mems;
  for (auto& item : GetFactorHostInParams()) {
    auto buffer_name = "in_mem" + std::to_string(device_mems.size());
    std::string size_expr_str = "1";
    if (auto sty = dyn_cast<SpannedType>(item.type))
      size_expr_str = ReplaceRuntimeNames(sty->ByteSizeExpression());

    os << "  void *" << buffer_name << " = nullptr;\n";
    os << "  CHECK(topsMalloc(&" << buffer_name << ", " << size_expr_str
       << "));\n";
    if (item.IsParameter()) {
      os << "  CHECK(topsMemcpy(" << buffer_name
         << ", reinterpret_cast<void *>(" << item.host_name << ".data()), "
         << size_expr_str << ", topsMemcpyHostToDevice));\n";
    }
    device_mems.push_back(buffer_name);
  }
  os << "  void * device_inputs[] = {" << DelimitedString(device_mems)
     << "};\n\n";

  std::string size_string = ReplaceRuntimeNames(out_size_expr);

  if (!out_size_expr.empty()) {
    os << "  void * out_mem = nullptr;\n";
    os << "  CHECK(topsMalloc(&out_mem, " << size_string << "));\n";
    os << "  void *device_outputs[] = {out_mem};\n";
  }

  std::vector<std::string> inputs; // factor input paramters

  os << "\n  // adaption: convert to the factor parameters\n";
  size_t index = 0;
  std::ostringstream tss;
  for (auto& item : GetFactorHostInParams()) {
    assert(item.h_index == (int)index);
    tss << "  std::unique_ptr<char[]> memref_raw" << index
        << "(new char[SizeOfRankedMemref(";
    if (auto sty = dyn_cast<SpannedType>(item.type))
      tss << sty->Dims();
    else
      tss << "1";
    tss << ")]);\n";
    tss << "  auto input" << index << " = CreateUnrankedMemref(in_mem" << index
        << ", memref_raw" << index << ".get(), ";
    if (auto sty = dyn_cast<SpannedType>(item.type)) {
      tss << LSTR(sty->GetShape());
    } else
      tss << "{1}";
    tss << ");\n";
    inputs.push_back("input" + std::to_string(index));
    ++index;
  }

  tss << "  std::vector<int64_t> out_shape = {";
  if (auto rty = dyn_cast<SpannedType>(fty.out_ty)) {
    tss << RSTR(rty->GetShape());
  } else if (!isa<VoidType>(fty.out_ty)) {
    tss << "1";
  }
  tss << "};\n";
  os << ReplaceRuntimeNames(tss.str(), "(int64_t)");

  if (!void_return) {
    os << "  std::unique_ptr<char[]> memref_raw" << index
       << "(new char[SizeOfRankedMemref(out_shape.size())]);\n";
    os << "  auto output = CreateUnrankedMemref(out_mem, memref_raw" << index
       << ".get(), out_shape);\n";
  }

  // phase 3: Execute the executable and fetch the output
  os << "\n  " << factor_fname << "(";
  for (auto& in : inputs) os << "&" << in << ", ";
  os << "stream" << ((void_return) ? "" : ", &output") << ");\n";
  os << "  CHECK(topsStreamSynchronize(stream));\n";

  size_t out_rank = 1;
  std::string shape_string = "{1}";
  if (auto rty = dyn_cast<SpannedType>(fty.out_ty)) {
    out_rank = rty->Dims();
    shape_string = ReplaceRuntimeNames(LSTR(rty->GetShape()));
  }

  if (!out_size_expr.empty()) {
    os << "  auto res = choreo::make_spandata<" << STR(GetBaseType(*rty))
       << ", " << out_rank << ">(" << shape_string << ");\n";
    os << "  // Copy output data from device to host\n";
    os << "  CHECK(topsMemcpy(reinterpret_cast<void *>(res.data()), out_mem,\n";
    os << "                  " << size_string
       << ", topsMemcpyDeviceToHost));\n";
  }

  // phase 4: Free up the resources
  os << "  // Free up the resources\n";
  for (auto& p : device_mems) os << "  topsFree(" << p << ");\n";
  os << "  topsFree(out_mem);\n\n";
  os << "  // TODO: figure out why stream destroying crash some "
        "applications.\n";
  os << "  // topsStreamDestroy(stream);\n";
  os << "  return res;\n";
  os << "}\n";
}

std::string FactorCodeGen::ReplaceRuntimeNames(const std::string& e,
                                               const std::string& prefix,
                                               bool host_code) {
  std::string expr = e;
  for (auto& s : dims_info) {
    size_t pos = 0;
    while ((pos = expr.find(s.first, pos)) != std::string::npos) {
      if (host_code)
        expr.replace(pos, s.first.length(), prefix + s.second.hd_name);
      else
        expr.replace(pos, s.first.length(), "-1");
    }
  }
  return expr;
}

std::string FactorCodeGen::ReplaceFactorDynDimName(const std::string& e) {
  std::string expr = e;
  for (auto& s : dims_info) {
    size_t pos = 0;
    while ((pos = expr.find(s.first, pos)) != std::string::npos) {
      std::string dim_value = "dim_(args[" +
                              std::to_string(s.second.param_index) + "], " +
                              std::to_string(s.second.dim_index) + ")";
      expr.replace(pos, s.first.length(), dim_value);
    }
  }
  return expr;
}

std::optional<std::string>
FactorCodeGen::ReplaceDynDimRef(const std::string& e) {
  std::string replaced = e;
  // match str begins with "::", thus "\\b" appears only in the suffix.
  for (auto& [id_name, sym_name] : idnm_rts)
    replaced = RegexReplaceAll(replaced, sym_name + "\\b",
                               named_dim_ref_prefix + id_name);
  if (replaced != e) return replaced;
  return std::nullopt;
}

void FactorCodeGen::EmitRuntimeCheck(std::ostream& os) {
  // check if the input shape is as declared in choreo
  if (cgi->ParameterCount(fname) == 0) return;

  struct Entry {
    size_t para_ordinal;
    size_t dim;
    std::string elem_name;
  };
  std::map<ValueExpr, std::vector<Entry>> ve_entries_map;

  size_t host_pindex = 0;
  for (auto& item : GetChoreoParameters()) {
    assert((int)host_pindex == item.p_index);
    auto name = item.host_name;
    if (auto sty = dyn_cast<SpannedType>(item.type)) {
      size_t dim_count = 0;
      for (auto vi : sty->GetShape().Value()) {
        auto elem_name = name + ".shape()[" + std::to_string(dim_count) + "]";
        if (auto vale = dyn_cast<int>(&vi)) {
          os << "  choreo::runtime_check(" << elem_name << " == " << *vale;
          os << ", \"shape inconsistent on the " << Ordinal(host_pindex + 1)
             << " parameter (dim: " << dim_count << ").\");\n";
        } else if (auto vale = dyn_cast<ValueExpr>(&vi)) {
          ve_entries_map[*vale].push_back(
              {host_pindex + 1, dim_count, elem_name});
        }
        dim_count++;
      }
    }
    host_pindex++;
  }

  // check if the named dims meet the constraint
  // eg. __co__ void foo(f32 [M, N] a, f32 [N, K] b)
  // then a.shape()[1] should be equal to b.shape()[0]
  for (auto& [_, entries] : ve_entries_map) {
    for (size_t i = 1; i < entries.size(); ++i) {
      auto& entry0 = entries[i - 1];
      auto& entry1 = entries[i];
      os << "  choreo::runtime_check(" << entry0.elem_name
         << " == " << entry1.elem_name;
      os << ", \"The shapes of the " << Ordinal(entry0.para_ordinal)
         << " parameter (dim: " << entry0.dim << ") and the "
         << Ordinal(entry1.para_ordinal) << " parameter (dim: " << entry1.dim
         << ") are inconsistent.\");\n";
    }
  }
}

void FactorCodeGen::EmitRuntimeMemUsageCheck(std::ostream& os) {
  // check if the input shape is as declared in choreo
  if (cgi->ParameterCount(fname) == 0) return;

  // there should be runtime memory usage check
  if (!rt_mem_usage_check_list.empty())
    os << "\n  // Check if the runtime memory usage exceeds the defined "
          "limits.\n";

  for (const auto& [useds, loc, limit] : rt_mem_usage_check_list) {
    std::ostringstream used_ss;
    used_ss << "  choreo::runtime_check((size_t)";
    for (auto& used : useds) {
      if (used.find(":") == std::string::npos) {
        // `used` is compile time memory usage
        used_ss << (used_ss.str().back() == ')' ? "" : " + ") << used;
        continue;
      }
      // `used` is runtime memory usage
      auto operands = SplitStringByDelimiter(used, "*");
      // `o` is dynamic dim. Should replace it with host name
      for (auto& o : operands) o = ReplaceRuntimeNames(o, "", true);
      // add (size_t) to avoid integer overflow
      used_ss << (used_ss.str().back() == ')' ? "" : " + ") << "(size_t)"
              << DelimitedString(operands, "*");
    }
    used_ss << " <= (size_t)" << limit
            << ", \"total memory usage(compile time and runtime) "
               "should not exceed limit, happends at "
            << loc << "\");\n";
    os << used_ss.str();
  }
}

void FactorCodeGen::EmitHostFuncDecl(std::ostringstream& oss,
                                     const FunctionType& fty,
                                     const std::string& name) {
  // emit the return type
  oss << HostTypeStringify(*fty.out_ty, true) << " " << name << "(";

  // emit the parameters
  size_t host_pindex = 0;
  for (auto& item : GetChoreoParameters()) {
    if (item.IsParameter()) assert((int)host_pindex == item.p_index);
    oss << ((host_pindex == 0) ? "" : ", ") << HostTypeStringify(*item.type)
        << " " << item.host_name;
    ++host_pindex;
  }
  oss << ")";

  if (debug_visit)
    VST_DEBUG(dbgs() << "Host function prototype:\n" << oss.str());
}

void FactorCodeGen::OutputScript(const ptr<FunctionType>& fty) {

  // a temporal path for the compilation process
  build_path = create_unique_path();
  std::string build_prefix = build_path + "/__choreo_" + fname;

  std::string kernel_fn = build_prefix + "_micro_kernel.cpp";
  std::string factor_fn = build_prefix + "_factor.cpp";
  std::string factor_bfn =
      build_path + "/${gcu_target_string}_lib" + factor_fname + ".o";
  host_filename = build_prefix + "_host.cpp";

  // Generate the host code
  std::string user_code = hs.str();
  // Reset the string to be empty
  hs.str("");
  // Clear any error flags that may be set
  hs.clear();

  // emit the fixed header
  EmitHostHead(hs);

  if (!user_code.empty()) {
    // The user code requires the choreo function be fwd-decalared for its call
    EmitHostFuncDecl(hs, *fty, fname);
    hs << ";\n" << user_code;
  }

  EmitHostFuncDecl(hs, *fty, fname);
  EmitHostFuncBody(hs, *fty, factor_bfn);

  // backpatch the factor bin filename
  std::string factor_src = fs.str();
  if (!alloc_in_fs.str().empty())
    factor_src.insert(alloc_pos, alloc_in_fs.str());
  ReplaceInString(&factor_src, std::string(backpatch_filename), kernel_fn);

  // Now generate the script
  outs() << "#!/usr/bin/env bash\n\n";
  outs()
      << "# This is the choreo generated bash script to compile factor code\n";
  // check for gcu_target_string first
  //
  outs() << R"script(
  gcu_arch=gcu210
  gcu_resource=2c24s
  gcu_target_string="dorado_2c"
)script";
  if (!cross_compile)
    outs() << R"script(
  # check the device
  # TODO: improve the target check with more solid code
  GCU_DEVICE_STR="$(lspci | grep Enflame | head -1)"
  GCU_DEVICE_STR_BACKUP="$(lspci | grep Tencent)"
  echo $GCU_DEVICE_STR
  if [[ "${GCU_DEVICE_STR}" == *"S60G"* ]]; then
    gcu_arch=gcu300
    gcu_resource=2c24s
    gcu_target_string="scorpio_${gcu_resource}"
  elif [[ "${GCU_DEVICE_STR}" == *"c035"* ]]; then
    gcu_arch=gcu300
    gcu_resource=1c12s
    gcu_target_string="scorpio_${gcu_resource}"
    export TOPS_VISIBLE_DEVICES=1
  elif [[ "${GCU_DEVICE_STR}" == *"S60"* ]]; then
    gcu_arch=gcu300
    gcu_resource=2c24s
    gcu_target_string="scorpio_${gcu_resource}"
  elif [[ "${GCU_DEVICE_STR}" == *"I20"* ]]; then
    gcu_arch=gcu210
    gcu_resource=2c24s
    gcu_target_string="dorado_2c"
  elif [[ "${GCU_DEVICE_STR_BACKUP}" != "" ]]; then
    gcu_arch=gcu210
    gcu_resource=2c24s
    gcu_target_string="dorado_2c"
  else
    echo "can not determine the GCU device type."
    exit 1
  fi
)script";

  outs() << "\n# step 0: set up the environment\n";
  outs() << "rm -fr " << build_path << "\n";
  outs() << "mkdir -p " << build_path << "\n";
  outs() << "cat <<'EOF' > " << build_path << "/factor_script.sh\n";
  outs() << __factor_script_as_string << "\nEOF\n";
  outs() << "chmod +x " << build_path << "/factor_script.sh\n";
  outs() << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  outs() << __choreo_header_as_string << "\nEOF\n\n";

  outs() << "\n# step 1: write the kernel source code into a temp file\n";
  outs() << "kernel_src=" << kernel_fn << "\n";
  outs() << "cat <<'EOF' > ${kernel_src}\n";
  outs() << ks.str() << "\nEOF\n";

  outs() << "\n# step 2: write the factor source code into a temp file\n";
  outs() << "factor_src=" << factor_fn << "\n";
  outs() << "cat <<'EOF' > ${factor_src}\n";
  outs() << factor_src << "\nEOF\n\n";

  outs() << "\n# step 3: set the factor binary file name\n";
  outs() << "factor_bin=" << factor_bfn << "\n";
}

const std::string FactorCodeGen::ExprSTR(AST::ptr<AST::Node> e) const {
  std::ostringstream oss;

  if (auto id = dyn_cast<AST::Identifier>(e)) {
    auto ty = NodeType(*id);
    if (ContainsLoopVar(id->name))
      oss << "iv_" << id->name;
    else if (isa<BoundedType>(ty) &&
             PrefixedWith(cast<BoundedType>(ty)->GetNote(), "pv")) {
      auto l = RemovePrefixOrNull("pv:", cast<BoundedType>(ty)->GetNote());
      assert(l.has_value());
      // is marked as parallel whose level is decided by target check
      if (*l == "0")
        oss << "thread_id";
      else if (*l == "1")
        oss << "block_id";
      else
        choreo_unreachable("invalid bounded type note.");
    } else
      oss << id->name;
  } else if (auto il = dyn_cast<AST::IntLiteral>(e)) {
    oss << "Value(" << il->value << ")";
  } else if (auto ii = dyn_cast<AST::IntIndex>(e)) {
    return ExprSTR(ii->value);
  } else if (auto expr = dyn_cast<AST::Expr>(e)) {
    if (isa<IntegerType>(NodeType(*e)) && (expr->s.IsValid())) {
      // prefer to use the deduced value when possible
      assert(expr->s.DimCount() == 1 && "A 1-dimensional value is expected.");
      return "Value(" + STR(expr->s.ValueAt(0)) + ")";
    }
    if (expr->IsReference()) {
      if (expr->GetInt())
        return ExprSTR(expr->GetReference());
      else if (expr->GetSymbol())
        return ExprSTR(expr->GetReference());
      else if (isa<AST::Expr>(NodeType(*expr->GetR()))) // should this happen?
        return ExprSTR(expr->GetR());
      else
        choreo_unreachable("Unsupported reference: " + PSTR(expr));
    } else if (expr->IsUnary()) {
      if (expr->op == "!") {
        oss << "!(" << ExprSTR(expr->GetR()) << ")";
      } else if (expr->op == "ubound") {
        auto rty = cast<BoundedType>(NodeType(*expr->GetR()));
        if (rty->Dims() == 1) oss << ValueSTR(rty->GetUpperBound());
      } else if (expr->op == "dataof") {
        assert(isa<FutureType>(expr->GetR()->GetType()) &&
               "expect a future operand.");
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
          if (fut_buf->at(fname).count(id->name))
            oss << fut_buf->at(fname).at(id->name);
          else
            choreo_unreachable("Future '" + id->name +
                               "' is not associated with a buffer.");
        } else
          choreo_unreachable("Can not retrive name of the future.");
      } else if (expr->op == "sizeof") {
        auto var = RemoveSuffix(*AST::GetName(*expr->GetR()), ".span");
        auto shape = GetShape(GetSymbolType(var));
        assert(shape.IsValid() && "Invalid shape is found");
        oss << shape.GetSizeExpression();
      } else
        choreo_unreachable("Unsupported choreo expression.");
    } else if (expr->IsBinary()) {
      if (expr->op == "cdiv") {
        std::string one = "Value(1)";
        oss << "((" << ExprSTR(expr->GetL()) << ")+(" << ExprSTR(expr->GetR())
            << "-" << one << ")/(" << ExprSTR(expr->GetR()) << ")";
      } else if (expr->op == "getith") {
        auto lty = cast<BoundedType>(NodeType(*expr->GetL()));
        if (cast<AST::IntIndex>(expr->GetR())->IsNegative()) {
          oss << "(" << ValueSTR(lty->GetUpperBound()) << "+("
              << ExprSTR(expr->GetR()) << "))";
        } else
          oss << "(" << ExprSTR(expr->GetR()) << ")";
      } else if (expr->IsArith() || expr->IsLogical()) {
        auto& l = expr->GetL();
        auto& r = expr->GetR();
        auto& op = expr->op;
        // handle bounded variable times
        if (op == "#" && IsActualBoundedIntegerType(l->GetType()) &&
            IsActualBoundedIntegerType(r->GetType())) {
          auto rty = cast<BoundedType>(NodeType(*r));
          assert(rty->Dims() == 1);
          oss << "((" << ExprSTR(l) << ")*(" << ValueSTR(rty->GetUpperBound())
              << ")+(" << ExprSTR(r) << "))";
        } else
          oss << "((" << ExprSTR(l) << ")" << op << "(" << ExprSTR(r) << "))";
      }
    } else if (expr->IsTernary()) {
      oss << "(" << ExprSTR(expr->GetC()) << ") ? (" << ExprSTR(expr->GetL())
          << ") : (" << ExprSTR(expr->GetR()) << ")";
    } else
      choreo_unreachable("unsupported expression '" + expr->op +
                         "': " + PSTR(expr) + ".");
  } else if (auto sl = dyn_cast<AST::Select>(e)) {
    size_t val_count = sl->expr_list->Count();
    // if val_count == 1, pingpong is meaningless?
    // (TODO: maybe assert when earlysema)
    assert(val_count >= 2);
    for (size_t i = 0; i < val_count - 1; i++) {
      std::string select_factor_str;
      auto factor = cast<IntegerType>(sl->select_factor->GetType());
      if (auto expr = factor->GetValidExpression())
        select_factor_str =
            STR(*expr); // use the expression simplified by value numbering
      else
        select_factor_str = ExprSTR(sl->select_factor);
      oss << "select_(" << select_factor_str << " == " << i << ", "
          << PSTR(sl->expr_list->ValueAt(i)) << (i < val_count - 1 ? ", " : "");
    }
    oss << PSTR(sl->expr_list->AllValues().back())
        << std::string(val_count - 1, ')');
  } else
    choreo_unreachable("unsupported expression '" + expr->op + "'.");

  return oss.str();
}
