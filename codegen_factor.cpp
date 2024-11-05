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

bool FactorCodeGen::ContainsLoopVar(const std::string& iv) const {
  for (auto& loop_var : loop_vars)
    if (loop_var.count(iv)) return true;
  return false;
}

bool FactorCodeGen::BeforeVisitImpl(AST::Node& n) {
  TraceEachVisit(n);
  if (isa<AST::Program>(&n)) {
    //    print_fixed_header(os);
  } else if (auto c = dyn_cast<AST::ChoreoFunction>(&n)) {
    sp_count = 0; // reset the count of stub parameter
    param_map.clear();
    rts_nmap.clear();
    rts_pidx.clear();
    rts_nidx.clear();
    idnm_rts.clear();
    host_params.clear();
    indent.clear();
    entry_fn = c->name;
    current_fn = "__choreo_" + entry_fn;
    // declare a factor function with proper name
    fs << R"(#include <vector>

#include "gcu/factor/factor.h"

using namespace factor;
)";
    fs << "void " << current_fn << "() {\n";
    this->incrementIndent();
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
    os << "\n# step 4: generate the host source\n";
    os << "host_src=" << host_fn << "\n";
    os << "echo \"#include \\\"\"${gcu_target_string}\"_lib" << current_fn
       << ".h\\\"\" > ${host_src}\n";
    os << "cat <<'EOF' >> ${host_src}\n";
    os << hs.str() << "\nEOF\n\n";

    os << "\n# step 5: JIT compile and execute\n";
    os << "# TODO: enable workflow of AOT compilation\n";
    os << "target=" << target_fn << "\n";
    os << R"(
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
    os << R"(
if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then
)";
    os << "  export FACTOR_INSTALL="
       << STRINGIZE(__CHOREO_FACTOR_DIR__) << "\n# JIT compile and execute\n";
    if (dyn_shaped) os << "VIEW_CONFIG=1 ENABLE_DYNSHAPE=1 ";
    os << build_path
       << "/factor_script.sh ${factor_src} ${factor_bin} ${host_src} ${target} "
          "${gcu_arch} ${gcu_resource}";
    os << R"script(
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
    auto& out_type = fty->out_ty;
    // TODO:need refactor
    auto out_size_expr = SizeExprOf(*out_type);
    fs << "}\n\n";

    fs << "MODULE_REGISTER(\"lib" << current_fn << "\", " << current_fn
       << ");"; // end the factor function definition

    // TODO:need refactor
    if (auto sty = dyn_cast<SpannedType>(out_type)) {
      OutputScript(fty, f->name, STR(GetBaseType(*out_type)), out_size_expr,
                   sty->GetShape());
    } else
      OutputScript(fty, f->name, STR(GetBaseType(*out_type)), out_size_expr,
                   Shape() /*invalid shape*/);
    ResetBuffers();
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level--;
    if (parallel_level == 0) {
      this->decrementIndent();
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
        decrementIndent();
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
  if (auto s = dyn_cast<AST::Select>(node.init_expr)) {
    assert(!s->inDMA);
    size_t val_count = s->expr_list->Count();
    assert(val_count >= 2);
    fs << this->indent << "auto " << node.name_str << " = ";
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
  auto nty = node.GetType();
  auto sym = node.name_str;
  if (auto sty = dyn_cast<SpannedType>(nty)) {
    assert(isa<SpannedType>(GetSymbolType(sym)) && "Inconsistent types!");
    if (factor_symbols.Exists(sym)) {
      fs << indent << "auto " << sym << " = alloc_("
         << factor_symbols.GetTypeName(sym) << ");\n";
    } else if (auto e = dyn_cast<AST::Expr>(node.init_expr);
               e && isa<AST::SpanAs>(e->GetR())) {
      assert(e->IsReference());
      auto sa = dyn_cast<AST::SpanAs>(e->GetR());
      int arg_idx = factor_symbols.GetSymbolIndex(sa->id->name);
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
    } else {
      std::string storage_type = stringify(sty->GetStorage());
      std::string base_type = stringify(Choreo::BaseType(sty->f_type));
      std::ostringstream _os;
      _os << "auto " << sym << " = alloc_(" << storage_type << "(" << base_type
          << "," << ReplaceRuntimeNames(LSTR(sty->GetShape()), "", false) << ")"
          << ");\n";
      if (storage_type == "DRAMType")
        fs << indent << _os.str();
      else
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
      assert(fut_buf->at(entry_fn).count(sa->id->name));
      buffer_name = fut_buf->at(entry_fn).at(sa->id->name);
    }
    if (arg_idx >= 0) buffer_name = "args[" + std::to_string(arg_idx) + "]";
    auto sty = dyn_cast<SpannedType>(node.GetType());
    assert(sty);
    std::string storage_type = stringify(sty->GetStorage());
    std::string base_type = stringify(Choreo::BaseType(sty->f_type));
    fs << indent << "auto " << node.name << " = bitcast_(" << storage_type
       << "(" << base_type << ", {";

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
  cur_params = &pl.values;
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::ParallelBy& by) {
  TraceEachVisit(by);
  parallel_factor *= by.bound;
  if (parallel_level == 1)
    pb_bound0 = by.bound;
  else if (parallel_level == 2)
    pb_bound1 = by.bound;
  else
    choreo_unreachable("parallel_level is invalid!");
  if (parallel_level > 1) { return true; }

  fs << this->indent << "Dim3 grid_dim(" << "$$pb_bound0$$" << ");\n";
  fs << this->indent << "Dim3 block_dim(" << "$$pb_bound1$$" << ");\n";
  fs << this->indent << "auto ts = launch_kernel_(\"" << current_fn
     << "_parallel\", grid_dim, block_dim, args.back(), {";
  if (cur_params->size() > 0) {
    fs << "args[0]";
    for (size_t i = 1; i < cur_params->size(); ++i) fs << ", args[" << i << "]";
  }
  fs << "}, {" << ((void_return) ? "" : "$$out$$")
     << "});\n"; // "$$out$$" : magic string for output, will be replaced later
  fs << this->indent << "return std::vector<Value>{"
     << ((void_return) ? "" : "$$out$$") << "};\n";
  this->decrementIndent();
  fs << this->indent << "}, true); // end of choreo-factor dataflow program\n";
  fs << "\n";

  fs << this->indent << "\n";
  fs << this->indent << "D(func_)(\"" << current_fn << "_parallel\", {";
  if (cur_params->size() > 0) {
    fs << (*cur_params)[0]->sym->name << "_type";
    for (unsigned i = 1; i < cur_params->size(); ++i)
      fs << ", " << (*cur_params)[i]->sym->name << "_type";
  }
  fs << "}, {" << ((void_return) ? "" : "output_type")
     << "}, [&](auto args, auto results) {\n";
  this->incrementIndent();
  // generate a reference name of the output
  if (!void_return) fs << indent << "auto & $$out$$ = results[0];\n";
  fs << this->indent << "auto thread_id = thread_id_();\n";
  fs << this->indent << "auto block_id = block_id_();\n";
  // dynamic-shape alias reference
  for (auto& [id_name, sym_name] : idnm_rts) {
    fs << indent << "auto " << named_dim_ref_prefix << id_name << " = "
       << ReplaceDynDimName(sym_name) << ";\n";
  }
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

// CLEAN
bool FactorCodeGen::Visit(AST::DMA& d) {
  TraceEachVisit(d);

  // do not emit code for the placeholder
  if (auto ph = dyn_cast<PlaceHolderType>(NodeType(d))) {
    assert(ph->Category() == TypeCategory::FUTURE);
    // TODO: optimize when it should be SDMA
    auto fty = cast<FutureType>(GetSymbolType(d.future));
    auto dma_str = "CDMA";
    if (fty->GetSpannedType()->GetStorage() == Storage::LOCAL) dma_str = "SDMA";
    alloc_in_fs << alloc_indent << "auto " << d.future << " = alloc_dma_("
                << dma_str << "Type());\n";
    return true;
  }

  // handle .to  in AST::Memory
  assert((isa<AST::ChunkAt>(d.from)) && "Unexpected type for DMA's source.");
  assert((isa<AST::Memory>(d.to) || isa<AST::ChunkAt>(d.to) ||
          isa<AST::Select>(d.to)) &&
         "Unexpected type for DMA's destination.");

  // retrieve the spanned type from a chunkat
  auto GetSpannedType = [this](AST::Node& ca) -> SpannedType* {
    auto sty = ca.GetType();
    if (auto fty = dyn_cast<FutureType>(sty))
      return fty->GetSpannedType().get();
    else
      return cast<SpannedType>(sty);
  };

  auto MemLevel = [](Storage s) -> int {
    switch (s) {
    case Storage::LOCAL: return 0;
    case Storage::SHARED: return 1;
    case Storage::GLOBAL:
    case Storage::DEFAULT: return 2;
    default: choreo_unreachable("Unexpected storage type."); return -1;
    }
  };

  auto ty = dyn_cast<FutureType>(d.GetType());
  assert(ty && "Invalid type of DMA statement!");

  // cook a valid future name
  auto future_name = d.future;
  if (future_name.empty()) {
    static size_t future_count = 0;
    future_name = "__choreo_anon_fut__" + std::to_string(future_count++);
  }

  assert(isa<AST::ChunkAt>(d.to));
  // cook a valid dst buffer name

  auto dst_buffer_name = cast<AST::ChunkAt>(d.to)->RefSymbol();
  auto src_buffer_name = cast<AST::ChunkAt>(d.from)->RefSymbol();

  auto sty = GetSpannedType(*d.from); // source spanned type
  auto tty = GetSpannedType(*d.to);   // dest spanned type
  size_t rank = sty->Dims();
  //  auto dst_shape = ty->GetShape();
  auto src_sto = sty->GetStorage();
  auto dst_sto = tty->GetStorage();

  int src_level = MemLevel(src_sto);
  int dst_level = MemLevel(dst_sto);

  auto GenerateOffsetString = [this, &GetSpannedType](AST::Node& n) {
    auto sty = GetSpannedType(n);
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
    std::vector<size_t> layout(rank);
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

// CLEAN
bool FactorCodeGen::Visit(AST::Call& c) {
  TraceEachVisit(c);
  fs << this->indent << "call_(\"";
  fs << STR(*c.function);
  fs << "\", {";
  auto args = c.arguments;
  assert(args && "Invalid kernel call args!");
  int arg_num = args->AllValues().size();
  for (int index = 0; index < arg_num; ++index) {
    auto arg = dyn_cast<AST::Expr>(args->ValueAt(index));
    assert(arg && "Invalid kernel call arg!");
    fs << ExprSTR(args->AllValues()[index]);
    if (isa<SpannedType>(NodeType(*arg))) fs << ".addr_()";
    if (index < arg_num - 1) fs << ",";
  }
  fs << "});\n";

  return true;
}

bool FactorCodeGen::Visit(AST::Swap& n) {
  TraceEachVisit(n);

  // TODO
  return true;
}

bool FactorCodeGen::Visit(AST::Select& c) {
  TraceEachVisit(c);
  assert(!c.inDMA);
  return true;
}

bool FactorCodeGen::Visit(AST::Return& returnNode) {
  TraceEachVisit(returnNode);
  if (returnNode.value) output_v = STR(*returnNode.value);
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
      fs << ", " << ReplaceDynDimName(STR(iv_sizes.ValueAt(0)));
      if (IsValidBound(loop_range->ubound))
        fs << " + (" << loop_range->ubound << ")";
      fs << ", ";
      if (IsValidStride(loop_range->stride))
        fs << loop_range->stride;
      else
        fs << 1;
      fs << ", [&](auto iv_" << iv_name << ") {\n";
      incrementIndent();
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
        fs << ", " << ReplaceDynDimName(STR(iv_sizes.ValueAt(i)));
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
        incrementIndent();
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
  auto ty = d.GetType();
  assert(isa<FunctionType>(ty) && "unexpected type.");
  auto& fty = *cast<FunctionType>(ty);
  std::string func_name = d.name;

  auto MapRuntimeShapeNames = [this, &func_name](SpannedType* sty,
                                                 const std::string& name,
                                                 size_t p_index) {
    size_t count = 0;
    for (auto vi : sty->GetShape().Value()) {
      if (auto vale = dyn_cast<ValueExpr>(&vi)) {
        auto elem_name = name + ".shape()[" + std::to_string(count) + "]";
        rts_nmap.emplace(*vale, elem_name);
        rts_pidx.emplace(*vale, p_index);
        rts_nidx.emplace(*vale, count);
        assert(PrefixedWith(*vale, "::" + func_name + "::"));
        idnm_rts.emplace(vale->substr(2 + func_name.size() + 2), *vale);
      }
      count++;
    }
  };

  // decide the host parameter names, and map the runtime shape dimensions
  // to the real host code name
  for (size_t i = 0; i < fty.in_tys.size(); ++i) {
    auto n = GenHostParamName();
    host_params.push_back(n);
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i]))
      MapRuntimeShapeNames(sty, n, i);
  }

  std::ostringstream dss; // for the shape string
  for (auto& param : *cur_params) {
    auto pname = param->sym->name;
    std::string type_name = pname + "_type";
    std::string type_string;
    if (auto sty = dyn_cast<SpannedType>(param->GetType())) {
      // define spanned type
      type_string = "DRAMType(" + stringify(sty->ElementType()) + ", " +
                    ReplaceRuntimeNames(LSTR(sty->GetShape()), "", false) + ")";
      factor_symbols.AddSymbol(pname, type_name, type_string);
    } else {
      type_string =
          "DRAMType(" + stringify(param->type->getBaseType()) + ", (1))";
      factor_symbols.AddSymbol(pname, type_name, type_string);
    }
    fs << indent << "auto " << type_name << " = " << type_string << ";\n";
  }

  if (auto rty = dyn_cast<SpannedType>(fty.out_ty)) {
    std::string name = "output";
    std::string type_name = "output_type";
    auto type_string = "DRAMType(" + stringify(rty->ElementType()) + ", " +
                       ReplaceRuntimeNames(LSTR(rty->GetShape()), "", false);

    // handle dynamic-typed output when necessary. Generate code snippet like:
    //
    //   auto output_rt_dim0 = dim_(args[0], 1);
    //   auto output = alloc_({output_rt_dim0}, output_type);
    //
    const auto& dyn_dims = rty->GetShape().GetDynamicDims();
    if (!dyn_dims.empty()) {
      dyn_shaped = true;
      type_name.clear();
      for (auto& ddim : dyn_dims) {
        auto ddim_name = name + "_rt_dim" + std::to_string(ddim.first);
        dss << indent << "  auto " << ddim_name << " = "
            << ReplaceDynDimName(ddim.second) << ";\n";
        if (type_name.size() == 0)
          type_name += ddim_name;
        else
          type_name += ", " + ddim_name;
      }
      type_name = "{" + type_name + "}, output_type";
    }

    fs << indent << "auto output_type = " << type_string << ");\n";
    factor_symbols.AddSymbol(name, type_name, type_string);

  } else if (isa<VoidType>(fty.out_ty)) {
    void_return = true;
  } else {
    auto name = "output";
    auto type_name = "output_type";
    auto type_string =
        "DRAMType(" + stringify(TC2BT(fty.out_ty->Category())) + ", (1));";
    factor_symbols.AddSymbol(name, type_name, type_string);
    fs << indent << "auto " << type_name << " = " << type_string << "\n";
  }

  fs << "\n";
  fs << this->indent << "// choreo-factor dataflow function\n";
  fs << this->indent << "D(host_func_)(\"" << current_fn << "\", {";

  for (auto& param : *cur_params) fs << param->sym->name + "_type, ";

  fs << "StreamType()}, [&](auto args) {\n";

  // dynamic-shape alias reference
  for (auto& [id_name, sym_name] : idnm_rts) {
    dss << indent << "  auto " << named_dim_ref_prefix << id_name << " = "
        << ReplaceDynDimName(sym_name) << ";\n";
  }

  this->incrementIndent();
  fs << dss.str(); // dynamic-shape specific

  return true;
}

bool FactorCodeGen::Visit(AST::ChoreoFunction&) { return true; }

bool FactorCodeGen::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);
  if (n.host) {
    hs << n.GetCode();
  } else {
    ks << n.GetCode();
  }
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

void FactorCodeGen::EmitHostFuncBody(std::ostream& os, const Type& ty,
                                     const std::string& f_n,
                                     const std::string& out_size_expr,
                                     const std::string& out_type,
                                     const Shape& out_shape) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto& fty = *cast<FunctionType>(&ty);

  // phase 1: create tops executable from a file
  os << "{\n";
  EmitRuntimeCheck(os, ty);
  EmitRuntimeMemUsageCheck(os, ty);
  os << R"(
  std::vector<char> binary;
  // Read bin file and store to a vector
)";
  os << "  std::ifstream ifs(\"" << f_n << "\", std::ios::binary);";
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
  for (auto& p : param_map) {
    auto mem_name = "in_mem" + std::to_string(device_mems.size());
    os << "  void *" << mem_name << " = nullptr;\n";
    os << "  CHECK(topsMalloc(&" << mem_name << ", " << p.second << "));\n";
    os << "  CHECK(topsMemcpy(" << mem_name << ", reinterpret_cast<void *>("
       << p.first << ".data()), " << p.second
       << ", topsMemcpyHostToDevice));\n";
    device_mems.push_back(mem_name);
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
  size_t i = 0;
  std::ostringstream tss;
  for (; i < fty.in_tys.size(); ++i) {
    tss << "  std::unique_ptr<char[]> memref_raw" << i
        << "(new char[SizeOfRankedMemref(";
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i]))
      tss << sty->Dims();
    else
      tss << "1";
    tss << ")]);\n";
    tss << "  auto input" << i << " = CreateUnrankedMemref(in_mem" << i
        << ", memref_raw" << i << ".get(), ";
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
      tss << LSTR(sty->GetShape());
    } else
      tss << "{1}";
    tss << ");\n";
    inputs.push_back("input" + std::to_string(i));
  }

  if (auto rty = dyn_cast<SpannedType>(fty.out_ty)) {
    tss << "  std::vector<int64_t> out_shape = {" << RSTR(rty->GetShape())
        << "};\n";
  } else if (!isa<VoidType>(fty.out_ty)) {
    tss << "  std::vector<int64_t> out_shape = {1};\n";
  }
  os << ReplaceRuntimeNames(tss.str(), "(int64_t)");

  if (!void_return) {
    os << "  std::unique_ptr<char[]> memref_raw" << i
       << "(new char[SizeOfRankedMemref(out_shape.size())]);\n";
    os << "  auto output = CreateUnrankedMemref(out_mem, memref_raw" << i
       << ".get(), out_shape);\n";
  }

  // phase 3: Execute the executable and fetch the output
  os << "\n  " << current_fn << "(";
  for (auto& in : inputs) os << "&" << in << ", ";
  os << "stream" << ((void_return) ? "" : ", &output") << ");\n";
  os << "  CHECK(topsStreamSynchronize(stream));\n";

  size_t out_rank = 1;
  std::string shape_string = "{1}";
  if (out_shape.IsValid()) {
    out_rank = out_shape.Dims();
    shape_string = ReplaceRuntimeNames(LSTR(out_shape));
  }

  if (!out_size_expr.empty()) {
    os << "  auto res = choreo::make_spandata<" << out_type << ", " << out_rank
       << ">(" << shape_string << ");\n";
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
  for (auto& s : rts_nmap) {
    size_t pos = 0;
    while ((pos = expr.find(s.first, pos)) != std::string::npos) {
      if (host_code)
        expr.replace(pos, s.first.length(), prefix + s.second);
      else
        expr.replace(pos, s.first.length(), "-1");
    }
  }
  return expr;
}

std::string FactorCodeGen::ReplaceDynDimName(const std::string& e) {
  std::string expr = e;
  for (auto& s : rts_nidx) {
    size_t pos = 0;
    while ((pos = expr.find(s.first, pos)) != std::string::npos) {
      std::string dim_value = "dim_(args[" + std::to_string(rts_pidx[s.first]) +
                              "], " + std::to_string(s.second) + ")";
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

void FactorCodeGen::EmitRuntimeCheck(std::ostream& os, const Type& ty) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto& fty = *cast<FunctionType>(&ty);

  assert(fty.in_tys.size() == host_params.size() &&
         "internal error when dealing with the host parameter size.");

  // check if the input shape is as declared in choreo
  if (fty.in_tys.size() == 0) return;

  struct Entry {
    size_t para_ordinal;
    size_t dim;
    std::string elem_name;
  };
  std::map<ValueExpr, std::vector<Entry>> ve_entries_map;

  for (size_t i = 0; i < fty.in_tys.size(); ++i) {
    auto name = host_params[i];
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
      size_t count = 0;
      for (auto vi : sty->GetShape().Value()) {
        auto elem_name = name + ".shape()[" + std::to_string(count) + "]";
        if (auto vale = dyn_cast<int>(&vi)) {
          os << "  choreo::runtime_check(" << elem_name << " == " << *vale;
          os << ", \"shape inconsistent on the " << Ordinal(i + 1)
             << " parameter (dim: " << count << ").\");\n";
        } else if (auto vale = dyn_cast<ValueExpr>(&vi)) {
          ve_entries_map[*vale].push_back({i + 1, count, elem_name});
        }
        count++;
      }
    }
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

void FactorCodeGen::EmitRuntimeMemUsageCheck(std::ostream& os, const Type& ty) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto& fty = *cast<FunctionType>(&ty);

  assert(fty.in_tys.size() == host_params.size() &&
         "internal error when dealing with the host parameter size.");

  // check if the input shape is as declared in choreo
  if (fty.in_tys.size() == 0) return;

  // there should be runtime memory usage check
  if (!rt_mem_usage_check_list.empty())
    os << "\n  // Check if the runtime memory usage will exceed the limit\n";

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

void FactorCodeGen::EmitHostFuncDecl(std::ostream& os, const Type& ty,
                                     const std::string& name, bool decl_only) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto& fty = *cast<FunctionType>(&ty);
  assert(host_params.size() == fty.in_tys.size() &&
         "inconsistent parameter count.");

  // emit the return type
  os << HostTypeStringify(*fty.out_ty, true) << " " << name << "(";

  if (fty.in_tys.size() > 0) {
    if (!decl_only) {
      if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0])) {
        param_map.push_back(std::make_pair(
            host_params[0], ReplaceRuntimeNames(sty->ByteSizeExpression())));
      } else
        param_map.push_back(std::make_pair(host_params[0], "1"));
    }
    os << HostTypeStringify(*fty.in_tys[0]) << " " << host_params[0];
    for (size_t i = 1; i < fty.in_tys.size(); ++i) {
      if (!decl_only) {
        if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
          param_map.push_back(std::make_pair(
              host_params[i], ReplaceRuntimeNames(sty->ByteSizeExpression())));
        } else
          param_map.push_back(std::make_pair(host_params[i], "1"));
      }
      os << ", " << HostTypeStringify(*fty.in_tys[i]) << " " << host_params[i];
    }
  }
  os << ")" << ((decl_only) ? ";\n" : " ");
}

void FactorCodeGen::OutputScript(FunctionType* fty, const std::string& name,
                                 const std::string& out_type,
                                 const std::string& out_size_expr,
                                 const Shape& out_shape) {
  // a temporal path for the compilation process
  build_path = create_unique_path();
  std::string build_prefix = build_path + "/__choreo_" + name;

  std::string kernel_fn = build_prefix + "_micro_kernel.cpp";
  std::string factor_fn = build_prefix + "_factor.cpp";
  std::string factor_bfn =
      build_path + "/${gcu_target_string}_lib" + current_fn + ".o";
  host_fn = build_prefix + "_host.cpp";
  target_fn = "__choreo_" + name;

  // Generate the host code
  std::string user_code = hs.str();
  hs.clear();

  EmitHostHead(hs);
  if (!user_code.empty()) {
    // user code needs the choreo function decal for call
    EmitHostFuncDecl(hs, *fty, name, true);
    hs << user_code;
  }
  EmitHostFuncDecl(hs, *fty, name);
  EmitHostFuncBody(hs, *fty, factor_bfn, out_size_expr, out_type, out_shape);

  // backpatch the factor bin filename
  std::string factor_src = fs.str();
  if (!alloc_in_fs.str().empty())
    factor_src.insert(alloc_pos, alloc_in_fs.str());
  ReplaceInString(&factor_src, std::string("$$out$$"), output_v);
  // only one `parallel by`
  if (pb_bound1 == -1) {
    pb_bound1 = pb_bound0;
    pb_bound0 = 1;
  }
  ReplaceInString(&factor_src, std::string("$$pb_bound0$$"),
                  std::to_string(pb_bound0));
  ReplaceInString(&factor_src, std::string("$$pb_bound1$$"),
                  std::to_string(pb_bound1));
  ReplaceInString(&factor_src, std::string(backpatch_filename), kernel_fn);

  // Now generate the script
  os << "#!/usr/bin/env bash\n\n";
  os << "# This is the choreo generated bash script to compile factor code\n";
  // check for gcu_target_string first
  //
  os << R"script(
  gcu_arch=gcu210
  gcu_resource=2c24s
  gcu_target_string="dorado_2c"
)script";
  if (!cross_compile)
    os << R"script(
  # check the device
  # TODO: improve the target check with more solid code
  GCU_DEVICE_STR="$(lspci | grep Enflame | head -1)"
  GCU_DEVICE_STR_BACKUP="$(lspci | grep Tencent)"
  echo $GCU_DEVICE_STR
  if [[ "${GCU_DEVICE_STR}" == *"S60G"* ]]; then
    gcu_arch=gcu300
    gcu_resource=1c12s
    gcu_target_string="scorpio_${gcu_resource}"
  elif [[ "${GCU_DEVICE_STR}" == *"c035"* ]]; then
    gcu_arch=gcu300
    gcu_resource=1c12s
    gcu_target_string="scorpio_${gcu_resource}"
    export TOPS_VISIBLE_DEVICES=1
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

  os << "\n# step 0: set up the environment\n";
  os << "rm -fr " << build_path << "\n";
  os << "mkdir -p " << build_path << "\n";
  os << "cat <<'EOF' > " << build_path << "/factor_script.sh\n";
  os << __factor_script_as_string << "\nEOF\n";
  os << "chmod +x " << build_path << "/factor_script.sh\n";
  os << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  os << __choreo_header_as_string << "\nEOF\n\n";

  os << "\n# step 1: write the kernel source code into a temp file\n";
  os << "kernel_src=" << kernel_fn << "\n";
  os << "cat <<'EOF' > ${kernel_src}\n";
  os << ks.str() << "\nEOF\n";

  os << "\n# step 2: write the factor source code into a temp file\n";
  os << "factor_src=" << factor_fn << "\n";
  os << "cat <<'EOF' > ${factor_src}\n";
  os << factor_src << "\nEOF\n\n";

  os << "\n# step 3: set the factor binary file name\n";
  os << "factor_bin=" << factor_bfn << "\n";
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
          if (fut_buf->at(entry_fn).count(id->name))
            oss << fut_buf->at(entry_fn).at(id->name);
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
