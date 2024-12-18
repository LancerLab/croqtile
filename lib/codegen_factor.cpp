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

extern Option<bool> native_f16;
extern Option<std::string> output;
extern Option<bool> cross_compile;
extern Option<bool> use_kernel_template;

bool FactorCodeGen::ContainsLoopVar(const std::string& iv) const {
  for (auto& loop_var : loop_vars)
    if (loop_var.count(iv)) return true;
  return false;
}

bool FactorCodeGen::BeforeVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "Before visiting " << n.TypeNameString() << "\n";

  if (isa<AST::Program>(&n)) {
    // decide the factor build environment
    build_path = CreateUniquePath();
    std::string build_prefix = build_path + "/__choreo_" + factor_pname;

    kernel_cpp_name = build_prefix + "_micro_kernel.cpp";
    factor_cpp_name = build_prefix + "_factor.cpp";
    topsfc_lib_name =
        build_path + "/${gcu_target_string}_lib" + factor_pname + ".o";
    host_cpp_name = build_prefix + "_host.cpp";

    // emit the fixed host header and factor header to their streams
    EmitFixedHostHead();
    EmitFixedFactorHead();
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ResetChoreoFunctionStates();
    factor_fname = "__choreo_" + fname;
    factor_fnames.push_back(factor_fname);

    fty = cast<FunctionType>(GetSymbolType(fname));

    fs << "void " << factor_fname << "() {\n";
    this->IncrementIndent();
    // include the kernel source file
    fs << this->indent << "include_(\"" << kernel_cpp_name << "\");\n";
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level++;
  } else if (isa<AST::ForeachBlock>(&n)) {
    loop_vars.push_back({});
  }

  return 0;
}

bool FactorCodeGen::AfterVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "After visiting " << n.TypeNameString() << "\n";

  if (isa<AST::Program>(&n)) {
    if (HasError()) return false; // do not generate code when error happens
    for (auto& name : factor_fnames) {
      // register factor functions
      std::ostringstream oss;
      oss << "MODULE_REGISTER(\"lib" << factor_pname << "\", " << name
          << ");\n";
      factor_code += oss.str();
    }
    host_code += ds.str() + cs.str() + hs.str();

    switch (CCtx().GetOutputKind()) {
    case OutputKind::TargetSourceCode: EmitFactorSource(); break;
    case OutputKind::TargetModule:
      choreo_unreachable("factor target module is yet to support.");
      break;
    case OutputKind::TargetExecutable: {
      std::ofstream tmpfs("./temp.sh");
      EmitScript(tmpfs);
      tmpfs.close();
      if (!ExecuteScript("./temp.sh", "--compile-binary")) {
        Error(n.LOC(), "failed to compile program.");
        error_count++;
      }
      break;
    }
    case OutputKind::ShellScript: EmitScript(outs()); break;
    default:
      choreo_unreachable("outputkind: " + STR(CCtx().GetOutputKind()) +
                         " is not supported.");
    }
  } else if (isa<AST::ChoreoFunction>(&n)) {
    // choreo-host function:
    // The user code may require the choreo function be fwd-decalared for its
    // call
    EmitHostFuncDecl(ds, fname);
    ds << "; // foward-declaration of choreo-host\n";

    EmitHostFunction(hs);

    // choreo-factor function handling
    if (factor_host_unbraced) {
      this->DecrementIndent();
      // factor requires a fake return
      if (void_return) fs << this->indent << "  return std::vector<Value>{};\n";

      fs << this->indent
         << "}, true); // end of choreo-factor host program\n\n";
    }

    fs << "} // end of " << factor_fname << "\n\n";

    // std::string placeholder = fs.str();
    // while (!alloc_fs_stack.empty()) {
    //   std::cout << "anchor" << std::endl;
    //   std::cout << alloc_fs_stack.size() << std::endl;
    //   std::cout << alloc_indent_stack.size() << std::endl;
    //   std::cout << alloc_pos_stack.size() << std::endl;
    //   std::cout << alloc_fs_stack.top().str() << std::endl;
    //   std::cout << alloc_indent_stack.top() << std::endl;
    //   std::cout << alloc_pos_stack.top() << std::endl;
    //   if (!alloc_fs_stack.top().str().empty())
    //     placeholder.insert(alloc_pos_stack.top(),
    //     alloc_fs_stack.top().str());
    //   alloc_fs_stack.pop();
    //   alloc_pos_stack.pop();
    //   alloc_indent_stack.pop();
    // }
    // fs.str("");
    // fs << placeholder;

    // append the function code to factor source code
    std::string factor_function_code = fs.str();
    factor_code += factor_function_code;
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level--;
    if (parallel_level == 0) {
      this->DecrementIndent();
      fs << this->indent << "}); // end of choreo-factor device function\n";
    }

    std::string placeholder = fs.str();
    if (!alloc_fs_stack.empty()) {
      // std::cout << "exit parallelby" << std::endl;
      // std::cout << alloc_fs_stack.size() << std::endl;
      // std::cout << alloc_indent_stack.size() << std::endl;
      // std::cout << alloc_pos_stack.size() << std::endl;
      // std::cout << alloc_fs_stack.top().str() << std::endl;
      // std::cout << alloc_indent_stack.top() << std::endl;
      // std::cout << alloc_pos_stack.top() << std::endl;
      if (!alloc_fs_stack.top().str().empty())
        placeholder.insert(alloc_pos_stack.top(), alloc_fs_stack.top().str());
      alloc_fs_stack.pop();
      alloc_pos_stack.pop();
      alloc_indent_stack.pop();
    }
    fs.str("");
    fs << placeholder;
  } else if (auto f = dyn_cast<AST::ForeachBlock>(&n)) {
    // erase the loop variables
    assert(!loop_vars.empty());
    loop_vars.pop_back();

    const auto& range_nodes = f->GetRangeNodes();
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
    // handle allocations of this within block
    std::string placeholder = fs.str();
    if (!alloc_fs_stack.empty()) {
      // std::cout << "exit within" << std::endl;
      // std::cout << alloc_fs_stack.size() << std::endl;
      // std::cout << alloc_indent_stack.size() << std::endl;
      // std::cout << alloc_pos_stack.size() << std::endl;
      // std::cout << alloc_fs_stack.top().str() << std::endl;
      // std::cout << alloc_indent_stack.top() << std::endl;
      // std::cout << alloc_pos_stack.top() << std::endl;
      if (!alloc_fs_stack.top().str().empty())
        placeholder.insert(alloc_pos_stack.top(), alloc_fs_stack.top().str());
      alloc_fs_stack.pop();
      alloc_pos_stack.pop();
      alloc_indent_stack.pop();
    }
    fs.str("");
    fs << placeholder;

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
      fs << "select_(" << ExprSTR(s->select_factor) << "== Value(" << i << "), "
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
      std::string storage_type = stringify(sty->GetStorage());
      std::string base_type = stringify(Choreo::BaseType(sty->f_type));
      fs << indent << "auto " << sym << " = bitcast_(" << storage_type << "("
         << base_type << ", ";

      auto shape = sty->GetShape();
      if (shape.IsDynamic()) {
        fs << "{";
        for (size_t i = 0; i < sa->list->Count(); ++i)
          fs << (i != 0 ? ", " : "") << "-1";
        fs << "}), " << buffer_name << ", {";
        for (size_t i = 0; i < sa->list->Count(); ++i) {
          auto value = sa->list->ValueAt(i);
          // TODO: how to utilize shape info
          auto expr_str = PSTR(cast<AST::Expr>(value));
          for (auto& [id_name, _] : idnm_rts)
            expr_str = RegexReplaceAll(expr_str, "\\b" + id_name + "\\b",
                                       named_dim_ref_prefix + id_name);
          fs << (i != 0 ? ", " : "") << expr_str;
        }
        fs << "}";
      } else {
        fs << LSTR(sty->GetShape()) << "), " << buffer_name;
      }
      fs << ");\n";
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
        alloc_fs_stack.top() << alloc_indent_stack.top() << _os_shared.str();
      } else
        alloc_fs_stack.top() << alloc_indent_stack.top() << _os.str();

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
  } else if (CanYieldAnInteger(nty) || isa<ITupleType>(nty)) {
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

bool FactorCodeGen::Visit(AST::Assignment& node) {
  if (auto sa = dyn_cast<AST::SpanAs>(node.value)) {
    int arg_idx = factor_symbols.GetSymbolIndex(sa->id->name);
    std::string buffer_name = sa->id->name;
    if (isa<FutureType>(GetSymbolType(sa->id->name))) {
      assert(FBInfo().count(InScopeName(sa->id->name)));
      buffer_name = UnScopedName(FBInfo().at(InScopeName(sa->id->name)).buffer);
    }
    if (arg_idx >= 0) buffer_name = "args[" + std::to_string(arg_idx) + "]";
    auto sty = cast<SpannedType>(node.GetType());
    fs << indent << "auto " << node.name << " = bitcast_("
       << stringify(sty->GetStorage()) << "(" << stringify(sty->ElementType())
       << ", ";

    auto shape = sty->GetShape();
    if (shape.IsDynamic()) {
      fs << "{";
      for (size_t i = 0; i < sa->list->Count(); ++i)
        fs << (i != 0 ? ", " : "") << "-1";
      fs << "}), " << buffer_name << ", {";
      for (size_t i = 0; i < sa->list->Count(); ++i) {
        auto value = sa->list->ValueAt(i);
        // TODO: how to utilize shape info
        auto expr_str = PSTR(cast<AST::Expr>(value));
        for (auto& [id_name, _] : idnm_rts)
          expr_str = RegexReplaceAll(expr_str, "\\b" + id_name + "\\b",
                                     named_dim_ref_prefix + id_name);
        fs << (i != 0 ? ", " : "") << expr_str;
      }
      fs << "}";
    } else {
      fs << LSTR(sty->GetShape()) << "), " << buffer_name;
    }
    fs << ");\n";
  } else if (isa<BoundedType>(NodeType(node)) ||
             isa<SpannedType>(NodeType(node)) ||
             isa<FutureType>(NodeType(node))) {
    fs << indent << "auto " << node.name << " = " << ExprSTR(node.value)
       << ";\n";
  }

  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::ParallelBy& by) {
  TraceEachVisit(by);
  if (parallel_level > 1) {
    alloc_pos_stack.push(fs.str().size());
    alloc_indent_stack.push(indent);
    alloc_fs_stack.push(std::ostringstream());
    // std::cout << "enter inner parallelly" << std::endl;
    // std::cout << alloc_fs_stack.size() << std::endl;
    // std::cout << alloc_indent_stack.size() << std::endl;
    // std::cout << alloc_pos_stack.size() << std::endl;
    // std::cout << alloc_fs_stack.top().str() << std::endl;
    // std::cout << alloc_indent_stack.top() << std::endl;
    // std::cout << alloc_pos_stack.top() << std::endl;
    return true;
  }

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
    // note: factor code always requires a return statement
    ret << this->indent << "return std::vector<Value>{"
        << ((!void_return) ? UnScopedName(cgi->GetReturnSymbol(fname)) : "")
        << "};\n";
    if (debug_visit)
      VST_DEBUG(dbgs() << "[Factor Host] Return:\n" << ret.str());
    fs << ret.str();
  }

  this->DecrementIndent();
  fs << this->indent << "}, true); // end of choreo-factor host program\n\n";
  factor_host_unbraced = false;

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

  // handle allocation stmts for global vars
  alloc_pos_stack.push(fs.str().size());
  alloc_indent_stack.push(indent);
  alloc_fs_stack.push(std::ostringstream());
  // std::cout << "enter parallelly" << std::endl;
  // std::cout << alloc_fs_stack.size() << std::endl;
  // std::cout << alloc_indent_stack.size() << std::endl;
  // std::cout << alloc_pos_stack.size() << std::endl;
  // std::cout << alloc_fs_stack.top().str() << std::endl;
  // std::cout << alloc_indent_stack.top() << std::endl;
  // std::cout << alloc_pos_stack.top() << std::endl;

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
  // record the position since some codes requires declaration in function scope
  alloc_pos_stack.push(fs.str().size());
  alloc_indent_stack.push(indent);
  alloc_fs_stack.push(std::ostringstream());
  // std::cout << "enter WithIn" << std::endl;
  // std::cout << alloc_fs_stack.size() << std::endl;
  // std::cout << alloc_indent_stack.size() << std::endl;
  // std::cout << alloc_pos_stack.size() << std::endl;
  // std::cout << alloc_fs_stack.top().str() << std::endl;
  // std::cout << alloc_indent_stack.top() << std::endl;
  // std::cout << alloc_pos_stack.top() << std::endl;

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

  if (auto shape = GetShape(NodeType(*n.in)); shape.IsDynamic())
    within_mdspan.emplace_back(STR(GetShape(NodeType(*n.in))), n.LOC());

  return true;
};

bool FactorCodeGen::Visit(AST::DMA& d) {
  TraceEachVisit(d);

  if (auto ph = dyn_cast<PlaceHolderType>(NodeType(d))) {
    assert(ph->Category() == TypeCategory::FUTURE);
    // TODO: optimize when it should be SDMA
    auto fty = cast<FutureType>(GetSymbolType(d.future));
    auto gcu_dma = "CDMA";
    if (GetSpannedType(fty)->GetStorage() == Storage::LOCAL) gcu_dma = "SDMA";
    alloc_fs_stack.top() << alloc_indent_stack.top() << "auto " << d.future
                         << " = alloc_dma_(" << gcu_dma << "Type());\n";
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
        if (bvn == "__choreo_tile_one") {
          offss << "0";
        } else {
          auto ty = cast<BoundedType>(NodeType(*id));
          // iterate over single bounded variables
          for (size_t it_idx = 0; it_idx < ty->Dims(); ++it_idx) {
            std::string name;
            if (within_map.count(InScopeName(bvn))) // with-matcher existed
              name = UnScopedName(within_map[InScopeName(bvn)][it_idx]);
            else
              name = bvn;
            auto iv_str = ExprSTR(AST::Make<AST::Identifier>(id->LOC(), name));
            offss << "Value(" << RSTR(shape.ValueAt(dim_cursor)) << ")*"
                  << iv_str;
          }
        }
        if (++dim_cursor < rank) offss << ",";
      } else if (auto gi_exp = dyn_cast<AST::Expr>(bv)) {
        auto id = cast<AST::Expr>(gi_exp->GetL())->GetSymbol();
        auto ty = cast<BoundedType>(NodeType(*id));
        assert((ty->Dims() == 1) &&
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
      alloc_fs_stack.top() << alloc_indent_stack.top() << "auto " << future_name
                           << " = alloc_dma_("
                           << DMATypeString(src_level, dst_level)
                           << "()).shared_();\n";
    else if (d.chained == true) {
      // hoist sdma for chained usage, to avoid use before definition
      std::stack<std::ostringstream> fs_container;
      std::stack<std::string> indent_container;
      fs_container.push(std::move(alloc_fs_stack.top()));
      indent_container.push(alloc_indent_stack.top());
      alloc_fs_stack.pop();
      alloc_indent_stack.pop();
      while (alloc_fs_stack.top().str().empty()) {
        fs_container.push(std::move(alloc_fs_stack.top()));
        indent_container.push(alloc_indent_stack.top());
        alloc_fs_stack.pop();
        alloc_indent_stack.pop();
      }
      alloc_fs_stack.top() << alloc_indent_stack.top() << "auto " << future_name
                           << " = alloc_dma_("
                           << DMATypeString(src_level, dst_level) << "());\n";
      while (!fs_container.empty()) {
        alloc_fs_stack.push(std::move(fs_container.top()));
        alloc_indent_stack.push(indent_container.top());
        fs_container.pop();
        indent_container.pop();
      }
    } else {
      alloc_fs_stack.top() << alloc_indent_stack.top() << "auto " << future_name
                           << " = alloc_dma_("
                           << DMATypeString(src_level, dst_level) << "());\n";
    }
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

  if (!use_kernel_template || !c.template_args)
    fs << this->indent << "call_(\"" << STR(*c.function) << "\", {";
  else
    fs << this->indent << "call_(\"" << STR(*c.function) << "_template_wrapper"
       << "\", {";
  size_t arg_num = c.arguments->Count();
  for (size_t index = 0; index < arg_num; ++index) {
    auto arg = c.arguments->ValueAt(index);
    fs << ExprSTR(arg)
       << ((isa<SpannedType>(NodeType(*arg))) ? ".addr_()" : "");
    if (index < arg_num - 1) fs << ",";
  }
  fs << "});\n";

  if (use_kernel_template && c.template_args) {
    // handle kernel template wrapper
    ks << "extern \"C\" void " << STR(*c.function) << "_template_wrapper(";
    for (size_t index = 0; index < arg_num; ++index) {
      auto arg = c.arguments->ValueAt(index);
      auto ft = GetUnderlyingType(arg->GetType());
      assert(ft != BaseType::UNKNOWN);
      ks << KernelTypeStringify(ft) << "* ";
      ks << "arg" << index;
      if (index < arg_num - 1) ks << ", ";
    }

    ks << ") {\n";
    ks << "  " << STR(*c.function);
    // c.template_args->SetDelimiter(", ");
    if (c.template_args != nullptr) {
      ks << "<";
      bool need_delimiter = false;
      for (size_t i = 0; i < c.template_args->Count(); ++i) {
        if (need_delimiter) ks << ", ";
        need_delimiter = true;
        ks << ExprSTR(c.template_args->ValueAt(i), false);
      }
      ks << ">";
    }
    ks << "(";
    bool need_delimiter = false;
    for (size_t index = 0; index < arg_num; ++index) {
      if (need_delimiter) ks << ", ";
      need_delimiter = true;
      ks << "arg" << index;
    }
    ks << ");\n";
    ks << "}\n";
  }

  return true;
}

bool FactorCodeGen::Visit(AST::Select& c) {
  TraceEachVisit(c);
  assert(!c.inDMA);
  return true;
}

bool FactorCodeGen::Visit(AST::ForeachBlock& forNode) {
  TraceEachVisit(forNode);
  // auto ty = this->GetSymbolType("l2_tile");
  // ty->Print(os);
  // auto l2_tile_idx = itervars->ValueAt(0);
  // auto l1_tile_idx = itervars->ValueAt(1);
  //
  auto ranges = forNode.GetRanges();
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

bool FactorCodeGen::Visit(AST::FunctionDecl& d) {
  TraceEachVisit(d);

  assert(d.name == fname && "incosistent in function names.");
  assert(isa<FunctionType>(d.GetType()) && "unexpected type.");

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
  if (isa<VoidType>(fty->out_ty)) {
    void_return = true;
    if (debug_visit) VST_DEBUG(dbgs() << "VOID\n");
  } else if (auto rty = dyn_cast<SpannedType>(fty->out_ty)) {
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
        "DRAMType(" + stringify(TC2BT(fty->out_ty->Category())) + ", (1))";
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

bool FactorCodeGen::Visit(AST::Return& n) {
  TraceEachVisit(n);

  if (factor_host_unbraced)
    fs << this->indent << "return std::vector<Value>{" << ExprSTR(n.value)
       << "};\n";

  return true;
}

bool FactorCodeGen::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);

  if (n.host)
    cs << n.GetCode();
  else
    ks << n.GetCode();

  return true;
}

void FactorCodeGen::EmitFixedFactorHead() {
  std::ostringstream oss;
  oss << R"(#include <vector>

#include "gcu/factor/factor.h"

using namespace factor;
)";
  factor_code = oss.str(); // reset factor code
}

void FactorCodeGen::EmitFixedHostHead() {
  std::ostringstream oss;
  oss <<
      R"(
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

// dependant on the topsruntime
#include "tops/tops_ext.h"
#include "tops/tops_runtime.h"

// include the choreo header;
)";
  if (native_f16) oss << "#define NATIVE_F16_SUPPORT\n";
  oss << R"(#include "choreo.h"

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
  host_code = oss.str(); // reset the host code
}

void FactorCodeGen::EmitHostFunction(std::ostream& os) {
  auto rty = fty->out_ty;
  auto out_size_expr = SizeExprOf(*rty);

  // host phase 0: Function declaration
  std::ostringstream fns;
  EmitHostFuncDecl(fns, fname);
  os << fns.str();

  // host phase 1: create tops executable from a file
  os << " {\n";

  EmitHostRuntimeCheck(os);
  EmitHostRuntimeMemUsageCheck(os);

  os << R"(
  std::vector<char> binary;
  // Read bin file and store to a vector
)";
  os << "  std::ifstream ifs(\"" << topsfc_lib_name << "\", std::ios::binary);";
  os << R"(
  std::copy(std::istreambuf_iterator<char>(ifs),
            std::istreambuf_iterator<char>(), std::back_inserter(binary));
  ifs.close();

  // Create stream
  topsStream_t stream = nullptr;
  CHECK(topsStreamCreate(&stream));

)";

  // host phase 2: allocate device memory and copy
  std::vector<std::string> device_mems;
  for (auto& item : GetFactorHostInParams()) {
    auto buffer_name = "in_mem" + std::to_string(device_mems.size());
    std::string size_expr_str = "1";
    if (auto sty = dyn_cast<SpannedType>(item.type))
      size_expr_str = ReplaceRuntimeNames(sty->ByteSizeExpression(true));

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

  // TODO: ULL suffix
  std::string size_string = ReplaceRuntimeNames(out_size_expr);

  if (!void_return) {
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
  if (auto rty = dyn_cast<SpannedType>(fty->out_ty)) {
    tss << RSTR(rty->GetShape());
  } else if (!isa<VoidType>(fty->out_ty)) {
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

  // host phase 3: Execute the executable and fetch the output
  os << "\n  " << factor_fname << "(";
  for (auto& in : inputs) os << "&" << in << ", ";
  os << "stream" << ((void_return) ? "" : ", &output") << ");\n";
  os << "  CHECK(topsStreamSynchronize(stream));\n";

  size_t out_rank = 1;
  std::string shape_string = "{1}";
  if (auto rty = dyn_cast<SpannedType>(fty->out_ty)) {
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
  if (!void_return) os << "  topsFree(out_mem);\n\n";
  os << "  // TODO: figure out why stream destroying crash some "
        "applications.\n";
  os << "  // topsStreamDestroy(stream);\n";
  if (!void_return)
    os << "  return res"
       << (GetChoreoHostReturnTypeString().has_value() ? "[0]" : "") << ";\n";
  os << "}\n";
}

const std::string FactorCodeGen::ReplaceRuntimeNames(const std::string& e,
                                                     const std::string& prefix,
                                                     bool host_code) const {
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

const std::string
FactorCodeGen::ReplaceFactorDynDimName(const std::string& e) const {
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

void FactorCodeGen::EmitHostRuntimeCheck(std::ostream& os) {
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

  // check if the mdspan of within is zero
  if (!within_mdspan.empty())
    os << "\n  // Check if the mdspan of within is zero.\n";
  for (auto& [mds, loc] : within_mdspan) {
    auto mds_vals = SplitStringByDelimiter(mds.substr(1, mds.size() - 2), ", ");
    int idx = 1;
    for (auto& mds_val : mds_vals) {
      os << "  choreo::runtime_check(" << ReplaceRuntimeNames(mds_val)
         << " != " << 0;
      os << ", \"zero is detected for the " << Ordinal(idx)
         << " dim of the mdspan inside the with-in statement, " << loc
         << "\");\n";
      idx++;
    }
  }
}

void FactorCodeGen::EmitHostRuntimeMemUsageCheck(std::ostream& os) {
  // check if the input shape is as declared in choreo
  if (cgi->ParameterCount(fname) == 0) return;

  if (!rt_mem_usage_check_lists.count(fname)) return;

  auto rt_mem_usage_check_list = rt_mem_usage_check_lists.at(fname);

  // there should be runtime memory usage check
  if (!rt_mem_usage_check_list.empty())
    os << "\n  // Check if the runtime memory usage exceeds the defined "
          "limits.\n";

  for (const auto& [useds, loc, limit, sto] : rt_mem_usage_check_list) {
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
      used_ss << (used_ss.str().back() == ')' ? "" : " + ")
              << DelimitedString(operands, "*");
    }
    used_ss << " <= (size_t)" << limit << ", \"total memory usage at "
            << __internal__::GetStringFrom(sto)
            << " level (compile time and runtime) "
               "should not exceed "
            << limit << " bytes, happends at " << loc << "\");\n";
    os << used_ss.str();
  }
}

std::optional<std::string>
FactorCodeGen::GetChoreoHostReturnTypeString() const {
  if (!void_return && cgi->HasReturnSymbol(fname)) {
    auto& item = cgi->GetReturnDetail(fname);
    if (item.rty_str != "$") return item.rty_str;
  }
  return {};
}

void FactorCodeGen::EmitHostFuncDecl(std::ostringstream& oss,
                                     const std::string& name) {
  auto rts = GetChoreoHostReturnTypeString();
  if (rts.has_value())
    oss << *rts;
  else
    oss << HostTypeStringify(*fty->out_ty, true);
  oss << " " << name << "(";

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

const std::string FactorCodeGen::ValueSTR(const ValueItem& vi,
                                          bool factor_value = true) const {
  if (auto i = dyn_cast<int>(&vi)) {
    return "Value(" + std::to_string(*i) + ")";
  } else if (factor_value) {
    // not int => this is a dynamic var or var bounded by dynamic var.
    return ReplaceFactorDynDimName(STR(vi));
  } else {
    return ReplaceRuntimeNames(STR(vi), "", false);
  }
}

const std::string FactorCodeGen::ExprSTR(AST::ptr<AST::Node> e,
                                         bool factor_value) const {
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
    oss << ((factor_value) ? "Value" : "") << "(" << il->value << ")";
  } else if (auto ii = dyn_cast<AST::IntIndex>(e)) {
    return ExprSTR(ii->value);
  } else if (auto expr = dyn_cast<AST::Expr>(e)) {
    // utilize the optimize value whenever possible
    if (auto sym = expr->GetSymbol()) {
      auto sname = InScopeName(sym->name);
      if (FCtx(fname).HasSymbolValues(sname)) {
        auto svs = FCtx(fname).GetSymbolValues(sname);
        if (IsValidValueItem(svs.int_expr))
          return std::string((factor_value) ? "Value" : "") + "(" +
                 STR(svs.int_expr) + ")";
      }
    }
    if (ConvertibleToInt(NodeType(*e))) {
      if (IsValidValueItem(expr->opt_vals.int_expr)) {
        return std::string((factor_value) ? "Value" : "") + "(" +
               STR(expr->opt_vals.int_expr) + ")";
      }
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
        // anchor
        if (rty->Dims() == 1) { oss << ValueSTR(rty->GetUpperBound(), true); }
      } else if (expr->op == "dataof") {
        assert(isa<FutureType>(expr->GetR()->GetType()) &&
               "expect a future operand.");
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
          if (FBInfo().count(InScopeName(id->name)))
            oss << UnScopedName(FBInfo().at(InScopeName(id->name)).buffer);
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
        if (!factor_value) one = "1";
        oss << "((" << ExprSTR(expr->GetL()) << ")+(" << ExprSTR(expr->GetR())
            << "-" << one << ")/(" << ExprSTR(expr->GetR()) << ")";
      } else if (expr->op == "getith") {
        auto lty = cast<BoundedType>(NodeType(*expr->GetL()));
        if (cast<AST::IntIndex>(expr->GetR())->IsNegative()) {
          oss << "(";
          oss << ValueSTR(lty->GetUpperBound(), true);
          oss << "+(" << ExprSTR(expr->GetR()) << "))";
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
      oss << "select_(" << ExprSTR(sl->select_factor, true) << " == ";
      oss << ((factor_value) ? "Value" : "") << "(" << i << ")";
      oss << ", " << PSTR(sl->expr_list->ValueAt(i))
          << (i < val_count - 1 ? ", " : "");
    }
    oss << PSTR(sl->expr_list->AllValues().back())
        << std::string(val_count - 1, ')');
  } else
    choreo_unreachable("unsupported expression '" + expr->op + "'.");

  return oss.str();
}

void FactorCodeGen::EmitScript(std::ostream& ss) {
  // Now generate the script
  ss << "#!/usr/bin/env bash\n\n";
  ss << "# This is the choreo generated bash script to compile factor code\n";

  // JIT: check for gcu_target_string first
  ss << R"script(
  gcu_arch=gcu210
  gcu_resource=2c24s
  gcu_target_string="dorado_2c"
)script";
  if (!cross_compile)
    ss << R"script(
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

  ss << "\n# step 0: set up the environment\n";
  ss << "rm -fr " << build_path << "\n";
  ss << "mkdir -p " << build_path << "\n";
  ss << "cat <<'EOF' > " << build_path << "/factor_script.sh\n";
  ss << __factor_script_as_string << "\nEOF\n";
  ss << "chmod +x " << build_path << "/factor_script.sh\n";
  ss << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  ss << __choreo_header_as_string << "\nEOF\n\n";

  ss << "\n# step 1: write the kernel source code into a temp file\n";
  ss << "kernel_src=" << kernel_cpp_name << "\n";
  ss << "cat <<'EOF' > ${kernel_src}\n";
  ss << ks.str() << "\nEOF\n";

  ss << "\n# step 2: write the factor source code into a temp file\n";
  ss << "factor_src=" << factor_cpp_name << "\n";
  ss << "cat <<'EOF' > ${factor_src}\n";
  ss << factor_code << "\nEOF\n\n";

  ss << "\n# step 3: set the factor binary file name\n";
  ss << "factor_bin=" << topsfc_lib_name << "\n";

  ss << "\n# step 4: generate the host source\n";
  ss << "host_src=" << host_cpp_name << "\n";
  ss << "echo \"#include \\\"\"${gcu_target_string}\"_lib" << factor_pname
     << ".h\\\"\" > ${host_src}\n";
  ss << "cat <<'EOF' >> ${host_src}\n";
  ss << host_code << "\nEOF\n\n";

  ss << "\n# step 5: JIT compile and execute\n";
  ss << "# TODO: enable workflow of AOT compilation\n";
  ss << R"(
if command -v nvim &> /dev/null
then
  EDITOR=nvim
else
  EDITOR=less
fi
set -x 
show_usage() {
    echo "    Usage: $0 | --execute           -> compile and execute choreo in factor
                    | --compile-binary    -> compile the binary code 
                    | --show-kernel       -> show the generated inner kernel code
                    | --show-tileflow     -> show the generated tileflow code scheduled by choreo
                    | --show-host         -> show the generated host side boilerplates"
    exit 1
}
)";
  ss << R"(
if [ "$1" == "--execute" ] || [ "$#" -eq 0 ] || [ "$1" == "--compile-binary" ]; then
  script_flags=$1
  if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then script_flags="--compile-execute"; fi
)";
  ss << "  export FACTOR_INSTALL=" << STRINGIZE(__CHOREO_FACTOR_DIR__) << "\n";
  ss << "  # JIT compile and execute\n";
  if (compile_with_dynshape) ss << "VIEW_CONFIG=1 ENABLE_DYNSHAPE=1 ";
  ss << " bash " << build_path << "/factor_script.sh";
  ss << " ${script_flags} ${factor_src} ${factor_bin} ${host_src} ";
  switch (CCtx().GetOutputKind()) {
  case OutputKind::ShellScript: ss << factor_pname; break;
  case OutputKind::TargetModule: ss << std::string(output); break;
  case OutputKind::TargetExecutable: ss << std::string(output); break;
  default:
    choreo_unreachable(
        "unsupported outputkind: " + STR(CCtx().GetOutputKind()) + ".");
  }
  ss << R"script( ${gcu_arch} ${gcu_resource}
elif [ "$1" == "--show-kernel" ]; then
  ${EDITOR} ${kernel_src}
elif [ "$1" == "--show-host" ]; then
  ${EDITOR} ${host_src}
elif [ "$1" == "--show-tileflow" ]; then
  ${EDITOR} ${factor_src}
else
  show_usage
fi
)script";
}

namespace {
inline static std::string AppendNameAheadOfSuffix(const std::string& filename,
                                                  const std::string& name) {
  size_t dotPos = filename.find_last_of('.');

  if (dotPos == std::string::npos || dotPos == 0)
    return filename + name; // Append name if no extension exists

  // Split the filename into the base name and extension
  std::string baseName = filename.substr(0, dotPos);
  std::string extension = filename.substr(dotPos);

  return baseName + name + extension;
}

} // end anonymous namespace

void FactorCodeGen::EmitFactorSource() {
  auto ifname = OptionRegistry::GetInstance().GetInputFileName();

  outs() << "// ------------------------------------------------------- //\n";
  outs() << "// Choreo generated HOST code (factor) for: \n";
  outs() << "//\n";
  outs() << "//   " << ifname << "\n";
  outs() << "// ------------------------------------------------------- //\n\n";
  outs() << __choreo_header_as_string << "\n";
  outs() << factor_code << "\n\n";
  outs() << host_code << "\n";

  if (!ks.str().empty()) {
    if (OptionRegistry::GetInstance().StdoutAsOutput()) {
      outs()
          << "// ------------------------------------------------------- //\n";
      outs() << "// Choreo generated DEVICE code (factor) for: \n";
      outs() << "//\n";
      outs() << "//   " << ifname << "\n";
      outs() << "// ------------------------------------------------------- "
                "//\n\n";
      outs() << ks.str() << "\n\n";
    } else {
      auto ofname = OptionRegistry::GetInstance().GetOutputFileName();
      auto kernel_filename = AppendNameAheadOfSuffix(
          ofname, "_" + ToLower(STR(CCtx().GetTarget())) + "_" +
                      ToLower(STR(CCtx().GetArch())) + "_device_kernel");
      std::ofstream knls(kernel_filename);
      knls << "// ------------------------------------------------------- //\n";
      knls << "// Choreo generated DEVICE code (factor) for: \n";
      knls << "//\n";
      knls << "//   " << ifname << "\n";
      knls << "// ------------------------------------------------------- "
              "//\n\n";
      knls << ks.str() << "\n\n";
      dbgs() << "[Info] Compiler 'factor' requires multiple inputs for further "
                "compilation:\n";
      dbgs() << " - host code: " << ofname << "\n";
      dbgs() << " - device code: " << kernel_filename << "\n";
    }
  }
}

bool FactorCodeGen::ExecuteScript(const std::string& filename,
                                  const std::string& option) {
  // Make the script executable
  std::string makeExecutableCmd = "chmod +x " + std::string(filename);
  if (system(makeExecutableCmd.c_str()) != 0) {
    VST_DEBUG(errs() << "Could not make the script executable.\n");
    return false;
  }

  // Execute the bash script
  std::string executeCmd = "bash " + filename + " " + option;
  if (system(executeCmd.c_str()) != 0) {
    VST_DEBUG(errs() << "Failed to execute the script.\n");
    return false;
  }
  return true;
}
