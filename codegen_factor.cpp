#include "codegen_factor.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "choreo_header.inc"
#include "codegen.hpp"
#include "factor_script.inc"
#include "types.hpp"

#ifndef __CHOREO_FACTOR_DIR__
#error "missing macro definition of __CHOREO_FACTOR_DIR__"
#endif

using namespace Choreo;
using namespace Choreo::Factor;

#define __TRACE_EACH_VISIT__(d)       \
  if (trace_visit) {                  \
    os << d.TypeNameString() << ": "; \
    os << "\n";                       \
  }

bool FactorCodeGen::ContainsLoopVar(const std::string &iv) const {
  for (auto &loop_var : loop_vars)
    if (loop_var.count(iv)) return true;
  return false;
}

std::string Shape::EmitTo(Target target) const {
  (void)target;
  std::ostringstream _os;
  if (!IsValidValueNumber(val_no))
    _os << "{}";
  else {
    assert(values.Exists(val_no) && "bad value number.");
    Factor::EmitFactorValueList(Value(), _os);
  }
  return _os.str();
}

static StringifyTable factor_symbols;

bool FactorCodeGen::BeforeVisitImpl(AST::Node &n) {
  __TRACE_EACH_VISIT__(n)
  if (isa<AST::Program>(&n)) {
    //    print_fixed_header(os);
  } else if (auto c = dyn_cast<AST::ChoreoFunction>(&n)) {
    sp_count = 0;  // reset the count of stub parameter
    param_map.clear();
    rts_nmap.clear();
    rts_pidx.clear();
    rts_nidx.clear();
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
    fs << indent << "include_(\"" << backpatch_filename << "\");\n";
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level++;
    // this->incrementIndent();
  } else if (isa<AST::ForeachBlock>(&n)) {
    loop_vars.push_back({});
    // this->incrementIndent();
  }
  return 0;
}

// CLEAN
bool FactorCodeGen::AfterVisitImpl(AST::Node &n) {
  __TRACE_EACH_VISIT__(n)
  if (isa<AST::Program>(&n)) {
    os << "\n# step 4: generate the host source\n";
    os << "host_src=" << host_fn << "\n";
    os << "echo \"#include \\\"${gcu_target_string}_lib" << current_fn
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
if [ "$#" -ne 1 ]; then
    echo "    Usage: $0 | --execute           -> compile and execute choreo in factor
                    | --statistics        -> show Line Of Code (LOC) statistic compare between kernel code boosted w./w.o. Choreo
                    | --show-kernel       -> show the generated inner kernel code
                    | --show-tileflow     -> show the generated tileflow code scheduled by choreo
                    | --show-host         -> show the generated host side boilerplates
                    | --show-choreo       -> show the choreo source code"
    exit 1
fi
)";
    os << R"(
if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then
)";
    os << R"script(
  # check the device
  # TODO: improve the target check with more solid code
  GCU_DEVICE_STR="$(lspci | grep Enflame | head -1)"
  echo $GCU_DEVICE_STR
  if [[ "${GCU_DEVICE_STR}" == *"S60G"* ]]; then
    gcu_arch=gcu300
    gcu_resource=1c12s
    gcu_target_string="scorpio_${gcu_resource}"
  elif [[ "${GCU_DEVICE_STR}" == *"c035"* ]]; then
    gcu_arch=gcu300
    gcu_resource=1c12s
    gcu_target_string="scorpio_${gcu_resource}"
    export TOP_VISIBLE_DEVICES=1
  elif [[ "${GCU_DEVICE_STR}" == *"I20"* ]]; then
    gcu_arch=gcu210
    gcu_resource=2c24s
    gcu_target_string="dorado_2c"
  elif [[ "$(lspci | grep Tencent)" != "" ]]; then
    gcu_arch=gcu210
    gcu_resource=2c24s
    gcu_target_string="dorado_2c"
  else
    echo "can not determine the GCU device type."
    exit 1
  fi
)script";
    os << "  export FACTOR_INSTALL=" << STRINGIZE(__CHOREO_FACTOR_DIR__)
       << "\n# JIT compile and execute\n";
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
    echo "    Usage: $0 | --execute           -> compile and execute choreo in factor
                    | --statistics        -> show Line Of Code (LOC) statistic compare between kernel code boosted w./w.o. Choreo
                    | --show-kernel       -> show the generated inner kernel code
                    | --show-tileflow     -> show the generated tileflow code scheduled by choreo
                    | --show-host         -> show the generated host side boilerplates
                    | --show-choreo       -> show the choreo source code"
    exit 1
fi
)script";

  } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    entry_fn = f->name;
    current_fn = "__choreo_" + entry_fn;
    auto fty = cast<FunctionType>(f->GetType());
    auto &out_type = fty->out_ty;
    auto out_size = GetByteSizeExprOf(*out_type);
    fs << "}\n\nMODULE_REGISTER(\"lib" << current_fn << "\", " << current_fn
       << ");";  // end the factor function definition
    if (auto sty = dyn_cast<SpannedType>(out_type)) {
      OutputScript(fty, f->name, GetBaseTypeStringOf(*out_type), out_size,
                   sty->GetShape());
    } else
      OutputScript(fty, f->name, GetBaseTypeStringOf(*out_type), out_size,
                   Shape() /*invalid shape*/);
    ResetBuffers();
  } else if (isa<AST::ParallelBy>(&n)) {
    parallel_level--;
    this->decrementIndent();
    if (parallel_level == 0)
      fs << this->indent << "}); // end of choreo-factor kernel function\n";
  } else if (auto f = dyn_cast<AST::ForeachBlock>(&n)) {
    // erase the loop variables
    assert(!loop_vars.empty());
    loop_vars.pop_back();

    for (auto id : f->ivs->AllValues()) {
      auto name = cast<AST::Identifier>(id)->name;
      int dec_by = 1;
      bool multiple_bounds = cur_bounded_vars.count(name);
      if (multiple_bounds) dec_by = cur_bounded_vars[name].size();
      for (int i = 0; i < dec_by; ++i) {
        decrementIndent();
        fs << indent << "}); // end of choreo-foreach block";
        if (multiple_bounds) fs << " on '" << cur_bounded_vars[name][i] << "'";
        fs << ".\n";
      }
    }
  } else if (auto wb = dyn_cast<AST::WithBlock>(&n)) {
    for (auto wi : wb->withins->AllSubs()) {
      auto w = cast<AST::WithIn>(wi);
      if (w->with && w->with_matchers) {
        cur_bounded_vars.erase(w->with->name);
      }
    }
  }
  return 0;
}

bool FactorCodeGen::Visit(AST::MultiNodes &) { return true; }
bool FactorCodeGen::Visit(AST::MultiValues &) { return true; }
bool FactorCodeGen::Visit(AST::IntLiteral &) { return true; }
bool FactorCodeGen::Visit(AST::Boolean &) { return true; }
bool FactorCodeGen::Visit(AST::Expr &) { return true; }
bool FactorCodeGen::Visit(AST::MultiDimSpans &) { return true; }
bool FactorCodeGen::Visit(AST::NamedTypeDecl &) { return true; }

// handle stmts like:
//   f32 [a.span] g_buffer;
//   local f32[f1.span] l_buffer;
//
// ast like:
//   NamedVariableDecl
//   CLEAN
bool FactorCodeGen::Visit(AST::NamedVariableDecl &node) {
  __TRACE_EACH_VISIT__(node)
  // TODO(albert): 'a.span' will be replace to the type-decl related to 'a'
  // TODO(albert): refine this function with TYPE_STR new API
  auto nty = node.GetType();
  auto sym = node.name_str;
  if (auto sty = dyn_cast<SpannedType>(nty)) {
    assert(isa<SpannedType>(GetSymbolType(sym)) && "Inconsistent types!");
    if (factor_symbols.Exists(sym)) {
      fs << indent << "auto " << sym << " = alloc_("
         << factor_symbols.GetTypeName(sym) << ");\n";
    } else {
      std::string storage_type = factor_storage_str(sty->GetStorage());
      std::string base_type = factor_typestr(Choreo::BaseType(sty->f_type));
      std::ostringstream _os;
      _os << "auto " << sym << " = alloc_(" << storage_type << "(" << base_type
          << "," << ReplaceRuntimeNames(LSTR(sty->GetShape()), "", false) << ")"
          << ");\n";
      if (storage_type == "DRAMType")
        fs << indent << _os.str();
      else
        alloc_in_fs << "    " << _os.str();

      fs << indent << "auto " << sym << "_init = alloc_dma_("
         << ((storage_type == "L1Type") ? "SDMAType()" : "CDMAType()")
         << ");\n";

      // generate "memset_()" action to initiate each alloc_memory with value 0
      fs << indent << "memset_(" << sym << "_init, " << sym << ", 0);\n";
    }
  } else {
    choreo_unreachable("non-spanned is not yet supported.");
    // TODO(albert): handle anon case
    fs << this->indent;
    fs << "auto " << node.name_str << " = alloc_(?";
    fs << ");\n";
  }

  return true;
}

bool FactorCodeGen::Visit(AST::IntTuple &) { return true; }
bool FactorCodeGen::Visit(AST::Assignment &) { return true; }
bool FactorCodeGen::Visit(AST::IntIndex &) { return true; }
bool FactorCodeGen::Visit(AST::DataType &) { return true; }

bool FactorCodeGen::Visit(AST::Identifier &n) {
  __TRACE_EACH_VISIT__(n)
  (void)n;
  return true;
}

bool FactorCodeGen::Visit(AST::Parameter &p) {
  __TRACE_EACH_VISIT__(p)
  (void)p;
  return true;
}

bool FactorCodeGen::Visit(AST::ParamList &pl) {
  __TRACE_EACH_VISIT__(pl)
  cur_params = &pl.values;
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::ParallelBy &by) {
  __TRACE_EACH_VISIT__(by)
  parallel_factor *= by.bound;
  if (parallel_level > 1) {
    return true;
  }
  fs << this->indent << "Dim3 grid_dim(1);\n";
  fs << this->indent << "Dim3 block_dim(" << by.bound << ");\n";
  fs << this->indent << "auto ts = launch_kernel_(\"" << current_fn
     << "_parallel\", grid_dim, block_dim, args.back(), {";
  if (cur_params->size() > 0) {
    fs << "args[0]";
    for (size_t i = 1; i < cur_params->size(); ++i) fs << ", args[" << i << "]";
  }
  fs << "}, {" << ((void_return) ? "" : "$$out$$")
     << "});\n";  // "$$out$$" : magic string for output, will be replaced later
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
  alloc_pos = fs.str().size();
  alloc_indent = indent;

  return true;
}

bool FactorCodeGen::Visit(AST::WhereBind &n) {
  __TRACE_EACH_VISIT__(n)
  // establish the binding
  auto lid = cast<AST::Identifier>(n.lhs);
  auto rid = cast<AST::Identifier>(n.rhs);
  bind_info.AddBind(SSTab().ScopedName(lid->name),
                    SSTab().ScopedName(rid->name));

  // also adds the value binding for the with-matchers
  if (cur_bounded_vars.count(lid->name)) {
    assert(cur_bounded_vars.count(rid->name));
    auto lbvs = cur_bounded_vars[lid->name];
    auto rbvs = cur_bounded_vars[rid->name];
    assert(lbvs.size() == rbvs.size());

    for (size_t i = 0; i < lbvs.size(); ++i) {
      bind_info.AddBind(SSTab().ScopedName(lbvs[i]),
                        SSTab().ScopedName(rbvs[i]));
    }
  }
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::WithIn &n) {
  __TRACE_EACH_VISIT__(n)
  assert(n.with_matchers && "expect matcher to be exist.");

  // associate with to the matcher.
  if (n.with && n.with_matchers) {
    std::vector<std::string> matchers;
    for (auto mn : n.with_matchers->AllValues()) {
      matchers.push_back(cast<AST::Identifier>(mn)->name);
    }
    cur_bounded_vars.emplace(n.with->name, matchers);
  }

  for (auto mn : n.with_matchers->AllValues()) {
    auto mname = cast<AST::Identifier>(mn)->name;
    fs << indent << "var_ " << mname << "(IntType(32));\n";
    fs << indent << mname << " = 0;\n";
  }

  return true;
};

bool FactorCodeGen::Visit(AST::WithBlock &) { return true; }

bool FactorCodeGen::Visit(AST::Memory &n) {
  __TRACE_EACH_VISIT__(n)
  (void)n;
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::DMA &d) {
  __TRACE_EACH_VISIT__(d)
  // handle .to  in AST::Memory
  assert((isa<AST::ChunkAt>(d.from)) && "Unexpected type for DMA's source.");
  assert((isa<AST::Memory>(d.to) || isa<AST::ChunkAt>(d.to) ||
          isa<AST::Select>(d.to)) &&
         "Unexpected type for DMA's destination.");

  // retrieve the spanned type from a chunkat
  auto GetSpannedType = [this](AST::Node &ca) -> SpannedType * {
    auto sty = ca.GetType();
    if (auto fty = dyn_cast<FutureType>(sty))
      return fty->GetSpannedType().get();
    else
      return cast<SpannedType>(sty);
  };

  auto MemLevel = [](Storage s) -> int {
    switch (s) {
      case Storage::LOCAL:
        return 0;
      case Storage::SHARED:
        return 1;
      case Storage::GLOBAL:
      case Storage::DEFAULT:
        return 2;
      default:
        choreo_unreachable("Unexpected storage type.");
        return -1;
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

  // cook a valid dst buffer name
  // note: for select, we also use the future_name + "_buffer" as handle name.
  auto dst_buffer_name = (isa<AST::ChunkAt>(d.to))
                             ? STR(cast<AST::ChunkAt>(d.to)->data)
                             : future_name + "_buffer";
  // if to node is AST::SELECT, use its future name, otherwise keep default one
  // dst_buffer_name = (isa<AST::Select>(d.to))
  //                            ? STR(cast<AST::Select>(d.to)->future)
  //                            : future_name + "_buffer";

  std::string src_node_name = STR(cast<AST::ChunkAt>(d.from)->data);
  assert(!src_node_name.empty() && "expect a named future/span in chunkat.");
  // use source symbol as the buffer name
  // lhs_load => lhs_load_buffer used by user of lhs_load
  std::string src_buffer_name = src_node_name;
  if (isa<FutureType>(GetSymbolType(
          RemoveSuffix(cast<AST::ChunkAt>(d.from)->data->name, ".data"))))
    src_buffer_name = src_node_name + "_buffer";

  auto sty = GetSpannedType(*d.from);  // source spanned type
  size_t rank = sty->Dims();
  auto dst_shape = ty->GetShape();
  auto src_sto = sty->GetStorage();
  auto dst_sto = Storage::DEFAULT;
  if (isa<AST::Memory>(d.to))
    dst_sto = cast<AST::Memory>(d.to)->Get();
  else if (isa<AST::Select>(d.to))
    // TODO(albert): get mem level from selects operands
    dst_sto = Storage::LOCAL;
  else
    dst_sto = GetSpannedType(*d.to)->GetStorage();
  // auto dst_sto = (isa<AST::Memory>(d.to)) ? cast<AST::Memory>(d.to)->Get()
  //                                         :
  //                                         GetSpannedType(*d.to)->GetStorage();
  // dst_sto = (isa<AST::Select>(d.to)) ? Storage::LOCAL
  //                                         :
  //                                         GetSpannedType(*d.to)->GetStorage();
  int src_level = MemLevel(src_sto);
  int dst_level = MemLevel(dst_sto);

  // allocate storage for DMA destination when it is not explicitly stated.
  if (auto mem_node = dyn_cast<AST::Memory>(d.to)) {
    // support
    static std::map<Storage, std::string> sto2alloc = {
        {Storage::LOCAL, "L1Type"},
        {Storage::SHARED, "SRAMType"},
        {Storage::GLOBAL, "DRAMType"},
    };
    // buffer in another stream
    alloc_in_fs << alloc_indent << "auto " << dst_buffer_name << " = alloc_("
                << sto2alloc.at(mem_node->Get()) << "("
                << factor_typestr(sty->ElementType()) << ","
                << ReplaceRuntimeNames(LSTR(dst_shape), "", false) << "));\n";
  }

  auto GenerateOffsetString = [this, &GetSpannedType](AST::Node &n) {
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
    for (auto &bv : ca->positions->AllValues()) {
      auto bvn = cast<AST::Identifier>(bv)->name;
      if (auto bity = dyn_cast<BoundedITupleType>(bv->GetType())) {
        for (size_t it_idx = 0; it_idx < bity->Dims(); ++it_idx) {
          std::string iv_str;
          if (within_map.count(bvn))  // with-matcher existed
            iv_str = within_map[bvn][it_idx];
          else
            iv_str = bvn;

          // prefix iteration variable
          if (ContainsLoopVar(iv_str)) iv_str = "iv_" + iv_str;

          // special handling for the parallel tiling factor
          auto l = RemovePrefixOrNull("pv:", bity->GetNote());
          if (l.has_value()) {
            // is marked as parallel whose level is decided by target check
            if (*l == "0")
              iv_str = "thread_id";
            else if (*l == "1")
              iv_str = "block_id";
            else
              choreo_unreachable("invalid type note.");
          }

          offss << "Value(" << RSTR(shape.ValueAt(dim_cursor)) << ")*"
                << iv_str;
          if (++dim_cursor < rank) offss << ",";
        }
      } else
        choreo_unreachable("unsupported type.");
    }
    return "{" + offss.str() + "}";
  };

  // factor_symbols.Print(fs);
  int arg_idx = factor_symbols.GetSymbolIndex(src_node_name);
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
  if (d.chained == true && ((d.chain_to != "" && src_level > dst_level) ||
                            (d.chain_from != "" && src_level < dst_level)))
    alloc_in_fs << alloc_indent << "auto " << future_name << " = alloc_dma_("
                << DMATypeString(src_level, dst_level) << "()).shared_();\n";
  else
    alloc_in_fs << alloc_indent << "auto " << future_name << " = alloc_dma_("
                << DMATypeString(src_level, dst_level) << "());\n";

  // decide the dma operation
  std::string dma_op = "";
  if (src_level >= dst_level)
    dma_op.append("async_load_");
  else
    dma_op.append("async_store_");

  ptr<AST::Node> chunkat_node = nullptr;

  if (isa<AST::Memory>(d.to) || isa<AST::Select>(d.to))
    chunkat_node = d.from;
  else if (cast<AST::ChunkAt>(d.to)->positions)
    chunkat_node = d.to;
  else
    choreo_unreachable("factor: unsupported chunkat.");

  fs << indent << dma_op << "(" << future_name << ", " << src_buffer_name
     << ", " << dst_buffer_name << ", " << GenerateOffsetString(*chunkat_node);

  if (auto pcfg = dyn_cast<PadConfig>(d.config)) {
    std::vector<size_t> layout(rank);
    std::iota(layout.begin(), layout.end(), 0);  // no transpose
    fs << ", {" << DelimitedString(layout) << "}, {"
       << DelimitedString(pcfg->pad_low) << "}, {"
       << DelimitedString(pcfg->pad_high) << "}, {"
       << DelimitedString(pcfg->pad_mid) << "}, " << pcfg->value.v;
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

bool FactorCodeGen::Visit(AST::ChunkAt &) { return true; }

bool FactorCodeGen::Visit(AST::Wait &w) {
  __TRACE_EACH_VISIT__(w)
  auto dmas = w.targets;
  assert(dmas && "Invalid wait target!");

  for (auto dma : dmas->AllValues()) {
    fs << this->indent << "wait_dma_(" << AST::STR(*dma) << ");\n";
  }

  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::Call &c) {
  __TRACE_EACH_VISIT__(c)
  fs << this->indent << "call_(\"";
  fs << STR(*c.function);
  fs << "\", {";
  auto args = c.arguments;
  assert(args && "Invalid kernel call args!");
  int arg_num = args->AllValues().size();
  for (int index = 0; index < arg_num;) {
    auto arg = dyn_cast<AST::Expr>(args->AllValues()[index]);
    assert(arg && "Invalid kernel call arg!");
    switch (arg->GetForm()) {
      case AST::Expr::Reference:
        try {
          std::stoi(STR(arg->GetR()));
          fs << STR(arg->GetR());
        } catch (const std::invalid_argument &e) {
          fs << STR(arg->GetR()) << ".addr_()";
        }
        break;
      case AST::Expr::Unary:
        if (arg->op == "sizeof") {
          auto var = STR(arg->GetR()).substr(0, STR(arg->GetR()).find('.'));
          assert(dyn_cast<FutureType>(this->GetSymbolType(var)) &&
                 "Unexpected !!!");
          auto ty_ptr = cast<FutureType>(this->GetSymbolType(var));
          auto shape = ty_ptr->GetShape();
#if 0
          auto shapes = shape.Value();
          auto dim = shape.values.values[0];
          int dim_sz = shape.Dims(), size = 1;
          for (int dim_cursor = 0; dim_cursor < dim_sz;)
            size = size * (*(std::get_if<int>(&shapes[dim_cursor++])));
          fs << std::to_string(size);
#endif
          fs << shape.GetSizeExpression();
        } else if (arg->op == "dataof") {
          fs << STR(arg->GetR()) << "_buffer"
             << ".addr_()";
        }
        break;
      default:
        choreo_unreachable("unhandled expression type: " +
                           std::to_string((int)(arg->GetForm())) + ".");
        break;
    }
    index++;
    if (index < arg_num) fs << ",";
  }
  fs << "});\n";

  return true;
}

bool FactorCodeGen::Visit(AST::Select &c) {
  __TRACE_EACH_VISIT__(c)
  size_t val_count = c.val_list->Count();
  // if val_count == 1, pingpong is meaningless? ( TODO: maybe assert when
  // earlysema)
  assert(val_count >= 2);
  fs << this->indent << "auto " << c.future << " = ";
  for (size_t i = 0; i < val_count - 1; i++) {
    fs << "select_(" << STR(c.select_factor) << "== " << i << ", "
       << STR(c.val_list->ValueAt(i)) << (i < val_count - 1 ? ", " : "");
  }
  fs << STR(c.val_list->AllValues().back()) << std::string(val_count - 1, ')')
     << ";\n";
  return true;
}

bool FactorCodeGen::Visit(AST::Return &returnNode) {
  __TRACE_EACH_VISIT__(returnNode)
  if (returnNode.value) output_v = STR(*returnNode.value);
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::ForeachBlock &forNode) {
  __TRACE_EACH_VISIT__(forNode)
  // auto ty = this->GetSymbolType("l2_tile");
  // ty->Print(os);
  // auto l2_tile_idx = itervars->ValueAt(0);
  // auto l1_tile_idx = itervars->ValueAt(1);
  //
  auto itervars = forNode.getIterationVars();
  for (size_t idx = 0; idx != itervars->Count(); ++idx) {
    // TODO(albert): support non-unit stride in loop
    std::ostringstream _os;
    auto id = cast<AST::Identifier>(itervars->ValueAt(idx));

    // get the lower/upper and stride for spanned iter var
    auto iv_type = this->GetSymbolType(id->name);
    auto iv_bounds = cast<BoundedITupleType>(iv_type)->GetBounds();

    // NOTES: foreach block ranges between [0, UB),
    // it always use one integer indicating the UB
    // we can certainly use idx=0 directly

    // synthesise the emitting string
    if (iv_type->Dims() == 1) {
      fs << this->indent << "for_(" << id->name << ", "
         << ReplaceDynDimName(STR(iv_bounds.ValueAt(0))) << ", "
         << 1 /* TODO(albert): need fix, unit stride is hardcoded for now*/
         << ", [&](auto iv_" << id->name << ") {\n";
      incrementIndent();
      loop_vars.back().insert(id->name);
      for (auto bind : bind_info.GetBinds(InScopeName(id->name))) {
        auto bname = SSTab().UnScopedName(bind);
        loop_vars.back().insert(bname);
        fs << indent << "auto iv_" << SSTab().UnScopedName(bind) << " = iv_"
           << id->name << ";\n";
      }
    } else {
      assert(cur_bounded_vars.count(id->name) &&
             "can not find the bounded name.");
      assert((cur_bounded_vars[id->name].size() == iv_bounds.Dims()) &&
             "can not find the bounded name.");
      size_t i = 0;
      for (auto name : cur_bounded_vars[id->name]) {
        fs << this->indent << "for_(" << name << ", "
           << ReplaceDynDimName(STR(iv_bounds.ValueAt(i))) << ", "
           << 1 /* TODO(albert): need fix, unit stride is hardcoded for now*/
           << ", [&](auto iv_" << name << ") {\n";
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
bool FactorCodeGen::Visit(AST::FunctionDecl &d) {
  __TRACE_EACH_VISIT__(d)
  auto ty = d.GetType();
  assert(isa<FunctionType>(ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(ty);

  auto MapRuntimeShapeNames = [this](SpannedType *sty, const std::string &name,
                                     size_t p_index) {
    size_t count = 0;
    for (auto vi : sty->GetShape().Value()) {
      if (auto vale = dyn_cast<ValueExpr>(&vi)) {
        auto elem_name = name + ".shape()[" + std::to_string(count) + "]";
        rts_nmap.emplace(*vale, elem_name);
        rts_pidx.emplace(*vale, p_index);
        rts_nidx.emplace(*vale, count);
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

  std::ostringstream dss;  // for the shape string
  for (auto &param : *cur_params) {
    auto pname = param->sym->name;
    std::string type_name = pname + "_type";
    std::string type_string;
    if (auto sty = dyn_cast<SpannedType>(param->GetType())) {
      // define spanned type
      type_string = "DRAMType(" + factor_typestr(sty->ElementType()) + ", " +
                    ReplaceRuntimeNames(LSTR(sty->GetShape()), "", false) + ")";
      factor_symbols.AddSymbol(pname, type_name, type_string);
    } else {
      type_string =
          "DRAMType(" + factor_typestr(param->type->getBaseType()) + ", (1))";
      factor_symbols.AddSymbol(pname, type_name, type_string);
    }
    fs << indent << "auto " << type_name << " = " << type_string << ";\n";
  }

  if (auto rty = dyn_cast<SpannedType>(fty.out_ty)) {
    std::string name = "output";
    std::string type_name = "output_type";
    auto type_string = "DRAMType(" + factor_typestr(rty->ElementType()) + ", " +
                       ReplaceRuntimeNames(LSTR(rty->GetShape()), "", false);

    // handle dynamic-typed output when necessary. Generate code snippet like:
    //
    //   auto output_rt_dim0 = dim_(args[0], 1);
    //   auto output = alloc_({output_rt_dim0}, output_type);
    //
    const auto &dyn_dims = rty->GetShape().GetDynamicDims();
    if (!dyn_dims.empty()) {
      dyn_shaped = true;
      type_name.clear();
      size_t i = 0;
      for (auto &ddim : dyn_dims) {
        auto ddim_name = name + "_rt_dim" + std::to_string(i);
        dss << indent << "  auto " << ddim_name << " = "
            << ReplaceDynDimName(ddim.second) << ";\n";
        if (type_name.size() == 0)
          type_name += ddim_name;
        else
          type_name = ddim_name + ", " + type_name;
        ++i;
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
        "DRAMType(" + factor_typestr(TC2BT(fty.out_ty->Category())) + ", (1));";
    factor_symbols.AddSymbol(name, type_name, type_string);
    fs << indent << "auto " << type_name << " = " << type_string << "\n";
  }

  fs << "\n";
  fs << this->indent << "// choreo-factor dataflow function\n";
  fs << this->indent << "D(host_func_)(\"" << current_fn << "\", {";

  for (auto &param : *cur_params) fs << param->sym->name + "_type, ";

  fs << "StreamType()}, [&](auto args) {\n";

  this->incrementIndent();
  fs << dss.str();  // dynamic-shape specific

  return true;
}

bool FactorCodeGen::Visit(AST::ChoreoFunction &) { return true; }

bool FactorCodeGen::Visit(AST::CppSourceCode &n) {
  __TRACE_EACH_VISIT__(n)
  if (n.host) {
    hs << n.GetCode();
  } else {
    ks << n.GetCode();
  }
  return true;
}

bool FactorCodeGen::Visit(AST::Program &) { return true; }

void FactorCodeGen::EmitHostHead(std::ostream &os) {
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

void FactorCodeGen::EmitHostFuncBody(std::ostream &os, const Type &ty,
                                     const std::string &f_n,
                                     const std::string &out_size,
                                     const std::string &out_type,
                                     const Shape &out_shape) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(&ty);

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
  for (auto &p : param_map) {
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

  std::string size_string = ReplaceRuntimeNames(out_size);

  if (!out_size.empty()) {
    os << "  void * out_mem = nullptr;\n";
    os << "  CHECK(topsMalloc(&out_mem, " << size_string << "));\n";
    os << "  void *device_outputs[] = {out_mem};\n";
  }

  std::vector<std::string> inputs;  // factor input paramters

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
  for (auto &in : inputs) os << "&" << in << ", ";
  os << "stream" << ((void_return) ? "" : ", &output") << ");\n";
  os << "  CHECK(topsStreamSynchronize(stream));\n";

  size_t out_rank = 1;
  std::string shape_string = "{1}";
  if (out_shape.IsValid()) {
    out_rank = out_shape.Dims();
    shape_string = ReplaceRuntimeNames(LSTR(out_shape));
  }

  if (!out_size.empty()) {
    os << "  auto res = choreo::make_spandata<" << out_type << ", " << out_rank
       << ">(" << shape_string << ");\n";
    os << "  // Copy output data from device to host\n";
    os << "  CHECK(topsMemcpy(reinterpret_cast<void *>(res.data()), out_mem,\n";
    os << "                  " << size_string
       << ", topsMemcpyDeviceToHost));\n";
  }

  // phase 4: Free up the resources
  os << "  // Free up the resources\n";
  for (auto &p : device_mems) os << "  topsFree(" << p << ");\n";
  os << "  topsFree(out_mem);\n\n";
  os << "  // TODO: figure out why stream destroying crash some "
        "applications.\n";
  os << "  // topsStreamDestroy(stream);\n";
  os << "  return res;\n";
  os << "}\n";
}

std::string FactorCodeGen::ReplaceRuntimeNames(const std::string &e,
                                               const std::string &prefix,
                                               bool host_code) {
  std::string expr = e;
  for (auto &s : rts_nmap) {
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

std::string FactorCodeGen::ReplaceDynDimName(const std::string &e) {
  std::string expr = e;
  for (auto &s : rts_nidx) {
    size_t pos = 0;
    while ((pos = expr.find(s.first, pos)) != std::string::npos) {
      std::string dim_value = "dim_(args[" + std::to_string(rts_pidx[s.first]) +
                              "], " + std::to_string(s.second) + ")";
      expr.replace(pos, s.first.length(), dim_value);
    }
  }
  return expr;
}

void FactorCodeGen::EmitRuntimeCheck(std::ostream &os, const Type &ty) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(&ty);

  assert(fty.in_tys.size() == host_params.size() &&
         "internal error when dealing with the host parameter size.");

  // check if the input shape is as declared in choreo
  if (fty.in_tys.size() == 0) return;

  if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0])) {
    auto name = host_params[0];
    size_t count = 0;
    for (auto vi : sty->GetShape().Value()) {
      if (auto vale = dyn_cast<int>(&vi)) {
        auto elem_name = name + ".shape()[" + std::to_string(count) + "]";
        os << "  choreo::runtime_check(" << elem_name << " == " << *vale;
        os << ", \"shape inconstant on 1st parameter (dim: " << count
           << ").\");\n";
      }
      count++;
    }
  }
  for (size_t i = 1; i < fty.in_tys.size(); ++i) {
    auto name = host_params[i];
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
      size_t count = 0;
      for (auto vi : sty->GetShape().Value()) {
        if (auto vale = dyn_cast<int>(&vi)) {
          auto elem_name = name + ".shape()[" + std::to_string(count) + "]";
          os << "  choreo::runtime_check(" << elem_name << " == " << *vale;
          os << ", \"shape inconstant on " << i + 1
             << "th parameter (dim: " << count << ").\");\n";
        }
        count++;
      }
    }
  }
}

void FactorCodeGen::EmitRuntimeMemUsageCheck(std::ostream &os, const Type &ty) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(&ty);

  assert(fty.in_tys.size() == host_params.size() &&
         "internal error when dealing with the host parameter size.");

  // check if the input shape is as declared in choreo
  if (fty.in_tys.size() == 0)
    return;

  // there should be runtime memory usage check
  if (!rt_mem_usage_check_list.empty())
    os << "\n  // Check if the runtime memory usage will exceed the limit\n";

  for (const auto &[useds, loc, limit] : rt_mem_usage_check_list) {
    std::ostringstream used_ss;
    used_ss << "  choreo::runtime_check((size_t)";
    for (auto &used : useds) {
      if (used.find(":") == std::string::npos) {
        // `used` is compile time memory usage
        used_ss << (used_ss.str().back() == ')' ? "" : " + ") << used;
        continue;
      }
      // `used` is runtime memory usage
      auto operands = SplitStringByDelimiter(used, "*");
      for (auto &o : operands) {
        if (o.find(":") != std::string::npos) {
          // `o` is dynamic dim. Should replace it with host name
          for (auto &[sig, name] : rts_nmap) {
            for (size_t p = o.find(sig); p != std::string::npos;
                 p = o.find(sig)) {
              o.replace(p, sig.size(), name);
            }
          }
        }
      }
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

void FactorCodeGen::EmitHostFuncDecl(std::ostream &os, const Type &ty,
                                     const std::string &n, bool decl_only) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(&ty);
  assert(host_params.size() == fty.in_tys.size() &&
         "inconsistent parameter count.");

  // emit the return type
  os << HostTypeString(*fty.out_ty, true) << " " << n << "(";

  if (fty.in_tys.size() > 0) {
    if (!decl_only) {
      if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0])) {
        param_map.push_back(std::make_pair(
            host_params[0], ReplaceRuntimeNames(sty->ByteSizeExpression())));
      } else
        param_map.push_back(std::make_pair(host_params[0], "1"));
    }
    os << HostTypeString(*fty.in_tys[0]) << " " << host_params[0];
    for (size_t i = 1; i < fty.in_tys.size(); ++i) {
      if (!decl_only) {
        if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
          param_map.push_back(std::make_pair(
              host_params[i], ReplaceRuntimeNames(sty->ByteSizeExpression())));
        } else
          param_map.push_back(std::make_pair(host_params[i], "1"));
      }
      os << ", " << HostTypeString(*fty.in_tys[i]) << " " << host_params[i];
    }
  }
  os << ")" << ((decl_only) ? ";\n" : " ");
}

void FactorCodeGen::OutputScript(FunctionType *fty, const std::string &n,
                                 const std::string &out_type,
                                 const std::string &out_size,
                                 const Shape &out_shape) {
  // a temporal path for the compilation process
  build_path = create_unique_path();
  std::string build_prefix = build_path + "/__choreo_" + n;

  std::string kernel_fn = build_prefix + "_micro_kernel.cpp";
  std::string factor_fn = build_prefix + "_factor.cpp";
  std::string factor_bfn =
      build_path + "/${gcu_target_string}_lib" + current_fn + ".o";
  host_fn = build_prefix + "_host.cpp";
  target_fn = "__choreo_" + n;

  // Generate the host code
  std::string user_code = hs.str();
  hs.clear();

  EmitHostHead(hs);
  if (!user_code.empty()) {
    // user code needs the choreo function decal for call
    EmitHostFuncDecl(hs, *fty, n, true);
    hs << user_code;
  }
  EmitHostFuncDecl(hs, *fty, n);
  EmitHostFuncBody(hs, *fty, factor_bfn, out_size, out_type, out_shape);

  // backpatch the factor bin filename
  std::string factor_src = fs.str();
  if (!alloc_in_fs.str().empty())
    factor_src.insert(alloc_pos, alloc_in_fs.str());
  ReplaceInString(factor_src, std::string("$$out$$"), output_v);
  ReplaceInString(factor_src, std::string(backpatch_filename), kernel_fn);

  // Now generate the script
  os << "#!/usr/bin/env bash\n\n";
  os << "# This is the choreo generated bash script to compile factor code\n";

//   os << R"script(
//   # check the device
//   # TODO: improve the target check with more solid code
//   GCU_DEVICE_STR="$(lspci | grep Enflame | head -1)"
//   echo $GCU_DEVICE_STR
//   if [[ "${GCU_DEVICE_STR}" == *"S60G"* ]]; then
//     gcu_arch=gcu300
//     gcu_resource=1c12s
//     gcu_target_string="scorpio_${gcu_resource}"
//   elif [[ "${GCU_DEVICE_STR}" == *"c035"* ]]; then
//     gcu_arch=gcu300
//     gcu_resource=1c12s
//     gcu_target_string="scorpio_${gcu_resource}"
//     export TOP_VISIBLE_DEVICES=1
//   elif [[ "${GCU_DEVICE_STR}" == *"I20"* ]]; then
//     gcu_arch=gcu210
//     gcu_resource=2c24s
//     gcu_target_string="dorado_2c"
//   elif [[ "$(lspci | grep Tencent)" != "" ]]; then
//     gcu_arch=gcu210
//     gcu_resource=2c24s
//     gcu_target_string="dorado_2c"
//   else
//     echo "can not determine the GCU device type."
//     exit 1
//   fi
// )script";
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
