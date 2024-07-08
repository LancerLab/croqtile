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

extern StringifyTable strtab;

bool FactorCodeGen::BeforeVisitImpl(AST::Node &n) {
  if (isa<AST::Program>(&n)) {
    //    print_fixed_header(os);
  } else if (auto c = dyn_cast<AST::ChoreoFunction>(&n)) {
    sp_count = 0;  // reset the count of stub parameter
    param_map.clear();
    rts_nmap.clear();
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
  if (isa<AST::Program>(&n)) {
    os << "# step 4: generate the host source\n";
    os << "host_src=" << host_fn << "\n";
    os << "cat <<'EOF' > ${host_src}\n";
    os << hs.str() << "\nEOF\n\n";

    os << "# step 5: compile the host source to target executable\n";
    os << "target=" << target_fn << "\n";
    os << "# TODO: sfc ${host_src} -o ${target}\n";
    os << R"(
if command -v nvim &> /dev/null
then
  EDITOR=nvim
else
  EDITOR=less
fi
if [ "$#" -gt 1 ]; then
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
    os << "  export FACTOR_INSTALL=" << STRINGIZE(__CHOREO_FACTOR_DIR__)
       << "\n";
    os << R"script(
  GCU_DEVICE_STR="$(lspci | grep Enflame | head -1)"
  echo $GCU_DEVICE_STR
  if [[ "${GCU_DEVICE_STR}" == *"S60G"* ]]; then
    gcu_device=gcu3
  elif [[ "${GCU_DEVICE_STR}" == *"c035"* ]]; then
    gcu_device=gcu3
  elif [[ "${GCU_DEVICE_STR}" == *"I20"* ]]; then
    gcu_device=gcu2
  elif [[ "$(lspci | grep Tencent)" != "" ]]; then
    gcu_device=gcu2
  else
    echo "can not determine the GCU device type."
    exit 1
  fi
  # JIT compile and execute
  /tmp/factor_script.sh ${factor_src} ${factor_bin} ${host_src} ${target} ${gcu_device}
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
    fs << "}\n\nMODULE_REGISTER(\"module" << current_fn << "\", " << current_fn
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
bool FactorCodeGen::Visit(AST::IntLiteral &) { return true; };
bool FactorCodeGen::Visit(AST::Expr &) { return true; };
bool FactorCodeGen::Visit(AST::MultiDimSpans &) { return true; };
bool FactorCodeGen::Visit(AST::NamedTypeDecl &) { return true; };

// handle stmts like:
//   f32 [a.span] g_buffer;
//   local f32[f1.span] l_buffer;
//
// ast like:
//   NamedVariableDecl
//   CLEAN
bool FactorCodeGen::Visit(AST::NamedVariableDecl &node) {
  // TODO(albert): 'a.span' will be replace to the type-decl related to 'a'
  // TODO(albert): refine this function with TYPE_STR new API
  // fs << AST::TYPE_STR(node);
  auto dtype = dyn_cast<AST::DataType>(node.type.get());
  auto ptype = dtype->getPartialType();
  if (ptype) {
    // get full name of mdspan type
    // NOTE: full ref name may use "a.span" to ref to a var's span partial type
    // the decl of new var will need this symbol "a", not "a.span"
    // we do a hardcode workaround here
    // auto full_ref_name = ptype->getRefName();
    // auto ref_symbol = full_ref_name.substr(0, full_ref_name.find('.'));
    // os << ref_symbol;
    auto ref_symbol = node.name_str;
    if (strtab.Exists(ref_symbol)) {
      fs << this->indent;
      fs << "auto " << node.name_str << " = alloc_(";
      fs << strtab.GetTypeSymbol(ref_symbol);
      fs << ");\n";
    } else {  // use GetSymbolType to get the required information
      auto ty = dyn_cast<SpannedType>(GetSymbolType(node.name_str));
      assert(ty && "Invalied type for variable declaration!");

      std::string storage_type = factor_storage_str(ty->GetStorage());
      std::string base_type = factor_typestr(Choreo::BaseType(ty->f_type));
      std::string shape_info = "";
      auto data_shape = ty->GetShape();
      std::ostringstream _os;
      _os << ReplaceRuntimeNames(LSTR(data_shape), false);
      shape_info += _os.str();
      // auto dim = data_shape.values.values[0];
      // int dim_sz = data_shape.Dims();
      // assert(dim_sz == (int)dim.size() && "Insonsistant sizes for variable
      // span."); for (int dim_cursor = 0; dim_cursor < dim_sz;) {
      //   auto dim_bound = *(std::get_if<int>(&dim[dim_cursor]));
      //   assert( dim_bound > 0 && "Invalid variable span!");
      //   // shape_info = shape_info + std::to_string(dim_bound/tf_bound);
      //   shape_info = shape_info + std::to_string(dim_bound);
      //   ++dim_cursor;
      //   if(dim_cursor < dim_sz)
      //     shape_info = shape_info + ",";
      //   else
      //     shape_info = shape_info + "}";
      // }
      _os.str("");
      _os.clear();
      _os << "    "
          << "auto " << node.name_str << " = alloc_(";
      _os << storage_type << "(" << base_type << "," << shape_info << ")";
      _os << ");\n";
      if (storage_type == "DRAMType")
        fs << _os.str();
      else
        alloc_in_fs << _os.str();

      // generate "memset_()" action to initiate each alloc_memory with value 0
      if (storage_type == "L1Type")
        fs << this->indent << "auto " << node.name_str
           << "_init = alloc_dma_(SDMAType());\n";
      else
        fs << this->indent << "auto " << node.name_str
           << "_init = alloc_dma_(CDMAType());\n";

      fs << this->indent << "memset_(" << node.name_str << "_init, "
         << node.name_str << ", 0);\n";
    }
  } else {
    // TODO(albert): handle anon case
    fs << this->indent;
    fs << "auto " << node.name_str << " = alloc_(?";
    fs << ");\n";
  }
  //
  // auto spantype = AST::dyn_cast<AST::MultiDimSpans>(ptype);
  // fs << dtype->isSpanned();
  //
  // ptype->Print(os);
  // fs << spantype->ref_name;
  // TODO: hardcode 'a', wait for expr eval

  return true;
};
bool FactorCodeGen::Visit(AST::IntTuple &) { return true; };
bool FactorCodeGen::Visit(AST::Assignment &) { return true; };
bool FactorCodeGen::Visit(AST::IntIndex &) { return true; };
bool FactorCodeGen::Visit(AST::DataType &) { return true; };

bool FactorCodeGen::Visit(AST::Identifier &n) {
  (void)n;
  return true;
}

bool FactorCodeGen::Visit(AST::Parameter &p) {
  (void)p;
  return true;
}

bool FactorCodeGen::Visit(AST::ParamList &pl) {
  cur_params = &pl.values;
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::ParallelBy &by) {
  parallel_factor *= by.bound;
  if (parallel_level > 1) {
    return true;
  }
  fs << this->indent << "Dim3 grid_dim(1);\n";
  fs << this->indent << "Dim3 block_dim(" << by.bound << ");\n";
  fs << this->indent << "Value stream = alloc_stream_();\n";
  fs << this->indent << "create_stream_(stream);\n";
  fs << this->indent << "auto ts = launch_kernel_(\"" << current_fn
     << "\", grid_dim, block_dim, stream, {";
  if (cur_params->size() > 0) {
    fs << "args[0]";
    for (size_t i = 1; i < cur_params->size(); ++i) {
      fs << ", "
         << "args[" << i << "]";
    }
  }
  fs << "}, {" << ((void_return) ? "" : "$$out$$")
     << "});\n";  // "$$out$$" : magic string for output, will be replaced later
  fs << this->indent << "destroy_stream_(stream);\n";
  fs << this->indent << "dealloc_stream_(stream);\n";
  fs << this->indent << "return std::vector<Value>{"
     << ((void_return) ? "" : "$$out$$") << "};\n";
  this->decrementIndent();
  fs << this->indent << "}); // end of choreo-factor dataflow program\n";
  fs << "\n";

  fs << this->indent << "\n";
  fs << this->indent << "D(func_)(\"" << current_fn << "\", {";
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
  (void)n;
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::DMA &d) {
  // handle .to  in AST::Memory
  assert((isa<AST::ChunkAt>(d.from)) && "Unexpected type for DMA's source.");
  assert((isa<AST::Memory>(d.to) || isa<AST::ChunkAt>(d.to)) &&
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
  auto dst_buffer_name = (isa<AST::Memory>(d.to))
                             ? future_name + "_buffer"
                             : STR(cast<AST::ChunkAt>(d.to)->data);

  std::string src_node_name = STR(cast<AST::ChunkAt>(d.from)->data);
  assert(!src_node_name.empty() && "expect a named future/span in chunkat.");
  // use source symbol as the buffer name
  std::string src_buffer_name = src_node_name;
  if (isa<FutureType>(GetSymbolType(
          RemoveSuffix(cast<AST::ChunkAt>(d.from)->data->name, ".data"))))
    src_buffer_name = src_node_name + "_buffer";

  auto sty = GetSpannedType(*d.from);  // source spanned type
  size_t rank = sty->Dims();
  auto dst_shape = ty->GetShape();
  auto src_sto = sty->GetStorage();
  auto dst_sto = (isa<AST::Memory>(d.to)) ? cast<AST::Memory>(d.to)->Get()
                                          : GetSpannedType(*d.to)->GetStorage();
  int src_level = MemLevel(src_sto);
  int dst_level = MemLevel(dst_sto);

  // allocate storage for DMA destination when it is not explicitly stated.
  if (auto mem_node = dyn_cast<AST::Memory>(d.to)) {
    // TODO(albert): generate 'local_buffer' with more smart naming way by valno
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
                << ReplaceRuntimeNames(LSTR(dst_shape), false) << "));\n";
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

  // strtab.Print(fs);
  int arg_idx = strtab.GetSymbolIndex(src_node_name);
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
  alloc_in_fs << alloc_indent << "auto " << future_name << " = alloc_dma_("
              << DMATypeString(src_level, dst_level) << "());\n";

  // decide the dma operation
  std::string dma_op = "";
  if (src_level >= dst_level)
    dma_op.append("async_load_");
  else
    dma_op.append("async_store_");

  ptr<AST::Node> chunkat_node = nullptr;

  if (isa<AST::Memory>(d.to))
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

  fs << ");\n";

  // synchornized dma must be waited
  if (!ty->IsAsync()) fs << indent << "wait_dma_(" << future_name << ");\n";

  return true;
}

bool FactorCodeGen::Visit(AST::ChunkAt &) { return true; }

bool FactorCodeGen::Visit(AST::Wait &w) {
  auto dmas = w.targets;
  assert(dmas && "Invalid wait target!");

  for (auto dma : dmas->AllValues()) {
    fs << this->indent << "wait_dma_(" << AST::STR(*dma) << ");\n";
  }

  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::Call &c) {
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
        } catch (const std::invalid_argument& e) {
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
        os << STR(arg->GetForm());
        choreo_unreachable("unhandled expression type.");
        break;
    }
    index++;
    if (index < arg_num) fs << ",";
  }
  fs << "});\n";

  return true;
}

bool FactorCodeGen::Visit(AST::Return &ReturnNode) {
  if (ReturnNode.value) output_v = STR(*ReturnNode.value);
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::ForeachBlock &forNode) {
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
    auto iv_bounds = dyn_cast<BoundedITupleType>(iv_type)->GetBounds();
    auto iv_values = iv_bounds.Value();

    // NOTES: foreach block ranges between [0, UB),
    // it always use one integer indicating the UB
    // we can certainly use idx=0 directly
    auto ub_value = GetAt<int>(iv_values, 0);

    // synthesise the emitting string
    if (iv_type->Dims() == 1) {
      fs << this->indent << "for_(" << id->name << ", "
         << std::to_string(ub_value) << ", "
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
      for (auto name : cur_bounded_vars[id->name]) {
        fs << this->indent << "for_(" << name << ", "
           << std::to_string(ub_value) << ", "
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
      }
    }
  }
  return true;
}

// CLEAN
bool FactorCodeGen::Visit(AST::FunctionDecl &d) {
  auto ty = d.GetType();
  assert(isa<FunctionType>(ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(ty);

  auto MapRuntimeShapeNames = [this](SpannedType *sty,
                                     const std::string &name) {
    size_t count = 0;
    for (auto vi : sty->GetShape().Value()) {
      if (auto vale = dyn_cast<ValueExpr>(&vi)) {
        auto elem_name = name + ".shape()[" + std::to_string(count) + "]";
        rts_nmap.emplace(*vale, elem_name);
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
      MapRuntimeShapeNames(sty, n);
  }

  for (auto &param : *cur_params) {
    auto name = param->sym->name;
    if (auto sty = dyn_cast<SpannedType>(param->GetType())) {
      // define spanned type
      auto type_symbol = name + "_type";
      auto type_string = "DRAMType(" + factor_typestr(sty->ElementType()) +
                         ", " +
                         ReplaceRuntimeNames(LSTR(sty->GetShape()), false);

      strtab.AddSymbol(name, type_symbol, type_string);
      fs << this->indent << "auto " << strtab.GetTypeSymbol(name) << " = "
         << strtab.GetTypeString(name);
      fs << ");\n";
    } else {
      auto type_symbol = name + "_type";
      auto type_string =
          "DRAMType(" + factor_typestr(param->type->getBaseType()) + ", (1));";
      strtab.AddSymbol(name, type_symbol, type_string);
      fs << this->indent << "auto " << strtab.GetTypeSymbol(name) << " = "
         << strtab.GetTypeString(name) << "\n";
    }
  }

  if (auto rty = dyn_cast<SpannedType>(fty.out_ty)) {
    auto name = "output";
    auto type_symbol = "output_type";
    auto type_string = "DRAMType(" + factor_typestr(rty->ElementType()) + ", " +
                       ReplaceRuntimeNames(LSTR(rty->GetShape()), false);

    strtab.AddSymbol(name, type_symbol, type_string);
    fs << this->indent << "auto " << strtab.GetTypeSymbol(name) << " = "
       << strtab.GetTypeString(name);
    fs << ");\n";
  } else if (isa<VoidType>(fty.out_ty)) {
    void_return = true;
  } else {
    auto name = "output";
    auto type_symbol = "output_type";
    auto type_string =
        "DRAMType(" + factor_typestr(TC2BT(fty.out_ty->Category())) + ", (1));";
    strtab.AddSymbol(name, type_symbol, type_string);
    fs << this->indent << "auto " << strtab.GetTypeSymbol(name) << " = "
       << strtab.GetTypeString(name) << "\n";
  }

  // fs << "    auto output_type = DRAMType(";
  // fs << factor_typestr(current_output->getBaseType());
  // fs << ", (1));\n";  // todo

  fs << "\n";
  fs << this->indent << "// choreo-factor dataflow function\n";
  fs << this->indent << "D(main_)({";

  bool first_param = true;
  for (auto &param : *cur_params) {
    auto name = param->sym->name;

    if (first_param) {
      fs << name + "_type";
      first_param = false;
    } else
      fs << ", " << name + "_type";
  }

  fs << "}, [&](auto args) {\n";

  this->incrementIndent();

  return true;
}

bool FactorCodeGen::Visit(AST::ChoreoFunction &) { return true; }

bool FactorCodeGen::Visit(AST::CppSourceCode &n) {
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
#include <chrono>
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
  // phase 1: create tops executable from a file
  os << "{\n";
  EmitRuntimeCheck(os, ty);
  os << R"(
  std::vector<char> binary;
  // Read bin file and store to a vector
)";
  os << "  std::ifstream ifs(\"" << f_n << "\", std::ios::binary);";
  os << R"(
  std::copy(std::istreambuf_iterator<char>(ifs),
            std::istreambuf_iterator<char>(), std::back_inserter(binary));
  ifs.close();

  // Create executable
  topsExecutable_t executable = nullptr;
  CHECK(topsCreateExecutable(&executable, binary.data(), binary.size()));
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
       << p.first << "), " << p.second << ", topsMemcpyHostToDevice));\n";
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

  // phase 3: Execute the executable and fetch the output
  auto &fty = *cast<FunctionType>(&ty);
  std::ostringstream tss;  // temporal stream
  if (fty.in_tys.size() > 0) {
    tss << "  int64_t input_dims[] = {";
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0])) {
      sty->GetShape().PrintPlain(tss);
      for (size_t i = 1; i < fty.in_tys.size(); ++i)
        if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
          tss << ", ";
          sty->GetShape().PrintPlain(tss);
        }
    }
    os << ReplaceRuntimeNames(tss.str());
    os << "};\n";
    os << "  size_t input_ranks[] = {";
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0])) {
      os << sty->GetShape().Dims();
      for (size_t i = 1; i < fty.in_tys.size(); ++i)
        if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i]))
          os << ", " << sty->GetShape().Dims();
    }
  }
  os << "};\n";

  os << R"(
  auto start = std::chrono::high_resolution_clock::now();

  CHECK(topsLaunchExecutableV2(
      executable, nullptr, device_inputs,
      sizeof(device_inputs) / sizeof(void *), (int64_t*)input_dims,
      (size_t*)input_ranks, device_outputs,
      sizeof(device_outputs) / sizeof(void *), stream));
  CHECK(topsStreamSynchronize(stream));

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;

)";

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
  os << R"(
  // Free up the resources";
)";
  for (auto &p : device_mems) os << "  topsFree(" << p << ");\n";
  os << R"(
  topsStreamDestroy(stream);
  topsDestroyExecutable(executable);
  return res;
}
)";
}

std::string FactorCodeGen::ReplaceRuntimeNames(const std::string &e,
                                               bool host_code) {
  std::string expr = e;
  for (auto &s : rts_nmap) {
    size_t start_pos = expr.find(s.first);
    if (start_pos != std::string::npos) {
      if (host_code)
        expr.replace(start_pos, s.first.length(), s.second);
      else
        expr.replace(start_pos, s.first.length(), "-1");
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
        param_map.push_back(
            std::make_pair(host_params[0] + ".data()",
                           ReplaceRuntimeNames(sty->ByteSizeExpression())));
      } else
        param_map.push_back(std::make_pair(host_params[0], "1"));
    }
    os << HostTypeString(*fty.in_tys[0]) << " " << host_params[0];
    for (size_t i = 1; i < fty.in_tys.size(); ++i) {
      if (!decl_only) {
        if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i])) {
          param_map.push_back(
              std::make_pair(host_params[i] + ".data()",
                             ReplaceRuntimeNames(sty->ByteSizeExpression())));
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
  // it requires temporal files for the compilation process
  std::string kernel_fn =
      create_unique_filename("__choreo_" + n + "_micro_kernel.cpp");
  std::string factor_fn =
      create_unique_filename("__choreo_" + n + "_factor.cpp");
  std::string factor_bfn =
      create_unique_filename("__choreo_" + n + "_factor.fb");
  host_fn = create_unique_filename("__choreo_" + n + "_host.cpp");
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
  os <<
      R"(#!/usr/bin/env bash

# This the the choreo generated bash script to compile factor code

)";
  os << "# copy factor scripts & environment to /tmp\n";
  os << "cat <<'EOF' > /tmp/factor_script.sh\n";
  os << __factor_script_as_string << "\nEOF\n";
  os << "chmod +x /tmp/factor_script.sh\n";

  os << "# copy choreo.h and factor scripts/env to /tmp\n";
  os << "cat <<'EOF' > /tmp/choreo.h\n";
  os << __choreo_header_as_string << "\nEOF\n\n";
  os << "# step 1: write the kernel source code into a temp file\n";
  os << "kernel_src=" << kernel_fn << "\n";
  os << "cat <<'EOF' > ${kernel_src}\n";
  os << ks.str() << "\nEOF\n\n";

  os << "# step 2: write the factor source code into a temp file\n";
  os << "factor_src=" << factor_fn << "\n";
  os << "cat <<'EOF' > ${factor_src}\n";
  os << factor_src << "\nEOF\n\n";

  os << "# step 3: compile factor code into a binary\n";
  os << "factor_bin=" << factor_bfn << "\n";
  os << "# TODO: sfc ${factor_src} -o ${factor_bin}\n\n";
}
