#include <iostream>

#include "ast.hpp"
#include "codegen.hpp"
#include "types.hpp"

using namespace Choreo;

extern StringifyTable strtab;

namespace {

using EntryParamType = std::vector<std::pair<std::string, size_t>>;

static inline void print_host_head(std::ostream &os) {
  os <<
R"(
#ifdef __CHOREO_HOST_CODE__  // this is the host code
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
static inline std::vector<uint8_t> ToFactorData(const spanned<T, Rank> &v) {
  auto u8_data = reinterpret_cast<const uint8_t *>(v.data);
    return std::vector<uint8_t>(u8_data, v.bytes());
}

template <int N, typename T>
static inline spanned<N, T> ToSpanned(const std::vector<uint8_t> &v) {
  return make_spanned<N, T, uint8_t>(v.data());
}

// must be true
#define CHECK(a) choreo_assert(a, ##a, __FILE__, __LINE__)

} // end anonymous namespace
#endif //__CHOREO_HOST_CODE__
)";
}

// phase 1: create tops executable from a file
static inline void print_host_phase1(std::ostream &os, const std::string & f_n) {
  os <<
R"(#ifdef __CHOREO_HOST_CODE__  // this is the host code
  std::vector<char> binary;
  // Read bin file and store to a vector
)";
  os << "  std::ifstream ifs(" << f_n << ", std::ios::binary);";
  os << R"(
  std::copy(std::istreambuf_iterator<char>(ifs),
            std::istreambuf_iterator<char>(), std::back_inserter(binary));
  ifs.close();

  // Create executable
  topsExecutable_t executable = nullptr;
  CHECK(topsCreateExecutable(&executable, binary.data(), binary.size()));
  topsStream_t stream = nullptr;
  CHECK(topsStreamCreate(&stream));

#endif //__CHOREO_HOST_CODE__
)";
}

// phase 2: allocate device memory and copy
static inline void print_host_phase2(std::ostream &os,
																		 const EntryParamType & params,
																		 size_t out_size,
																		 std::vector<std::string> & d_params) {
  assert((d_params.size() == 0) && "expecting an empty vector.");

  os << "#ifdef __CHOREO_HOST_CODE__  // this is the host code\n";

  // input parameters
	for (auto & p : params) {
		auto mem_name = "in_mem" + std::to_string(d_params.size());
    os << "  void *" << mem_name << " = nullptr;\n";
    os << "  CHECK(topsMalloc(&" << mem_name << ", " << p.second << "));\n";
    os << "  CHECK(topsMemcpy(" << mem_name << ", reinterpret_cast<void *>("
			 << p.first << ", " << p.second << ", topsMemcpyHostToDevice));\n";
		d_params.push_back(mem_name);
	}
  os << "  void * device_inputs[] = {" << DelimitedString(d_params) << "};\n\n";

  // output parameter
	if (out_size) {
		os << "  void * out_mem = nullptr;\n";
		os << "  CHECK(topsMalloc(&out_mem, " << out_size << "));\n";
		os << "  void *device_outputs[] = {out_mem};\n";
	}
  os << "#endif //__CHOREO_HOST_CODE__";
}

// phase 3: Execute the executable and fetch the output
static inline void print_host_phase3(std::ostream &os,
																		 size_t parallel_factor,
																		 size_t out_size) {
  os << " size_t input_dim = " << parallel_factor << ";\n";
  os << R"(
  size_t input_rank = 1;\n";

  CHECK(topsLaunchExecutableV2(
      executable, nullptr, device_inputs,
      sizeof(device_inputs) / sizeof(void *), &input_dim,
      &input_rank, device_outputs,
      sizeof(device_outputs) / sizeof(void *), stream));
  CHECK(topsStreamSynchronize(stream));

)";

	if (out_size) {
		os << R"(
  // Copy output data from device to host
  std::vector<uint8_t> host_mem2 = {0, 0, 0, 0}; // TODO: manage the memory by mdspan
  CHECK(topsMemcpy(reinterpret_cast<void *>(host_mem2.data()), out_mem,
)";
		os << "                 " << out_size << ", topsMemcpyDeviceToHost));\n";
	}
}

// resource deallocation
static inline void print_host_phase4(std::ostream &os,
																		 std::vector<std::string> & d_params) {
  os << "// Free up the resources\n";
  for (auto & p : d_params)
    os << "  topsFree(" << p << ");\n";
  os << R"(
  topsStreamDestroy(stream);
  topsDestroyExecutable(executable);

  return 0;
})";
}

static inline std::string factor_storage_str(Choreo::Storage s) {
  switch (s) {
    case Storage::LOCAL:
      return "L1Type";
    case Storage::SHARED:
      return "SRAMType";
    case Storage::GLOBAL:
    case Storage::DEFAULT:
      return "DRAMType";
    default:
      choreo_unreachable();
  }
}

static inline std::string stub_type_str(const Choreo::Type & ty,
                                        bool is_ret = false) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (auto sty = dyn_cast<SpannedType>(&ty)) {
    if (is_ret)  // return by value
      return "choreo::spanned<choreo::" +
        getStringFrom((BaseType)sty->f_type) + ", " +
        std::to_string(sty->Dims()) + ">";
    else  // pass by reference
      return "const choreo::spanned<choreo::" +
        getStringFrom((BaseType)sty->f_type) + ", " +
        std::to_string(sty->Dims()) + "> &";
  }
  choreo_unreachable("unsupported stub function type.");
  return "";
}

static inline std::string factor_typestr(Choreo::BaseType t) {
  switch (t) {
    case AST::BaseType::F32:
      return "FloatType(32)";
      break;
    case AST::BaseType::F16:
      return "FloatType(16)";
      break;
    case AST::BaseType::BF16:
      return "BFloatType(16)";
      break;
    case AST::BaseType::U32:
    case AST::BaseType::S32:
      return "IntType(32)";
      break;
    case AST::BaseType::U16:
    case AST::BaseType::S16:
      return "IntType(16)";
      break;
    case AST::BaseType::U8:
    case AST::BaseType::S8:
      return "IntType(8)";
      break;
    // should it be passed in?
    case AST::BaseType::INT:
      return "IntType(32)";
      break;
    case AST::BaseType::BOOL:
      return "BoolType(32)";
      break;
    default:
      choreo_unreachable();
  }
}

}  // end anonymous namespace

bool FactorCodeGen::BeforeVisitImpl(AST::Node &n) {
  if (isa<AST::Program>(&n)) {
//    print_fixed_header(os);
  } else if (auto c = dyn_cast<AST::ChoreoFunction>(&n)) {
    sp_count = 0; // reset the count of stub parameter
    entry_fn = c->name;
    current_fn = "__choreo_" + entry_fn;
    // declare a factor function with proper name
    bs << "using namespace factor;\n";
    bs << "FACTOR_PROGRAM(" << current_fn << ");\n\n";
    bs << "" << current_fn << "([&](auto target_name) {\n";
    this->incrementIndent();
  } else if (isa<AST::ParallelBy>(&n)) {
    // this->incrementIndent();
  } else if (isa<AST::ForeachBlock>(&n)) {
    // this->incrementIndent();
  }
  return 0;
}

bool FactorCodeGen::AfterVisitImpl(AST::Node &n) {
  if (isa<AST::ChoreoFunction>(&n)) {
    size_t out_size = GetByteSizeOf(*(cast<FunctionType>(cur_fty)->out_ty));
    bs << "});\n\n"; // end the factor function definition
    print_host_head(bs);
		GenerateHostFunction(bs, *cur_fty, entry_fn);
		print_host_phase1(bs, bin_fn);
    std::vector<std::string> device_mems;
		print_host_phase2(bs, entry_data, out_size, device_mems);
		print_host_phase3(bs, parallel_factor, out_size);
		print_host_phase4(bs, device_mems);

#if 0
    bs << " // compile the choreo-factor program\n";
    bs << " CompileOptions compile_options;\n";
    bs << " compile_options.gcu_arch = target_name.c_str();\n";
    std::string exe_name = "choreo_" + p->name + "_exe";
    std::string res_name = "choreo_" + p->name + "_res";
    bs << " auto " << exe_name << " = Compile(" << current_fn << ", compile_options);\n";

    // This is the ugly part, we have to copy spanned data into a std::vector
    std::vector<std::string> inputs;
    if (entry_data.size() > 0) {
      bs << " std::vector<std::vector<uint8_t>> inputs_data;\n";
      for (auto & data_size : ) {
        auto input_name = "input" + std::to_string(inputs.size());
        bs << " std::vector<uint8_t> " << input_name
          << "(reinterpret_cast<uint8_t*>(" << data_size.first
          << "), reinterpret_cast<uint8_t*>(" << data_size.first
          << ") + " << data_size.second << ");\n";
        inputs.push_back(input_name);
      }
    }

    bs << " auto " << res_name << " = factor::experimental::Run(";
    for (auto &in: inputs)
      bs << in << ", ";
    bs <<  "compile_options.gcu_arch, reinterpret_cast<char*>(std::get<0>("
       << exe_name << ").get()), std::get<1>(" << exe_name << "));\n";
    bs << "}\n";
#endif

    // now flush both buffer to the output
    FlushBuffers();
    cur_fty = nullptr;
  } else if (isa<AST::ParallelBy>(&n)) {
    this->decrementIndent();
    bs << this->indent << "}); // end of choreo-factor kernel function\n";
  } else if (isa<AST::ForeachBlock>(&n)) {
    this->decrementIndent();
    bs << this->indent << "}); // end of choreo-foreach block\n";
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
bool FactorCodeGen::Visit(AST::NamedVariableDecl &node) {
  // TODO(albert): 'a.span' will be replace to the type-decl related to 'a'
  // TODO(albert): refine this function with TYPE_STR new API
  // bs << AST::TYPE_STR(node);
  auto dtype = dyn_cast<AST::DataType>(node.type.get());
  auto ptype = dtype->getPartialType();
  if (ptype) {
    // get full name of mdspan type
    // NOTE: full ref name may use "a.span" to ref to a var's span partial type
    // the decl of new var will need this symbol "a", not "a.span"
    // we do a hardcode workaround here
    auto full_ref_name = ptype->getRefName();
    auto ref_symbol = full_ref_name.substr(0, full_ref_name.find('.'));
    if (strtab.Exists(ref_symbol)) {
      bs << this->indent;
      bs << "auto " << node.name_str << " = alloc_(";
      bs << strtab.GetTypeSymbol(ref_symbol);
      bs << ");\n";
    } else { //use GetSymbolType to get the required information
      auto ty = dyn_cast<SpannedType>(GetSymbolType(node.name_str));
      assert(ty && "Invalied type for variable declaration!");

      std::string storage_type = factor_storage_str(ty->GetStorage());
      std::string base_type = factor_typestr(Choreo::BaseType(ty->f_type));
      std::string shape_info = "";
      auto data_shape = ty->GetShape();
      std::ostringstream _os;
      _os << data_shape.EmitTo(Target::Factor);
      shape_info += _os.str();
      // auto dim = data_shape.values.values[0];
      // int dim_sz = data_shape.Dims();
      // assert(dim_sz == (int)dim.size() && "Insonsistant sizes for variable span.");
      // for (int dim_cursor = 0; dim_cursor < dim_sz;) {
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

      bs << this->indent << "auto " << node.name_str << " = alloc_(";
      bs << storage_type << "(" << base_type << "," << shape_info;
      bs << ");\n";
    }
  } else {
    // TODO(albert): handle anon case
    bs << this->indent;
    bs << "auto " << node.name_str << " = alloc_(?";
    bs << ");\n";
  }
  //
  // auto spantype = AST::dyn_cast<AST::MultiDimSpans>(ptype);
  // bs << dtype->isSpanned();
  //
  // ptype->Print(os);
  // bs << spantype->ref_name;
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

bool FactorCodeGen::Visit(AST::ParallelBy &by) {
  parallel_factor *= by.bound;
  bs << this->indent << "Dim3 grid_dim(1);\n";
  bs << this->indent << "Dim3 block_dim(" << by.bound << ");\n";
  bs << this->indent << "Value stream = alloc_stream_();\n";
  bs << this->indent << "create_stream_(stream);\n";
  bs << this->indent << "auto ts = launch_kernel_(\"" << current_fn
     << "\", grid_dim, block_dim, stream, {";
  if (cur_params->size() > 0) {
    bs << "args[0]";
    for (size_t i = 1; i < cur_params->size(); ++i) {
      bs << ", "
         << "args[" << i << "]";
    }
  }
  bs << "}, {" << ((void_return) ? "" : "output") << "});\n";
  bs << this->indent << "destroy_stream_(stream);\n";
  bs << this->indent << "dealloc_stream_(stream);\n";
  bs << this->indent << "return std::vector<Value>{" << ((void_return) ? "" : "output") << "};\n";
  this->decrementIndent();
  bs << this->indent << "}); // end of choreo-factor dataflow program\n";
  bs << "\n";

  bs << this->indent << "\n";
  bs << this->indent << "D(func_)(\"" << current_fn << "\", {";
  if (cur_params->size() > 0) {
    bs << (*cur_params)[0]->sym->name << "_type";
    for (unsigned i = 1; i < cur_params->size(); ++i)
      bs << ", " << (*cur_params)[i]->sym->name << "_type";
  }
  bs << "}, {" << ((void_return) ? "" : "choreo_output_type") << "}, [&](auto args, auto results) {\n";
  this->incrementIndent();
  bs << this->indent << "auto thread_id = thread_id_();\n";
  // int i = 0;
  // NOTE: remove unused aliasing 'auto k_a = args[0];'
  // for (auto &param : *cur_params) {
  //   bs << "      "
  //      << "auto k_" << param->sym->name << " = args[" << i++ << "];\n";
  // }

  return true;
}

bool FactorCodeGen::Visit(AST::RequireBind &) { return true; };
bool FactorCodeGen::Visit(AST::WithIn &) { return true; };

bool FactorCodeGen::Visit(AST::WithBlock &) { return true; }

bool FactorCodeGen::Visit(AST::Memory &n) {
  (void)n;
  return true;
}

bool FactorCodeGen::Visit(AST::DMA &d) {
  // handle .to  in AST::Memory
  // d.to->Print(os); // shared
  assert((dyn_cast<AST::ChunkAt>(d.from)) && "Unexpected type for DMA's source.");
  assert((dyn_cast<AST::Memory>(d.to) || dyn_cast<AST::ChunkAt>(d.to)) && "Unexpected type for DMA's destination.");

  auto future_name = d.future;
  auto to_node_name = (dyn_cast<AST::Memory>(d.to))? future_name + "_buffer" :
                      STR(cast<AST::ChunkAt>(d.to)->data) ;
  std::string from_node_name = STR(cast<AST::ChunkAt>(d.from)->data);
  std::string offset_string = "";
  std::string dma_op = "";

  auto getMemLevel = [](Storage s) -> int {
    switch(s) {
      case Storage::LOCAL:
        return 0;
      case Storage::SHARED:
        return 1;
      case Storage::GLOBAL:
      case Storage::DEFAULT:
        return 2;
      default:
        assert(false && "Unexpected storage type.");
        return -1;
    }
  };
  auto storage_level = (dyn_cast<AST::Memory>(d.to))? cast<AST::Memory>(d.to)->getStorageLevel():
                      dyn_cast<SpannedType>(this->GetSymbolType(STR(cast<AST::ChunkAt>(d.to)->data)))->GetStorage();
  int des_level = getMemLevel(storage_level);
  storage_level = dyn_cast<SpannedType>(this->GetSymbolType(STR(cast<AST::ChunkAt>(d.from)->data)))->GetStorage();
  int src_level = getMemLevel(storage_level);

  auto chunkat_node = (src_level >= des_level)? dyn_cast<AST::ChunkAt>(d.from) : dyn_cast<AST::ChunkAt>(d.to);
  assert(chunkat_node && "Unexpected !!!");
  auto chunkat_node_name = STR(chunkat_node->data);
  auto data_type = dyn_cast<SpannedType>(this->GetSymbolType(chunkat_node_name))->f_type;

  if (auto mem_node = dyn_cast<AST::Memory>(d.to)) {
    // TODO(albert): generate 'local_buffer' with more smart naming way by valno support
    bs << this->indent << "auto " << to_node_name << " = alloc_(";
    switch(mem_node->getStorageLevel()) {
      case Storage::LOCAL:
        bs << "L1Type(";
        break;
      case Storage::SHARED:
        bs << "SRAMType(";
        break;
      case Storage::GLOBAL:
        bs << "DRAMType(";
        break;
      default:
        assert(false && "Unexpected storage type.");
    }
    bs << factor_typestr((Choreo::BaseType)data_type) << ",";

    auto ty = dyn_cast<FutureType>(GetSymbolType(future_name));
    assert(ty && "Invalied return type of DMA op!");
    std::string shape_info = "";
    auto data_shape = ty->GetShape();
    std::ostringstream _os;
    _os << data_shape.EmitTo(Target::Factor);
    shape_info += _os.str();
    // WE USE node.SHAPE, not node.DIM_BOUND
    // auto dim = data_shape.values.values[0];
    // bs << dim;
    // int dim_sz = data_shape.Dims();
    // assert(dim_sz == (int)dim.size() && "Insonsistant sizes for variable span.");
    // for (int dim_cursor = 0; dim_cursor < dim_sz;) {
    //   auto dim_bound = *(std::get_if<int>(&dim[dim_cursor]));
    //   assert( dim_bound > 0 && "Invalid variable span!");
    //   shape_info = shape_info + std::to_string(dim_bound);
    //   ++dim_cursor;
    //   if(dim_cursor < dim_sz)
    //     shape_info = shape_info + ",";
    //   else
    //     shape_info = shape_info + "}";
    // }
    bs << shape_info << ");\n";
  }

  if(src_level >= des_level)
    dma_op.append("async_load_(");
  else
    dma_op.append("async_store_(");


  auto tile_factors = chunkat_node->positions;
  auto data_shape = dyn_cast<SpannedType>(this->GetSymbolType(chunkat_node_name))->GetShape();
  int dim_sz = data_shape.Dims();
  offset_string.append("{");
  if(tile_factors){
    assert(dim_sz == (int)tile_factors->getValues().size() && "Insonsistant sizes for DMA offset.");
    auto dim = data_shape.values.values[0];
    assert(dim_sz == (int)dim.size() && "Insonsistant sizes for DMA offset.");
    for (int dim_cursor = 0; dim_cursor < dim_sz;) {
      auto tile_factor = tile_factors->getValues()[dim_cursor];
      auto tf_symbol = STR(tile_factor);
      auto tf_bounds = dyn_cast<BoundedITupleType>(this->GetSymbolType(tf_symbol))->GetBounds().Value();
      auto tf_bound = *(std::get_if<int>(&tf_bounds[0]));
      auto dim_bound = *(std::get_if<int>(&dim[dim_cursor]));
      assert( (tf_bound > 0 && dim_bound > 0) && "Invalid Dim size or Tile factor!");
      // auto offset = (dim_cursor == 0)? std::to_string(dim_bound/tf_bound) + "*thread_id" :
      //                                  std::to_string(dim_bound/tf_bound) + "*" + STR(tile_factor);
      auto offset = (dim_cursor == 0)? "thread_id" :
                                       STR(tile_factor);
      offset_string = offset_string + offset;
      ++dim_cursor;
      if(dim_cursor < dim_sz)
        offset_string = offset_string + ",";
    }
  } else {
    for (int dim_cursor = 0; dim_cursor < dim_sz;) {
      offset_string = offset_string + "0";
      ++dim_cursor;
      if(dim_cursor < dim_sz)
        offset_string = offset_string + ",";
    }
  }
  offset_string = offset_string + "}";

  int arg_idx = strtab.GetSymbolIndex(from_node_name);
  from_node_name = arg_idx < 0 ? from_node_name : "args[" + std::to_string(arg_idx) + "]";
  arg_idx = strtab.GetSymbolIndex(to_node_name);
  to_node_name = arg_idx < 0 ? to_node_name : "args[" + std::to_string(arg_idx) + "]";

  bs << this->indent << "auto " << future_name << " = alloc_dma_(SDMAType());\n";
  bs << this->indent << dma_op << future_name
     << ", " << from_node_name << ", "
     << to_node_name << ", " << offset_string <<  ");\n";

  return true;
}

bool FactorCodeGen::Visit(AST::ChunkAt &) { return true; };

bool FactorCodeGen::Visit(AST::Wait &w) {
  auto dmas = dyn_cast<AST::MultiValues>(w.target);
  assert(dmas && "Invalid wait target!");

  for(auto dma : dmas->getValues()){
    bs << this->indent << "wait_dma_(" << AST::STR(*dma) << ");\n";
  }

  return true;
};

bool FactorCodeGen::Visit(AST::Call &c) {
  bs << this->indent << "call_(\"";
  bs << STR(*c.function);
  bs << "\", {";
  auto args = dyn_cast<AST::MultiValues>(c.arguments);
  assert(args && "Invalid kernel call args!");
  int arg_num = args->getValues().size();
  for(int index = 0; index < arg_num;){
    auto arg = dyn_cast<AST::Expr>(args->getValues()[index]);
    assert(arg && "Invalid kernel call arg!");
    switch (arg->t) {
      case AST::Expr::Reference:
        bs << STR(arg->value_r);
        break;
      case AST::Expr::Unary:
        if(arg->op == "sizeof"){
          auto var = STR(arg->value_r).substr(0,STR(arg->value_r).find('.'));
          assert(dyn_cast<FutureType>(this->GetSymbolType(var)) && "Unexpected !!!");
          auto ty_ptr = cast<FutureType>(this->GetSymbolType(var));
          auto shape = ty_ptr->GetShape();
          auto dim = shape.values.values[0];
          int dim_sz = shape.Dims(), size = 1;
          for (int dim_cursor = 0; dim_cursor < dim_sz;)
            size = size * (*(std::get_if<int>(&dim[dim_cursor++])));
          bs << std::to_string(size);
        }
        else if(arg->op == ".data"){
          bs << STR(arg->value_r) << "_buffer";
        }
        break;
      default:
        bs << STR(*arg);
        choreo_unreachable("unhandled expression type.");
        break;
    }
    index++;
    if(index < arg_num)
      bs << ",";
  }
  bs << "});\n";

  return true;
};


bool FactorCodeGen::Visit(AST::Return &) { return true; };

bool FactorCodeGen::Visit(AST::ForeachBlock &forNode) {
  // auto ty = this->GetSymbolType("l2_tile");
  // ty->Print(os);
  // auto l2_tile_idx = itervars->getValueAt(0);
  // auto l1_tile_idx = itervars->getValueAt(1);
  //
  auto itervars = forNode.getIterationVars();
  for (size_t idx = 0; idx != itervars->Count(); ++idx) {
    // TODO(albert): support non-unit stride in loop
    std::ostringstream _os;
    itervars->getValueAt(idx)->Print(_os);
    auto iv_str = _os.str();

    // get the lower/upper and stride for spanned iter var
    auto iv_type = this->GetSymbolType(iv_str);
    auto iv_bounds = dyn_cast<BoundedITupleType>(iv_type)->GetBounds();
    auto iv_values = iv_bounds.Value();
    // for (const auto &value : iv_values) {
    //   if (auto intValue = std::get_if<int>(&value)) bs << *intValue;
    // }
    // NOTES: bounds always has one integer indicating the upperbound value
    // we can certainly use idx=0 directly
    auto upper_bound = *(std::get_if<int>(&iv_values[0]));

    // synthesise the emitting string
    bs << this->indent << "for_(0, " << std::to_string(upper_bound)
       << ", " << 1 /* TODO(albert): need fix, unit stride is hardcoded for now*/
       << ", " << "[&](auto "
       << iv_str << ") {\n";
  }
  this->incrementIndent();
  return true;
}

void FactorCodeGen::GenerateHostFunction(std::ostream &os, const Type & ty,
																				 const std::string & n) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(&ty);
  os << "#ifdef __CHOREO_HOST_CODE__  // this is the host code\n";
  os << stub_type_str(*fty.out_ty, true) << " " << n << "(";
  if (fty.in_tys.size() > 0) {
    auto n = GenEntryParamName();
    if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0]))
      entry_data.push_back(std::make_pair(n + ".data", sty->ByteSize()));
    else
      entry_data.push_back(std::make_pair(n, 1));
    os << stub_type_str(*fty.in_tys[0]) << " " << n;
    for (size_t i = 1; i < fty.in_tys.size(); ++i) {
      auto n = GenEntryParamName();
      if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i]))
        entry_data.push_back(std::make_pair(n + ".data", sty->ByteSize()));
      else
        entry_data.push_back(std::make_pair(n, 1));
      os << ", " << stub_type_str(*fty.in_tys[i]) << " " << n;
    }
  }
  os << ") {\n";
  os << "#endif //__CHOREO_HOST_CODE__\n";
}

bool FactorCodeGen::Visit(AST::FunctionDecl &d) {
  current_output = d.ret_type;

  assert(isa<FunctionType>(d.GetType()) && "expecting a function type.");
  cur_fty = d.GetType();

  for (auto &param : *cur_params) {
    auto name = param->sym->name;
    if (AST::typeof<SpannedType>(param.get())) {
      // define spanned type
      auto type_symbol = name+"_type";
      std::ostringstream _os;
      // param->type->Print(os, "");
      if(param->type->getPartialType()) {
        _os << param->type->getPartialType()->EmitTo("", Target::Factor);
      } else {
        // TODO: this guard code may not needed
        _os << "{?}";
      }
      auto type_string = "DRAMType(" +
                         factor_typestr(param->type->getBaseType()) +
                         ", " + _os.str();

      strtab.AddSymbol(name, type_symbol, type_string);
      bs << this->indent << "auto " << strtab.GetTypeSymbol(name) << " = " << strtab.GetTypeString(name);
      bs << ");\n";
    } else {
      auto type_symbol = name+"_type";
      auto type_string = "DRAMType(" + factor_typestr(param->type->getBaseType()) + ", (1));";
      strtab.AddSymbol(name, type_symbol, type_string);
      bs << this->indent << "auto " << strtab.GetTypeSymbol(name) << " = " << strtab.GetTypeString(name) << "\n";
    }
  }

  if (AST::typeof<SpannedType>(current_output.get())) {
    auto type_symbol = "choreo_output_type";
    std::ostringstream _os;
    // param->type->Print(os, "");
    if(current_output->getPartialType()) {
      _os << current_output->getPartialType()->EmitTo("", Target::Factor);
    } else {
      // TODO: this guard code may not needed
      _os << "{?}";
    }
    auto type_string = "DRAMType(" +
                       factor_typestr(current_output->getBaseType()) +
                       ", " + _os.str();

    strtab.AddSymbol(type_symbol, type_symbol, type_string);
    bs << this->indent << "auto " << type_symbol << " = " << strtab.GetTypeString(type_symbol);
    bs << ");\n";
  } else if (AST::typeof<VoidType>(current_output)) {
    void_return = true;
  } else {
    auto type_symbol = "choreo_output_type";
    auto type_string = "DRAMType(" + factor_typestr(current_output->getBaseType()) + ", (1));";
    strtab.AddSymbol(type_symbol, type_symbol, type_string);
    bs << this->indent << "auto " << strtab.GetTypeSymbol(type_symbol) << " = " << strtab.GetTypeString(type_symbol) << "\n";
  }

  // bs << "    auto choreo_output_type = DRAMType(";
  // bs << factor_typestr(current_output->getBaseType());
  // bs << ", (1));\n";  // todo

  bs << "\n";
  bs << this->indent << "// choreo-factor dataflow function\n";
  bs << this->indent << "D(main_)({";

  bool first_param = true;
  for (auto &param : *cur_params) {
    auto name = param->sym->name;

    if (first_param) {
      bs << name + "_type";
      first_param = false;
    } else
      bs << ", " << name + "_type";
  }

  bs << "}, [&](auto args) {\n";

  this->incrementIndent();

  return true;
}

bool FactorCodeGen::Visit(AST::ChoreoFunction &) {
  return true;
}

bool FactorCodeGen::Visit(AST::CppSourceCode &n) {
  os << n.GetCode();
  return true;
}

bool FactorCodeGen::Visit(AST::Program &) { return true; }
