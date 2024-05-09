#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "codegen.hpp"
#include "types.hpp"
#include "choreo_header.inc"

using namespace Choreo;

extern StringifyTable strtab;

namespace {

constexpr const char *backpatch_filename =
    "__choreo_kernel_file_name_that_will_be_back_patched_soon_ok_enough_i_am_"
    "bored__";

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

inline static std::string create_unique_filename(
    const std::string &custom_string) {
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

  // Construct the filename
  std::string filename = "/tmp/" + std::to_string(nanoseconds) + "_" +
                         thread_id + "_" + custom_string;

  return filename;
}

static inline void print_host_head(std::ostream &os) {
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

// phase 1: create tops executable from a file
static inline void print_host_phase1(std::ostream &os, const std::string &f_n) {
  os <<
      R"(std::vector<char> binary;
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
}

// phase 2: allocate device memory and copy
static inline void print_host_phase2(std::ostream &os,
                                     const FactorCodeGen::EntryParamsInfo &params,
                                     const std::string & out_size,
                                     std::vector<std::string> &d_params) {
  assert((d_params.size() == 0) && "expecting an empty vector.");

  // input parameters
  for (auto &p : params) {
    auto mem_name = "in_mem" + std::to_string(d_params.size());
    os << "  void *" << mem_name << " = nullptr;\n";
    os << "  CHECK(topsMalloc(&" << mem_name << ", " << p.second << "));\n";
    os << "  CHECK(topsMemcpy(" << mem_name << ", reinterpret_cast<void *>("
       << p.first << "), " << p.second << ", topsMemcpyHostToDevice));\n";
    d_params.push_back(mem_name);
  }
  os << "  void * device_inputs[] = {" << DelimitedString(d_params) << "};\n\n";

  // output parameter
  if (!out_size.empty()) {
    os << "  void * out_mem = nullptr;\n";
    os << "  CHECK(topsMalloc(&out_mem, " << out_size << "));\n";
    os << "  void *device_outputs[] = {out_mem};\n";
  }
}

// phase 3: Execute the executable and fetch the output
static inline void print_host_phase3(std::ostream &os,
                                     const FactorCodeGen::EntryParamsInfo &params,
                                     const std::string & out_size,
                                     const std::string &out_type,
                                     size_t out_rank, const std::string &out_shape) {
  // TODO: resolve hardcode in input dims and ranks
  // os << "  int64_t input_dims[] = {";
  // if (params.size() > 0) {
  //   os << params[0].second;
  //   for (size_t i = 1; i < params.size(); ++i) os << ", " << params[i].second;
  // }
  // os << "};";
  os << R"(
  int64_t input_dims[] = {6, 17, 128, 6, 17, 128};
  size_t input_ranks[] = {3, 3};

  CHECK(topsLaunchExecutableV2(
      executable, nullptr, device_inputs,
      sizeof(device_inputs) / sizeof(void *), (int64_t*)input_dims,
      (size_t*)input_ranks, device_outputs,
      sizeof(device_outputs) / sizeof(void *), stream));
  CHECK(topsStreamSynchronize(stream));

)";

  if (!out_size.empty()) {
    os << "  auto res = choreo::make_spandata<" << out_type << ", " << out_rank << ">("
     << out_shape << ");\n";
    os << "  // Copy output data from device to host\n";
    os << "  CHECK(topsMemcpy(reinterpret_cast<void *>(res.data()), out_mem,\n";
    os << "                  " << out_size << ", topsMemcpyDeviceToHost));\n";
  }
}

// resource deallocation
static inline void print_host_phase4(std::ostream &os,
                                     std::vector<std::string> &d_params) {
  os << R"(
  // Free up the resources";
)";
  for (auto &p : d_params) os << "  topsFree(" << p << ");\n";
  os << R"(
  topsStreamDestroy(stream);
  topsDestroyExecutable(executable);
  return res;
}
)";
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

static inline std::string stub_type_str(const Choreo::Type &ty,
                                        bool is_ret = false) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<IntegerType>(&ty))
    return "int";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (auto sty = dyn_cast<SpannedType>(&ty)) {
    if (is_ret)  // return by value
      return "choreo::spanned<choreo::" + getStringFrom((BaseType)sty->f_type) +
             ", " + std::to_string(sty->Dims()) + ">";
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
      choreo_unreachable("Type '" + getStringFrom(t) + "' is not supported.");
  }
}

}  // end anonymous namespace

bool FactorCodeGen::BeforeVisitImpl(AST::Node &n) {
  if (isa<AST::Program>(&n)) {
    //    print_fixed_header(os);
  } else if (auto c = dyn_cast<AST::ChoreoFunction>(&n)) {
    sp_count = 0;  // reset the count of stub parameter
    entry_fn = c->name;
    current_fn = "__choreo_" + entry_fn;
    // declare a factor function with proper name
    fs <<
        R"(
#include <vector>

#include "gcu/factor/factor.h"

using namespace factor;
)";
    fs << "void " << current_fn << "() {\n";
    this->incrementIndent();
    fs << indent << "include_(\"" << backpatch_filename << "\");\n";
  } else if (isa<AST::ParallelBy>(&n)) {
    // this->incrementIndent();
  } else if (isa<AST::ForeachBlock>(&n)) {
    // this->incrementIndent();
  }
  return 0;
}

bool FactorCodeGen::AfterVisitImpl(AST::Node &n) {
  if (isa<AST::Program>(&n)) {
    os << "# step 4.1: generate the host source\n";
    os << "host_src=" << host_fn << "\n";
    os << "cat <<EOF > ${host_src}\n";
    os << hs.str() << "\nEOF\n\n";

    os << "# step 5: compile the host source to target executable\n";
    os << "target=" << target_fn << "\n";
    os << "# TODO: sfc ${host_src} -o ${target}\n";
    os << R"(
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
if [ "$1" == "--execute" ]; then
  ~/choreo/scripts/factor_compile_and_exec.sh ${factor_src} ${factor_bin} ${host_src} ${target}
elif [ "$1" == "--statistics" ]; then
  echo ">>>> Line of Code without Choreo"
  wc -l ${factor_src} ${host_src} ${kernel_src}
  echo ">>>> Line of Code with Choreo"
  wc -l ~/choreo/demo/elementwise_add.co
  # grep -v '^ *//' ~/choreo/demo/elementwise_add.co | wc -l
elif [ "$1" == "--show-kernel" ]; then
  nvim ${kernel_src}
elif [ "$1" == "--show-host" ]; then
  nvim ${host_src}
elif [ "$1" == "--show-tileflow" ]; then
  nvim ${factor_src}
elif [ "$1" == "--show-choreo" ]; then
  nvim ~/choreo/demo/elementwise_add.co
else
    echo "    Usage: $0 | --execute           -> compile and execute choreo in factor
                    | --statistics        -> show Line Of Code (LOC) statistic compare between kernel code boosted w./w.o. Choreo
                    | --show-kernel       -> show the generated inner kernel code
                    | --show-tileflow     -> show the generated tileflow code scheduled by choreo
                    | --show-host         -> show the generated host side boilerplates
                    | --show-choreo       -> show the choreo source code"
    exit 1
fi
    )";

  } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    entry_fn = f->name;
    current_fn = "__choreo_" + entry_fn;
    auto &out_type = cast<FunctionType>(cur_fty)->out_ty;
    auto out_size = GetByteSizeExprOf(*out_type);
    fs << "}\n\nMODULE_REGISTER(\"module" << current_fn << "\", " << current_fn
       << ");";  // end the factor function definition
    if (auto sty = dyn_cast<SpannedType>(out_type)) {
      OutputScript(f->name, GetBaseTypeStringOf(*out_type), out_size,
                   sty->GetShape());
    } else
      OutputScript(f->name, GetBaseTypeStringOf(*out_type), out_size,
                   Shape() /*invalid shape*/);
    ResetBuffers();
    cur_fty = nullptr;
  } else if (isa<AST::ParallelBy>(&n)) {
    this->decrementIndent();
    fs << this->indent << "}); // end of choreo-factor kernel function\n";
  } else if (isa<AST::ForeachBlock>(&n)) {
    this->decrementIndent();
    fs << this->indent << "}); // end of choreo-foreach block\n";
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
  // fs << AST::TYPE_STR(node);
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
      _os << data_shape.EmitTo(Target::Factor);
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

      fs << this->indent << "auto " << node.name_str << " = alloc_(";
      fs << storage_type << "(" << base_type << "," << shape_info << ")";
      fs << ");\n";
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

bool FactorCodeGen::Visit(AST::ParallelBy &by) {
  parallel_factor *= by.bound;
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
  fs << "}, {" << ((void_return) ? "" : "output") << "});\n";
  fs << this->indent << "destroy_stream_(stream);\n";
  fs << this->indent << "dealloc_stream_(stream);\n";
  fs << this->indent << "return std::vector<Value>{"
     << ((void_return) ? "" : "output") << "};\n";
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
  fs << this->indent << "auto thread_id = thread_id_();\n";
  // int i = 0;
  // NOTE: remove unused aliasing 'auto k_a = args[0];'
  // for (auto &param : *cur_params) {
  //   fs << "      "
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
  assert((dyn_cast<AST::ChunkAt>(d.from)) &&
         "Unexpected type for DMA's source.");
  assert((dyn_cast<AST::Memory>(d.to) || dyn_cast<AST::ChunkAt>(d.to)) &&
         "Unexpected type for DMA's destination.");

  auto future_name = d.future;
  auto to_node_name = (dyn_cast<AST::Memory>(d.to))
                          ? future_name + "_buffer"
                          : STR(cast<AST::ChunkAt>(d.to)->data);
  std::string from_node_name = STR(cast<AST::ChunkAt>(d.from)->data);
  std::string offset_string = "";
  std::string dma_op = "";

  auto getMemLevel = [](Storage s) -> int {
    switch (s) {
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
  auto storage_level =
      (dyn_cast<AST::Memory>(d.to))
          ? cast<AST::Memory>(d.to)->getStorageLevel()
          : dyn_cast<SpannedType>(
                this->GetSymbolType(STR(cast<AST::ChunkAt>(d.to)->data)))
                ->GetStorage();
  int des_level = getMemLevel(storage_level);
  storage_level =
      dyn_cast<SpannedType>(
          this->GetSymbolType(STR(cast<AST::ChunkAt>(d.from)->data)))
          ->GetStorage();
  int src_level = getMemLevel(storage_level);

  auto chunkat_node = (src_level >= des_level) ? dyn_cast<AST::ChunkAt>(d.from)
                                               : dyn_cast<AST::ChunkAt>(d.to);
  assert(chunkat_node && "Unexpected !!!");
  auto chunkat_node_name = STR(chunkat_node->data);
  auto data_type =
      dyn_cast<SpannedType>(this->GetSymbolType(chunkat_node_name))->f_type;

  if (auto mem_node = dyn_cast<AST::Memory>(d.to)) {
    // TODO(albert): generate 'local_buffer' with more smart naming way by valno
    // support
    fs << this->indent << "auto " << to_node_name << " = alloc_(";
    switch (mem_node->getStorageLevel()) {
      case Storage::LOCAL:
        fs << "L1Type(";
        break;
      case Storage::SHARED:
        fs << "SRAMType(";
        break;
      case Storage::GLOBAL:
        fs << "DRAMType(";
        break;
      default:
        assert(false && "Unexpected storage type.");
    }
    fs << factor_typestr((Choreo::BaseType)data_type) << ",";

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
    // assert(dim_sz == (int)dim.size() && "Insonsistant sizes for variable
    // span."); for (int dim_cursor = 0; dim_cursor < dim_sz;) {
    //   auto dim_bound = *(std::get_if<int>(&dim[dim_cursor]));
    //   assert( dim_bound > 0 && "Invalid variable span!");
    //   shape_info = shape_info + std::to_string(dim_bound);
    //   ++dim_cursor;
    //   if(dim_cursor < dim_sz)
    //     shape_info = shape_info + ",";
    //   else
    //     shape_info = shape_info + "}";
    // }
    fs << shape_info << "));\n";
  }

  if (src_level >= des_level)
    dma_op.append("async_load_(");
  else
    dma_op.append("async_store_(");

  auto tile_factors = chunkat_node->positions;
  auto data_shape =
      dyn_cast<SpannedType>(this->GetSymbolType(chunkat_node_name))->GetShape();
  int dim_sz = data_shape.Dims();
  offset_string.append("{");
  if (tile_factors) {
    assert(dim_sz == (int)tile_factors->getValues().size() &&
           "Insonsistant sizes for DMA offset.");
    auto dim = data_shape.values.values[0];
    assert(dim_sz == (int)dim.size() && "Insonsistant sizes for DMA offset.");
    for (int dim_cursor = 0; dim_cursor < dim_sz;) {
      auto tile_factor = tile_factors->getValues()[dim_cursor];
      auto tf_symbol = STR(tile_factor);
      auto tf_bounds =
          dyn_cast<BoundedITupleType>(this->GetSymbolType(tf_symbol))
              ->GetBounds()
              .Value();
#if 0
      auto tf_bound = *(std::get_if<int>(&tf_bounds[0]));
      auto dim_bound = *(std::get_if<int>(&dim[dim_cursor]));
      assert((tf_bound > 0 && dim_bound > 0) &&
             "Invalid Dim size or Tile factor!");
      // auto offset = (dim_cursor == 0)? std::to_string(dim_bound/tf_bound) +
      // "*thread_id" :
      //                                  std::to_string(dim_bound/tf_bound) +
      //                                  "*" + STR(tile_factor);
#endif
      auto offset = (dim_cursor == 0) ? "thread_id" : STR(tile_factor);
      offset_string = offset_string + offset;
      ++dim_cursor;
      if (dim_cursor < dim_sz) offset_string = offset_string + ",";
    }
  } else {
    for (int dim_cursor = 0; dim_cursor < dim_sz;) {
      offset_string = offset_string + "0";
      ++dim_cursor;
      if (dim_cursor < dim_sz) offset_string = offset_string + ",";
    }
  }
  offset_string = offset_string + "}";

  // strtab.Print(fs);
  int arg_idx = strtab.GetSymbolIndex(from_node_name);
  from_node_name =
      arg_idx < 0 ? from_node_name : "args[" + std::to_string(arg_idx) + "]";
  arg_idx = strtab.GetSymbolIndex(to_node_name);
  // TODO(albert): need a param table to resolve hardcode, connecting symbol
  // with results, and symbols with args
  to_node_name = arg_idx < 0 ? to_node_name
                             : "results[" + std::to_string(arg_idx - 2) + "]";

  fs << this->indent << "auto " << future_name
     << " = alloc_dma_(SDMAType());\n";
  fs << this->indent << dma_op << future_name << ", " << from_node_name << ", "
     << to_node_name << ", " << offset_string << ");\n";

  return true;
}

bool FactorCodeGen::Visit(AST::ChunkAt &) { return true; }

bool FactorCodeGen::Visit(AST::Wait &w) {
  auto dmas = dyn_cast<AST::MultiValues>(w.target);
  assert(dmas && "Invalid wait target!");

  for (auto dma : dmas->getValues()) {
    fs << this->indent << "wait_dma_(" << AST::STR(*dma) << ");\n";
  }

  return true;
}

bool FactorCodeGen::Visit(AST::Call &c) {
  fs << this->indent << "call_(\"";
  fs << STR(*c.function);
  fs << "\", {";
  auto args = dyn_cast<AST::MultiValues>(c.arguments);
  assert(args && "Invalid kernel call args!");
  int arg_num = args->getValues().size();
  for (int index = 0; index < arg_num;) {
    auto arg = dyn_cast<AST::Expr>(args->getValues()[index]);
    assert(arg && "Invalid kernel call arg!");
    switch (arg->t) {
      case AST::Expr::Reference:
        fs << STR(arg->value_r) << ".addr_()";
        break;
      case AST::Expr::Unary:
        if (arg->op == "sizeof") {
          auto var = STR(arg->value_r).substr(0, STR(arg->value_r).find('.'));
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
          fs << STR(arg->value_r) << "_buffer"
             << ".addr_()";
        }
        break;
      default:
        fs << STR(*arg);
        choreo_unreachable("unhandled expression type.");
        break;
    }
    index++;
    if (index < arg_num) fs << ",";
  }
  fs << "});\n";

  return true;
}

bool FactorCodeGen::Visit(AST::Return &) { return true; }

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
    //   if (auto intValue = std::get_if<int>(&value)) fs << *intValue;
    // }
    // NOTES: bounds always has one integer indicating the upperbound value
    // we can certainly use idx=0 directly
    auto upper_bound = *(std::get_if<int>(&iv_values[0]));

    // synthesise the emitting string
    fs << this->indent << "for_(0, " << std::to_string(upper_bound) << ", "
       << 1 /* TODO(albert): need fix, unit stride is hardcoded for now*/
       << ", "
       << "[&](auto " << iv_str << ") {\n";
  }
  this->incrementIndent();
  return true;
}

void FactorCodeGen::GenerateHostFunction(std::ostream &os, const Type &ty,
                                         const std::string &n, bool decl_only) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(&ty);
  os << stub_type_str(*fty.out_ty, true) << " " << n << "(";
  if (fty.in_tys.size() > 0) {
    auto n = GenEntryParamName();
    if (!decl_only) {
      if (auto sty = dyn_cast<SpannedType>(fty.in_tys[0]))
        entry_params.push_back(std::make_pair(n + ".data", sty->ByteSizeExpression()));
      else
        entry_params.push_back(std::make_pair(n, "1"));
    }
    os << stub_type_str(*fty.in_tys[0]) << " " << n;
    for (size_t i = 1; i < fty.in_tys.size(); ++i) {
      auto n = GenEntryParamName();
      if (!decl_only) {
        if (auto sty = dyn_cast<SpannedType>(fty.in_tys[i]))
          entry_params.push_back(std::make_pair(n + ".data", sty->ByteSizeExpression()));
        else
          entry_params.push_back(std::make_pair(n, "1"));
      }
      os << ", " << stub_type_str(*fty.in_tys[i]) << " " << n;
    }
  }
  os << ")" << ((decl_only) ? ";" : " {") << "\n";
  if (decl_only) ResetEntryParamCount();
}

bool FactorCodeGen::Visit(AST::FunctionDecl &d) {
  current_output = d.ret_type;

  assert(isa<FunctionType>(d.GetType()) && "expecting a function type.");
  cur_fty = d.GetType();

  for (auto &param : *cur_params) {
    auto name = param->sym->name;
    if (AST::typeof<SpannedType>(param.get())) {
      // define spanned type
      auto type_symbol = name + "_type";
      std::ostringstream _os;
      // param->type->Print(os, "");
      if (param->type->getPartialType()) {
        _os << param->type->getPartialType()->EmitTo("", Target::Factor);
      } else {
        // TODO: this guard code may not needed
        _os << "{?}";
      }
      auto type_string = "DRAMType(" +
                         factor_typestr(param->type->getBaseType()) + ", " +
                         _os.str();

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

  if (auto out_ty = dyn_cast<SpannedType>(cast<FunctionType>(cur_fty)->out_ty)) {
    auto name = "output";
    auto type_symbol = "output_type";
    std::ostringstream _os;
    // param->type->Print(os, "");
    if (const auto & pty = out_ty->GetMDSpanType()) {
      _os << pty->EmitTo(Target::Factor);
    } else {
      // TODO: this guard code may not needed
      _os << "{?}";
    }
    auto type_string = "DRAMType(" +
                       factor_typestr(out_ty->ElementType()) + ", " +
                       _os.str();

    strtab.AddSymbol(name, type_symbol, type_string);
    fs << this->indent << "auto " << strtab.GetTypeSymbol(name) << " = "
       << strtab.GetTypeString(name);
    fs << ");\n";
  } else if (AST::typeof<VoidType>(current_output)) {
    void_return = true;
  } else {
    auto name = "output";
    auto type_symbol = "output_type";
    auto type_string =
        "DRAMType(" + factor_typestr(current_output->getBaseType()) + ", (1));";
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

void FactorCodeGen::OutputScript(const std::string &n,
                                 const std::string &out_type, const std::string & out_size,
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
  print_host_head(hs);
  GenerateHostFunction(hs, *cur_fty, n, true);
  hs << user_code;
  GenerateHostFunction(hs, *cur_fty, n);
  print_host_phase1(hs, factor_bfn);
  std::vector<std::string> device_mems;
  print_host_phase2(hs, entry_params, out_size, device_mems);
  if (out_shape.IsValid()) {
    std::ostringstream oss;
    out_shape.PrintAsList(oss);
    print_host_phase3(hs, entry_params, out_size, out_type, out_shape.Dims(), oss.str());
  } else
    print_host_phase3(hs, entry_params, out_size, out_type, 1, "{1}");
  print_host_phase4(hs, device_mems);

  // backpatch the factor bin filename
  std::string factor_src = fs.str();
  ReplaceInString(factor_src, std::string(backpatch_filename), kernel_fn);

  // Now generate the script
  os <<
      R"(#!/usr/bin/env bash

# This the the choreo generated bash script to compile factor code

)";
  os << "# copy choreo.h to /tmp\n";
  os << "cat <<EOF > /tmp/choreo.h\n";
  os << __choreo_header_as_string;
  os << "# step 1: write the kernel source code into a temp file\n";
  os << "kernel_src=" << kernel_fn << "\n";
  os << "cat <<EOF > ${kernel_src}\n";
  os << ks.str() << "\nEOF\n\n";

  os << "# step 2: write the factor source code into a temp file\n";
  os << "factor_src=" << factor_fn << "\n";
  os << "cat <<EOF > ${factor_src}\n";
  os << factor_src << "\nEOF\n\n";

  os << "# step 3: compile factor code into a binary\n";
  os << "factor_bin=" << factor_bfn << "\n";
  os << "# TODO: sfc ${factor_src} -o ${factor_bin}\n\n";

  os << "# step 4: generate the host source\n";
  os << "host_src=" << host_fn << "\n";
  os << "cat <<EOF > ${host_src}\n";
  os << hs.str() << "\nEOF\n\n";

#if 0
  os << "# step 5: compile the host source to target executable\n";
  os << "target=" << target_fn << "\n";
  os << "# TODO: sfc ${host_src} -o ${target}\n";
  os << "~/choreo/scripts/factor_compile_and_exec.sh ${factor_src} ${factor_bin} ${host_src} ${target}\n";
#endif
}
