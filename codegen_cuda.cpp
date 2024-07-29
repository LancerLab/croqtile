#include "codegen_cuda.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
#include "choreo_cuda_header.inc"
#include "codegen.hpp"
#include "cuda_script.inc"
#include "types.hpp"

#ifndef __CHOREO_CUDA_DIR__
#error "missing macro definition of __CHOREO_CUDA_DIR__"
#endif

using namespace Choreo;
using namespace Choreo::CUDA;

#define __TRACE_EACH_VISIT__(d)       \
  if (trace_visit) {                  \
    os << d.TypeNameString() << ": "; \
    os << "\n";                       \
  }

bool CUDACodeGen::ContainsLoopVar(const std::string &iv) const {
  for (auto &loop_var : loop_vars)
    if (loop_var.count(iv)) return true;
  return false;
}

// TODO(albert) rename to EmitTo and extract to common file
std::string CUDACodeGen::EmitTo(Target target) {
  (void)target;
  // std::ostringstream _os;
  // if (!IsValidValueNumber(val_no))
  //   _os << "{}";
  // else {
  //   assert(values.Exists(val_no) && "bad value number.");
  //   CUDA::EmitCUDAValueList(Value(), _os);
  // }
  // return _os.str();
}

static StringifyTable cuda_symbols;


bool CUDACodeGen::BeforeVisitImpl(AST::Node &n) {
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
    // declare a cuda function with proper name
    fs << "void " << current_fn << "() {\n";
    this->incrementIndent();
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
bool CUDACodeGen::AfterVisitImpl(AST::Node &n) {
  __TRACE_EACH_VISIT__(n)
  if (isa<AST::Program>(&n)) {
    os << "\n# step 4: generate the host source\n";
    os << "host_src=" << host_fn << "\n";
    os << "cat <<'EOF' >> ${host_src}\n";
    os << hs.str()
       << "\nEOF\n\n";

    os << "\n# step 5: JIT compile and execute\n";
    os << "target=" << target_fn << "\n";
    os << "build_path=" << build_path << "\n";
    os << "cuda_script=" << build_path << "/cuda_script.sh\n";
    os << R"(
if command -v nvim &> /dev/null
then
  EDITOR=nvim
else
  EDITOR=less
fi
if [ "$#" -ne 1 ]; then
    echo "    Usage: $0 | --execute           -> compile and execute choreo in cuda
                    | --statistics        -> show Line Of Code (LOC) statistic compare between kernel code boosted w./w.o. Choreo
                    | --list-sources      -> show the tree view of all sources
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
  NVCC=nvcc
  CUDA_SYS_INCLUDES="-I/usr/local/cuda/include"
  CUDA_CHOREO_INCLUDES="-I./demos/cuda/sgemm_ref/"
  CUDA_INCLUDES="${CUDA_SYS_INCLUDES} ${CUDA_CHOREO_INCLUDES}"
  CUDA_CC="sm_35"
  CUDA_ARCH="compute_35"

# TODO, device id
  GPU_CC=$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader)

  case "${GPU_CC}" in
      3.0)
          CUDA_ARCH="compute_30"
          CUDA_CC="sm_30"
          ;;
      3.5)
          CUDA_ARCH="compute_35"
          CUDA_CC="sm_35"
          ;;
      3.7)
          CUDA_ARCH="compute_37"
          CUDA_CC="sm_37"
          ;;
      5.0)
          CUDA_ARCH="compute_50"
          CUDA_CC="sm_50"
          ;;
      5.2)
          CUDA_ARCH="compute_52"
          CUDA_CC="sm_52"
          ;;
      5.3)
          CUDA_ARCH="compute_53"
          CUDA_CC="sm_53"
          ;;
      6.0)
          CUDA_ARCH="compute_60"
          CUDA_CC="sm_60"
          ;;
      6.1)
          CUDA_ARCH="compute_61"
          CUDA_CC="sm_61"
          ;;
      6.2)
          CUDA_ARCH="compute_62"
          CUDA_CC="sm_62"
          ;;
      7.0)
          CUDA_ARCH="compute_70"
          CUDA_CC="sm_70"
          ;;
      7.2)
          CUDA_ARCH="compute_72"
          CUDA_CC="sm_72"
          ;;
      7.5)
          CUDA_ARCH="compute_75"
          CUDA_CC="sm_75"
          ;;
      8.0)
          CUDA_ARCH="compute_80"
          CUDA_CC="sm_80"
          ;;
      8.6)
          CUDA_ARCH="compute_86"
          CUDA_CC="sm_86"
          ;;
      8.9)
          CUDA_ARCH="compute_89"
          CUDA_CC="sm_89"
          ;;
      9.0)
          CUDA_ARCH="compute_90"
          CUDA_CC="sm_90"
          ;;
      *)
          echo "Unsupported GPU compute capability: ${GPU_CC}"
          exit 1
          ;;
  esac

  echo "CUDA_ARCH: ${CUDA_ARCH}"
  echo "CUDA_CC: ${CUDA_CC}"

)script";
    os << "  export cuda_INSTALL=" << STRINGIZE(__CHOREO_cuda_DIR__)
       << "\n# JIT compile and execute\n";
    if (dyn_shaped) os << "VIEW_CONFIG=1 ENABLE_DYNSHAPE=1 ";
    // os << "${cuda_script} ${build_path} ./demos/cuda/test_dir/sgemm_main.cu ${target}\n";
    os << "${cuda_script} ${build_path} ${host_src} ${target}\n";
    // os << "diff ${host_src} ./demos/cuda/test_dir/sgemm_main.cu";
    os << R"script(
elif [ "$1" == "--list-sources" ]; then
  tree ${build_path} -L 1
elif [ "$1" == "--statistics" ]; then
  echo ">>>> Line of Code without Choreo"
  wc -l ${cuda_src} ${host_src} ${kernel_src}
  echo ">>>> Line of Code with Choreo"
  wc -l ~/choreo/demo/elementwise_add.co
  # grep -v '^ *//' ~/choreo/demo/elementwise_add.co | wc -l
elif [ "$1" == "--show-kernel" ]; then
  ${EDITOR} ${kernel_src}
elif [ "$1" == "--show-host" ]; then
  ${EDITOR} ${host_src}
elif [ "$1" == "--show-tileflow" ]; then
  ${EDITOR} ${cuda_src}
elif [ "$1" == "--show-choreo" ]; then
  ${EDITOR} ~/choreo/demo/elementwise_add.co
else
    echo "    Usage: $0 | --execute           -> compile and execute choreo in cuda
                    | --statistics        -> show Line Of Code (LOC) statistic compare between kernel code boosted w./w.o. Choreo
                    | --list-sources      -> show the tree view of all sources
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
    fs << R"(
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  // blockIdx.y * BLOCKSIZE
  // threadIdx.x++ => x++ | y==
  // x to M

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
)";

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
      fs << this->indent << "}); // end of choreo-cuda kernel function\n";
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

bool CUDACodeGen::Visit(AST::MultiNodes &) { return true; }
bool CUDACodeGen::Visit(AST::MultiValues &) { return true; }
bool CUDACodeGen::Visit(AST::IntLiteral &) { return true; };
bool CUDACodeGen::Visit(AST::Boolean &) { return true; };
bool CUDACodeGen::Visit(AST::Expr &) { return true; };
bool CUDACodeGen::Visit(AST::MultiDimSpans &) { return true; };
bool CUDACodeGen::Visit(AST::NamedTypeDecl &) { return true; };

bool CUDACodeGen::Visit(AST::NamedVariableDecl &node) {
  __TRACE_EACH_VISIT__(node)
  auto nty = node.GetType();
  auto sym = node.name_str;
  if (auto sty = dyn_cast<SpannedType>(nty)) {
    assert(isa<SpannedType>(GetSymbolType(sym)) && "Inconsistent types!");
    if (cuda_symbols.Exists(sym)) {
      fs << indent << "auto " << sym << " = alloc_("
         << cuda_symbols.GetTypeName(sym) << ");\n";
    } else {
      auto storage_type = sty->GetStorage();
      auto base_type = Choreo::BaseType(sty->f_type);
      std::ostringstream _os;
      // if (storage_type == Choreo::Storage::SHARED) {
      _os << cuda_storage_str(storage_type) << " " << cuda_type_str(base_type) << " " << sym
        << ReplaceRuntimeNames(CUDASIZE(sty->GetShape()), "", false) << ";\n";
      fs << indent << _os.str();
      // }
      // if (storage_type == "DRAMType")
      //   fs << indent << _os.str();
      // else
      //   alloc_in_fs << "    " << _os.str();
    }
  } else {
    choreo_unreachable("non-spanned is not yet supported.");
    fs << this->indent;
    fs << "auto " << node.name_str << " = alloc_(?";
    fs << ");\n";
  }
  return true;
};
bool CUDACodeGen::Visit(AST::IntTuple &) { return true; };
bool CUDACodeGen::Visit(AST::Assignment &) { return true; };
bool CUDACodeGen::Visit(AST::IntIndex &) { return true; };
bool CUDACodeGen::Visit(AST::DataType &) { return true; };

bool CUDACodeGen::Visit(AST::Identifier &n) {
  __TRACE_EACH_VISIT__(n)
  (void)n;
  return true;
}

bool CUDACodeGen::Visit(AST::Parameter &n) {
  __TRACE_EACH_VISIT__(n)
  (void)n;
  return true;
}

bool CUDACodeGen::Visit(AST::ParamList &n) {
  __TRACE_EACH_VISIT__(n)
  cur_params = &n.values;
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::ParallelBy &by) {
  __TRACE_EACH_VISIT__(by)
  // parallel_cuda *= by.bound;
  // if (parallel_level > 1) {
  //   return true;
  // }
  // fs << this->indent << "Dim3 grid_dim(1);\n";
  // fs << this->indent << "Dim3 block_dim(" << by.bound << ");\n";
  // fs << this->indent << "auto ts = launch_kernel_(\"" << current_fn
  //    << "_parallel\", grid_dim, block_dim, args.back(), {";
  // if (cur_params->size() > 0) {
  //   fs << "args[0]";
  //   for (size_t i = 1; i < cur_params->size(); ++i) fs << ", args[" << i << "]";
  // }
  // fs << "}, {" << ((void_return) ? "" : "$$out$$")
  //    << "});\n";  // "$$out$$" : magic string for output, will be replaced later
  // fs << this->indent << "return std::vector<Value>{"
  //    << ((void_return) ? "" : "$$out$$") << "};\n";
  // this->decrementIndent();
  // fs << this->indent << "}, true); // end of choreo-cuda dataflow program\n";
  // fs << "\n";
  //
  // fs << this->indent << "\n";
  // fs << this->indent << "D(func_)(\"" << current_fn << "_parallel\", {";
  // if (cur_params->size() > 0) {
  //   fs << (*cur_params)[0]->sym->name << "_type";
  //   for (unsigned i = 1; i < cur_params->size(); ++i)
  //     fs << ", " << (*cur_params)[i]->sym->name << "_type";
  // }
  // fs << "}, {" << ((void_return) ? "" : "output_type")
  //    << "}, [&](auto args, auto results) {\n";
  // this->incrementIndent();
  // // generate a reference name of the output
  // if (!void_return) fs << indent << "auto & $$out$$ = results[0];\n";
  // fs << this->indent << "auto thread_id = thread_id_();\n";
  // fs << this->indent << "auto block_id = block_id_();\n";
  // alloc_pos = fs.str().size();
  // alloc_indent = indent;

  return true;
}

bool CUDACodeGen::Visit(AST::WhereBind &n) {
  __TRACE_EACH_VISIT__(n)
  // auto lid = cast<AST::Identifier>(n.lhs);
  // auto rid = cast<AST::Identifier>(n.rhs);
  // bind_info.AddBind(SSTab().ScopedName(lid->name),
  //                   SSTab().ScopedName(rid->name));
  //
  // // also adds the value binding for the with-matchers
  // if (cur_bounded_vars.count(lid->name)) {
  //   assert(cur_bounded_vars.count(rid->name));
  //   auto lbvs = cur_bounded_vars[lid->name];
  //   auto rbvs = cur_bounded_vars[rid->name];
  //   assert(lbvs.size() == rbvs.size());
  //
  //   for (size_t i = 0; i < lbvs.size(); ++i) {
  //     bind_info.AddBind(SSTab().ScopedName(lbvs[i]),
  //                       SSTab().ScopedName(rbvs[i]));
  //   }
  // }
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::WithIn &n) {
  __TRACE_EACH_VISIT__(n)
  // assert(n.with_matchers && "expect matcher to be exist.");
  //
  // // associate with to the matcher.
  // if (n.with && n.with_matchers) {
  //   std::vector<std::string> matchers;
  //   for (auto mn : n.with_matchers->AllValues()) {
  //     matchers.push_back(cast<AST::Identifier>(mn)->name);
  //   }
  //   cur_bounded_vars.emplace(n.with->name, matchers);
  // }
  //
  // for (auto mn : n.with_matchers->AllValues()) {
  //   auto mname = cast<AST::Identifier>(mn)->name;
  //   fs << indent << "var_ " << mname << "(IntType(32));\n";
  //   fs << indent << mname << " = 0;\n";
  // }
  return true;
};

bool CUDACodeGen::Visit(AST::WithBlock &) { return true; }

bool CUDACodeGen::Visit(AST::Memory &n) {
  __TRACE_EACH_VISIT__(n)
  (void)n;
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::DMA &d) {
  __TRACE_EACH_VISIT__(d)
  // handle .to  in AST::Memory
  // assert((isa<AST::ChunkAt>(d.from)) && "Unexpected type for DMA's source.");
  // assert((isa<AST::Memory>(d.to) || isa<AST::ChunkAt>(d.to) || isa<AST::Select>(d.to)) &&
  //        "Unexpected type for DMA's destination.");
  //
  // // retrieve the spanned type from a chunkat
  // auto GetSpannedType = [this](AST::Node &ca) -> SpannedType * {
  //   auto sty = ca.GetType();
  //   if (auto fty = dyn_cast<FutureType>(sty))
  //     return fty->GetSpannedType().get();
  //   else
  //     return cast<SpannedType>(sty);
  // };
  //
  // auto MemLevel = [](Storage s) -> int {
  //   switch (s) {
  //     case Storage::LOCAL:
  //       return 0;
  //     case Storage::SHARED:
  //       return 1;
  //     case Storage::GLOBAL:
  //     case Storage::DEFAULT:
  //       return 2;
  //     default:
  //       choreo_unreachable("Unexpected storage type.");
  //       return -1;
  //   }
  // };
  //
  // auto ty = dyn_cast<FutureType>(d.GetType());
  // assert(ty && "Invalid type of DMA statement!");
  //
  // // cook a valid future name
  // auto future_name = d.future;
  // if (future_name.empty()) {
  //   static size_t future_count = 0;
  //   future_name = "__choreo_anon_fut__" + std::to_string(future_count++);
  // }
  //
  // // cook a valid dst buffer name
  // // note: for select, we also use the future_name + "_buffer" as handle name.
  // auto dst_buffer_name = (isa<AST::ChunkAt>(d.to))
  //                            ? STR(cast<AST::ChunkAt>(d.to)->data)
  //                            : future_name + "_buffer";
  // // if to node is AST::SELECT, use its future name, otherwise keep default one
  // // dst_buffer_name = (isa<AST::Select>(d.to))
  // //                            ? STR(cast<AST::Select>(d.to)->future)
  // //                            : future_name + "_buffer";
  //
  // std::string src_node_name = STR(cast<AST::ChunkAt>(d.from)->data);
  // assert(!src_node_name.empty() && "expect a named future/span in chunkat.");
  // // use source symbol as the buffer name
  // // lhs_load => lhs_load_buffer used by user of lhs_load
  // std::string src_buffer_name = src_node_name;
  // if (isa<FutureType>(GetSymbolType(
  //         RemoveSuffix(cast<AST::ChunkAt>(d.from)->data->name, ".data"))))
  //   src_buffer_name = src_node_name + "_buffer";
  //
  // auto sty = GetSpannedType(*d.from);  // source spanned type
  // size_t rank = sty->Dims();
  // auto dst_shape = ty->GetShape();
  // auto src_sto = sty->GetStorage();
  // auto dst_sto = Storage::DEFAULT;
  // if (isa<AST::Memory>(d.to))
  //   dst_sto = cast<AST::Memory>(d.to)->Get();
  // else if (isa<AST::Select>(d.to))
  //   // TODO(albert): get mem level from selects operands
  //   dst_sto = Storage::LOCAL;
  // else
  //   dst_sto = GetSpannedType(*d.to)->GetStorage();
  // // auto dst_sto = (isa<AST::Memory>(d.to)) ? cast<AST::Memory>(d.to)->Get()
  // //                                         : GetSpannedType(*d.to)->GetStorage();
  // // dst_sto = (isa<AST::Select>(d.to)) ? Storage::LOCAL
  // //                                         : GetSpannedType(*d.to)->GetStorage();
  // int src_level = MemLevel(src_sto);
  // int dst_level = MemLevel(dst_sto);
  //
  // // allocate storage for DMA destination when it is not explicitly stated.
  // if (auto mem_node = dyn_cast<AST::Memory>(d.to)) {
  //   // support
  //   static std::map<Storage, std::string> sto2alloc = {
  //       {Storage::LOCAL, "L1Type"},
  //       {Storage::SHARED, "SRAMType"},
  //       {Storage::GLOBAL, "DRAMType"},
  //   };
  //   // buffer in another stream
  //   alloc_in_fs << alloc_indent << "auto " << dst_buffer_name << " = alloc_("
  //               << sto2alloc.at(mem_node->Get()) << "("
  //               << cuda_type_str(sty->ElementType()) << ","
  //               << ReplaceRuntimeNames(LSTR(dst_shape), "", false) << "));\n";
  // }
  //
  // auto GenerateOffsetString = [this, &GetSpannedType](AST::Node &n) {
  //   auto sty = GetSpannedType(n);
  //   auto shape = sty->GetShape();
  //   size_t rank = sty->Dims();
  //
  //   auto ca = cast<AST::ChunkAt>(&n);
  //   if (!ca->positions) {
  //     // symbol only, the offset is a multi-dim-zeros
  //     return "{" + DelimitedString(std::vector<size_t>(rank, 0)) + "}";
  //   }
  //
  //   std::ostringstream offss;
  //   size_t dim_cursor = 0;
  //   for (auto &bv : ca->positions->AllValues()) {
  //     auto bvn = cast<AST::Identifier>(bv)->name;
  //     if (auto bity = dyn_cast<BoundedITupleType>(bv->GetType())) {
  //       for (size_t it_idx = 0; it_idx < bity->Dims(); ++it_idx) {
  //         std::string iv_str;
  //         if (within_map.count(bvn))  // with-matcher existed
  //           iv_str = within_map[bvn][it_idx];
  //         else
  //           iv_str = bvn;
  //
  //         // prefix iteration variable
  //         if (ContainsLoopVar(iv_str)) iv_str = "iv_" + iv_str;
  //
  //         // special handling for the parallel tiling cuda
  //         auto l = RemovePrefixOrNull("pv:", bity->GetNote());
  //         if (l.has_value()) {
  //           // is marked as parallel whose level is decided by target check
  //           if (*l == "0")
  //             iv_str = "thread_id";
  //           else if (*l == "1")
  //             iv_str = "block_id";
  //           else
  //             choreo_unreachable("invalid type note.");
  //         }
  //
  //         offss << "Value(" << RSTR(shape.ValueAt(dim_cursor)) << ")*"
  //               << iv_str;
  //         if (++dim_cursor < rank) offss << ",";
  //       }
  //     } else
  //       choreo_unreachable("unsupported type.");
  //   }
  //   return "{" + offss.str() + "}";
  // };
  //
  // // cuda_symbols.Print(fs);
  // int arg_idx = cuda_symbols.GetSymbolIndex(src_node_name);
  // src_buffer_name =
  //     arg_idx < 0 ? src_buffer_name : "args[" + std::to_string(arg_idx) + "]";
  //
  // // decide the dma allocation type
  // auto DMATypeString = [](int src_lvl, int dst_lvl) {
  //   if ((src_lvl == 2 && dst_lvl == 2) || (src_lvl == 2 && dst_lvl == 1) ||
  //       (src_lvl == 1 && dst_lvl == 2) || (src_lvl == 1 && dst_lvl == 1))
  //     return "CDMAType";
  //   else
  //     return "SDMAType";
  // };
  //
  // // buffer the allocation in another stream
  // // if use pipeline-mode, make all cdma with shared_ annotation
  // if (d.chained == true && 
  //     ((d.chain_to != "" && src_level > dst_level) ||
  //     (d.chain_from != "" && src_level < dst_level)))
  //   alloc_in_fs << alloc_indent << "auto " << future_name << " = alloc_dma_("
  //             << DMATypeString(src_level, dst_level) << "()).shared_();\n";
  // else
  //   alloc_in_fs << alloc_indent << "auto " << future_name << " = alloc_dma_("
  //             << DMATypeString(src_level, dst_level) << "());\n";
  //
  // // decide the dma operation
  // std::string dma_op = "";
  // if (src_level >= dst_level)
  //   dma_op.append("async_load_");
  // else
  //   dma_op.append("async_store_");
  //
  // ptr<AST::Node> chunkat_node = nullptr;
  //
  // if (isa<AST::Memory>(d.to) || isa<AST::Select>(d.to))
  //   chunkat_node = d.from;
  // else if (cast<AST::ChunkAt>(d.to)->positions)
  //     chunkat_node = d.to;
  // else
  //   choreo_unreachable("cuda: unsupported chunkat.");
  //
  // fs << indent << dma_op << "(" << future_name << ", " << src_buffer_name
  //    << ", " << dst_buffer_name << ", " << GenerateOffsetString(*chunkat_node);
  //
  // if (auto pcfg = dyn_cast<PadConfig>(d.config)) {
  //   std::vector<size_t> layout(rank);
  //   std::iota(layout.begin(), layout.end(), 0);  // no transpose
  //   fs << ", {" << DelimitedString(layout) << "}, {"
  //      << DelimitedString(pcfg->pad_low) << "}, {"
  //      << DelimitedString(pcfg->pad_high) << "}, {"
  //      << DelimitedString(pcfg->pad_mid) << "}, " << pcfg->value.v;
  // }
  //
  // if (d.chained == false) {
  //   fs << ");\n";
  //   // synchornized dma must be waited
  //   if (!ty->IsAsync()) fs << indent << "wait_dma_(" << future_name << ");\n";
  // } else {
  //   assert(ty->IsAsync() && "Notifying DMA only apply to async primitives in cuda lang.");
  //   if (d.chain_from != "") 
  //     if (src_level >= dst_level)
  //       fs << ").wait_on_(" << d.chain_from << ");\n";
  //     else
  //       fs << ").multi_wait_on_(" << d.chain_from << ");\n";
  //
  //   if (d.chain_to != "")
  //     if (src_level >= dst_level)
  //       fs << ").multi_notify_(" << d.chain_to << ");\n";
  //     else 
  //       fs << ").notify_(" << d.chain_to << ");\n";
  // }

  return true;
}

bool CUDACodeGen::Visit(AST::ChunkAt &) { return true; }

bool CUDACodeGen::Visit(AST::Wait &w) {
  __TRACE_EACH_VISIT__(w)
  // auto dmas = w.targets;
  // assert(dmas && "Invalid wait target!");
  //
  // for (auto dma : dmas->AllValues()) {
  //   fs << this->indent << "wait_dma_(" << AST::STR(*dma) << ");\n";
  // }

  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::Call &c) {
  __TRACE_EACH_VISIT__(c)
//   fs << this->indent << "call_(\"";
//   fs << STR(*c.function);
//   fs << "\", {";
//   auto args = c.arguments;
//   assert(args && "Invalid kernel call args!");
//   int arg_num = args->AllValues().size();
//   for (int index = 0; index < arg_num;) {
//     auto arg = dyn_cast<AST::Expr>(args->AllValues()[index]);
//     assert(arg && "Invalid kernel call arg!");
//     switch (arg->GetForm()) {
//       case AST::Expr::Reference:
//         try {
//           std::stoi(STR(arg->GetR()));
//           fs << STR(arg->GetR());
//         } catch (const std::invalid_argument &e) {
//           fs << STR(arg->GetR()) << ".addr_()";
//         }
//         break;
//       case AST::Expr::Unary:
//         if (arg->op == "sizeof") {
//           auto var = STR(arg->GetR()).substr(0, STR(arg->GetR()).find('.'));
//           assert(dyn_cast<FutureType>(this->GetSymbolType(var)) &&
//                  "Unexpected !!!");
//           auto ty_ptr = cast<FutureType>(this->GetSymbolType(var));
//           auto shape = ty_ptr->GetShape();
// #if 0
//           auto shapes = shape.Value();
//           auto dim = shape.values.values[0];
//           int dim_sz = shape.Dims(), size = 1;
//           for (int dim_cursor = 0; dim_cursor < dim_sz;)
//             size = size * (*(std::get_if<int>(&shapes[dim_cursor++])));
//           fs << std::to_string(size);
// #endif
//           fs << shape.GetSizeExpression();
//         } else if (arg->op == "dataof") {
//           fs << STR(arg->GetR()) << "_buffer"
//              << ".addr_()";
//         }
//         break;
//       default:
//         choreo_unreachable("unhandled expression type: " +
//                            std::to_string((int)(arg->GetForm())) + ".");
//         break;
//     }
//     index++;
//     if (index < arg_num) fs << ",";
//   }
//   fs << "});\n";

  return true;
}

bool CUDACodeGen::Visit(AST::Select &c) {
  __TRACE_EACH_VISIT__(c)
  // size_t val_count = c.val_list->Count();
  // // if val_count == 1, pingpong is meaningless? ( TODO: maybe assert when earlysema)
  // assert(val_count >= 2);
  // fs << this->indent << "auto " << c.future << " = ";
  // for (size_t i = 0; i < val_count - 1; i++) {
  //   fs << "select_(" << STR(c.select_cuda) << "== " << i << ", " << STR(c.val_list->ValueAt(i)) << (i < val_count-1 ? ", " : "");
  // }
  // fs << STR(c.val_list->AllValues().back()) << std::string(val_count-1, ')') << ";\n";
  return true;
}

bool CUDACodeGen::Visit(AST::Return &returnNode) {
  __TRACE_EACH_VISIT__(returnNode)
  // if (returnNode.value) output_v = STR(*returnNode.value);
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::ForeachBlock &forNode) {
  __TRACE_EACH_VISIT__(forNode)
  // auto itervars = forNode.getIterationVars();
  // for (size_t idx = 0; idx != itervars->Count(); ++idx) {
  //   // TODO(albert): support non-unit stride in loop
  //   std::ostringstream _os;
  //   auto id = cast<AST::Identifier>(itervars->ValueAt(idx));
  //
  //   // get the lower/upper and stride for spanned iter var
  //   auto iv_type = this->GetSymbolType(id->name);
  //   auto iv_bounds = cast<BoundedITupleType>(iv_type)->GetBounds();
  //
  //   // NOTES: foreach block ranges between [0, UB),
  //   // it always use one integer indicating the UB
  //   // we can certainly use idx=0 directly
  //
  //   // synthesise the emitting string
  //   if (iv_type->Dims() == 1) {
  //     fs << this->indent << "for_(" << id->name << ", "
  //        << ReplaceDynDimName(STR(iv_bounds.ValueAt(0))) << ", "
  //        << 1 /* TODO(albert): need fix, unit stride is hardcoded for now*/
  //        << ", [&](auto iv_" << id->name << ") {\n";
  //     incrementIndent();
  //     loop_vars.back().insert(id->name);
  //     for (auto bind : bind_info.GetBinds(InScopeName(id->name))) {
  //       auto bname = SSTab().UnScopedName(bind);
  //       loop_vars.back().insert(bname);
  //       fs << indent << "auto iv_" << SSTab().UnScopedName(bind) << " = iv_"
  //          << id->name << ";\n";
  //     }
  //   } else {
  //     assert(cur_bounded_vars.count(id->name) &&
  //            "can not find the bounded name.");
  //     assert((cur_bounded_vars[id->name].size() == iv_bounds.Dims()) &&
  //            "can not find the bounded name.");
  //     size_t i = 0;
  //     for (auto name : cur_bounded_vars[id->name]) {
  //       fs << this->indent << "for_(" << name << ", "
  //          << ReplaceDynDimName(STR(iv_bounds.ValueAt(i))) << ", "
  //          << 1 /* TODO(albert): need fix, unit stride is hardcoded for now*/
  //          << ", [&](auto iv_" << name << ") {\n";
  //       std::string scoped_var = InScopeName(name);
  //       loop_vars.back().insert(name);
  //       incrementIndent();
  //       for (auto bind : bind_info.GetBinds(InScopeName(name))) {
  //         auto bname = SSTab().UnScopedName(bind);
  //         loop_vars.back().insert(bname);
  //         if (bname != name)
  //           fs << indent << "auto iv_" << bname << " = iv_" << name << ";\n";
  //       }
  //       ++i;
  //     }
  //   }
  // }
  return true;
}

// CLEAN
bool CUDACodeGen::Visit(AST::FunctionDecl &d) {
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
      type_string = "DRAMType(" + cuda_type_str(sty->ElementType()) + ", " +
                    ReplaceRuntimeNames(LSTR(sty->GetShape()), "", false) + ")";
      cuda_symbols.AddSymbol(pname, type_name, type_string);
    } else {
      type_string =
          "DRAMType(" + cuda_type_str(param->type->getBaseType()) + ", (1))";
      cuda_symbols.AddSymbol(pname, type_name, type_string);
    }
    // fs << indent << "auto " << type_name << " = " << type_string << ";\n";
  }

  if (auto rty = dyn_cast<SpannedType>(fty.out_ty)) {
    std::string name = "output";
    std::string type_name = "output_type";
    auto type_string = "DRAMType(" + cuda_type_str(rty->ElementType()) + ", " +
                       ReplaceRuntimeNames(LSTR(rty->GetShape()), "", false);

    // handle dynamic-typed output when necessary. Generate code snippet like:
    //
    //   auto output_rt_dim0 = dim_(args[0], 1);
    //   auto output = alloc_({output_rt_dim0}, output_type);
    //
    const auto &dyn_dims = rty->GetShape().GetDynamicDims();
    if (!dyn_dims.empty()) {
      dyn_shaped = true;
      type_name = "{";
      size_t i = 0;
      for (auto &ddim : dyn_dims) {
        auto ddim_name = name + "_rt_dim" + std::to_string(i);
        dss << "auto " << ddim_name << " = " << ReplaceDynDimName(ddim.second)
            << ";\n";
        type_name += ddim_name;
        if (++i != dyn_dims.size()) type_name += ", ";
      }
      type_name += "}, output_type";
    }

    // fs << indent << "auto output_type = " << type_string << ");\n";
    cuda_symbols.AddSymbol(name, type_name, type_string);

  } else if (isa<VoidType>(fty.out_ty)) {
    void_return = true;
  } else {
    auto name = "output";
    auto type_name = "output_type";
    auto type_string =
        "DRAMType(" + cuda_type_str(TC2BT(fty.out_ty->Category())) + ", (1));";
    cuda_symbols.AddSymbol(name, type_name, type_string);
    // fs << indent << "auto " << type_name << " = " << type_string << "\n";
  }

  // fs << "\n";
  // fs << this->indent << "// choreo-cuda dataflow function\n";
  // fs << this->indent << "D(host_func_)(\"" << current_fn << "\", {";

  // for (auto &param : *cur_params) fs << param->sym->name + "_type, ";

  // fs << "StreamType()}, [&](auto args) {\n";

  this->incrementIndent();
  // fs << indent << dss.str();  // dynamic-shape specific
  return true;
}

bool CUDACodeGen::Visit(AST::ChoreoFunction &) { return true; }

bool CUDACodeGen::Visit(AST::CppSourceCode &n) {
  __TRACE_EACH_VISIT__(n)
  if (n.host) {
    hs << n.GetCode();
  } else {
    ks << n.GetCode();
  }
  return true;
}

bool CUDACodeGen::Visit(AST::Program &) { return true; }

void CUDACodeGen::EmitHostHead(std::ostream &os) {
  os <<
      R"(// choreo header
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <iterator>
#include <string>
#include <chrono>


using namespace choreo;
using namespace choreo::cuda;

// UPDATE: from kernel-6, it only occupies half of the L1 mem, which is not optimised enough for reuse.
// for kernel-7, let us try a <256,256,256> setting, where occupies 3/4 of L1 MEM

// Analysis 1:
//
// HW: in DORADO: 1 card = 2 clusters = 6 csb = 6 L2 = 24 SIP = 24 L1, each CSB = 8MB, each L1 = 1 MB
//
// GMEM: unchanged from kernel 5 
// SMEM: unchanged from kernel 5 
// LMEM: lhs_load_s=<64x1024>=256KB, rhs_load=<1024x64>=256KB, l2_out=<64x64>=4KB  < 1MB

// Analysis: compute intensity
// kernel 6 calculate <32x32> results per thread requires:
//   32x1024 loads from lhs
//   32x1024 loads from rhs
//   32x32 loads and stores from out
//   => 65 loads + 1 store per result
//
// kernel 7 calculate <256x256> results per thread requires:
//   256x256x4 loads from lhs
//   256x256x4 loads from rhs
//   256x256x4 loads and stores from out
//   => 12 loads and 4 store per result



namespace {

// Nasty data copy. Need optimization together with cuda
template <typename T, int Rank>
static inline std::vector<uint8_t>
ToCUDAData(const spanned_view<T, Rank> &v) {
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

void CUDACodeGen::EmitHostTail(std::ostream &os) {
  // compare results
  os << R"(
  unsigned long flops = 2 * hp0.shape()[0] * hp1.shape()[1] * hp0.shape()[1];
  printf(
      "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
      "(%lu).\n",
      elapsed_time,
      (flops * 1e-9) / elapsed_time, hp0.shape()[0]);
  fflush(stdout);

  if (!verify_matrix(res_cublas.data(), res_choreo.data(), res_choreo.shape()[0] * res_choreo.shape()[1])) {
    std::cout
        << "Failed to pass the correctness verification against NVIDIA "
           "cuBLAS."
        << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Compute Correct with Cublas";
  )";
  os << "  return res_choreo;\n";
  os << "}\n";
}

void CUDACodeGen::EmitHostFuncBody(std::ostream &os, const Type &ty,
                                     const std::string &f_n,
                                     const std::string &out_size,
                                     const std::string &out_type,
                                     const Shape &out_shape) {
  assert(isa<FunctionType>(&ty) && "unexpected type.");
  auto &fty = *cast<FunctionType>(&ty);

  // TODO
  // 1. make alpha and beta into arguments
  // 3. 
  os << "{\n";
  EmitRuntimeCheck(os, ty);
  os << R"(
  int deviceIdx = 0;
  printf("Running on device %d.\n", deviceIdx);

  cublasHandle_t cublas_handle;
  if (cublasCreate(&cublas_handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };

  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  float alpha = 1.0;
  float beta = 0.0;
)";

  // phase 2: allocate device memory and copy
  std::vector<std::string> device_mems;
  for (auto &p : param_map) {
    auto mem_name = "in_mem" + std::to_string(device_mems.size());
    os << "  float *" << mem_name << " = nullptr;\n";
    os << "  CHECK(cudaMalloc(&" << mem_name << ", " << p.second << "));\n";
    os << "  CHECK(cudaMemcpy(" << mem_name << ", "
       << p.first << ".data(), " << p.second
       << ", cudaMemcpyHostToDevice));\n";
    device_mems.push_back(mem_name);
  }
  os << "  float * device_inputs[] = {" << DelimitedString(device_mems)
     << "};\n\n";

  std::string size_string = ReplaceRuntimeNames(out_size);

  if (!out_size.empty()) {
    os << "  float * out_mem_choreo = nullptr;\n";
    os << "  CHECK(cudaMalloc(&out_mem_choreo, " << size_string << "));\n";
    os << "  float *device_outputs[] = {out_mem_choreo};\n";
    os << "  // result for cublas ref\n";
    os << "  float * out_mem_cublas = nullptr;\n";
    os << "  CHECK(cudaMalloc(&out_mem_cublas, " << size_string << "));\n";
  }

  std::vector<std::string> inputs;  // cuda input paramters

  // go ref impl with cublas
  os << "\n  run_kernel(0, hp0.shape()[0], hp1.shape()[1], hp0.shape()[1], alpha, " << DelimitedString(device_mems) << ", beta, out_mem_cublas, cublas_handle);\n";
  os << "  CUDACheck(cudaDeviceSynchronize());\n";
  size_t out_rank = 1;
  std::string shape_string = "{1}";
  if (out_shape.IsValid()) {
    out_rank = out_shape.Dims();
    shape_string = ReplaceRuntimeNames(LSTR(out_shape));
  }
  if (!out_size.empty()) {
    os << "  auto res_cublas = choreo::make_spandata<" << out_type << ", " << out_rank
       << ">(" << shape_string << ");\n";
    os << "  // Copy output data from device to host\n";
    os << "  CUDACheck(cudaMemcpy(res_cublas.data(), out_mem_cublas,\n";
    os << "                  " << size_string
       << ", cudaMemcpyDeviceToHost));\n";
  }

  // go our impl
  // TODO(albert) cleanup HC, make size and alpha/beta into interface arguments
  os << "\n  cudaEventRecord(beg);\n";
  os << "\n  run_kernel(1, hp0.shape()[0], hp1.shape()[1], hp0.shape()[1], alpha, " << DelimitedString(device_mems) << ", beta, out_mem_choreo, cublas_handle);\n";
  os << "  CUDACheck(cudaDeviceSynchronize());\n";
  os << R"(
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  elapsed_time /= 1000.; // Convert to seconds
  )";
  if (!out_size.empty()) {
    os << "auto res_choreo = choreo::make_spandata<" << out_type << ", " << out_rank
       << ">(" << shape_string << ");\n";
    os << "  // Copy output data from device to host\n";
    os << "  CUDACheck(cudaMemcpy(res_choreo.data(), out_mem_choreo,\n";
    os << "                  " << size_string
       << ", cudaMemcpyDeviceToHost));\n";
  }


  // phase 4: Free up the resources
  os << "  // Free up the resources\n";
  os << "  cublasDestroy(cublas_handle);\n";
  for (auto &p : device_mems) os << "  cudaFree(" << p << ");\n";
  os << "  cudaFree(out_mem_choreo);\n";
  os << "  cudaFree(out_mem_cublas);\n\n";
}

std::string CUDACodeGen::ReplaceRuntimeNames(const std::string &e,
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

std::string CUDACodeGen::ReplaceDynDimName(const std::string &e) {
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

void CUDACodeGen::EmitRuntimeCheck(std::ostream &os, const Type &ty) {
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

void CUDACodeGen::EmitHostFuncDecl(std::ostream &os, const Type &ty,
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

void CUDACodeGen::OutputScript(FunctionType *fty, const std::string &n,
                                 const std::string &out_type,
                                 const std::string &out_size,
                                 const Shape &out_shape) {
  // a temporal path for the compilation process
  build_path = create_unique_path();
  std::string build_prefix = build_path + "/__choreo_" + n;
  std::string build_prefix_anonymous = build_path + "/__choreo";

  std::string kernel_fn = build_prefix + "_kernel_inlined.cuh";
  std::string cuda_fn = build_prefix_anonymous + "_cuda_kernel.cuh";
  std::string cuda_bfn =
      build_path + "/${gcu_target_string}_lib" + current_fn + ".o";
  host_fn = build_prefix + "_host.cu";
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
  EmitHostFuncBody(hs, *fty, cuda_bfn, out_size, out_type, out_shape);
  EmitHostTail(hs);
  //
  // // backpatch the cuda bin filename
  std::string cuda_src = fs.str();
  if (!alloc_in_fs.str().empty())
    cuda_src.insert(alloc_pos, alloc_in_fs.str());
  ReplaceInString(cuda_src, std::string("$$out$$"), output_v);
  ReplaceInString(cuda_src, std::string(backpatch_filename), kernel_fn);

  // Now generate the script
  os << "#!/usr/bin/env bash\n\n";
  os << "# This is the choreo generated bash script to compile cuda code\n";
  os << "\n# step 0: set up the environment\n";
  os << "rm -fr " << build_path << "\n";
  os << "mkdir -p " << build_path << "\n";

  // no need to gen run shell, since cuda compile is simple enough to handle
  os << "cat <<'EOF' > " << build_path << "/cuda_script.sh\n";
  os << __cuda_script_as_string << "\nEOF\n";
  os << "chmod +x " << build_path << "/cuda_script.sh\n";

  os << "cat <<'EOF' > " << build_path << "/choreo_cuda.h\n";
  os << __choreo_header_as_string << "\nEOF\n\n";

  // TODO(albert): support INLINED ASM FOR CUDA
  os << "\n# step 1: write the kernel source code into a temp file\n";
  os << "kernel_src=" << kernel_fn << "\n";
  os << "cat <<'EOF' > ${kernel_src}\n";
  os << ks.str() << "\nEOF\n";

  os << "\n# step 2: write the cuda source code into a temp file\n";
  os << "cuda_src=" << cuda_fn << "\n";
  os << "cat <<'EOF' > ${cuda_src}\n";
  os << cuda_src << "\nEOF\n\n";

  os << "\n# step 3: set the cuda binary file name\n";
  os << "cuda_bin=" << cuda_bfn << "\n";
}
