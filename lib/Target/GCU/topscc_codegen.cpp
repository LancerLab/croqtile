#include "topscc_codegen.hpp"
#include "codegen_utils.hpp"
#include "topscc_device_codegen.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <system_error>
#include <vector>

#include "ast.hpp"
#include "choreo_header.inc"
#include "choreo_types_header.inc"
#include "codegen.hpp"
#include "io.hpp"
#include "lower_libcall.hpp"
#include "operator_info.hpp"
#include "target.hpp"
#include "target_utils.hpp"
#include "topscc_header.inc"
#include "types.hpp"

#ifdef __CHOREO_GCU_ACORE_DIR__
  #include "acore_runtime.inc"
  #define __CHOREO_ACORE_RUNTIME_AVAILABLE__
#endif

#ifndef __CHOREO_TOPSCC_DIR__
  #error "missing macro definition of __CHOREO_TOPSCC_DIR__"
#endif

Option<bool> use_system_toolchain(OptionKind::Hidden, "--use-system-toolchain",
                                  "-st",
#ifdef __CHOREO_INSTALLATION_PACKAGE__
                                  true,
#else
                                  false,
#endif
                                  "(Experimental) Use system installed "
                                  "toolchain: topscc, topsrt, etc for choreo.");
using namespace Choreo;
using namespace Choreo::Topscc;

extern Option<bool> use_system_toolchain;

Option<bool> emit_fatbin(OptionKind::Hidden, "-fb", "", false,
                         "Emit fatbin file.");
Option<bool> split_8byte_dma_transfer(
    OptionKind::Hidden, "-fsplit-8b-dma", "", true,
    "8-byte DMA transfers will be split into 1-byte chunks when platforms that "
    "do not support native 8-byte DMA.");
Option<bool> raw_dte_mode(
    OptionKind::User, "-fraw-dte", "", false,
    "Use raw DTE pool references instead of choreo::future for anonymous sync "
    "DMA on GCU300. Reduces per-DMA overhead by eliminating future "
    "construct/destruct.");
Option<bool> use_dte_pool(
    OptionKind::User, "--use-dte-pool", "", true,
    "Reuse a persistent DTE pool for anonymous and named DMAs instead of "
    "creating per-DMA DTE instances. Prevents DTE resource exhaustion "
    "(SIP asserts) when total init/destroy cycles exceed hardware limits.");
Option<bool> no_future_mode(
    OptionKind::User, "-fno-future", "", false,
    "Eliminate ALL choreo::future objects on GCU300. Named async futures are "
    "replaced with raw DTE pool refs + tops::event tracking. Rotate/swap "
    "become raw pointer swaps. Subsumes -fraw-dte.");
Option<bool> named_dte_mode(
    OptionKind::User, "-fnamed-dte", "", false,
    "Generate named tops::private_dte variables instead of an array pool. "
    "Enables better register allocation by eliminating indirect addressing.");
Option<bool> dte_merge_mode(
    OptionKind::User, "-fdte-merge", "", false,
    "Merge DTE slots based on liveness: reuse a waited future's DTE for "
    "new allocations. Implies -fnamed-dte. Reduces DTE instance count "
    "(e.g. 8 -> 4 for conv1d). Off by default.");

namespace {

inline const char* TopsMdsStorage(Storage st) {
  switch (st) {
  case Storage::DEFAULT:
  case Storage::GLOBAL: return "tops::Global";
  case Storage::SHARED: return "tops::Shared";
  case Storage::LOCAL: return "tops::Private";
  default: choreo_unreachable("storage type is not supported.");
  }
  return "";
}

inline const char* TopsDeviceMemory(Storage st) {
  switch (st) {
  case Storage::SHARED: return "__shared__";
  case Storage::LOCAL: return "__local__";
  default: choreo_unreachable("device storage type is not supported.");
  }
  return "";
}

inline std::string TopsParamStorage(Storage st) {
  switch (st) {
  case Storage::SHARED: return "__shared__";
  case Storage::LOCAL: return "__private__";
  default: return "";
  }
  return "";
}

inline const std::string GetDTEContextName() {
  static unsigned i = 0;
  return "choreo_topscc_ctx" + std::to_string(i++);
}

} // namespace

bool TopsccCodeGen::ShouldEmitLineDirective(AST::Node& n) const {
  return isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n) ||
         isa<AST::InThreadsBlock>(&n) || isa<AST::IfElseBlock>(&n) ||
         isa<AST::WhileBlock>(&n) || isa<AST::Assignment>(&n) ||
         isa<AST::ParallelBy>(&n) || isa<AST::DMA>(&n) || isa<AST::Wait>(&n) ||
         isa<AST::Trigger>(&n) || isa<AST::Break>(&n) ||
         isa<AST::Continue>(&n) || isa<AST::Rotate>(&n) ||
         isa<AST::Synchronize>(&n) || isa<AST::Call>(&n) ||
         isa<AST::NamedVariableDecl>(&n) || isa<AST::Return>(&n);
}

std::string TopsccCodeGen::EscapeLineDirectivePath(const std::string& path) {
  return EscapeLinePathForDirective(path);
}

std::string TopsccCodeGen::ResolveLineDirectivePath(const location& loc) const {
  return ResolveDebugLinePath(loc, CCtx().GetDebugLinePathMode());
}

void TopsccCodeGen::EmitLineDirective(AST::Node& n) {
  if (!EnableLineDirective() || !ShouldEmitLineDirective(n)) return;

  auto loc = n.LOC();
  if (loc.begin.line <= 0) return;

  auto file = ResolveLineDirectivePath(loc);
  if (file.empty()) return;

  auto& line_state = IsHost() ? host_line_state : device_line_state;
  if (line_state.valid && line_state.line == loc.begin.line &&
      line_state.file == file)
    return;

  auto& os = IsHost() ? hs : ds;
  os << "#line " << loc.begin.line << " \"" << EscapeLineDirectivePath(file)
     << "\"\n";

  line_state.line = loc.begin.line;
  line_state.file = file;
  line_state.valid = true;
}

void TopsccCodeGen::ResetLineDirectiveState() {
  host_line_state = {};
  device_line_state = {};
}

inline const std::string
TopsccCodeGen::VectorTypeSTR(const ptr<Type>& ty) const {
  auto smi = cur_loop ? cur_loop->GetScopedMaskInfo() : nullptr;
  auto vty = dyn_cast<VectorType>(ty);
  if (!vty) choreo_unreachable("expecting a vector type.");
  auto elem_ty = vty->e_type;
  auto ec = vty->ec;
  auto elem_size = SizeOf(elem_ty);
  auto vector_size = elem_size * ec;
  if (elem_ty == BaseType::BOOL) {
    assert(smi && "missing scoped mask info.");
    auto mask_elem_type = smi->GetMaskEType();
    vector_size *= SizeOf(mask_elem_type);
  }
  std::string vty_str;
  if (vector_size == CCtx().GetVectorLength())
    vty_str = "__vector ";
  else if (vector_size == 2 * CCtx().GetVectorLength())
    vty_str = "__vector2 ";
  else if (vector_size == 4 * CCtx().GetVectorLength())
    vty_str = "__vector4 ";
  else
    choreo_unreachable(
        "unsupported vector size: " + std::to_string(vector_size) + ".");
  auto ELEM_TYPE_STR = [&]() -> std::string {
    switch (elem_ty) {
    case BaseType::F64: return "double";
    case BaseType::F32: return "float";
    case BaseType::F16: return "__fp16";
    case BaseType::BF16: return "__bf16";
    case BaseType::U64: return "unsigned long long";
    case BaseType::U32: return "unsigned int";
    case BaseType::U16: return "unsigned short";
    case BaseType::U8: return "unsigned char";
    case BaseType::S64: return "long long";
    case BaseType::S32: return "int";
    case BaseType::S16: return "short";
    case BaseType::S8: return "char";
    case BaseType::BOOL: return "bool";
    default: choreo_unreachable("unsupported base-type: " + STR(elem_ty) + ".");
    }
  };
  vty_str += ELEM_TYPE_STR();

  if (elem_ty == BaseType::BOOL) {
    assert(smi && "missing scoped mask info.");
    auto mask_elem_type = smi->GetMaskEType();
    if (SizeOf(mask_elem_type) == 8)
      vty_str += " long long";
    else if (SizeOf(mask_elem_type) == 4)
      vty_str += " int";
    else if (SizeOf(mask_elem_type) == 2)
      vty_str += " short";
    else if (SizeOf(mask_elem_type) == 1)
      vty_str += " char";
    else
      choreo_unreachable("unsupported vector boolean type.");
  }
  return vty_str;
}

const std::string TopsccCodeGen::DMATypeSTR(Storage sto,
                                            bool block_level) const {
  if (CCtx().GetArch() == "gcu400") {
    if (sto == Storage::GLOBAL)
      return "tops::shared_dte"; // to confirm
    else if (sto == Storage::SHARED)
      return "tops::shared_dte";
    else if (sto == Storage::LOCAL)
      return "tops::local_dte";
    else
      choreo_unreachable("unsupported storage for DMA context.");
  } else if (CCtx().GetArch() == "gcu300") {
    // GCU300 CDTE type selection:
    //   block_level=true  -> choreo_cdte (shared_dte): block-shared,
    //   single-thread init block_level=false -> choreo_cdte_priv
    //   (private_cdte): per-thread, RAII init
    // SDTE always uses choreo_sdte (private_dte)
    if (sto == Storage::SHARED)
      return block_level ? "choreo::choreo_cdte" : "choreo::choreo_cdte_priv";
    else
      return "choreo::choreo_sdte";
  } else
    return "tops_dte_ctx_t";
}

void TopsccCodeGen::EmitDTEDecl(std::ostringstream& os,
                                const std::string& indent, Storage sto,
                                const std::string& varname, bool with_scope,
                                bool block_level) const {
  auto type = DMATypeSTR(sto, block_level);
  os << indent << type << " " << varname << ";\n";
  if (with_scope && CCtx().GetArch() != "gcu300" &&
      CCtx().GetArch() != "gcu400") {
    os << indent << "tops::dte_scope s_" << varname << "(" << varname << ");\n";
  }
}

const std::string TopsccCodeGen::ShapeSTR(const Shape& s,
                                          const std::string& delimiter,
                                          BaseType cast_to) const {
  auto& vl = s.Value();
  assert(!vl.empty());

  std::ostringstream oss;
  for (unsigned i = 0; i < vl.size(); ++i) {
    if (i > 0) oss << delimiter;
    bool need_static_cast = (cast_to != BaseType::UNKNOWN && !VIIsInt(vl[i]));
    if (need_static_cast)
      oss << "static_cast<" << NameBaseType(cast_to) << ">(";
    oss << ValueSTR(vl[i]);
    if (need_static_cast) oss << ")";
  }
  return oss.str();
}

bool TopsccCodeGen::BeforeVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "Before visiting " << n.TypeNameString() << "\n";

  EmitPreSiteAssertions(n);

  EmitLineDirective(n);

  if (isa<AST::Program>(&n)) {
    VST_DEBUG(dbgs() << STR(FBInfo()) << "\n");
    if (CCtx().GetArch() != "gcu300") {
      use_dte_pool = false;
      no_future_mode = false;
      named_dte_mode = false;
      dte_merge_mode = false;
    }
    if ((no_future_mode || named_dte_mode || dte_merge_mode) && !use_dte_pool) {
      std::string flags;
      if (no_future_mode) flags += " -fno-future";
      if (named_dte_mode) flags += " -fnamed-dte";
      if (dte_merge_mode) flags += " -fdte-merge";
      choreo_unreachable("flags" + flags +
                         " require --use-dte-pool to be enabled.");
    }
    // emit the fixed headers
    EmitFixedHostHead();
    EmitFixedDeviceHead();
    ssm.EnterScope();
    ssm.MapDeviceSymbolIfNotExist("::__choreo_no_tiling__", "0");
    levels.push(ParallelLevel::NONE);
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ResetChoreoFunctionStates();
    BuildSiteAssertionMap();
    device_fn = "__choreo_device_" + fname;
    fty = cast<FunctionType>(GetSymbolType(fname));
    ssm.EnterScope();
    levels.push(ParallelLevel::SEQ);
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.push(pb->GetLevel());
    if (pb->IsOuter() && pb->GetLevel() == ParallelLevel::DEVICE) {
      device_defers_launch = true;
      deferred_device_pb = pb;
      parallel_idx += 1;
      hs << h_indent << "// device parallel-by: " << n.LOC() << "\n";
    } else if (pb->IsOuter() ||
               (!pb->IsOuter() && pb->GetLevel() == ParallelLevel::BLOCK &&
                device_defers_launch)) {
      if (!pb->IsOuter()) {
        // BLOCK under deferred DEVICE -- treat as the actual outer for codegen
      } else {
        parallel_idx += 1;
      }
      emitted_device_names_.clear();
      if (cgi.GetFunctionTrait(fname).multiple_parallelby)
        device_fn = "__choreo_device_" + fname + std::to_string(parallel_idx);
      current_pb_is_cooperative =
          pb->GetLevel() == ParallelLevel::BLOCK && pb->IsCooperative();
      EmitDeviceFuncDecl(ds);
      ds << " {\n";
      IncrDeviceIndent();
      extern_smem = false;
      if (auto dev_name = SSTab().ScopeName();
          FCtx(fname).HaveDynamicBuffer(dev_name, Storage::SHARED)) {
        auto mri = FCtx(fname).GetDynMemReuseInfo(dev_name);
        assert(mri);
        shared_spm_size =
            sbe::sym(mri->infos[Storage::SHARED].spm_size)->Normalize();
        if (!sbe::ceq(shared_spm_size, sbe::nu(0))) {
          ds << d_indent << "extern __shared__ char " << device_fn
             << "__runtime_shared_buffer__[];\n";
          extern_smem = true;
        }
      }
      // The placeholder will be replaced with the actual DTE pool decl.
      if (use_dte_pool)
        ds << R"(  /*__CHOREO_DTE_POOL_PLACEHOLDER__*/)" << "\n";
      ds << d_indent << "{ // parallel-by: " << n.LOC() << "\n";
      VST_DEBUG(pb->InlinePrint(dbgs());
                dbgs() << " (max-level: " << STR(TargetMaxLevel()) << ")\n");
    }
  } else if (isa<AST::WithBlock>(&n)) {
    IndStream() << "// with-in: " << n.LOC() << "\n";
    IndStream() << "{\n";
    IncrIndent();
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    IndStream() << "// foreach: " << n.LOC() << "\n";
    auto loop = fb->loop;
    if (loop && loop->CanVectorize()) {
      // create vector loop induction variable
      std::string sname = InScopeName(UnScopedName(loop->IVSym()));
      auto ivs = within_map.at(sname);
      assert(ivs.size() == 1 &&
             "vectorized foreach only supports one induction variable.");
      auto iv = UnScopedName(ivs[0]);
      std::string iv_name = "__iv_" + iv;
      int vector_width = loop->GetVectorFactor();
      std::string vec_iv_plus_name = "__vec_iv_" + iv + "_plus";
      std::string vec_iv_base_name = "__vec_iv_" + iv + "_base";
      auto vector_type = MakeVectorType(BaseType::U32, vector_width);
      ds << d_indent << VectorTypeSTR(vector_type) << " " << vec_iv_base_name
         << " = "
         << "(" << VectorTypeSTR(vector_type) << ")(" << iv_name << ");\n";

      ds << d_indent << VectorTypeSTR(vector_type) << " " << vec_iv_plus_name
         << " = ";
      if (CCtx().GetArch() == "gcu300") {
        ds << "{";
        for (int i = 0; i < vector_width; ++i) {
          if (i > 0) ds << ", ";
          ds << i;
        }
        ds << "};\n";
      } else if (CCtx().GetArch() == "gcu400") {
        ds << "tcle::mid<" << VectorTypeSTR(vector_type) << ", 0>(0);\n";
      } else
        choreo_unreachable("unsupported target architecture.");

      std::string vec_iv_name = "__vec_iv_" + iv;
      ds << d_indent << VectorTypeSTR(vector_type) << " " << vec_iv_name
         << " = " << vec_iv_base_name << " + " << vec_iv_plus_name << ";\n";
      return true;
    }
  }

  if (isa<AST::IfElseBlock>(&n) || isa<AST::NamedVariableDecl>(&n)) {
    emit_call = false;
  }

#if 0
  if (!n.IsBlock() && NeedLevelPred()) {
    ds << d_indent << "if (" << LevelPred(Level())
       << ") { // implicit inthreads\n";
    IncrDeviceIndent();
  }
#endif

  return true;
}

bool TopsccCodeGen::InMidVisitImpl(AST::Node& n) {
  if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    if (!ie->HasElse()) return true;
    PopEmittedNames();
    PushEmittedNames();
    DecrIndent();
    IndStream() << "} else {\n";
    IncrIndent();
  }
  return true;
}

bool TopsccCodeGen::AfterVisitImpl(AST::Node& n) {
  if (trace_visit) dbgs() << "After visiting " << n.TypeNameString() << "\n";

  if (isa<AST::Program>(&n)) {
    ssm.LeaveScope();

    // Auto-include acore_op.h when acore:: calls are detected
#ifdef __CHOREO_GCU_ACORE_DIR__
    if (has_acore_call && !code_segments.empty()) {
      code_segments[0] += "#ifdef __ACORE_OP__\n"
                          "#include <common/acore_op.h>\n"
                          "#endif // __ACORE_OP__\n\n";
    }
#endif
    if (!acore_mma.stub_code.empty() && !code_segments.empty())
      code_segments[0] += acore_mma.stub_code;
    if ((has_lib_gemm_general || has_lib_fallback) && !code_segments.empty())
      code_segments[0] += "#include \"gcu/lib_fallback.h\"\n\n";

    // internal functionality: fatbin generation
    if (emit_fatbin) {
      if (!CompileWithScript("--gen-fatbin")) {
        error_count++;
        return false;
      } else
        return true;
    }

    switch (CCtx().GetOutputKind()) {
    case OutputKind::TargetSourceCode:
    case OutputKind::DeviceSourceOnly: EmitSource(); break;
    case OutputKind::TargetModule: {
      if (!CompileWithScript("--compile-module")) {
        error_count++;
        return false;
      }
      break;
    }
    case OutputKind::TargetExecutable: {
      if (!CompileWithScript("--compile-link")) {
        error_count++;
        return false;
      }
      break;
    }
    case OutputKind::TargetLibrary: {
      if (!CompileWithScript("--lib")) {
        error_count++;
        return false;
      }
      break;
    }
    case OutputKind::ShellScript: {
      EmitScript(outs());
      break;
    }
    default:
      choreo_unreachable("outputkind: " + STR(CCtx().GetOutputKind()) +
                         " is not supported.");
    }
  } else if (isa<AST::ChoreoFunction>(&n)) {
    PLDCheck();
    ssm.LeaveScope();
    // ds: __global__ kernel. hs: target launch entry for this __co__ function
    // (malloc/H2D/launch/D2H/free). For DeviceSourceOnly (hetero offload) hs
    // supplies the __hetero_* symbol the hetero host .o links against; it is
    // not the hetero orchestration program and does not duplicate main().
    // Ensure choreo function code lands in a CS_CO segment, even if user code
    // pushed a CS_USER segment between choreo functions.
    if (segment_tags.empty() || segment_tags.back() != CS_CO) {
      code_segments.push_back("");
      segment_tags.push_back(CS_CO);
    }
    {
      auto device_code = ds.str();
      if (dte_pool_size > 0) {
        // Disable named-dte when no-future rotate is used: rotation requires
        // runtime slot indexing that named variable replacement would break.
        bool use_named =
            (named_dte_mode || dte_merge_mode) && !has_nofuture_rotate;
        std::string pool_decl;
        if (use_named) {
          for (int i = 0; i < dte_pool_size; ++i) {
            std::vector<std::string> slot_users;
            for (const auto& kv : dte_pool_slots) {
              // Skip the internal sentinel key used for anonymous DMA
              // merge tracking; it's not a real future name.
              if (kv.first == "__anon__") continue;
              if (kv.second == i) slot_users.push_back(kv.first);
            }
            std::string var_name;
            if (slot_users.empty()) {
              var_name = "__dte_anon_" + std::to_string(i) + "__";
            } else if (slot_users.size() == 1) {
              var_name = "__dte_" + slot_users[0] + "__";
            } else {
              std::sort(slot_users.begin(), slot_users.end());
              var_name = "__dte_" + slot_users[0] + "__";
            }
            dte_named_vars[i] = var_name;
            pool_decl += "  tops::private_dte " + var_name + ";\n";
          }
        } else {
          pool_decl = "  choreo::choreo_sdte __choreo_dte_pool__[" +
                      std::to_string(dte_pool_size) +
                      "];\n"
                      "  for (int __i = 0; __i < " +
                      std::to_string(dte_pool_size) +
                      "; ++__i) __choreo_dte_pool__[__i].init();";
        }
        // Replace ALL placeholders with DTE declarations
        const std::string placeholder = "  /*__CHOREO_DTE_POOL_PLACEHOLDER__*/";
        size_t pos = 0;
        while ((pos = device_code.find(placeholder, pos)) !=
               std::string::npos) {
          device_code.replace(pos, placeholder.length(), pool_decl);
          pos += pool_decl.length();
        }
        if (use_named) {
          // Replace all __choreo_dte_pool__[N] refs with named vars
          for (int i = 0; i < dte_pool_size; ++i) {
            std::string old_ref =
                "__choreo_dte_pool__[" + std::to_string(i) + "]";
            std::string new_ref = dte_named_vars[i];
            size_t search_pos = 0;
            while ((search_pos = device_code.find(old_ref, search_pos)) !=
                   std::string::npos) {
              device_code.replace(search_pos, old_ref.length(), new_ref);
              search_pos += new_ref.length();
            }
          }
          // Replace dynamic slot refs: __choreo_dte_pool__[X_slot]
          for (const auto& kv : dte_pool_slots) {
            std::string old_ref = "__choreo_dte_pool__[" + kv.first + "_slot]";
            std::string new_ref = dte_named_vars[kv.second];
            size_t search_pos = 0;
            while ((search_pos = device_code.find(old_ref, search_pos)) !=
                   std::string::npos) {
              device_code.replace(search_pos, old_ref.length(), new_ref);
              search_pos += new_ref.length();
            }
          }
        }
      } else {
        const std::string placeholder =
            "  /*__CHOREO_DTE_POOL_PLACEHOLDER__*/\n";
        size_t pos = 0;
        while ((pos = device_code.find(placeholder, pos)) !=
               std::string::npos) {
          device_code.erase(pos, placeholder.length());
        }
      }
      code_segments.back() += device_code + hs.str();
    }
    ds.str(""); // reset the streams
    hs.str("");
    return_stream.str("");
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    bool was_device_deferred_block = !pb->IsOuter() &&
                                     pb->GetLevel() == ParallelLevel::BLOCK &&
                                     device_defers_launch;
    levels.pop();
    if (pb->IsOuter() && pb->GetLevel() == ParallelLevel::DEVICE) {
      if (h_indent.size() >= 2) h_indent.resize(h_indent.size() - 2);
      hs << h_indent << "} // end device parallel-by\n";
      auto pv_name = pb->BPV()->name;
      auto bound = ValueSTR(pb->BoundValue());
      hs << h_indent << "for (int __sync_" << pv_name << " = 0; __sync_"
         << pv_name << " < " << bound << "; ++__sync_" << pv_name << ") {\n";
      hs << h_indent << "  choreo::abend_true(topsSetDevice(__sync_" << pv_name
         << "));\n";
      hs << h_indent << "  choreo::abend_true(topsDeviceSynchronize());\n";

      for (auto& item : GetChoreoFuncIns(cgi)) {
        auto sty = dyn_cast<SpannedType>(item.type);
        if (!sty || item.attr == ParamAttr::GLOBAL_INPUT) continue;
        auto oname = UnScopedName(item.name);
        auto buf_sym = oname + "__device";
        if (item.IsReference()) {
          hs << h_indent << "  choreo::abend_true(topsMemcpy(" << oname
             << ".data(), " << buf_sym << "_vec[__sync_" << pv_name << "], "
             << UnScopedSizeExpr(*sty) << ", topsMemcpyDeviceToHost));\n";
        }
        hs << h_indent << "  choreo::abend_true(topsFree(" << buf_sym
           << "_vec[__sync_" << pv_name << "]));\n";
      }

      hs << h_indent << "}\n";
      device_defers_launch = false;
      deferred_device_pb = nullptr;
    } else if (pb->IsOuter() || was_device_deferred_block) {
      // Emit pending buffer unmaps before closing the parallel scope.
      for (auto &[src_sym, entry] : pending_mapped_buffers_) {
        auto &res_name = std::get<0>(entry);
        ds << d_indent << "tops::unmap_mem_m(" << res_name << "_mmu, "
           << std::get<2>(entry) << ");\n";
      }
      pending_mapped_buffers_.clear();
      ds << d_indent << "} // end parallel-by\n";
      DecrDeviceIndent();
      ds << "}\n\n";
    }
  } else if (isa<AST::WithBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    const auto& ranges = fb->GetRangeNodes();
    for (int j = ranges->Count() - 1; j >= 0; --j) {
      auto rng = cast<AST::LoopRange>(ranges->ValueAt(j));
      auto cname = rng->GetRVName();

      auto ivs = within_map.at(InScopeName(cname));
      auto loop = fb->loop;
      if (loop && loop->CanVectorize()) {
        auto iv_name = UnScopedName(cur_loop->IVSym());
        std::string sname = InScopeName(iv_name);
        int vector_width = cur_loop->GetVectorFactor();
        auto vector_type = MakeVectorType(BaseType::U32, vector_width);

        IndStream() << SSMName(sname, IsHost()) << " += ("
                    << VectorTypeSTR(vector_type) << ")(" << vector_width
                    << ");\n";
        // remap symbol of loop induction variable to its scalar version.
        ssm.RemapDeviceSymbol(sname, "__iv_" + iv_name);
        ssm.RemapHostSymbol(sname, "__iv_" + iv_name);
      }

      for (auto iv_itr = ivs.rbegin(); iv_itr != ivs.rend(); ++iv_itr) {
        DecrIndent();
        IndStream() << "} // " << UnScopedName(*iv_itr) << "\n";
        IndStream() << ssm.DeviceName(*iv_itr) << " = 0;\n"; // must reset
      }
    }
  } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
    // only on device-side
    DecrDeviceIndent();
    if (!it->stmts->None()) {
      ds << d_indent << "}";
      if (!it->async && it->outer) ds << "\n" << d_indent << "__syncthreads();";
      ds << " // end inthreads\n";
    }
    PopEmittedNames();
  } else if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    auto pred = ie->GetPred();
    if (!pred->GetDiversityShape().Varying()) {
      DecrIndent();
      IndStream() << "} // end if-else: " << ie->LOC() << "\n";
    } else {
      IndStream() << "// end if-else: " << ie->LOC() << "\n";
    }
    PopEmittedNames();
  } else if (auto ie = dyn_cast<AST::WhileBlock>(&n)) {
    DecrIndent();
    IndStream() << "} // end while: " << ie->LOC() << "\n";
  } else if (isa<AST::NamedVariableDecl>(&n)) {
    emit_call = true;
  }

  EmitPostSiteAssertions(n);

#if 0
  if (!n.IsBlock() && NeedLevelPred()) {
    DecrDeviceIndent();
    ds << d_indent << "} // end implicit inthreads\n";
  }
#endif

  return true;
}

// tops::mdspan style offset
std::pair<std::string, size_t>
TopsccCodeGen::GenMdsOffset(const ptr<AST::ChunkAt> ca,
                            ptr<DMAConfig> config) const {
  auto& sops = ca->AllOperations();
  assert(!sops.empty());

  size_t sop_base = 0;
  if (auto li = ca->IndexOfLastSpanAs()) sop_base = *li + 1;

  if (sop_base == sops.size()) {
    // span_as is the tail spannedoperation
    std::ostringstream oss;
    auto sz = ca->GetBlockShape().DimCount();
    for (size_t i = 0; i < sz; ++i) {
      if (i > 0) oss << ", ";
      oss << "0";
    }
    return {oss.str(), sz};
  }

  // Handle each SOP in the chain via symbolic algebra.
  // Dispatch is interface-based:
  //   GetIndices non-null -- index-based (Tiling/SubSpan/TileAt)
  //   GetOffsets non-null -- offset-based (View)
  std::vector<sbe::ExprSum> offsets;

  for (size_t sop_idx = sop_base; sop_idx < sops.size(); ++sop_idx) {
    assert(!isa<AST::SOP::Reshape>(sops[sop_idx]));

    auto& shape = sops[sop_idx]->GetBlockShape();
    auto op = sops[sop_idx];
    ValueList coords;
    bool scale_by_shape = true;

    if (op->GetIndices()) {
      // Index-based SOPs (Tiling, SubSpan, TileAt): idx * block_shape.
      // Per-node Opts carry pre-expanded bounded vars and preserve the
      // ::__choreo_no_tiling__ symbol -- collect them directly.
      for (auto p : op->IndexNodes()) {
        if (auto e = dyn_cast<AST::Expr>(p); e && e->Opts().HasVals())
          for (auto& val : e->Opts().GetVals()) coords.push_back(val);
        // Non-Expr nodes (e.g. raw Identifiers injected by LateNorm) or nodes
        // without per-node vals defer to the aggregated MultiValues path below.
      }
      // Use aggregated Opts when per-node collection doesn't cover all dims
      // (e.g. bounded vars set on the MultiValues level by HostSliceBufferGen).
      if (coords.size() != shape.DimCount()) {
        assert(op->GetIndices()->Opts().HasVals() &&
               "aggregated index vals missing after per-node collection");
        coords = op->GetIndices()->Opts().GetVals();
      }
    } else if (auto off = op->GetOffsets()) {
      // Offset-based SOPs (View): offsets NOT scaled by block_shape.
      // View offset nodes are always Expr with vals from ShapeInference.
      scale_by_shape = false;
      for (auto p : off->AllValues()) {
        auto e = dyn_cast<AST::Expr>(p);
        assert(e && e->Opts().HasVals() && "View offset node missing vals");
        for (auto& val : e->Opts().GetVals()) coords.push_back(val);
      }
    } else {
      // No indices or offsets -- zero offset per dimension
      coords = ValxN(sbe::nu(0), shape.DimCount());
    }

    if (auto tc = dyn_cast<TransposeConfig>(config)) {
      assert(tc->dim_values.size() == coords.size());
      assert(ca->TilingOperationCount() == 1);
    }

    // Accumulate per-dimension symbolic offsets across chained SOPs.
    if (offsets.empty()) offsets.resize(coords.size());

    for (size_t i = 0; i < coords.size(); ++i) {
      if (sbe::ceq(coords[i], sbe::sym("::__choreo_no_tiling__"))) continue;
      if (scale_by_shape)
        offsets[i] += coords[i] * shape.ValueAt(i);
      else
        offsets[i] += coords[i];
    }
  }

  std::ostringstream offset;
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (i != 0) offset << ", ";
    auto norm = offsets[i].Get();
    if (sbe::ceq(norm, sbe::nu(0)))
      offset << "0";
    else
      offset << "(int)(" << ValueSTR(norm) << ")";
  }

  if (split_8byte_dma_transfer) {
    auto sty = GetSpannedType(GetSymbolType(ca->data->name));
    if (SizeOf(sty->ElementType()) == 8) offset << ", 0";
  }

  VST_DEBUG(dbgs() << "Offset for chunkat (" << PSTR(ca)
                   << "): " << offset.str() << "\n");

  return {offset.str(), offsets.size()};
}

// Example:
//
//   f32 [10, 9, 8] a;
//   ... a.subspan(2, 9, 8).at(p, _, _).span_as(...);
//
// The "Tile Base Offset" (offset ahead of last span_as) is:
//
//    tbo = p * 2 * (9 * 8)
//
const std::string
TopsccCodeGen::TileBaseOffset(const ptr<AST::ChunkAt>& ca) const {
  auto lidx = ca->IndexOfLastSpanAs();
  if (!lidx.has_value()) choreo_unreachable("unexpect");
  return GenOffset(ca, lidx.value());
}

// given i.sop(...).sop(...)..., generate the flat element-offset of the final
// span in the original span. Each SOP in [0, end_idx) contributes additively.
//
// Dispatch is based on the SOP interface, not concrete types:
//   GetIndices non-null (Tiling/SubSpan/TileAt) -> idx * blk * stride
//   GetOffsets non-null (View)                  -> off * stride
//   Reshape                                     -> boundary (skip)
const std::string TopsccCodeGen::GenOffset(const ptr<AST::ChunkAt>& ca,
                                           size_t end_idx) const {
  if (ca->NoOperation()) return "";

  end_idx = std::min(end_idx, ca->OpCount());

  sbe::ExprSum offset;
  auto cur_strd = GetSpannedType(GetSymbolType(ca->RefSymbol()))->GetStrides();

  for (size_t i = 0; i < end_idx; ++i) {
    const auto& sop = ca->OpAt(i);
    if (isa<AST::SOP::Reshape>(sop)) continue;

    auto strd = sop->GetBlockStrides();

    if (auto indices = sop->GetIndices()) {
      auto& vals = indices->Opts().GetVals();
      auto blk = sop->GetBlockShape();
      for (size_t dim = 0; dim < vals.size(); ++dim)
        offset += vals[dim] * blk.ValueAt(dim) * strd[dim];
      cur_strd = strd;
    } else if (auto off_mv = sop->GetOffsets()) {
      auto& vals = off_mv->Opts().GetVals();
      auto off_strd = strd;
      for (size_t dim = 0; dim < vals.size(); ++dim)
        offset += vals[dim] * off_strd[dim];
      cur_strd = strd;
    }
  }

  return ValueSTR(offset.Get());
}

void TopsccCodeGen::EmitFixedHostHead() {
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

#if __GCU_ARCH__ >= 300
#include "tcle.h"
#endif // __GCU_ARCH__ >= 300

// Choreo intrinsic-prefix pragma is emitted as a no-op macro in the
// generated source so that topscc does not reject it as an unknown
// identifier. The choreo parser consumes the pragma before codegen.
#define __pragma_croq_intrinsic_prefix(x)
#define __pragma_croq_intrinsic_namespace(x)
)";

  oss << "// include the choreo header;\n";
  if (native_f16)
    oss << "#define __CHOREO_TARGET_NATIVE_HALF_FLOAT_SUPPORT__\n";
  if (native_bf16) oss << "#define __CHOREO_TARGET_NATIVE_BF16_SUPPORT__\n";
  oss << R"(#include "choreo.h"

using namespace choreo;

// Force line-buffered stdout so host printf output merges correctly
// with device output when piped (e.g. `2>&1 | FileCheck`).
__attribute__((constructor))
static void __co_init_stdout() { setvbuf(stdout, NULL, _IOLBF, 0); }

)";

  code_segments.push_back(oss.str());
  segment_tags.push_back(CS_CO);
}

void TopsccCodeGen::EmitFixedDeviceHead() {}

bool TopsccCodeGen::Visit(AST::FunctionDecl& n) {
  TraceEachVisit(n);

  assert(n.name == fname && "inconsistent in function names.");
  assert(isa<FunctionType>(n.GetType()) && "unexpected type.");

  auto HandleSymbolicDimensions = [this](const ptr<SpannedType>& sty,
                                         const std::string& hp_name,
                                         size_t hp_index) {
    size_t dim_index = 0;
    for (auto vi : sty->GetShape().Value()) {
      if (auto vale = VISym(vi)) { // the dimension is symbolic
        assert(PrefixedWith(*vale, "::" + fname + "::") &&
               "unexpected symbolic dimension name.");

        auto dim_expr = hp_name + ".shape()[" + std::to_string(dim_index) + "]";
        if (symbolic_dimensions.count(*vale) == 0) {
          symbolic_dimensions[*vale] = {dim_expr, hp_index, dim_index};
          ssm.MapDeviceSymbol(*vale, UnScopedName(*vale));
        }
      }
      assert(!VIIsBop(vi) && "unexpected binary operation.");
      dim_index++;
    }
  };

  // Go through all the symbols appeared in topscc host function, do:
  //
  //  - decide the host parameter names,
  //  - map the runtime shape dimensions to the real host code expression
  //  - decide the topscc-host parameter names,
  //  - decide the topscc-host parameter indices,
  //
  size_t host_pindex = 0;
  for (auto& item : GetChoreoFuncIns(cgi)) {
    if (item.IsParameter()) {
      assert((int)host_pindex == item.p_index);
      item.host_name = UnScopedName(item.name);
      if (auto sty = dyn_cast<SpannedType>(item.type)) {
        ssm.MapHostSymbol(item.name, item.host_name + ".data()");
        HandleSymbolicDimensions(sty, item.host_name, host_pindex);
      } else
        ssm.MapHostSymbol(item.name, item.host_name);
    } else
      item.host_name = UnScopedName(item.name);

    item.h_name = "args[" + std::to_string(host_pindex) + "]";
    item.h_index = host_pindex;
    host_pindex++;
  }

  if (isa<VoidType>(fty->out_ty)) {
    void_return = true;
    VST_DEBUG(dbgs() << fname << ": void return\n");
  }

  EmitHostFuncDecl(hs);

  hs << " {\n";
  IncrHostIndent();

  // name the symbolic dimensions for better readability
  for (auto item : symbolic_dimensions) {
    hs << h_indent << "auto &" << UnScopedName(item.first) << " = "
       << item.second.hsd_expr << ";\n";
    ssm.MapHostSymbol(item.first, UnScopedName(item.first));
  }

  // emit the runtime checks
  EmitHostRuntimeCheck();

  // do not generate device function unless parallel-by exists
  if (NeedDeviceFunc()) {
    // map the choreo input to device memory
    for (auto& item : GetChoreoFuncIns(cgi)) {
      if (auto sty = dyn_cast<SpannedType>(item.type)) {
        if (item.attr == ParamAttr::GLOBAL_INPUT) {
          ssm.MapHostSymbol(item.name + "__device",
                            UnScopedName(item.name) + ".data()");
          continue;
        }
        auto sym = UnScopedName(item.name);
        std::string bts = NameBaseType(sty->ElementType(), false);
        auto buf_sym = sym + "__device";
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        if (!FCtx(fname).HasDeviceParallel()) {
          hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym << ", "
             << UnScopedSizeExpr(*sty) << "));\n";
          hs << h_indent << "choreo::abend_true(topsMemcpy(" << buf_sym << ", "
             << ssm.HostName(item.name) << ", " << UnScopedSizeExpr(*sty)
             << ", topsMemcpyHostToDevice));\n";
        }
        ssm.MapHostSymbol(item.name + "__device", buf_sym);
        global_buffers.insert(buf_sym);
      }
    }
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);

  // If there is no AST::Return
  if (return_stream.str().empty() && NeedDeviceFunc()) EmitTopsFree();

  DecrHostIndent();
  hs << "}\n\n";

  return true;
}

bool TopsccCodeGen::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  auto sym = n.name_str;

  SSTab().DefineSymbol(sym, nty);

  bool ref = n.HasNote("ref");
  // workaround:
  // if a symbol is declared but have no symbol value(optimized value)
  // pass it to device func even it is unused.
  auto sname = InScopeName(sym);
  if (!FCtx(fname).HasSymbolValues(sname))
    updating_cgi.AddSymbolDetail(fname, {sname, GetSymbolType(sym), true});
  else {
    auto sv = FCtx(fname).GetSymbolValues(sname);
    // workround: for symbolic valno, treat it as ref to do codegen.
    if (n.IsMutable() && sv.HasVals() && sv.GetVals().size() == 1 &&
        VIIsSym(sv.GetVal()))
      ref = true;
    updating_cgi.AddSymbolDetail(fname, {sname, GetSymbolType(sym), ref});
  }

  // The type is determined first, and then
  // the device or host side is determined
  if (auto s = dyn_cast<AST::Select>(n.init_expr)) {
    assert(!IsHost() && "select should be on device side.");
    assert(!s->inDMA);
    size_t val_count = s->expr_list->Count();
    assert(val_count >= 2);
    std::string array_sym = sym + "_select_array__";
    if (isa<FutureType>(NodeType(*s))) {
      ds << d_indent << "future * " << array_sym << "[] = {";
      for (size_t i = 0; i < val_count; i++) {
        if (i > 0) ds << ", ";
        ds << "&" << OpExprSTR(s->expr_list->ValueAt(i), "&", false, false);
      }
      ds << "};\n";
      // make symbol a reference
      ds << d_indent << "future & " << sym << " = *" << array_sym << "["
         << ExprSTR(s->select_factor, false) << "];\n";
    } else
      choreo_unreachable("select of " + PSTR(NodeType(*s)) +
                         " is yet to implement.");

    return true;
  }

  if (auto e = dyn_cast<AST::Expr>(n.init_expr))
    if (auto sa = dyn_cast<AST::SpanAs>(e->GetReference())) {
      if (IsHost()) choreo_unreachable("span-as should be on device side.");
      ds << d_indent << "auto* " << sym << " = ";
      auto tty = GetSymbolType(sa->id->name);
      if (isa<FutureType>(tty)) {
        auto sn = InScopeName(sa->id->name);
        auto nf_it =
            no_future_mode ? nofuture_vars.find(sn) : nofuture_vars.end();
        if (nf_it != nofuture_vars.end())
          ds << nf_it->second.data_ptr << ";\n";
        else
          ds << sa->id->name << ".data();\n";
      } else
        ds << sa->id->name << ";\n";
      ssm.MapDeviceSymbol(InScopeName(sym), sym);
      return true;
    }

  if (auto sty = dyn_cast<SpannedType>(nty)) {
    auto sym__init = sym + "__init";
    auto buf_sym = sym + "__device";
    // globals are declared in host, while shareds/locals are declared in device
    auto shape = sty->GetShape();
    std::string bts{NameBaseType(sty->ElementType())};
    auto sto = sty->GetStorage();

    bool spmem = false; // allocatable scratchpad memory: share, local

    auto HandleGlobal = [&]() -> void {
      bts = NameBaseType(sty->ElementType(), false); // use the device type name

      if (IsChoreoOutput(InScopeName(sym))) {
        // the sym is choreo output
        std::string sym_data = sym + ".data()";
        hs << h_indent << "auto " << sym
           << " = choreo::make_spandata<choreo::" << STR(sty->e_type) << ", "
           << shape.Rank() << ">({" << ShapeSTR(shape, ", ", BaseType::U64)
           << "});\n";
        if (n.init_value) {
          // support initialization of output
          hs << h_indent << "std::fill(" << sym_data << ", " << sym_data << "+"
             << sym << ".element_count()"
             << ", "
             << ExprCastSTR(n.init_value, std::nullopt, GetBaseType(*sty),
                            GetBaseType(*n.init_value->GetType()), true)
             << ");\n";
        }
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << "));\n";
        if (n.init_value) {
          hs << h_indent << "choreo::abend_true(topsMemcpy(" << buf_sym << ", "
             << sym_data << ", " << UnScopedSizeExpr(*sty)
             << ", topsMemcpyHostToDevice));\n";
        }
        global_buffers.insert(buf_sym);
        return;
      }

      // the sym is not choreo output
      if (FBIContainsBuffer(FBInfo(), InScopeName(sym)) &&
          use_hetero_tileflow && IsHost()) {
        // a non-init global var decl tied with future
        // this hint is enough to say a host side dataflow
        VST_DEBUG(dbgs() << "Found " << buf_sym << " in FBInfo - "
                         << STR(FBInfo()) << "\n");
      } else {
        hs << h_indent << bts << " * " << buf_sym << " = nullptr;\n";
        hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym << ", "
           << UnScopedSizeExpr(*sty) << "));\n";

        if (!n.init_value) return;

        // support int/float-point literal initialization
        std::string sym_init_val = sym + "_init_val";
        hs << h_indent << bts << " " << sym_init_val << " = "
           << ExprSTR(n.init_value) << ";\n";
        std::string sym_init_vptr = sym + "_init_vptr";
        size_t data_len = SizeOf(sty->ElementType()) * 8;
        std::string init_val_type;
        switch (data_len) {
        case 32: init_val_type = "int"; break;
        case 16: init_val_type = "unsigned short"; break;
        case 8: init_val_type = "unsigned char"; break;
        default:
          choreo_unreachable("unsupported data length " +
                             std::to_string(data_len) +
                             " in global span init.");
        }
        hs << h_indent << init_val_type << "* " << sym_init_vptr
           << " = reinterpret_cast<" << init_val_type << "*>(&" << sym_init_val
           << ");\n";
        hs << h_indent << "choreo::abend_true(topsMemsetD" << data_len << "("
           << buf_sym << ", *" << sym_init_vptr << ", "
           << UnScopedExpr(ElemCountExprOf(*sty)) << "));\n";
      }
    };

    auto HandleSharedLocal = [&]() -> void {
      if (IsChoreoOutput(InScopeName(n.name_str)))
        choreo_unreachable(
            "error: shared/local buffer cannot be Choreo output.");

      auto type_modifiers =
          (sto == Storage::SHARED ? "__shared__ " : "__local__ __valigned__ ");

      if (!CCtx().MemReuse()) {
        ds << d_indent << type_modifiers << bts << " " << sym;
        for (const auto& dim : GetArrayDimensions(nty))
          ds << "[" << ValueSTR(dim) << "]";
        ds << "[" << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
        return;
      }

      // memory reuse is enabled

      if (n.HasNote("spm")) {
        if (!extern_smem || sto == Storage::LOCAL) {
          ds << d_indent << type_modifiers << bts << " " << sym << "["
             << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
        } else {
          ds << d_indent << bts << "* " << sym << " = (" << bts << "*)"
             << device_fn << "__runtime_shared_buffer__;\n";
        }
        return;
      }

      // the buffer is not the declared whole spm.
      if (n.HasNote("reuse")) {
        auto reuse = n.GetNote("reuse");
        auto offset = n.GetNote("offset");
        ds << d_indent << bts << "* " << sym << " = (" << bts << "*)"
           << "(" << reuse << " + " << offset << ");\n";
      } else {
        // the buffer is not reused
        assert(!n.HasNote("offset"));
        if (ref) {
          ds << d_indent << bts << "* " << sym << " = (" << bts << "*)"
             << "(" << ExprSTR(n.init_expr, false) << ");\n";
        } else {
          // which means that it is declared but never used.
          // TODO: should we DCE the unused buffer?
          ds << d_indent << type_modifiers << bts << " " << sym << "["
             << UnScopedExpr(ElemCountExprOf(*sty)) << "];\n";
        }
      }
    };

    if (sto == Storage::GLOBAL) {
      if (!IsHost()) choreo_unreachable("error: global var decl in device.");
      HandleGlobal();
      ssm.MapHostSymbol(InScopeName(sym) + "__device", buf_sym);
      ssm.MapHostSymbol(InScopeName(sym), buf_sym);
      ssm.MapDeviceSymbolIfNotExist(InScopeName(sym), sym);
      global_buffers.insert(buf_sym);
    } else if (sto == Storage::SHARED || sto == Storage::LOCAL) {
      if (IsHost()) choreo_unreachable("error: shared/local var decl in host.");
      sym = UniqueDeviceName(n.name_str);
      HandleSharedLocal();
      ssm.MapDeviceSymbol(InScopeName(n.name_str), sym);
      spmem = true;
    } else
      choreo_unreachable("unsupported storage type.");

    // initialize the spm buffer if needed
    if (spmem && n.init_value) {
      if (sto != Storage::SHARED && sto != Storage::LOCAL)
        choreo_unreachable(
            "error: unexpected storage type in spm initialization.");
      ds << d_indent << BufferInitPred(sto) << "{\n";
      IncrDeviceIndent();
      EmitDTEDecl(ds, d_indent, sto, sym__init, true, false);
      ds << d_indent << "tops::memset(" << sym__init << ", tops::mdspan("
         << TopsMdsStorage(sto) << ", (" << NameBaseType(sty->ElementType())
         << "*)" << sym << ", " << ShapeSTR(sty->GetShape()) << "), "
         << ExprCastSTR(n.init_value, std::nullopt, GetBaseType(*sty),
                        GetBaseType(*n.init_value->GetType()), false)
         << ");\n";
      DecrDeviceIndent();
      ds << d_indent << "} // single instance\n";
    }
    return true;
  }

  // vector variable
  if (IsActualVectorType(nty)) {
    bool masking = false;
    std::string mask;
    auto smi = cur_loop->GetScopedMaskInfo();
    if (smi->NeedMask() && !n.HasNote("masking")) {
      mask = smi->GetMaskInScope(SSTab().ScopeName());
      if (!smi->IsMaskAlltrue(mask)) masking = true;
    }

    auto vector_width = ElementCount(nty);
    auto ele_ty = ElementType(nty);
    auto vty = MakeVectorType(ele_ty, vector_width);
    IndStream() << VectorTypeSTR(vty) << " " << sym << " = ";
    if (n.init_expr) {
      auto rhs_ty = n.init_expr->GetType();
      if (masking) Stream() << "tcle::vsel(exec, ";

      bool rparen = false;
      if (!IsActualVectorType(rhs_ty)) {
        if (n.init_expr->HasNote("broadcast")) {
          Stream() << "(" << VectorTypeSTR(vty) << ")"
                   << "(";
          rparen = true;
        } else {
          choreo_unreachable("not supported scalar -> vector");
        }
      }
      Stream() << ExprSTR(n.init_expr, false);
      if (rparen) Stream() << ")";
      // if need masking, fill the inactive lanes with zero
      if (masking) Stream() << ", (" << VectorTypeSTR(vty) << ")(0))";
    }

    Stream() << ";\n";
    if (!IsHost()) ssm.MapDeviceSymbol(InScopeName(sym), sym);
  }

  if (auto bty = dyn_cast<BoundedType>(nty)) {
    // bounded variable is not with a fixed value
    if (!IsActualBoundedIntegerType(bty))
      choreo_unreachable(
          "yet to support: bounded ituple variable code generation.");
    IndStream() << "int " << sym << " = " << ExprSTR(n.init_expr, false)
                << ";\n";
    return true;
  }

  // when symbol is not valued
  if (isa<ScalarType>(nty) &&
      (IsMutable(*nty) || !FCtx(fname).HasSymbolValues(InScopeName(sym)))) {
    auto mem = n.GetMemory();
    IndStream();
    if (mem != nullptr) {
      auto st = mem->Get();
      Stream() << TopsDeviceMemory(st) << " ";
    }
    Stream() << NameBaseType(GetBaseType(*nty), false) << " " << sym;
    if (n.init_expr) Stream() << " = " << ExprSTR(n.init_expr, false);
    Stream() << ";\n";

    // mutables have references
    if (IsMutable(*nty))
      if (!IsHost()) ssm.MapDeviceSymbol(InScopeName(sym), sym);

    return true;
  }

  // handle events
  if (auto ety = dyn_cast<EventArrayType>(nty)) {
    switch (ety->GetStorage()) {
    case Storage::GLOBAL: {
      assert(IsHost());
      auto sym = InScopeName(n.name_str);
      auto buf_sym = n.name_str + "__device";
      hs << h_indent << "bool * " << buf_sym << " = nullptr; // global event\n";
      hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym << ", "
         << ety->ElemCount() << "));\n";
      hs << h_indent << "choreo::abend_true(topsMemset(&" << buf_sym << ", 0, "
         << ety->ElemCount() << "));\n";
      ssm.MapHostSymbol(sym, buf_sym);
      ssm.MapDeviceSymbol(sym, n.name_str);
      global_buffers.insert(buf_sym);
    } break;
    case Storage::SHARED:
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << TopsDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << n.name_str;
      ety->PrintAsCArray(ds);
      ds << "; // " << STR(ety->GetStorage()) << " event\n";
      ds << d_indent << "// initialize the event\n";
      ds << d_indent << BufferInitPred(ety->GetStorage()) << "{\n";
      GenerateSubscriptions(ds, "  " + d_indent + n.name_str, " = false;\n",
                            ety->Dimensions());
      ds << d_indent << "}\n";
      ds << d_indent << EmitSync(ety->GetStorage()) << ";\n";
    } break;
    default: break;
    }
  } else if (auto ety = dyn_cast<EventType>(nty)) {
    auto ename = UniqueDeviceName(n.name_str);
    switch (ety->GetStorage()) {
    case Storage::GLOBAL: {
      assert(IsHost());
      auto sym = InScopeName(n.name_str);
      auto buf_sym = ename + "__device";
      hs << h_indent << "bool * " << buf_sym << " = nullptr; // global event\n";
      hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym
         << ", 1));\n";
      hs << h_indent << "choreo::abend_true(topsMemset(&" << buf_sym
         << ", 0, 1));\n";
      ssm.MapHostSymbol(sym, buf_sym);
      ssm.MapDeviceSymbol(sym, ename);
      global_buffers.insert(buf_sym);
    } break;
    case Storage::SHARED:
    case Storage::LOCAL: {
      assert(!IsHost());
      ds << d_indent << TopsDeviceMemory(ety->GetStorage())
         << " __volatile__ bool " << ename << "; // " << STR(ety->GetStorage())
         << " event\n";
      ds << d_indent << BufferInitPred(ety->GetStorage()) << " {\n";
      ds << d_indent << "  " << ename << " = false;\n"; // inited as untriggered
      ds << d_indent << "}\n";
      ds << d_indent << EmitSync(ety->GetStorage()) << ";\n";
      ssm.MapDeviceSymbol(InScopeName(n.name_str), ename);
    } break;
    default: break;
    }
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  if (!n.AssignToDataElement()) {
    auto name = n.GetName();
    bool ref = n.HasNote("ref");
    if (!SSTab().IsDeclared(name) && !isa<AST::SpanAs>(n.value))
      updating_cgi.AddSymbolDetail(
          fname, {InScopeName(name), GetSymbolType(name), ref});
  }

  auto nty = NodeType(n);

  if (auto s = dyn_cast<AST::Select>(n.value)) {
    assert(!IsHost() && "select should be on device side.");
    assert(!s->inDMA);
    size_t val_count = s->expr_list->Count();
    assert(val_count >= 2);
    std::string array_sym = n.GetName() + "_select_array__";
    if (isa<FutureType>(nty)) {
      ds << d_indent << "future * " << array_sym << "[] = {";
      for (size_t i = 0; i < val_count; i++) {
        if (i > 0) ds << ", ";
        ds << "&" << OpExprSTR(s->expr_list->ValueAt(i), "&", false, false);
      }
      ds << "};\n";
      // make symbol a reference
      ds << d_indent << "future & " << n.GetName() << " = *" << array_sym << "["
         << ExprSTR(s->select_factor, false) << "];\n";
    } else if (auto sty = dyn_cast<SpannedType>(nty)) {
      auto bts = NameBaseType(sty->ElementType());
      ds << d_indent << bts << " * " << array_sym << "[] = {";
      for (size_t i = 0; i < val_count; i++) {
        if (i > 0) ds << ", ";
        ds << ExprSTR(s->expr_list->ValueAt(i), false);
      }
      ds << "};\n";
      // make symbol a reference
      ds << d_indent << bts << " & " << n.GetName() << " = *" << array_sym
         << "[" << ExprSTR(s->select_factor, false) << "];\n";
    } else
      choreo_unreachable("select of " + PSTR(NodeType(*s)) +
                         " is yet to implement.");

    return true;
  }

  if (auto sa = dyn_cast<AST::SpanAs>(n.value)) {
    assert(!IsHost() && "span-as should be on device side.");
    ds << d_indent << "auto * " << n.GetName() << " = ";
    auto tty = GetSymbolType(sa->id->name);
    if (isa<FutureType>(tty)) {
      auto sn = InScopeName(sa->id->name);
      auto nf_it =
          no_future_mode ? nofuture_vars.find(sn) : nofuture_vars.end();
      if (nf_it != nofuture_vars.end())
        ds << nf_it->second.data_ptr << ";\n";
      else
        ds << sa->id->name << ".data();\n";
    } else
      ds << sa->id->name << ";\n";
    ssm.MapDeviceSymbol(InScopeName(n.GetName()), n.GetName());
    return true;
  }

  // vector type assign
  if (IsActualVectorType(nty)) {
    bool masking = false;
    std::string mask;
    auto smi = cur_loop->GetScopedMaskInfo();
    if (smi->NeedMask() && !n.HasNote("masking")) {
      mask = smi->GetMaskInScope(SSTab().ScopeName());
      if (!smi->IsMaskAlltrue(mask)) masking = true;
    }

    if (!n.AssignToDataElement()) {
      IndStream() << ((!n.IsDecl()) ? "" : "auto ") << n.GetName() << " = ";
      // if need masking, the inactive lanes are not modified
      if (masking) Stream() << "tcle::vsel(exec, ";
      Stream() << ExprSTR(n.value, IsHost());
      if (masking) Stream() << ", " << n.GetName() << ")";
      Stream() << ";\n";
    } else {
      // if it is assigned to a memory reference, which is a store
      if (!IsHost()) {
        auto rhs = n.value;
        if (auto id = AST::GetIdentifier(rhs)) {
          IndStream() << DASTR(n.da, ExprSTR(n.value, false), false, masking)
                      << ";\n";
        } else {
          // need a temporary variable to hold the rhs value
          auto tmp_val_name = symtab.GetAnonName();
          ds << d_indent << VectorTypeSTR(nty) << " " << tmp_val_name << " = "
             << ExprSTR(n.value, false) << ";\n";
          ds << d_indent << DASTR(n.da, tmp_val_name, false, masking) << ";\n";
        }
      } else
        choreo_unreachable(
            "error: assignment to data element should be on device side.");
    }
    return true;
  }

  if (n.AssignToDataElement()) {
    if (!IsHost())
      ds << d_indent << DASTR(n.da, ExprSTR(n.value, false), false) << ";\n";
    else
      choreo_unreachable(
          "error: assignment to data element should be on device side.");

    return true;
  }

  if (isa<BoundedType>(nty) || isa<SpannedType>(nty) || isa<FutureType>(nty)) {
    assert(!IsHost() && "bounded/spanned/future should be on device side.");
    ds << d_indent << ((IsMutable(*nty)) ? "" : "auto ") << n.GetName() << " = "
       << ExprSTR(n.value, false) << ";\n";
    return true;
  }

  if (isa<ScalarType>(nty)) {
    if (IsHost())
      hs << h_indent << ((!n.IsDecl()) ? "" : "auto ") << n.GetName() << " = "
         << ExprSTR(n.value, true) << ";\n";
    else
      ds << d_indent << ((!n.IsDecl()) ? "" : "auto ") << n.GetName() << " = "
         << ExprSTR(n.value, false) << ";\n";
    return true;
  }

  errs() << "Assignment " << STR(n) << " unprocessed, not supported "
         << PSTR(nty) << "\n";
  return false;
}

bool TopsccCodeGen::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  // add the device name map
  std::string dname[] = {"x", "y", "z"};
  switch (n.GetLevel()) {
  case ParallelLevel::DEVICE: {
    auto pv_name = n.BPV()->name;
    ssm.MapHostSymbol(InScopeName(pv_name), pv_name);
    ssm.MapDeviceSymbol(InScopeName(pv_name), "__device_id_" + pv_name);
    for (size_t i = 0; i < n.AllSubPVs().size(); ++i) {
      ssm.MapHostSymbol(InScopeName(n.GetSubPV(i)->name), pv_name);
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(i)->name),
                          "__device_id_" + pv_name);
    }
  } break;
  case ParallelLevel::BLOCK:
    for (size_t i = 0; i < n.AllSubPVs().size(); ++i)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(i)->name),
                          "__tops_bid_" + dname[i] + "()");
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name), "__tops_bid_x()");
    break;
  case ParallelLevel::GROUP:
    for (size_t i = 0; i < n.AllSubPVs().size(); ++i)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(i)->name),
                          "__tops_tid_" + dname[i] + "()");
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name), "__tops_tid_x()");
    break;
  case ParallelLevel::THREAD: {
    std::string id_str = TargetHasLevel(ParallelLevel::GROUP) ? "stid" : "tid";
    for (size_t i = 0; i < n.AllSubPVs().size(); ++i)
      ssm.MapDeviceSymbol(InScopeName(n.GetSubPV(i)->name),
                          "__tops_" + id_str + "_" + dname[i] + "()");
    if (n.AllSubPVs().size() == 1)
      ssm.MapDeviceSymbol(InScopeName(n.BPV()->name),
                          "__tops_" + id_str + "_x()");
  } break;
  default:
    choreo_unreachable("unsupported parallel-by level: " + STR(n.GetLevel()) +
                       ".");
  }

  // DEVICE level: emit host-side topsSetDevice loop with runtime guard
  if (n.IsOuter() && n.GetLevel() == ParallelLevel::DEVICE) {
    auto pv_name = n.BPV()->name;
    auto bound = ValueSTR(n.BoundValue());

    if (!CCtx().DisableRuntimeCheck()) {
      hs << h_indent << "{\n";
      hs << h_indent << "  int __choreo_device_count = 0;\n";
      hs << h_indent
         << "  choreo::abend_true(topsGetDeviceCount(&__choreo_device_count));"
            "\n";
      hs << h_indent
         << "  choreo::runtime_check(__choreo_device_count >= " << bound
         << ", \"device parallelism requires " << bound
         << " device(s), but only \""
            "\n"
         << h_indent
         << "    + std::to_string(__choreo_device_count) + \" available.\");\n";
      hs << h_indent << "}\n";
    }

    for (auto& item : GetChoreoFuncIns(cgi)) {
      auto sty = dyn_cast<SpannedType>(item.type);
      if (!sty || item.attr == ParamAttr::GLOBAL_INPUT) continue;
      auto sym = UnScopedName(item.name);
      auto buf_sym = sym + "__device";
      std::string bts = NameBaseType(sty->ElementType(), false);
      hs << h_indent << "std::vector<" << bts << "*> " << buf_sym << "_vec("
         << bound << ", nullptr);\n";
    }

    hs << h_indent << "for (int " << pv_name << " = 0; " << pv_name << " < "
       << bound << "; ++" << pv_name << ") {\n";
    h_indent += "  ";
    hs << h_indent << "choreo::abend_true(topsSetDevice(" << pv_name << "));\n";

    for (auto& item : GetChoreoFuncIns(cgi)) {
      auto sty = dyn_cast<SpannedType>(item.type);
      if (!sty || item.attr == ParamAttr::GLOBAL_INPUT) continue;
      auto sym = UnScopedName(item.name);
      auto buf_sym = sym + "__device";
      hs << h_indent << "choreo::abend_true(topsMalloc(&" << buf_sym << "_vec["
         << pv_name << "], " << UnScopedSizeExpr(*sty) << "));\n";
      hs << h_indent << "choreo::abend_true(topsMemcpy(" << buf_sym << "_vec["
         << pv_name << "], " << ssm.HostName(item.name) << ", "
         << UnScopedSizeExpr(*sty) << ", topsMemcpyHostToDevice));\n";
      hs << h_indent << buf_sym << " = " << buf_sym << "_vec[" << pv_name
         << "];\n";
    }

    return true;
  }

  bool emit_launch =
      n.IsOuter() || (!n.IsOuter() && n.GetLevel() == ParallelLevel::BLOCK &&
                      device_defers_launch);

  // only do the whole codegen when accessing the outer parallel-by
  if (!emit_launch) return true;

  EmitMemReuse(SSTab().ScopeName());

  // note: `thread_dims` for gcu400 is generated in `EmitDeviceFuncDecl`
  auto& lconfig = cgi.GetFunctionLaunches(fname)[parallel_idx];
  hs << h_indent << "dim3 __" << fname << "_gdims" << parallel_idx << "("
     << ValueSTR(lconfig.block_count.x) << ", "
     << ValueSTR(lconfig.block_count.y) << ", "
     << ValueSTR(lconfig.block_count.z) << ");\n";
  hs << h_indent << "dim3 __" << fname << "_bdims" << parallel_idx << "(";
  if (TargetHasLevel(ParallelLevel::GROUP))
    hs << ValueSTR(lconfig.group_count.x) << ", "
       << ValueSTR(lconfig.group_count.y) << ", "
       << ValueSTR(lconfig.group_count.z) << ");\n";
  else
    hs << ValueSTR(lconfig.thread_count.x) << ", "
       << ValueSTR(lconfig.thread_count.y) << ", "
       << ValueSTR(lconfig.thread_count.z) << ");\n";

  std::string effective_stream;
  if (n.HasStream()) effective_stream = STR(n.StreamExpr());

  if (n.IsCooperative()) {
    // Cooperative launch via topsLaunchCooperativeKernel (required by GCU3+ for
    // kernels that access global memory via device pointers).
    hs << h_indent << "{\n";
    hs << h_indent << "  void* __choreo_coop_args[] = {";
    size_t i = 0;
    for (auto& item : GetDeviceFuncIns(updating_cgi)) {
      auto sname = item.name;
      if (isa<SpannedType>(item.type)) sname += "__device";
      if (!PrefixedWith(scoped_symtab.ScopeName(), GetScope(sname))) continue;
      hs << ((i++ == 0) ? "" : ", ");
      if (ssm.HasHostName(sname))
        hs << "&" << ssm.HostName(sname);
      else
        hs << "&" << UnScopedName(ssm.DeviceName(sname));
    }
    for (auto item : symbolic_dimensions) {
      hs << ((i++ > 0) ? ", " : "");
      hs << "&" << UnScopedName(item.first);
    }
    if (const auto& mri = FCtx(fname).GetDynMemReuseInfo(SSTab().ScopeName()))
      for (const auto& [sto, ie] : mri->infos)
        for (size_t idx = 0; idx < ie.offset_args.size(); ++idx)
          hs << ((i++ > 0) ? ", " : "") << "&" << ie.offsets_name << "[" << idx
             << "]";
    if (deferred_device_pb) {
      auto dpv = deferred_device_pb->BPV()->name;
      hs << ((i++ > 0) ? ", " : "") << "&" << dpv;
    }
    hs << "};\n";
    hs << h_indent << "  topsLaunchCooperativeKernel((const void*)" << device_fn
       << ", __" << fname << "_gdims" << parallel_idx << ", __" << fname
       << "_bdims" << parallel_idx << ", __choreo_coop_args, "
       << (extern_smem ? ValueSTR(shared_spm_size) : "0") << ", "
       << (effective_stream != "" ? effective_stream : "0") << ");\n";
    hs << h_indent << "}\n";
  } else {
    hs << h_indent << device_fn << "<<<__" << fname << "_gdims" << parallel_idx
       << ", __" << fname << "_bdims" << parallel_idx;
    if (extern_smem) hs << ", " << shared_spm_size;
    if (effective_stream != "")
      hs << (extern_smem ? "" : ", 0") << ", " << effective_stream;
    hs << ">>>(";

    size_t i = 0;
    for (auto& item : GetDeviceFuncIns(updating_cgi)) {
      auto sname = item.name;
      if (isa<SpannedType>(item.type)) sname += "__device";
      if (!PrefixedWith(scoped_symtab.ScopeName(), GetScope(sname))) continue;
      hs << ((i++ == 0) ? "" : ", ");
      if (ssm.HasHostName(sname))
        hs << ssm.HostName(sname);
      else
        hs << UnScopedName(ssm.DeviceName(sname));
    }
    for (auto item : symbolic_dimensions) {
      hs << ((i++ > 0) ? ", " : "");
      hs << UnScopedName(item.first);
    }

    if (const auto& mri = FCtx(fname).GetDynMemReuseInfo(SSTab().ScopeName()))
      for (const auto& [sto, ie] : mri->infos)
        for (size_t idx = 0; idx < ie.offset_args.size(); ++idx)
          hs << ((i++ > 0) ? ", " : "") << ie.offsets_name << "[" << idx << "]";

    if (deferred_device_pb) {
      auto dpv = deferred_device_pb->BPV()->name;
      hs << ((i++ > 0) ? ", " : "") << dpv;
    }

    hs << ");\n";
  }

  if (!n.IsAsync() && !device_defers_launch) {
    if (effective_stream != "")
      hs << h_indent << "choreo::abend_true(topsStreamSynchronize("
         << effective_stream << "));\n";
    else
      hs << h_indent << "choreo::abend_true(topsDeviceSynchronize());\n";
  }

  if (!device_defers_launch) {
    for (const auto& item : GetChoreoFuncIns(updating_cgi)) {
      if (isa<SpannedType>(item.type)) {
        auto oname = UnScopedName(item.name);
        if (item.attr != ParamAttr::GLOBAL_INPUT && item.IsReference())
          hs << h_indent << "choreo::abend_true(topsMemcpy(" << oname
             << ".data(), " << oname + "__device" << ", "
             << UnScopedSizeExpr(*item.type) << ", topsMemcpyDeviceToHost));\n";
      }
    }
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::DMA& n) {
  TraceEachVisit(n);

  // Currently, DMA in host-side:
  // - will not generate any future.
  // - are performed directly by manipulating pointers.
  // - not support tiling.
  // - not support async.

  // Generate tops dte and choreo::future in device-side
  // When raw_dte_mode is enabled, anonymous sync DMA on GCU300 bypasses
  // choreo::future entirely, using __choreo_dte_pool__[slot] directly.
  bool is_raw_dte = false;
  std::string raw_dte_ctx; // e.g. "__choreo_dte_pool__[0]"
  auto EmitFutureClaim =
      [this, &n, &is_raw_dte, &raw_dte_ctx](
          const std::string& buf_expr, Storage sto = Storage::DEFAULT,
          const std::string& mdata_expr = "") -> std::string {
    if (!n.future.empty() && claimed_dte.count(InScopeName(n.future)))
      return n.future;

    std::string dte_ctx;
    bool use_pool = (use_dte_pool && sto != Storage::SHARED);

    if (use_pool) {
      // GCU300 SDTE pool: all Private-level DMA (anonymous and named)
      // reuse persistent DTE pool slots to avoid resource exhaustion from
      // repeated init/destroy in loops. Each unique named future gets its
      // own stable slot; anonymous DMA shares slot 0.
      int slot = 0;
      if (!n.future.empty()) {
        // Use unscoped name as pool key so that same-named futures in
        // different scopes (e.g., `fd` in tile-0 vs foreach vs tail)
        // share one DTE slot — they are never active simultaneously.
        auto key = n.future;
        auto it = dte_pool_slots.find(key);
        if (it != dte_pool_slots.end()) {
          slot = it->second;
          waited_futures.erase(key);
        } else {
          // DTE merge (-fdte-merge): reuse a waited future's DTE slot,
          // but only if the slot is truly free (no un-waited future uses it).
          int reuse_slot = -1;
          if (dte_merge_mode) {
            // Collect slots that are currently in-flight (have un-waited
            // users). A slot is in-flight only if ALL its users are non-waited;
            // if any user is waited, the non-waited entries are stale.
            std::set<int> in_flight_slots;
            for (const auto& kv : dte_pool_slots) {
              if (!waited_futures.count(kv.first)) {
                bool has_waited_user = false;
                for (const auto& kv2 : dte_pool_slots) {
                  if (kv2.second == kv.second &&
                      waited_futures.count(kv2.first)) {
                    has_waited_user = true;
                    break;
                  }
                }
                if (!has_waited_user) in_flight_slots.insert(kv.second);
              }
            }
            for (const auto& kv : dte_pool_slots) {
              if (waited_futures.count(kv.first) &&
                  !in_flight_slots.count(kv.second)) {
                reuse_slot = kv.second;
                waited_futures.erase(kv.first);
                break;
              }
            }
          }
          if (reuse_slot >= 0) {
            slot = reuse_slot;
          } else {
            slot = dte_pool_size++;
          }
          dte_pool_slots[key] = slot;
          waited_futures.erase(key);
        }
      } else {
        // Anonymous DMA: in merge mode, dynamically find a free slot
        // and track it so named futures can reuse it. Sync DMA completes
        // immediately, so its slot is always available for reuse.
        if (dte_merge_mode) {
          static const std::string anon_key = "__anon__";
          auto it = dte_pool_slots.find(anon_key);
          if (it != dte_pool_slots.end()) {
            // Check if the current anonymous slot is in-flight (used by an
            // un-waited named future). If so, find a new free slot.
            int cur_slot = it->second;
            bool slot_in_flight = false;
            for (const auto& kv : dte_pool_slots) {
              if (kv.first != anon_key && kv.second == cur_slot &&
                  !waited_futures.count(kv.first)) {
                slot_in_flight = true;
                break;
              }
            }
            if (slot_in_flight) {
              // Find a waited future's free slot, or allocate new
              std::set<int> in_flight_slots;
              for (const auto& kv : dte_pool_slots) {
                if (!waited_futures.count(kv.first)) {
                  bool has_waited_user = false;
                  for (const auto& kv2 : dte_pool_slots) {
                    if (kv2.second == kv.second &&
                        waited_futures.count(kv2.first)) {
                      has_waited_user = true;
                      break;
                    }
                  }
                  if (!has_waited_user) in_flight_slots.insert(kv.second);
                }
              }
              int reuse_slot = -1;
              for (const auto& kv : dte_pool_slots) {
                if (waited_futures.count(kv.first) &&
                    !in_flight_slots.count(kv.second)) {
                  reuse_slot = kv.second;
                  break;
                }
              }
              if (reuse_slot >= 0)
                it->second = reuse_slot;
              else
                it->second = dte_pool_size++;
            }
            slot = it->second;
          } else {
            // First anonymous DMA: try to reuse a waited future's slot
            std::set<int> in_flight_slots;
            for (const auto& kv : dte_pool_slots) {
              if (!waited_futures.count(kv.first)) {
                bool has_waited_user = false;
                for (const auto& kv2 : dte_pool_slots) {
                  if (kv2.second == kv.second &&
                      waited_futures.count(kv2.first)) {
                    has_waited_user = true;
                    break;
                  }
                }
                if (!has_waited_user) in_flight_slots.insert(kv.second);
              }
            }
            int reuse_slot = -1;
            for (const auto& kv : dte_pool_slots) {
              if (waited_futures.count(kv.first) &&
                  !in_flight_slots.count(kv.second)) {
                reuse_slot = kv.second;
                break;
              }
            }
            if (reuse_slot >= 0)
              slot = reuse_slot;
            else
              slot = dte_pool_size++;
            dte_pool_slots[anon_key] = slot;
          }
          // Sync DMA completes immediately; mark slot as available
          waited_futures.insert(anon_key);
        } else {
          if (anon_dte_slot < 0) anon_dte_slot = dte_pool_size++;
          slot = anon_dte_slot;
        }
      }
      dte_ctx = "__choreo_dte_pool__[" + std::to_string(slot) + "]";
    } else {
      dte_ctx = GetDTEContextName();
      EmitDTEDecl(ds, d_indent, sto, dte_ctx, false, NeedLevelPred());
    }

    // No-future / raw-dte mode: bypass choreo::future entirely
    bool skip_future = (no_future_mode && use_pool) ||
                       (raw_dte_mode && use_pool && n.future.empty());
    if (skip_future) {
      is_raw_dte = true;
      raw_dte_ctx = dte_ctx;
      if (!n.future.empty()) {
        // Named future: track raw data pointer, event, and DTE slot
        int slot = dte_pool_slots[n.future];
        claimed_dte.emplace(InScopeName(n.future), dte_ctx);
        ds << d_indent << "void* " << n.future << "_data = "
           << (buf_expr.empty() ? "nullptr" : "(void*)" + buf_expr) << ";\n";
        ds << d_indent << "tops::event " << n.future << "_evt;\n";
        ds << d_indent << "int " << n.future << "_slot = " << slot << ";\n";
        nofuture_vars[InScopeName(n.future)] = {n.future + "_data",
                                                n.future + "_evt", slot};
        // Map symbols to raw accessors
        ssm.MapDeviceSymbol(InScopeName(n.future), n.future);
        ssm.MapDeviceSymbol(InScopeName(n.future) + ".data",
                            n.future + "_data");
        if (n.IsSparse())
          ssm.MapDeviceSymbol(InScopeName(n.future) + ".mdata",
                              n.future + "_mdata");
      }
      return n.future.empty() ? dte_ctx : n.future;
    }
    is_raw_dte = false;

    auto future_name = n.future;
    if (future_name.empty()) {
      static size_t future_count = 0;
      future_name = "__choreo_anon_fut__" + std::to_string(future_count++);
    } else {
      claimed_dte.emplace(InScopeName(n.future), dte_ctx);
      ssm.MapDeviceSymbol(InScopeName(n.future), n.future);
      ssm.MapDeviceSymbol(InScopeName(n.future) + ".data",
                          n.future + ".data()");
      if (n.IsSparse())
        ssm.MapDeviceSymbol(InScopeName(n.future) + ".mdata",
                            n.future + ".mdata()");
    }
    ds << d_indent << "choreo::future " << future_name << "(" << dte_ctx
       << ", \"" << n.future << "\", " << n.LOC().begin.line << ", "
       << n.LOC().begin.column;
    if (!buf_expr.empty()) ds << ", " << buf_expr;
    if (!mdata_expr.empty()) ds << ", " << mdata_expr;
    ds << ");\n";

    return future_name;
  };

  auto nty = NodeType(n);
  if (auto ph = dyn_cast<PlaceHolderType>(nty)) {
    assert(ph->GetBaseType() == BaseType::FUTURE);
    // must set the buffer
    auto buf_name = FBInfo().at(InScopeName(n.future)).buffer;

    // Handle placeholder checks that need to postpone after all lv processed
    // Currently, the only case is the plder tied to global buffer

    // assert(ssm.HasDeviceName(buf_name) && "buffer has been defined");
    if (!ssm.HasDeviceName(buf_name)) pld_checklist.push_back(buf_name);

    // dma.any in host-side is of no practical use.
    // It should not be claimed. And there is no future to remap to.
    if (IsHost()) return true;

    // Determine buffer storage so that the DTE pool decision matches
    // the actual DMA that will be assigned later (avoid future/raw mismatch
    // when rotate/swap pairs a dma.any placeholder with a real DMA).
    Storage ph_sto = Storage::DEFAULT;
    auto ph_buf_ty = GetSymbolType(UnScopedName(buf_name));
    if (auto sty = GetSpannedType(ph_buf_ty)) ph_sto = sty->GetStorage();
    EmitFutureClaim(UnScopedName(buf_name), ph_sto);
    // For auto-alloc (DOK_SYMBOL), redirect buffer references through
    // future.data() so that rotate() transparently updates the pointer.
    // For explicit buffer chunks (DOK_CHUNK), the user manages buffers
    // via l_xbuf[idx] directly; remapping would break after rotate()
    // because future.data() no longer points to the array base.
    // For auto-alloc (DOK_SYMBOL), redirect buffer references through
    // future.data() so that rotate() transparently updates the pointer.
    // For explicit buffer arrays (ArrayType), the user manages buffers
    // via l_xbuf[idx] directly; remapping would break after rotate()
    // because future.data() no longer points to the array base.
    auto buf_ty = GetSymbolType(UnScopedName(buf_name));
    if (!isa<ArrayType>(buf_ty)) {
      if (no_future_mode && nofuture_vars.count(InScopeName(n.future)))
        ssm.RemapDeviceSymbol(buf_name, n.future + "_data");
      else
        ssm.RemapDeviceSymbol(buf_name, n.future + ".data()");
    }
    return true;
  }

  assert(isa<AST::ChunkAt>(n.from) && "Unexpected type for DMA's source.");
  assert(isa<AST::ChunkAt>(n.to) && "Unexpected type for DMA's destination.");

  auto fty = dyn_cast<FutureType>(nty);
  assert(fty && "Invalid type of DMA statement!");
  if (fty->IsAsync()) assert(!n.future.empty());

  auto f_ca = cast<AST::ChunkAt>(n.from);
  auto t_ca = cast<AST::ChunkAt>(n.to);
  auto f_sym = f_ca->data->name;
  auto t_sym = t_ca->data->name;
  auto f_idx = f_ca->indices;
  auto t_idx = t_ca->indices;
  auto f_ty = GetSymbolType(f_sym);
  auto t_ty = GetSymbolType(t_sym);
  auto f_sty = GetSpannedType(f_ty);
  auto t_sty = GetSpannedType(t_ty);

  assert(f_sty && "can not retrieve data from 'from'.");
  assert(t_sty && "can not retrieve data from 'to'.");

  Storage f_sto = f_sty->GetStorage();
  Storage t_sto = t_sty->GetStorage();
  Storage dma_sto = f_sto < t_sto ? f_sto : t_sto;

  auto SymbolToSymbol = [f_ca, t_ca]() -> bool {
    return f_ca->NoTilingOperation() && t_ca->NoTilingOperation();
  };
  auto SymbolToTile = [f_ca, t_ca]() -> bool {
    return f_ca->NoTilingOperation() && t_ca->HasTilingOperation();
  };
  auto TileToSymbol = [f_ca, t_ca]() -> bool {
    return f_ca->HasTilingOperation() && t_ca->NoTilingOperation();
  };
  auto TileToTile = [f_ca, t_ca]() -> bool {
    return f_ca->HasTilingOperation() && t_ca->HasTilingOperation();
  };
  auto HasReshape = [f_ca, t_ca]() -> bool {
    return f_ca->HasReshape() || t_ca->HasReshape();
  };

  if (t_sty->GetStorage() == Storage::GLOBAL && use_hetero_tileflow &&
      IsHost()) {
    std::string bts = NameBaseType(t_sty->ElementType(), false);
    auto buf_sym = t_sym + "__device";
    auto buf_sym_from = f_sym + "__device";
    if (n.operation == ".copy") {
      if (SymbolToSymbol()) {
        // direct copy
        hs << h_indent << bts << " * " << buf_sym << " = " << buf_sym_from
           << ";\n";
      } else if (SymbolToTile()) {
        static int s_cnt = 0;
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(t_ca);
        hs << h_indent << "int " << off_name << " = " << offset << ";\n";
        hs << h_indent << bts << " * " << buf_sym << " + " << off_name << " = "
           << buf_sym_from << ";\n";
      } else if (TileToSymbol()) {
        static int s_cnt = 0;
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(f_ca);
        hs << h_indent << "int " << off_name << " = " << offset << ";\n";
        hs << h_indent << bts << " * " << buf_sym << " = " << buf_sym_from
           << " + " << off_name << ""
           << ";\n";
      } else
        choreo_unreachable("not support dual-side chunkat in dma copy");
    } else
      choreo_unreachable("not support host-side dma other than copy");

    return true;
  }

  // TODO: how to do tiling in host-side?
  if ((t_sty->GetStorage() == Storage::GLOBAL ||
       IsChoreoInput(InScopeName(t_sym))) &&
      IsHost()) {
    if (n.IsAsync()) choreo_unreachable("not support host-side async dma yet");
    std::string bts = NameBaseType(t_sty->ElementType(), false);
    std::string buf_sym_from;
    std::string buf_sym;
    std::string tops_dma_kind = "topsMemcpy";
    if (global_buffers.count(f_sym + "__device")) {
      buf_sym_from = f_sym + "__device";
      tops_dma_kind.append("Device");
    } else {
      buf_sym_from = f_sym + ".data()";
      if (f_sty->GetStorage() == Storage::GLOBAL)
        tops_dma_kind.append("Device");
      else
        tops_dma_kind.append("Host");
    }

    if (global_buffers.count(t_sym + "__device")) {
      buf_sym = t_sym + "__device";
      tops_dma_kind.append("ToDevice");
    } else {
      buf_sym = t_sym + ".data()";
      if (f_sty->GetStorage() == Storage::GLOBAL)
        tops_dma_kind.append("ToDevice");
      else
        tops_dma_kind.append("ToHost");
    }

    if (n.operation == ".copy") {
      if (SymbolToSymbol()) {
        // direct copy
        hs << h_indent << "choreo::abend_true(topsMemcpy(" << buf_sym << ", "
           << buf_sym_from << ", " << UnScopedSizeExpr(*f_sty) << ", "
           << tops_dma_kind << "));\n";
      } else
        choreo_unreachable(
            "not support tiling chunkat in dma copy at host side for now");
    } else
      choreo_unreachable("not support host-side dma other than copy");

    return true;
  }

  // TODO: correct?
  if (IsHost()) choreo_unreachable("the dma is not supported in host side!");

  // return {buf_name, buf_expr}
  // `buf_name` is the base buffer name without subscription, used for mds
  // declaration and DTE association.
  // `buf_expr` is the actual expression to access the buffer element, used for
  // future.data() association and async mds offset calculation
  auto GetBufferExpr = [this](const std::string& sym,
                              const ptr<AST::MultiValues> subscription,
                              const ptr<Type>& sym_ty) {
    std::string buf_expr = "";
    std::string sname = InScopeName(sym);
    if (isa<FutureType>(sym_ty) && !IsHostSymbol(sname)) {
      std::string buf_name = sname + ".data";
      buf_expr = ssm.DeviceName(buf_name);
    } else if (isa<FutureType>(sym_ty)
               // This only matches the host-side buffer that is defined in
               // choreo DMA and tied to future but host-side data copy does
               // not really do device-level DMA, and the future is basically
               // a phantom handle do not emit any concrete code at host-side.
               && IsHostSymbol(sname) && !IsChoreoInput(sname) &&
               !IsChoreoOutput(sname)) {
      buf_expr =
          UnScopedName(const_cast<FutureBufferInfo&>(FBInfo())[sname].buffer);
    } else
      buf_expr = ssm.DeviceName(sname);

    std::string buf_name = buf_expr;
    if (subscription != nullptr) {
      if (auto array_ty = dyn_cast<ArrayType>(sym_ty);
          array_ty && CCtx().MemReuse()) {
        // When MemReuse is on, `i` is a flat pointer, so `i[1]` must use
        // pointer arithmetic.  Because the base may be void* (e.g.
        // future.data()), we use byte-level arithmetic with (char*) cast:
        //   (char*)base + idx * sub_byte_size
        std::string array_idx = "";
        auto subscriptions = subscription->AllValues();
        const ValueList& array_sizes = array_ty->Dimensions();
        for (size_t i = 0; i < subscriptions.size(); ++i) {
          if (array_idx.empty())
            array_idx = ExprSTR(subscriptions[i], IsHost());
          else
            array_idx = "(" + array_idx + ")*" + ValueSTR(array_sizes[i]) +
                        "+" + ExprSTR(subscriptions[i], IsHost());
        }
        auto inner_sty = cast<SpannedType>(sym_ty);
        std::string sub_bytes = inner_sty->ByteSizeExpression();
        buf_expr = "((char*)" + buf_expr + " + (" + array_idx + ") * " +
                   sub_bytes + ")";
      } else {
        for (auto expr : subscription->AllValues())
          buf_expr += "[" + ExprSTR(expr, IsHost()) + "]";
      }
    }
    return std::make_pair(buf_name, buf_expr);
  };

  // return mds name and the declaration string.
  // If offset is not empty, means that need to do memory viewing.
  //   Just add offset to buf_expr, then utilize new_shape to decl mdspan.
  auto GenMDSDecl =
      [this](const std::string& buf_name, const std::string& buf_expr,
             const ptr<SpannedType>& sty, const std::string& offset = "",
             const Shape& new_shape =
                 Shape()) -> std::pair<std::string, std::string> {
    static int mds_cnt = 0;
    auto mds_name = "__mds" + std::to_string(mds_cnt++) + "_" +
                    RemoveSuffix(buf_name, ".data()");
    auto bt = sty->ElementType();
    bool split_to_char = false;
    if (split_8byte_dma_transfer && SizeOf(bt) == 8) split_to_char = true;
    std::string bts{split_to_char ? "char" : NameBaseType(bt)};
    std::ostringstream mds_decl;
    mds_decl << d_indent << "tops::mdspan " << mds_name << "("
             << TopsMdsStorage(sty->GetStorage()) << ", (" << bts << "*)"
             << buf_expr;
    if (offset == "")
      mds_decl << ", "
               << ShapeSTR(new_shape.IsValid() ? new_shape : sty->GetShape());
    else {
      mds_decl << " + ";
      if (split_to_char)
        mds_decl << offset << " * 8";
      else
        mds_decl << offset;
      mds_decl << ", " << ShapeSTR(new_shape);
    }
    if (split_to_char) mds_decl << ", 8";
    mds_decl << ");\n";
    return {mds_name, mds_decl.str()};
  };

  const auto f_buf = GetBufferExpr(f_sym, f_idx, f_ty);
  const auto t_buf = GetBufferExpr(t_sym, t_idx, t_ty);

  std::string mdata_expr = "";
  if (n.IsSparse()) {
    static size_t mdata_count = 0;
    auto mdata_sym = "__choreo_mdata__" + std::to_string(mdata_count++);
    if (t_sty->GetStorage() == Storage::SHARED)
      ds << d_indent << "__shared__ uint32_t " << mdata_sym << "[1];\n";
    else
      ds << d_indent << "uint32_t " << mdata_sym << "[1];\n";
    mdata_expr = mdata_sym;
  }

  std::string future_name = n.future;
  bool bind_data = SymbolToSymbol() || TileToSymbol() || TileToTile();
  std::string bound_mdata_expr = (n.IsSparse() && bind_data) ? mdata_expr : "";
  // bind the data to the future
  if (bind_data)
    future_name = EmitFutureClaim(t_buf.second, dma_sto, bound_mdata_expr);
  else
    future_name = EmitFutureClaim("", dma_sto, "");

  std::string event_name;
  // nf_named: true when this DMA targets a named future tracked in nofuture
  // mode.
  // Two cases: (1) EmitFutureClaim just created the tracking (is_raw_dte=true),
  //            (2) future was previously claimed — look it up in nofuture_vars.
  bool nf_named = is_raw_dte && !n.future.empty() && no_future_mode;
  if (!nf_named && no_future_mode && !n.future.empty()) {
    auto sn = InScopeName(n.future);
    if (nofuture_vars.find(sn) != nofuture_vars.end()) {
      nf_named = true;
      is_raw_dte = true;
    }
  }
  if (fty->IsAsync())
    event_name = future_name + (nf_named ? "_evt" : "__event__");

  // DTE context expression for DMA API calls:
  // no-future/raw-dte mode uses pool reference directly; normal mode
  // dereferences future
  std::string dte_ref_expr;
  if (is_raw_dte) {
    dte_ref_expr = nf_named ? ("__choreo_dte_pool__[" + future_name + "_slot]")
                            : raw_dte_ctx;
  } else {
    dte_ref_expr = "*" + future_name + ".get_ctx()";
  }

  // No-future event helpers:
  // event_decl_prefix: prepended before async DMA call
  //   normal mode: "tops::event fA__event__ = " (declares new variable)
  //   no-future:   "fA_evt = "                 (assigns to existing variable)
  std::string event_decl_prefix;
  if (!event_name.empty()) {
    event_decl_prefix =
        nf_named ? (event_name + " = ") : ("tops::event " + event_name + " = ");
  }
  // skip_set_event: whether to skip future_name.set_event() emission
  bool skip_set_event = nf_named;
  auto DMACodeGen = [&]() {
    std::string f_mds_offset = "";
    std::string t_mds_offset = "";
    Shape f_shape = f_sty->GetShape();
    Shape t_shape = t_sty->GetShape();
    const auto& f_buf_name = f_buf.first;
    const auto& f_buf_expr = f_buf.second;
    const auto& t_buf_name = t_buf.first;

    const auto& t_buf_expr = t_buf.second;
    if (auto idx = f_ca->IndexOfLastSpanAs()) {
      f_mds_offset = TileBaseOffset(f_ca);
      f_shape = f_ca->OpAt(*idx)->GetBlockShape();
    }
    if (auto idx = t_ca->IndexOfLastSpanAs()) {
      t_mds_offset = TileBaseOffset(t_ca);
      t_shape = t_ca->OpAt(*idx)->GetBlockShape();
    }

    // When a View operation specifies non-default strides (e.g.
    // buf.view(x, y : P, Q).from(i, j)), the stride {P, Q, ...} changes
    // the memory layout interpretation.  tops::mdspan encodes layout
    // solely through its shape (row-major), so we derive an equivalent
    // shape whose implied row-major strides equal the View strides.
    //
    // For N-dim stride {P0, P1, ..., P_{N-1}} (P_{N-1}==1):
    //   new_shape[k] = P_{k-1} / P_k   for 1 <= k <= N-1
    //   new_shape[0]  kept from original (only bounds-check, not pitch)
    auto ApplyStrideToShape = [this](Shape& shape, const ptr<AST::ChunkAt>& ca,
                                     const ptr<SpannedType>& sty) {
      auto ca_ty = dyn_cast<SpannedType>(ca->GetType());
      if (!ca_ty) return;
      const auto& strides = ca_ty->GetStrides();
      if (strides.empty()) return;
      // Check if strides differ from the default row-major strides of shape
      auto default_strides = sty->GetStrides();
      bool differs = (strides.size() != default_strides.size());
      if (!differs) {
        for (size_t i = 0; i < strides.size(); ++i) {
          if (!sbe::ceq(strides[i], default_strides[i])) {
            differs = true;
            break;
          }
        }
      }
      if (!differs) return;
      // Derive equivalent shape whose row-major strides equal the View
      // strides.  For stride {P0, P1, ..., P_{N-1}} with P_{N-1}=1:
      //   new_shape[k] = P_{k-1} / P_k  for 1 <= k <= N-1
      //   new_shape[0] = original dim-0  (only affects bounds, not pitch)
      //
      // When all strides are compile-time constants we compute exact
      // integer division; otherwise fall back to symbolic division.
      size_t N = strides.size();
      ValueList vl(N);
      vl[0] = shape.ValueAt(0);
      for (size_t k = 1; k < N; ++k) {
        auto pk_1 = strides[k - 1];
        auto pk = strides[k];
        if (auto nv1 = dyn_cast<sbe::NumericValue>(pk_1)) {
          if (auto nv2 = dyn_cast<sbe::NumericValue>(pk)) {
            if (nv2->Value() != 0) {
              vl[k] = sbe::nu(nv1->Value() / nv2->Value());
              continue;
            }
          }
        }
        vl[k] = pk_1 / pk;
      }
      shape = Shape(N, vl);
    };
    ApplyStrideToShape(f_shape, f_ca, f_sty);
    ApplyStrideToShape(t_shape, t_ca, t_sty);

    // For LOCAL/SHARED dest without its own View/SpanAs, use the
    // source ChunkAt type shape (includes View adjustments like
    // .view(block_n)) so the dest mdspan matches the actual DMA
    // transfer size rather than the declared upper-bound buffer size.
    // Only applies to .copy (not .pad/.transp) since shape-modifying
    // operations intentionally change the destination shape.
    if (n.operation == ".copy" && !t_ca->IndexOfLastSpanAs().has_value()) {
      auto t_storage = t_sty->GetStorage();
      if (t_storage == Storage::LOCAL || t_storage == Storage::SHARED) {
        // Only apply when the source ChunkAt type differs from
        // the base symbol type (indicating a View/SpanAs was used).
        auto f_chunkat_shape =
            dyn_cast<SpannedType>(f_ca->GetType())->GetShape();
        auto f_base_shape = f_sty->GetShape();
        bool f_has_view = false;
        for (size_t d = 0; d < f_chunkat_shape.Value().size() &&
                           d < f_base_shape.Value().size();
             ++d) {
          if (!sbe::ceq(f_chunkat_shape.Value()[d], f_base_shape.Value()[d])) {
            f_has_view = true;
            break;
          }
        }
        if (f_has_view &&
            f_chunkat_shape.Value().size() == t_shape.Value().size()) {
          t_shape = f_chunkat_shape;
        }
      }
    }
    // the source actually has a View/SpanAs that changes its shape.
    if (n.operation == ".copy" && !t_ca->IndexOfLastSpanAs().has_value()) {
      auto t_storage = t_sty->GetStorage();
      if (t_storage == Storage::LOCAL || t_storage == Storage::SHARED) {
        // Check if source has a View/SpanAs by comparing shapes
        auto f_base_shape = f_sty->GetShape();
        bool f_has_view = false;
        for (size_t d = 0; d < f_shape.Value().size(); ++d) {
          // View changed the shape if operands differ
          auto fsv = f_shape.Value()[d];
          auto fbv = f_base_shape.Value()[d];
          if (!sbe::ceq(fsv, fbv)) {
            f_has_view = true;
            break;
          }
        }
        if (f_has_view && f_shape.Value().size() == t_shape.Value().size()) {
          t_shape = f_shape;
        }
      }
    }
    const auto f_mds =
        GenMDSDecl(f_buf_name, f_buf_expr, f_sty, f_mds_offset, f_shape);
    const auto t_mds =
        GenMDSDecl(t_buf_name, t_buf_expr, t_sty, t_mds_offset, t_shape);

    std::string f_mds_name{f_mds.first};
    std::string f_mds_decl{f_mds.second};
    std::string t_mds_name{t_mds.first};
    std::string t_mds_decl{t_mds.second};

    ds << f_mds_decl;
    ds << t_mds_decl;

    // handles dma in block, where only single thread in block can
    // operate
    if (NeedLevelPred()) {
      ds << d_indent << LevelPred(Level()) << "{\n";
      IncrDeviceIndent();
    }

    if (n.operation == ".copy") {
      auto LinearCopy = [&]() -> void {
        ds << d_indent;
        if (!event_name.empty()) ds << event_decl_prefix;
        ds << "tops::memcpy" << (fty->IsAsync() ? "_async" : "") << "("
           << dte_ref_expr << ", " << t_mds_name << ", " << f_mds_name
           << ");\n";
        VerboseDMA(ds, d_indent, t_sym, f_sym, "copy", "", 0,
                   ", line " + std::to_string(n.LOC().begin.line));
        if (!event_name.empty())
          if (!skip_set_event)
            ds << d_indent << future_name << ".set_event(" << event_name
               << ");\n";
      };
      auto Deslice = [&]() -> void {
        static int ds_cnt = 0;
        auto off_name = "__deslice_offset" + std::to_string(ds_cnt++) + "__" +
                        t_sym + "_2_" + f_sym;
        auto [offset, offcnt] = GenMdsOffset(t_ca);
        VerboseDMA(ds, d_indent, t_sym, f_sym, "deslice", offset, offcnt,
                   ", line " + std::to_string(n.LOC().begin.line));
        ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
        if (nf_named && fty->IsAsync()) {
          ds << d_indent << dte_ref_expr << ".config_deslice(" << t_mds_name
             << ", " << f_mds_name << ", " << off_name << ");\n";
          ds << d_indent << event_name << " = " << dte_ref_expr
             << ".trigger();\n";
        } else {
          ds << d_indent;
          if (!event_name.empty()) ds << event_decl_prefix;
          ds << "tops::deslice" << (fty->IsAsync() ? "_async" : "") << "("
             << dte_ref_expr << ", " << t_mds_name << ", " << f_mds_name << ", "
             << off_name << ");\n";
          if (!event_name.empty())
            if (!skip_set_event)
              ds << d_indent << future_name << ".set_event(" << event_name
                 << ");\n";
        }
      };
      auto Slice = [&]() -> void {
        static int s_cnt = 0;
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(f_ca);
        VerboseDMA(ds, d_indent, f_sym, t_sym, "slice", offset, offcnt,
                   ", line " + std::to_string(n.LOC().begin.line));
        ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
        if (nf_named && fty->IsAsync()) {
          ds << d_indent << dte_ref_expr << ".config_slice(" << t_mds_name
             << ", " << f_mds_name << ", " << off_name << ", 0);\n";
          ds << d_indent << event_name << " = " << dte_ref_expr
             << ".trigger();\n";
        } else {
          ds << d_indent;
          if (!event_name.empty()) ds << event_decl_prefix;
          ds << "tops::slice" << (fty->IsAsync() ? "_async" : "") << "("
             << dte_ref_expr << ", " << t_mds_name << ", " << f_mds_name << ", "
             << off_name << ");\n";
          if (!event_name.empty())
            if (!skip_set_event)
              ds << d_indent << future_name << ".set_event(" << event_name
                 << ");\n";
        }
      };
      auto SliceDeslice = [&]() -> void {
        static int s_cnt = 0;
        auto s_off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                          f_sym + "_2_" + t_sym;
        auto [s_offset, s_offcnt] = GenMdsOffset(f_ca);

        static int ds_cnt = 0;
        auto ds_off_name = "__deslice_offset" + std::to_string(ds_cnt++) +
                           "__" + t_sym + "_2_" + f_sym;
        auto [ds_offset, ds_offcnt] = GenMdsOffset(t_ca);

        VerboseDMA(ds, d_indent, t_sym, f_sym, "deslice", ds_offset, ds_offcnt,
                   ", line " + std::to_string(n.LOC().begin.line));

        if (nf_named && fty->IsAsync()) {
          ds << d_indent << "int " << s_off_name << "[] = {" << s_offset
             << "};\n";
          ds << d_indent << "int " << ds_off_name << "[] = {" << ds_offset
             << "};\n";
          auto slice_shape_name = "__slice_shape" + std::to_string(s_cnt) +
                                  "__" + f_sym + "_2_" + t_sym;
          ds << d_indent << "unsigned int " << slice_shape_name << "[] = {"
             << ShapeSTR(f_ca->GetBlockShape(), ", ", BaseType::U32) << "};\n";
          ds << d_indent << dte_ref_expr << ".config_slice_deslice("
             << t_mds_name << ", " << f_mds_name << ", " << s_off_name << ", "
             << slice_shape_name << ", " << ds_off_name << ");\n";
          ds << d_indent << event_name << " = " << dte_ref_expr
             << ".trigger();\n";
        } else {
          ds << d_indent << "int " << s_off_name << "[] = {" << s_offset
             << "};\n";
          ds << d_indent << "int " << ds_off_name << "[] = {" << ds_offset
             << "};\n";

          auto slice_shape_name = "__slice_shape" + std::to_string(s_cnt) +
                                  "__" + f_sym + "_2_" + t_sym;
          ds << d_indent << "unsigned int " << slice_shape_name << "[] = {"
             << ShapeSTR(f_ca->GetBlockShape(), ", ", BaseType::U32) << "};\n";
          ds << d_indent;
          if (!event_name.empty()) ds << event_decl_prefix;
          ds << "tops::slice_deslice" << (fty->IsAsync() ? "_async" : "") << "("
             << dte_ref_expr << ", " << t_mds_name << ", " << f_mds_name << ", "
             << s_off_name << ", " << slice_shape_name << ", " << ds_off_name
             << ");\n";

          if (!event_name.empty())
            if (!skip_set_event)
              ds << d_indent << future_name << ".set_event(" << event_name
                 << ");\n";
        }
      };

      if (SymbolToSymbol()) {
        LinearCopy();
      } else if (SymbolToTile()) {
        Deslice();
      } else if (TileToSymbol()) {
        Slice();
      } else if (TileToTile()) {
        SliceDeslice();
      } else {
        choreo_unreachable("unexpected situation.");
      }
    } else if (n.operation == ".pad") {
      static int p_cnt = 0;
      p_cnt++;
      auto pcmvSTR = [&](ptr<AST::MultiValues> mv) -> std::string {
        std::string res;
        for (const auto& v : mv->AllValues()) {
          if (!res.empty()) res += ", ";
          res += ExprSTR(v, IsHost());
        }
        return res;
      };
      auto pad_config = cast<PadConfig>(n.GetConfig());
      auto f_buf_name = RemoveSuffix(f_buf_expr, ".data()");
      std::string pad_low = "__pad_low_" + f_buf_name + std::to_string(p_cnt);
      std::string pad_high = "__pad_high_" + f_buf_name + std::to_string(p_cnt);
      std::string pad_mid = "__pad_mid_" + f_buf_name + std::to_string(p_cnt);
      ds << d_indent << "unsigned int " << pad_low << "[] = {"
         << pcmvSTR(pad_config->pad_low) << "};\n";
      ds << d_indent << "unsigned int " << pad_high << "[] = {"
         << pcmvSTR(pad_config->pad_high) << "};\n";
      ds << d_indent << "unsigned int " << pad_mid << "[] = {"
         << pcmvSTR(pad_config->pad_mid) << "};\n";

      auto Pad = [&]() -> void {
        ds << d_indent;
        if (!event_name.empty()) ds << event_decl_prefix;
        ds << "tops::pad" << (fty->IsAsync() ? "_async" : "") << "("
           << dte_ref_expr << ", " << t_mds_name << ", " << f_mds_name << ", "
           << pad_low << ", " << pad_high << ", " << pad_mid << ", "
           << ExprSTR(pad_config->value, IsHost()) << ");\n";
        // set the device future
        if (!event_name.empty())
          if (!skip_set_event)
            ds << d_indent << future_name << ".set_event(" << event_name
               << ");\n";
      };

      auto SlicePad = [&]() -> void {
        static int s_cnt = 0;
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(f_ca);
        VerboseDMA(ds, d_indent, t_sym, f_sym, "slice+pad", offset, offcnt,
                   ", line " + std::to_string(n.LOC().begin.line));
        ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
        auto slice_shape_name = "__slice_shape" + std::to_string(s_cnt) + "__" +
                                f_sym + "_2_" + t_sym;
        ds << d_indent << "unsigned int " << slice_shape_name << "[] = {"
           << ShapeSTR(f_ca->GetBlockShape(), ", ", BaseType::U32) << "};\n";
        ds << d_indent;
        if (!event_name.empty()) ds << event_decl_prefix;
        ds << "tops::slice_pad" << (fty->IsAsync() ? "_async" : "") << "("
           << dte_ref_expr << ", " << t_mds_name << ", " << f_mds_name << ", "
           << off_name << ", " << slice_shape_name << ", " << pad_low << ", "
           << pad_high << ", " << pad_mid << ", "
           << ExprSTR(pad_config->value, IsHost()) << ");\n";
        // set the device future
        if (!event_name.empty())
          if (!skip_set_event)
            ds << d_indent << future_name << ".set_event(" << event_name
               << ");\n";
      };

      if (SymbolToSymbol()) {
        Pad();
      } else if (TileToSymbol()) {
        SlicePad();
      } else {
        choreo_unreachable(
            "only support dma.pad with (symbol=>symbol), (tile=>symbol).");
      }
    } else if (n.operation == ".transp") {
      static int t_cnt = 0;
      auto transp_config = cast<TransposeConfig>(n.GetConfig());
      auto f_buf_name = RemoveSuffix(f_buf_expr, ".data()");
      auto t_buf_name = RemoveSuffix(t_buf_expr, ".data()");
      auto layout_name =
          "__transpose_layout" + std::to_string(t_cnt++) + "__" + f_buf_name;
      ds << d_indent << "int " << layout_name << "[] = {"
         << DelimitedString(transp_config->dim_values) << "};\n";

      auto Transpose = [&]() -> void {
        ds << d_indent;
        if (!event_name.empty()) ds << event_decl_prefix;
        ds << "tops::transpose" << (fty->IsAsync() ? "_async" : "") << "("
           << dte_ref_expr << ", " << t_mds_name << ", " << f_mds_name << ", "
           << layout_name << ");\n";
        if (!event_name.empty())
          if (!skip_set_event)
            ds << d_indent << future_name << ".set_event(" << event_name
               << ");\n";
      };
      auto SliceTranspose = [&]() -> void {
        static int s_cnt = 0;
        auto off_name = "__slice_offset" + std::to_string(s_cnt++) + "__" +
                        f_sym + "_2_" + t_sym;
        auto [offset, offcnt] = GenMdsOffset(f_ca, n.GetConfig());
        ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
        ds << d_indent;
        if (!event_name.empty()) ds << event_decl_prefix;
        ds << "tops::slice_transpose" << (fty->IsAsync() ? "_async" : "") << "("
           << dte_ref_expr << ", " << t_mds_name << ", " << f_mds_name << ", "
           << off_name << ", " << layout_name << ");\n";
        if (!event_name.empty())
          if (!skip_set_event)
            ds << d_indent << future_name << ".set_event(" << event_name
               << ");\n";
      };
      auto TransposeDeslice = [&]() -> void {
        static int ds_cnt = 0;
        auto off_name = "__deslice_offset" + std::to_string(ds_cnt++) + "__" +
                        t_sym + "_2_" + f_sym;
        auto [offset, offcnt] = GenMdsOffset(t_ca, n.GetConfig());
        ds << d_indent << "int " << off_name << "[] = {" << offset << "};\n";
        ds << d_indent;
        if (!event_name.empty()) ds << event_decl_prefix;
        ds << "tops::transpose_deslice" << (fty->IsAsync() ? "_async" : "")
           << "(" << dte_ref_expr << ", " << t_mds_name << ", " << f_mds_name
           << ", " << layout_name << ", " << off_name << ");\n";
        if (!event_name.empty())
          if (!skip_set_event)
            ds << d_indent << future_name << ".set_event(" << event_name
               << ");\n";
      };

      if (SymbolToSymbol()) {
        Transpose();
      } else if (SymbolToTile()) {
        TransposeDeslice();
      } else if (TileToSymbol()) {
        SliceTranspose();
      } else if (TileToTile()) {
        choreo_unreachable("slice-transpose-deslice is not supported now.");
      }
    }

    if (NeedLevelPred()) {
      DecrDeviceIndent();
      ds << d_indent << "} // single instance\n";
      if (!fty->IsAsync() && dma_sto < Storage::GLOBAL) {
        // not async, must syncthreads immediately
        // else, defer the sync till the wait time
        ds << d_indent << EmitSync(dma_sto) << ";\n";
      }
    }
  };

  DMACodeGen();

  // Mark sync named futures as immediately available for DTE merge reuse.
  // Sync DMA completes immediately, so its DTE slot is free for reuse.
  if (dte_merge_mode && !n.future.empty() && !fty->IsAsync())
    waited_futures.insert(n.future);

  return true;
}

bool TopsccCodeGen::Visit(AST::BufferMap& n) {
  TraceEachVisit(n);

  if (IsHost()) return true;

  auto src_ca = dyn_cast<AST::ChunkAt>(n.source.get());
  if (!src_ca || !src_ca->data) {
    Error1(n.LOC(), "buffer.map/remap requires a valid source buffer.");
    return false;
  }
  auto src_sym = src_ca->data->name;

  auto src_ty = GetSymbolType(src_sym);
  auto src_sty = GetSpannedType(src_ty);
  if (!src_sty) {
    Error1(n.LOC(),
           "buffer.map/remap: could not resolve spanned type for source.");
    return false;
  }

  std::string bts = NameBaseType(src_sty->ElementType(), false);
  std::string src_name = SSMName(src_sym, false);
  std::string res_name = SSMName(n.result, false);

  // Emit offset and size as device-side C++ expressions. Parenthesize so
  // the trailing `* sizeof(...)` byte conversion applies to the whole
  // expression (e.g. an offset of `slide * 2` must not become
  // `slide * 2 * sizeof` -> `slide + 2 * sizeof`).
  std::string offset_str = "(" + ExprSTR(n.offset, false) + ")";
  std::string size_str = "(" + ExprSTR(n.size, false) + ")";

  // Compute the raw global pointer from the source address. This is the
  // address handed to the MMU map/remap API (and any DTE that consumes the
  // buffer directly), distinct from the mapped handle used for vld/st.
  ds << d_indent << bts << "* " << res_name << " = (" << bts << "*)((char*)"
     << src_name << " + " << offset_str << " * sizeof(" << bts << "));\n";

  std::string size_bytes =
      "(int)(" + size_str + " * sizeof(" + bts + "))";

  if (n.IsMap()) {
    ds << d_indent << "mapped_ptr " << res_name << "_mmu = "
       << "tops::map_mem_m((generic_ptr)" << res_name << ", "
       << size_bytes << ");\n";
  } else {
    // Remap: look up the existing mapped handle for the same source.
    auto it = pending_mapped_buffers_.find(src_sym);
    if (it == pending_mapped_buffers_.end()) {
      Error1(n.LOC(),
             "buffer.remap: no existing mapping found for source `" +
                 src_sym + "'.");
      return false;
    }
    auto &existing_name = std::get<0>(it->second);
    auto &old_size_bytes = std::get<2>(it->second);
    ds << d_indent << "mapped_ptr " << res_name << "_mmu = "
       << "tops::remap_mem_m(" << existing_name << "_mmu, "
       << old_size_bytes << ", (generic_ptr)" << res_name << ", "
       << size_bytes << ");\n";
  }

  // vld/st (vector load/store) on GCU can only access L3-mapped addresses, so
  // derive the typed access pointer from the returned mapped_ptr handle rather
  // than the raw global pointer.
  ds << d_indent << bts << "* " << res_name
     << "_l3 = reinterpret_cast<" << bts << "*>(" << res_name << "_mmu);\n";

  // Map .data element accesses on the mapped buffer to the L3 pointer.
  ssm.MapDeviceSymbol(InScopeName(n.result) + ".data", res_name + "_l3");

  // Track this mapping for subsequent remap lookups and scope-exit unmap.
  pending_mapped_buffers_[src_sym] = {n.result, bts, size_bytes};

  return true;
}

bool TopsccCodeGen::Visit(AST::Rotate& n) {
  TraceEachVisit(n);

  if (IsHost())
    choreo_unreachable(
        "rotate is only support in device side(inside parallel-by)!");

  // No-future mode: emit raw pointer/event/slot rotation
  if (no_future_mode) {
    auto& ids = n.GetIds();
    std::vector<std::string> names;
    bool all_nofuture = true;
    for (auto& id : ids) {
      auto sym = cast<AST::Identifier>(id);
      names.push_back(sym->name);
      if (!nofuture_vars.count(InScopeName(sym->name))) all_nofuture = false;
    }
    if (all_nofuture && names.size() >= 2) {
      // Rotate: a <- b <- c <- a (for 3 elements)
      // Implemented as: tmp = a; a = b; b = c; c = tmp;
      auto emit_swap_field = [&](const std::string& suffix) {
        ds << d_indent << "{ auto __tmp = " << names[0] << suffix << "; ";
        for (size_t i = 0; i < names.size() - 1; ++i)
          ds << names[i] << suffix << " = " << names[i + 1] << suffix << "; ";
        ds << names.back() << suffix << " = __tmp; }\n";
      };
      emit_swap_field("_data");
      emit_swap_field("_evt");
      // Always swap slot indices — needed for DTE context tracking after
      // rotation. Forces array pool mode (disables named-dte text replacement).
      emit_swap_field("_slot");
      has_nofuture_rotate = true;
      return true;
    }
  }

  ds << d_indent << "choreo::rotate(";
  int i = 0;
  for (auto& id : n.GetIds()) {
    assert(isa<FutureType>(NodeType(*id)) &&
           "only rotating futures are supported.");
    if (i++ > 0) ds << ", ";
    ds << ExprSTR(id, false);
  }
  ds << ");\n";

  return true;
}

bool TopsccCodeGen::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);

  switch (n.Resource()) {
  case Storage::GLOBAL:
    hs << h_indent << "choreo::abend_true(topsDeviceSynchronize());\n";
    break;
  case Storage::SHARED: ds << d_indent << "__syncthreads();\n"; break;
  case Storage::LOCAL:
    // GCU4+ has __syncsubthreads(); GCU2/GCU3 fall back to __syncthreads().
    if (CCtx().HasTarget() &&
        CCtx().GetTarget().ArchNum(CCtx().GetArch()) >= 400)
      ds << d_indent << "__syncsubthreads();\n";
    else
      ds << d_indent << "__syncthreads();\n";
    break;
  default:
    choreo_unreachable(
        "unsupported synchronization type: " + STR(n.Resource()) + ".");
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Barrier& n) {
  TraceEachVisit(n);

  switch (n.GetLevel()) {
  case ParallelLevel::THREAD:
    // GCU4+ has __syncsubthreads(); GCU2/GCU3 fall back to __syncthreads().
    if (CCtx().HasTarget() &&
        CCtx().GetTarget().ArchNum(CCtx().GetArch()) >= 400)
      ds << d_indent << "__syncsubthreads();\n";
    else
      ds << d_indent << "__syncthreads();\n";
    break;
  case ParallelLevel::GROUP: ds << d_indent << "__syncthreads();\n"; break;
  case ParallelLevel::BLOCK:
    if (!current_pb_is_cooperative) {
      // Inside a non-cooperative parallel scope, degrade block-level barrier
      // to the highest available level (GROUP on gcu400+, THREAD on gcu300).
      if (CCtx().HasTarget() &&
          CCtx().GetTarget().ArchNum(CCtx().GetArch()) >= 400)
        ds << d_indent << "__syncthreads(); // sync.barrier:block degraded\n";
      else
        ds << d_indent << "__syncthreads(); // sync.barrier:block degraded\n";
    } else {
      ds << d_indent << "__syncblocks();\n";
    }
    break;
  default:
    choreo_unreachable("unsupported barrier level: " + STR(n.GetLevel()) + ".");
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Fence& n) {
  TraceEachVisit(n);

  auto memory = n.GetMemory();
  // Default memory scope from visibility level when no explicit <storage>.
  if (memory == Storage::NONE) {
    memory = CCtx().GetTarget().GetDefaultFenceMemory(CCtx().GetArch(),
                                                      n.GetVisibility());
  }

  switch (memory) {
  case Storage::LOCAL:
    ds << d_indent << "tcle::fence<FenceType::L1_VDMEM>();\n";
    break;
  case Storage::SHARED:
    ds << d_indent << "tcle::fence<FenceType::L2_MEM>();\n";
    break;
  case Storage::GLOBAL:
    ds << d_indent << "tcle::fence<FenceType::L3_MEM>();\n";
    break;
  default: choreo_unreachable("unsupported fence memory: " + STR(memory) + ".");
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Wait& n) {
  TraceEachVisit(n);

  for (auto& f : n.GetTargets()) {
    if (!isa<FutureType>(NodeType(*f))) continue;
    assert(cast<AST::Expr>(f)->GetSymbol());
    auto name = cast<AST::Expr>(f)->GetSymbol()->name;
  }

  if (NeedLevelPred()) {
    ds << d_indent << LevelPred(Level()) << "{\n";
    IncrDeviceIndent();
  }

  for (auto& f : n.GetTargets()) {
    auto expr = cast<AST::Expr>(f);
    bool is_array_ref = (expr->op == Op::ElemOf);
    auto fty = is_array_ref
                   ? GetSymbolType(AST::GetArrayBaseSymbol(*expr)->name)
                   : NodeType(*f);
    if (isa<FutureType>(fty)) {
      assert(!IsHost());
      auto fut_sym = cast<AST::Expr>(f)->GetSymbol();
      auto sname = fut_sym ? InScopeName(fut_sym->name) : "";
      auto nf_wait_it =
          no_future_mode ? nofuture_vars.find(sname) : nofuture_vars.end();
      if (nf_wait_it != nofuture_vars.end()) {
        ds << d_indent << "tops::wait(" << nf_wait_it->second.event_var
           << ");\n";
      } else {
        ds << d_indent << ExprSTR(f, false) << ".wait();\n";
      }
      // Track that this future has been waited (DTE is now free for reuse)
      if (fut_sym && !fut_sym->name.empty())
        waited_futures.insert(fut_sym->name);
    } else if (auto ety = dyn_cast<EventArrayType>(fty)) {
      if (IsHost())
        choreo_unreachable("yet to support: wait global event in host.");
      switch (ety->GetStorage()) {
      case Storage::GLOBAL:
      case Storage::SHARED:
      case Storage::LOCAL: {
        ds << d_indent << "// wait event " << PSTR(f) << "\n";
        ds << d_indent << "while (";
        if (is_array_ref) {
          size_t lvl = GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          // TODO: "!" same here
          GenerateSubscriptions(ds, "!" + ExprSTR(f, false), " || ",
                                bty->RemainderDimensions(lvl));
        } else
          GenerateSubscriptions(ds, "!" + ExprSTR(f, false), " || ",
                                ety->RemainderDimensions(0));
        ds << "false) continue;\n";
        ds << d_indent << "// reset event " << PSTR(f) << "\n";
        if (is_array_ref) {
          size_t lvl = GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          GenerateSubscriptions(ds, d_indent + ExprSTR(f, false), " = false;\n",
                                bty->RemainderDimensions(lvl));
        } else
          GenerateSubscriptions(ds, d_indent + ExprSTR(f, false), " = false;\n",
                                ety->RemainderDimensions(0));
      } break;
      default:
        choreo_unreachable("unsupported event array storage '" +
                           STR(ety->GetStorage()) + "'.");
      }
    } else if (auto ety = dyn_cast<EventType>(fty)) {
      if (IsHost())
        choreo_unreachable("yet to support: wait global event in host.");
      switch (ety->GetStorage()) {
      case Storage::GLOBAL:
      case Storage::SHARED:
      case Storage::LOCAL: {
        ds << d_indent << "while (" << ExprSTR(f, false)
           << " == false) continue; // spinlock\n";
        if (is_array_ref) {
          ds << d_indent << "// reset event " << PSTR(f) << "\n";
          size_t lvl = GetSubScriptLevel(*expr);
          auto bid = AST::GetArrayBaseSymbol(*expr);
          auto bty =
              cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
          GenerateSubscriptions(ds, d_indent + ExprSTR(f, false), " = false;\n",
                                bty->RemainderDimensions(lvl));
        } else
          ds << d_indent << ExprSTR(f, false) << " = false; // reset event\n";
      } break;
      default:
        choreo_unreachable("unsupported event storage '" +
                           STR(ety->GetStorage()) + "'.");
      }
    }
  }

  if (NeedLevelPred()) {
    DecrDeviceIndent();
    ds << d_indent << "}\n";
    ds << d_indent << "__syncthreads();\n";
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Break& n) {
  TraceEachVisit(n);
  IndStream() << "break;\n";
  return true;
}

bool TopsccCodeGen::Visit(AST::AsmStmt& n) {
  TraceEachVisit(n);

  auto& tgt = CCtx().GetTarget();
  auto arch = CCtx().GetArch();

  // Collect operands that need temp variables because they are not
  // simple C variable names (e.g., array accesses like lz.at(i)).
  // GCC extended asm requires lvalue variable names, so we auto-
  // lower non-Identifier expressions through temporaries.
  struct TempOp {
    AST::ptr<AST::AsmOperand> op;
    std::string tempName;
    bool isOutput;    // in output operand list
    bool isReadWrite; // "+" constraint prefix
  };
  std::vector<TempOp> tempOps;

  auto CollectTemps = [&](AST::ptr<AST::AsmOperand>& op, bool isOutput) {
    if (!op->expression->GetSymbol()) {
      TempOp t;
      t.op = op;
      t.tempName = symtab.GetAnonName();
      t.isOutput = isOutput;
      t.isReadWrite =
          !op->constraint.empty() && op->constraint[0] == '+';
      tempOps.push_back(t);
    }
  };

  for (auto& op : n.outputOperands) CollectTemps(op, true);
  for (auto& op : n.inputOperands) CollectTemps(op, false);

  // Emit compiler barrier for volatile asm
  if (n.isVolatile) {
    IndStream() << "asm volatile(\"\" ::: \"memory\");\n";
  }

  // Emit temp variable declarations with copy-in from the expression.
  // For output-only operands the initial value is unused but gives
  // the compiler a well-typed declaration.
  for (auto& t : tempOps) {
    IndStream()
        << "auto " << t.tempName << " = "
        << ExprSTR(t.op->expression, false) << ";\n";
  }

  // Helper: get the C variable name to use for an operand (temp or
  // direct identifier).
  auto OpName = [&](AST::ptr<AST::AsmOperand>& op) -> std::string {
    for (auto& t : tempOps)
      if (t.op == op) return t.tempName;
    return tgt.LowerAsmOperand(ExprSTR(op->expression, false));
  };

  IndStream() << (n.isVolatile ? "asm volatile(" : "asm(");
  IndStream() << "\"" << n.templateStr << "\"";

  bool hasOutputs = !n.outputOperands.empty();
  bool hasInputs = !n.inputOperands.empty();
  bool hasClobbers = !n.clobbers.empty();
  bool hasAny = hasOutputs || hasInputs || hasClobbers;

  // Output operands (first colon section)
  if (hasAny) {
    IndStream() << "\n    :";
    if (hasOutputs) {
      IndStream() << " ";
      bool first = true;
      for (auto& op : n.outputOperands) {
        if (!first) IndStream() << ",\n      ";
        first = false;
        IndStream() << "\"" << op->constraint << "\"("
                    << OpName(op) << ")";
      }
    }
  }

  // Input operands (second colon section)
  if (hasInputs || hasClobbers) {
    IndStream() << "\n    :";
    if (hasInputs) {
      IndStream() << " ";
      bool first = true;
      for (auto& op : n.inputOperands) {
        if (!first) IndStream() << ",\n      ";
        first = false;
        IndStream() << "\"" << op->constraint << "\"("
                    << OpName(op) << ")";
      }
    }
  }

  // Clobbers (third colon section)
  if (hasClobbers) {
    IndStream() << "\n    : ";
    bool first = true;
    for (auto& c : n.clobbers) {
      if (!first) IndStream() << ", ";
      first = false;
      IndStream() << "\"" << c << "\"";
    }
  }

  IndStream() << ");\n";

  // Write back temps for output and read-write operands.
  for (auto& t : tempOps) {
    if (t.isOutput) {
      IndStream() << ExprSTR(t.op->expression, false) << " = "
                  << t.tempName << ";\n";
    }
  }

  // Emit compiler barrier after volatile asm
  if (n.isVolatile) {
    IndStream() << "asm volatile(\"\" ::: \"memory\");\n";
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::Continue& n) {
  TraceEachVisit(n);
  IndStream() << "continue;\n";
  return true;
}

bool TopsccCodeGen::Visit(AST::Trigger& n) {
  TraceEachVisit(n);

  for (auto& f : n.GetEvents()) {
    auto expr = cast<AST::Expr>(f);
    bool is_array_ref = (expr->op == Op::ElemOf);
    assert(IsSymbolOrArrayRef(*f) &&
           "expect either symbol or array reference.");
    auto fty = is_array_ref
                   ? GetSymbolType(AST::GetArrayBaseSymbol(*expr)->name)
                   : NodeType(*f);
    if (auto ety = dyn_cast<EventArrayType>(fty)) {
      if (IsHost()) {
        assert(ety->GetStorage() == Storage::GLOBAL);
        // TODO: make & into OpExprSTR?
        hs << h_indent << "choreo::abend_true(topsMemset(&" << ExprSTR(f, true)
           << ", 1, " << ety->ElemCount() << ")); // trigger event\n";
        // TODO: support array reference
      } else {
        switch (ety->GetStorage()) {
        case Storage::GLOBAL:
        case Storage::SHARED:
        case Storage::LOCAL:
          ds << d_indent << "// trigger event " << PSTR(f) << "\n";
          if (is_array_ref) {
            size_t lvl = GetSubScriptLevel(*expr);
            auto bid = AST::GetArrayBaseSymbol(*expr);
            auto bty =
                cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
            GenerateSubscriptions(ds, d_indent + ExprSTR(f, false),
                                  " = true;\n", bty->RemainderDimensions(lvl));
          } else
            GenerateSubscriptions(ds, d_indent + ExprSTR(f, false),
                                  " = true;\n", ety->RemainderDimensions(0));
          break;
        default:
          choreo_unreachable("unsupported event array storage '" +
                             STR(ety->GetStorage()) + "' to trigger.");
          break;
        }
      }
    } else if (auto ety = dyn_cast<EventType>(fty)) {
      if (IsHost()) {
        assert(ety->GetStorage() == Storage::GLOBAL);
        hs << h_indent << "choreo::abend_true(topsMemset(&" << ExprSTR(f, true)
           << ", 1, 1)); // trigger event\n";
        // TODO: support array reference
      } else {
        switch (ety->GetStorage()) {
        case Storage::GLOBAL:
        case Storage::SHARED:
        case Storage::LOCAL:
          if (is_array_ref) {
            ds << d_indent << "// trigger event " << PSTR(f) << "\n";
            size_t lvl = GetSubScriptLevel(*expr);
            auto bid = AST::GetArrayBaseSymbol(*expr);
            auto bty =
                cast<EventArrayType>(GetSymbolType(UnScopedName(bid->name)));
            GenerateSubscriptions(ds, d_indent + ExprSTR(f, false),
                                  " = true; // trigger event\n",
                                  bty->RemainderDimensions(lvl));
          } else
            ds << d_indent << ExprSTR(f, false)
               << " = true; // trigger event\n";
          break;
        default:
          choreo_unreachable("unsupported event array storage '" +
                             STR(ety->GetStorage()) + "' to trigger.");
          break;
        }
      }
    }
  }
  return true;
}

bool TopsccCodeGen::Visit(AST::Call& n) {
  TraceEachVisit(n);

  if (!emit_call) return true;

  auto& os = (IsHost()) ? hs : ds;
  auto& indent = (IsHost()) ? h_indent : d_indent;

  // Track acore:: library usage for auto-include
  if (!n.IsBIF() && PrefixedWith(n.function->name, "acore::"))
    has_acore_call = true;

  // generate the built-in functions
  if (n.IsBIF()) {
    const auto func_name = n.function->name;
    if (func_name == "assert") {
      if (IsHost()) {
        os << indent << "choreo_assert(" << ExprSTR(n.GetArguments().at(0))
           << ", \"" << ExprSTR(n.GetArguments().at(1)) << "\", \""
           << n.LOC().begin.get_filename() << "\", " << n.LOC().begin.get_line()
           << ");\n";
      } else {
        os << indent << "if (!(" << ExprSTR(n.GetArguments().at(0), false)
           << ")) {\n";
        os << indent << "  printf(\"" << n.LOC() << ": choreo assertion abort: "
           << ExprSTR(n.GetArguments().at(1), false) << "\");\n";
        os << indent << "  __co_abort__();\n";
        os << indent << "}\n";
      }
      return true;
    } else if (func_name == "print" || func_name == "println") {
      if (n.CompileTimeEval()) return true;
      std::string print_format;
      print_format += "\"";
      std::string print_args;
      auto GenFormatAndArgsFromValueList = [&](const ValueList& vl) {
        std::string format;
        std::ostringstream oss;
        for (size_t i = 0; i < vl.size(); ++i) {
          if (i != 0) {
            format += ", ";
            oss << ", ";
          }
          format += "%lld";
          oss << "static_cast<long long>(" << ValueSTR(vl[i]) << ")";
        }
        std::string args = UnScopedExpr(oss.str());
        return std::make_pair(format, args);
      };
      auto GenFormatAndArgsFromShape = [&](const Shape& shape) {
        return GenFormatAndArgsFromValueList(shape.Value());
      };
      for (const auto& arg : n.GetArguments()) {
        const auto type = NodeType(*arg);
        auto e = cast<AST::Expr>(arg);
        if (isa<StringType>(type)) {
          print_format += ExprSTR(arg, IsHost());
        } else if (isa<ScalarIntegerType>(type)) {
          print_format += "%lld";
          print_args += ExprCastSTR(arg, std::nullopt, BaseType::S64,
                                    type->GetBaseType(), IsHost());
          print_args += ", ";
        } else if (isa<BooleanType>(type) || isa<EventType>(type)) {
          if (CCtx().GetArch() == "gcu200" || CCtx().GetArch() == "gcu210") {
            print_format += "%d";
            print_args += "(" + ExprSTR(arg, IsHost()) + " ? 1 : 0), ";
          } else {
            print_format += "%s";
            print_args +=
                "(" + ExprSTR(arg, IsHost()) + " ? \"true\" : \"false\"), ";
          }
        } else if (BaseType bt = type->GetBaseType(); IsFloatType(bt)) {
          print_format += "%f";
          print_args +=
              ExprCastSTR(arg, std::nullopt,
                          bt == BaseType::F64 ? BaseType::F64 : BaseType::F32,
                          bt, IsHost());
          print_args += ", ";
        } else if (isa<ITupleType>(type)) {
          print_format += "{";
          auto [format, args] =
              GenFormatAndArgsFromValueList(e->Opts().GetVals());
          print_format += format;
          print_format += "}";
          print_args += args + ", ";
        } else if (isa<MDSpanType>(type)) {
          print_format += "[";
          auto [format, args] = GenFormatAndArgsFromShape(e->s);
          print_format += format;
          print_format += "]";
          print_args += args + ", ";
        } else if (isa<BoundedIntegerType>(type)) {
          choreo_unreachable("All the BoundedIntegerType vars should have been "
                             "normed to BoundedITupleType vars.");
        } else if (isa<BoundedITupleType>(type)) {
          assert(e->Opts().HasVals() &&
                 "BoundedITupleType print arg missing symbolic vals");
          auto [format, args] =
              GenFormatAndArgsFromValueList(e->Opts().GetVals());
          print_format += "{" + format + "}";
          print_args += args + ", ";
        } else if (isa<AddrType>(type)) {
          print_format += "%p";
          print_args += "static_cast<void*>(" + ExprSTR(arg, IsHost()) + "), ";
        } else
          choreo_unreachable(
              "unsupported type for print: " + AST::TYPE_STR(*arg) +
              "\n\targ: " + ExprSTR(arg, IsHost()));
      }
      if (func_name == "println") print_format += "\\n";
      print_format += "\"";
      os << indent << "printf(" << print_format;
      if (auto len = print_args.length(); len > 2) {
        assert(print_args[len - 2] == ',');
        print_args = print_args.substr(0, len - 2); // remove last ", "
        os << ", " << print_args;
      }
      os << ");\n";
      return true;
    } else if (n.IsArith()) {
    } else if (n.IsLibCall()) {
      EmitLibCall(n, func_name, os, indent);
      return true;
    } else
      choreo_unreachable("the bif '" + n.function->name +
                         "' is not supported by this target.");
  }

  if (!n.IsExpr()) os << indent << CallSTR(n) << ";\n";

  return true;
}

bool TopsccCodeGen::Visit(AST::ParamList& n) {
  int index = 0;
  for (auto param : n.values) {
    auto ty = GetSymbolType(param->sym->name);
    if (isa<StreamType>(ty)) continue;
    SSTab().DefineSymbol(param->sym->name, ty);
    updating_cgi.AddSymbolDetail(fname, {InScopeName(param->sym->name),
                                         param->GetType(), param->pass_by_ref,
                                         index++, param->GetAttr()});
  }
  return true;
}

bool TopsccCodeGen::Visit(AST::WithIn& n) {
  TraceEachVisit(n);

  if (n.with)
    ssm.MapDeviceSymbol(InScopeName(n.with->name), "__iv_" + n.with->name);

  assert(n.with_matchers && "expected matchers exist.");

  for (auto& v : n.GetMatchers()) {
    auto id = cast<AST::Identifier>(v);
    ssm.RemapDeviceSymbol(InScopeName(id->name), "__iv_" + id->name);
    ssm.RemapHostSymbol(InScopeName(id->name), "__iv_" + id->name);
    // Keep the device side decl, even for host side iv.
    // for visibility of shapes
    if (IsHost()) {
      hs << h_indent << "int __iv_" << id->name << " = 0;\n";
      updating_cgi.AddSymbolDetail(fname,
                                   {InScopeName(id->name), id->GetType(), true,
                                    -1, ParamAttr::NONE, "", true});
    } else
      ds << d_indent << "int __iv_" << id->name << " = 0;\n";
  }

  if (n.with && (n.GetMatchers().size() == 1)) {
    auto m1 = cast<AST::Identifier>(n.GetMatchers()[0]);
    ssm.RemapDeviceSymbol(InScopeName(n.with->name), "__iv_" + m1->name);
    ssm.RemapHostSymbol(InScopeName(n.with->name), "__iv_" + m1->name);
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);

  // TODO
  choreo_unreachable("where bind is yet to support.");

  return true;
}

bool TopsccCodeGen::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);
  // anything required?
  return true;
}

bool TopsccCodeGen::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);
  cur_loop = n.loop;
  for (auto& rn : n.GetRanges()) {
    auto rng = cast<AST::LoopRange>(rn);
    auto cname = rng->GetRVName();
    for (auto iv_name : within_map.at(InScopeName(cname))) {
      auto iv_ty = GetSymbolType(UnScopedName(iv_name));
      assert(IsActualBoundedIntegerType(iv_ty));
      auto iv_bty = cast<BoundedITupleType>(iv_ty);
      assert(iv_bty);
      auto step = iv_bty->GetStep(0);
      auto width = iv_bty->GetWidth(0);
      int increment = step * width;
      IndStream() << "for (" << SSMName(iv_name, IsHost()) << " = "
                  << (rng->lbound ? ("(" + ExprSTR(rng->lbound, IsHost()) + ")")
                                  : "0")
                  << "; " << SSMName(iv_name, IsHost()) << " < "
                  << UnScopedExpr(ValueSTR(iv_bty->GetUpperBound()))
                  << (rng->ubound ? (" + " + ExprSTR(rng->ubound, IsHost()))
                                  : "")
                  << "; "
                  << (increment != 1 ? (SSMName(iv_name, IsHost()) +
                                        " += " + std::to_string(increment))
                                     : ("++" + SSMName(iv_name, IsHost())))
                  << ") {\n";

      IncrIndent();
      // if this loop needs vectorization, remap the iv to vec_iv
      if (cur_loop && cur_loop->CanVectorize()) {
        std::string vector_iv_name = "__vec_iv_" + cname;
        std::string scalar_iv_name = ssm.DeviceName(iv_name);
        ssm.RemapDeviceSymbol(iv_name, vector_iv_name);
        ssm.RemapHostSymbol(iv_name, vector_iv_name);
      }
    }
  }

  return true;
}

bool TopsccCodeGen::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  assert(!IsHost());
  PushEmittedNames();
  ds << d_indent << "// inthreads: " << n.LOC() << "\n";
  if (!n.stmts->None())
    ds << d_indent << "if (" << ExprSTR(n.pred, false) << ") {\n";
  IncrDeviceIndent();
  return true;
}

bool TopsccCodeGen::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  PushEmittedNames();
  IndStream() << "// if-else: " << n.LOC() << "\n";
  auto pred = n.GetPred();
  if (!pred->GetDiversityShape().Varying()) {
    if (auto c = dyn_cast<AST::Call>(n.pred->GetReference()))
      IndStream() << "if (" << CallSTR(*c) << ") {\n";
    else
      IndStream() << "if (" << ExprSTR(n.pred, IsHost()) << ") {\n";
    IncrIndent();
  }
  emit_call = true;
  return true;
}

bool TopsccCodeGen::Visit(AST::WhileBlock& n) {
  TraceEachVisit(n);

  IndStream() << "// while: " << n.LOC() << "\n";
  IndStream() << "while (" << ExprSTR(n.pred, IsHost()) << ") {\n";
  IncrIndent();

  return true;
}

bool TopsccCodeGen::Visit(AST::Return& n) {
  TraceEachVisit(n);

  return_stream.str("");

  auto vty = NodeType(*n.value);

  if (isa<ScalarType>(vty)) {
    return_stream << "return " << ExprSTR(n.value, true) << ";\n";
  } else if (auto sty = dyn_cast<SpannedType>(vty)) {
    if (auto id = AST::GetIdentifier(*n.value)) {
      auto sym = id->name;
      if (IsChoreoInput(InScopeName(sym))) {
        // return the global storage, must map back
        hs << h_indent << "choreo::abend_true(topsMemcpy(" << sym << ".data(), "
           << sym << "__device, " << UnScopedSizeExpr(*sty)
           << ", topsMemcpyDeviceToHost));\n";
        return_stream << "return choreo::copy_as_spanned(" << sym << ".data(), "
                      << sym << ".shape());\n";
      } else if (IsChoreoOutput(InScopeName(sym))) {
        // return the global storage, must map back
        hs << h_indent << "choreo::abend_true(topsMemcpy(" << sym << ".data(), "
           << sym << "__device, " << UnScopedSizeExpr(*sty)
           << ", topsMemcpyDeviceToHost));\n";
        return_stream << "return " << sym << ";\n";
      } else {
        choreo_unreachable("unexpected situation");
      }
    } else if (auto expr = cast<AST::Expr>(n.value);
               expr && (expr->op == Op::DataOf || expr->op == Op::MDataOf)) {
      // return future.data, must map back
      auto id = cast<AST::Expr>(expr->GetR())->GetSymbol();
      assert(id && "expect a symbol");
      auto sym = id->name + "__buf__";
      hs << h_indent << "choreo::abend_true(topsMemcpy(" << sym << ".data(), "
         << sym << "__device, " << UnScopedSizeExpr(*sty)
         << ", topsMemcpyDeviceToHost));\n";
      return_stream << "return " << ExprSTR(n.value, true) << ";\n";
    } else {
      choreo_unreachable("not support return value of type: " + PSTR(vty));
    }
  } else {
    choreo_unreachable("not support return value of type: " + PSTR(vty));
  }

  EmitTopsFree();

  hs << h_indent << return_stream.str();

  return true;
}

bool TopsccCodeGen::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);

  if (n.kind == AST::CppSourceCode::Inline) {
    Stream() << n.GetCode();
  } else {
    CodeSegment cur_cs =
        (n.kind == AST::CppSourceCode::Host) ? CS_USER : CS_COK;
    if (cur_cs != cs) {
      code_segments.push_back("");
      segment_tags.push_back(cur_cs);
    }

    code_segments.back() += n.GetCode();
  }

  return true;
}

void TopsccCodeGen::EmitHostFuncDecl(std::ostringstream& oss) {
  // handle the return type
  if (!void_return) {
    if (cgi.HasReturnSymbol(fname)) {
      auto& item = cgi.GetReturnDetail(fname);
      if (item.rty_str != "$")
        oss << item.rty_str;
      else
        oss << HostTypeStringify(*fty->out_ty, true);
    } else {
      oss << HostTypeStringify(*fty->out_ty, true);
    }
  } else
    oss << "void";
  oss << " " << fname << "(";

  // emit the parameters
  size_t host_pindex = 0;
  for (auto& item : GetChoreoFuncIns(cgi)) {
    if (item.IsParameter()) assert((int)host_pindex == item.p_index);
    oss << ((host_pindex == 0) ? "" : ", ")
        << HostTypeStringify(*item.type, false, item.IsReference()) << " "
        << item.host_name;
    ++host_pindex;
  }
  oss << ")";

  VST_DEBUG(dbgs() << "Host function prototype:\n" << oss.str() << "\n");
}

void TopsccCodeGen::EmitHostRuntimeCheck() {
  if (CCtx().DisableRuntimeCheck()) return;
  // check if the input shape is as declared in choreo
  if (cgi.ParameterCount(fname) == 0) return;

  struct Entry {
    size_t para_ordinal;
    size_t dim;
    std::string elem_name;
  };
  std::map<std::string, std::vector<Entry>> ve_entries_map;

  size_t host_pindex = 0;
  for (const auto& item : GetChoreoFuncIns(cgi)) {
    assert((int)host_pindex == item.p_index);
    auto name = UnScopedName(item.name);
    if (auto sty = dyn_cast<SpannedType>(item.type)) {
      size_t dim_count = 0;
      for (auto vi : sty->GetShape().Value()) {
        auto elem_name = name + ".shape()[" + std::to_string(dim_count) + "]";
        if (auto vale = VIInt(vi)) {
          hs << h_indent << "choreo::runtime_check(" << elem_name
             << " == " << *vale;
          hs << ", \"shape inconsistent on the " << Ordinal(host_pindex + 1)
             << " parameter (\'" << name << "\', dim: " << dim_count
             << "): expect: " << *vale << ", but got \" + std::to_string("
             << elem_name << ") + \".\");\n";
        } else if (VIIsNil(vi)) {
          hs << h_indent << "choreo::runtime_check(" << elem_name
             << " == choreo::__inf__, \"must set 'choreo::__inf__' to the "
                "unbounded dimension on the "
             << Ordinal(host_pindex + 1) << " parameter (\'" << name
             << "\', dim: " << dim_count << "): got \" + std::to_string("
             << elem_name << ") + \".\");\n";
        } else if (auto vale = VISym(vi))
          ve_entries_map[*vale].push_back(
              {host_pindex + 1, dim_count, elem_name});
        dim_count++;
      }
    }
    host_pindex++;
  }

  // Check if the named dims meet the constraint. Eg.
  //
  //   __co__ void foo(f32 [M, N] a, f32 [N, K] b)
  //
  // then a.shape()[1] should be equal to b.shape()[0]
  for (const auto& [_, entries] : ve_entries_map) {
    for (size_t i = 1; i < entries.size(); ++i) {
      auto& entry0 = entries[i - 1];
      auto& entry1 = entries[i];
      hs << h_indent << "choreo::runtime_check(" << entry0.elem_name
         << " == " << entry1.elem_name;
      hs << ", \"The shapes of the " << Ordinal(entry0.para_ordinal)
         << " parameter (dim: " << entry0.dim << ") and the "
         << Ordinal(entry1.para_ordinal) << " parameter (dim: " << entry1.dim
         << ") are inconsistent.\");\n";
    }
  }

  for (const auto& rc : FCtx(fname).GetRtChecks()) {
    hs << h_indent << "choreo::runtime_check(" << ValueSTR(sbe::sym(rc.lhs))
       << " " << rc.op << " " << ValueSTR(sbe::sym(rc.rhs)) << ", \""
       << rc.message << ", " << rc.loc << "\");\n";
  }

  for (const auto& ar : FCtx(fname).GetAssertions(AssessType::ENTRY)) {
    if (!ar.enabled) continue;
    hs << h_indent << "choreo::runtime_check(" << ValueSTR(ar.expr, true)
       << ", \"" << ar.message << ", " << ar.loc << "\");\n";
  }

  // USE_SITE and DEF_SITE assertions are emitted in device code (inside the
  // kernel) via EmitSiteAssertions, which is called from AfterVisitImpl during
  // the AST traversal.  They depend on iteration-varying or locally-redefined
  // values that only exist on the device side.
}

void TopsccCodeGen::EmitMemReuse(const std::string& df_name) {
  const auto& mri = FCtx(fname).GetDynMemReuseInfo(df_name);
  if (!mri) return;
  hs << h_indent << R"(// JIT memory reuse begin)" << "\n";
  for (const auto& [sto, ie] : mri->infos) {
    hs << h_indent << "HeapSimulator::Chunks " << ie.chunks_name << ";\n";
    for (const auto& c : ie.chunks)
      hs << h_indent << ie.chunks_name << ".push_back(" << c << ");\n";
  }
  for (const auto& [sto, ie] : mri->infos) {
    hs << h_indent << "HeapSimulator " << ie.simulator << ";\n";
    size_t align = ie.alignment ? ie.alignment : 512;
    if (ie.n_buffers > 0 && !ie.interference.empty()) {
      std::string imat_name = ie.chunks_name + "_imat";
      hs << h_indent << "std::vector<bool> " << imat_name << " = {";
      for (size_t k = 0; k < ie.interference.size(); ++k) {
        if (k) hs << ",";
        hs << (ie.interference[k] ? "true" : "false");
      }
      hs << "};\n";
      hs << h_indent << "HeapSimulator::Result " << ie.result << " = "
         << ie.simulator << ".Allocate(" << ie.chunks_name << ", " << align
         << ", " << imat_name << ");\n";
    } else {
      hs << h_indent << "HeapSimulator::Result " << ie.result << " = "
         << ie.simulator << ".Allocate(" << ie.chunks_name << ", " << align
         << ");\n";
    }
    hs << h_indent << "unsigned " << ie.spm_size << " = " << ie.result
       << ".heap_size;\n";
    // special host runtime check
    std::string mem_capacity = std::to_string(CCtx().GetMemCapacity(sto));
    if (!CCtx().DisableRuntimeCheck())
      hs << h_indent << "choreo::runtime_check(" << ie.spm_size
         << " <= (size_t)" << mem_capacity
         << ", \"In the memory reuse of dynamic shapes"
         << ", the size of the initial " << STR(sto)
         << " spm should not exceed the memory usage limit " << mem_capacity
         << " bytes.\");\n";
    hs << h_indent << "unsigned long " << ie.offsets_name << "["
       << mri->infos[sto].offset_args.size() << "];"
       << "\n";
    std::string idx = ie.chunks_name + "_idx";
    hs << h_indent << "size_t " << idx << " = 0;\n";
    hs << h_indent << "for (const auto& [buffer_id, offset] : " << ie.result
       << ".chunk_offsets)\n";
    hs << h_indent << "  " << ie.offsets_name << "[" << idx
       << "++] = offset;\n";
    // --- memory reuse diagnostics ---
    VST_DEBUG(
        hs << h_indent
           << "printf(\"[mem-reuse] heap_size = %u bytes (%.1f KB), "
           << "capacity = " << mem_capacity << " bytes (%.1f KB), "
           << "usage = %.1f%%\\n\", " << ie.spm_size << ", " << ie.spm_size
           << " / 1024.0, " << mem_capacity << " / 1024.0, " << ie.spm_size
           << " * 100.0 / " << mem_capacity << ");\n";
        hs << h_indent << "printf(\"[mem-reuse] %zu chunks allocated:\\n\", "
           << ie.chunks_name << ".size());\n";
        hs << h_indent << "{\n"; hs << h_indent << "  size_t __mr_i = 0;\n";
        hs << h_indent << "  for (const auto& __mr_c : " << ie.chunks_name
           << ") {\n";
        hs << h_indent << "    printf(\"[mem-reuse]   [%zu] %-60s  size=%8zu  "
           << "offset=%8lu\\n\",\n";
        hs << h_indent
           << "           __mr_i, __mr_c.buffer_id.c_str(), __mr_c.size, "
           << ie.offsets_name << "[__mr_i]);\n";
        hs << h_indent << "    __mr_i++;\n"; hs << h_indent << "  }\n";
        hs << h_indent << "}\n";);
    // --- end memory reuse diagnostics ---
  }
  hs << h_indent << R"(// JIT memory reuse end)" << "\n";
}

static inline const std::string
DeviceParamTypeStringify(const Choreo::Type& ty) {
  if (isa<VoidType>(&ty))
    return "void";
  else if (isa<S8Type>(&ty))
    return "char";
  else if (isa<U8Type>(&ty))
    return "unsigned char";
  else if (isa<S16Type>(&ty))
    return "short";
  else if (isa<U16Type>(&ty))
    return "unsigned short";
  else if (isa<S32Type>(&ty))
    return "int";
  else if (isa<U32Type>(&ty))
    return "unsigned int";
  else if (isa<S64Type>(&ty))
    return "int64_t";
  else if (isa<U64Type>(&ty))
    return "uint64_t";
  else if (isa<BooleanType>(&ty))
    return "bool";
  else if (isa<FloatE4M3Type>(&ty))
    return "choreo::float_e4m3_t";
  else if (isa<FloatE5M2Type>(&ty))
    return "choreo::float_e5m2_t";
  else if (isa<F16Type>(&ty))
    return "choreo::half";
  else if (isa<BF16Type>(&ty))
    return "choreo::bfp16";
  else if (isa<F32Type>(&ty))
    return "float";
  else if (isa<F64Type>(&ty))
    return "double";
  else if (isa<EventArrayType>(&ty))
    return "bool *";
  else if (isa<EventType>(&ty))
    return "bool"; // use bool for event
  else if (auto sty = dyn_cast<SpannedType>(&ty))
    return std::string(NameBaseType(sty->ElementType(), false)) + " *";
  else if (auto bitt = dyn_cast<BoundedITupleType>(&ty)) {
    // There should have no BoundedIntegerType.
    // They have been normalized to BoundedITupleType.
    assert(bitt->Dims() == 1);
    (void)bitt;
    return "int";
  } else
    choreo_unreachable("unsupported host function type: " + STR(ty) + ".");
  return "";
}

void TopsccCodeGen::EmitTopsFree() {
  assert(IsHost());
  if (FCtx(fname).HasDeviceParallel()) return;
  for (const auto& item : GetDeviceFuncIns(updating_cgi)) {
    if (!PrefixedWith(scoped_symtab.ScopeName(), GetScope(item.name))) continue;
    if (!isa<SpannedType>(item.type)) continue;
    if (item.attr == ParamAttr::GLOBAL_INPUT) continue;
    if (!NeedDeviceFunc() && !IsChoreoOutput(item.name)) continue;
    hs << h_indent << "choreo::abend_true(topsFree(" << UnScopedName(item.name)
       << "__device));\n";
  }
}

void TopsccCodeGen::EmitDeviceFuncDecl(std::ostringstream& oss) {
  if (TargetHasLevel(ParallelLevel::GROUP)) {
    auto& lconfig = cgi.GetFunctionLaunches(fname)[parallel_idx];
    oss << "__thread_dims__(" << lconfig.thread_count.x << ", "
        << lconfig.thread_count.y << ", " << lconfig.thread_count.z << ")\n";
  }

  oss << (current_pb_is_cooperative ? "__cooperative__ " : "")
      << "__global__ void " << device_fn << "(";

  size_t index = 0;
  for (auto& item : GetDeviceFuncIns(updating_cgi)) {
    if (!PrefixedWith(scoped_symtab.ScopeName(), GetScope(item.name))) continue;
    auto dname = (item.need_iv_prefix ? "__iv_" : "") + UnScopedName(item.name);
    if (index++ > 0) oss << ", ";
    oss << DeviceParamTypeStringify(*item.type) << " " << dname;
    ssm.MapDeviceSymbolIfNotExist(item.name, dname);
  }

  for (auto item : symbolic_dimensions) {
    oss << ((index++ > 0) ? ", unsigned " : "unsigned ");
    oss << UnScopedName(item.first);
    ssm.MapDeviceSymbolIfNotExist(item.first, UnScopedName(item.first));
  }

  if (const auto& mri = FCtx(fname).GetDynMemReuseInfo(SSTab().ScopeName()))
    for (const auto& [sto, ie] : mri->infos)
      for (size_t idx = 0; idx < ie.offset_args.size(); ++idx) {
        auto dname = RegexReplaceAll(ie.offset_args[idx], "::", "_");
        oss << ((index++ > 0) ? ", " : "") << "unsigned long " << dname;
      }

  // Pass device parallel variable as a kernel parameter so the kernel
  // knows which device partition it is operating on.
  if (deferred_device_pb) {
    auto dpv = deferred_device_pb->BPV()->name;
    auto dev_param = "__device_id_" + dpv;
    oss << ((index++ > 0) ? ", " : "") << "int " << dev_param;
  }

  oss << ")";

  VST_DEBUG(dbgs() << "Device function prototype:\n" << oss.str() << "\n");
}

// EmitLibCall is defined in lower_libcall.cpp for maintainability.

// ============================================================================
// MMA codegen for GCU acore/VACC
// ============================================================================

std::string TopsccCodeGen::ResolveFragAddr(const AST::ptr<AST::Expr>& frag) {
  auto sym = AST::FragName(frag);
  auto scoped = InScopeName(sym);
  if (acore_mma.frag_addr.count(scoped)) return acore_mma.frag_addr.at(scoped);
  if (ssm.HasDeviceName(scoped)) return ssm.DeviceName(scoped);
  return sym;
}

bool TopsccCodeGen::Visit(AST::MMA& n) {
  TraceEachVisit(n);

  auto& op = *n.GetOperation();
  auto& indent = d_indent;

  switch (op.Tag()) {
  case AST::MMAOperation::Fill: {
    auto frag = op.FillingTo();
    auto frag_sym = AST::FragName(frag);
    auto scoped = InScopeName(frag_sym);

    if (!FCtx(fname).FragIsUKERNEL(scoped)) break;

    AcoreAccumState state;
    state.first_exec = true;
    state.vab_off = acore_mma.next_vab_off;
    state.ws_name = "__mma_ws_" + frag_sym;

    auto frag_ty = frag->GetType();
    if (auto sty = dyn_cast<SpannedType>(frag_ty)) {
      auto shape = sty->GetShape();
      if (shape.Rank() >= 2) {
        auto m_vi = shape.Value()[0];
        auto n_vi = shape.Value()[1];
        if (auto mv = VIInt(m_vi))
          if (auto nv = VIInt(n_vi)) {
            int vacc_usage = AcoreVACCUsage(*mv, *nv);
            int aligned =
                ((vacc_usage + ACORE_VAB_ALIGN - 1) / ACORE_VAB_ALIGN) *
                ACORE_VAB_ALIGN;
            acore_mma.next_vab_off += aligned;
          }
      }
    }

    acore_mma.accum_states[scoped] = state;

    ds << indent << "int " << state.ws_name << "[" << ACORE_WS_SIZE << "];\n";
  } break;

  case AST::MMAOperation::Load:
  case AST::MMAOperation::LoadR: {
    // On GCU, mma.load records the source buffer address for use at exec time.
    auto ld_to = op.LoadTo();
    if (!ld_to) break;
    auto frag_sym = AST::FragName(ld_to);
    auto scoped = InScopeName(frag_sym);

    auto ld_from = op.LoadFrom();
    if (!ld_from) break;

    auto ref_sym = ld_from->RefSymbol();
    auto scoped_ref = InScopeName(ref_sym);
    std::string buf_name;
    if (ssm.HasDeviceName(scoped_ref))
      buf_name = ssm.DeviceName(scoped_ref);
    else
      buf_name = ref_sym;

    auto offset = GenOffset(ld_from);
    if (!offset.empty())
      acore_mma.frag_addr[scoped] = "(" + buf_name + " + (" + offset + "))";
    else
      acore_mma.frag_addr[scoped] = buf_name;
  } break;

  case AST::MMAOperation::Exec: {
    auto c_frag = op.ExecOperand(0);
    auto a_frag = op.ExecOperand(1);
    auto b_frag = op.ExecOperand(2);

    auto c_sym = AST::FragName(c_frag);
    auto scoped_c = InScopeName(c_sym);

    if (!FCtx(fname).FragIsUKERNEL(scoped_c)) break;

    auto& state = acore_mma.accum_states[scoped_c];

    auto a_sty = GetSpannedType(a_frag->GetType());
    auto b_sty = GetSpannedType(b_frag->GetType());
    if (!a_sty || !b_sty)
      choreo_unreachable("MMA exec operands must have spanned types.");

    auto a_shape = a_sty->GetShape();
    auto b_shape = b_sty->GetShape();
    auto method = op.GetMethod();

    int static_M = 0;
    if (a_shape.Rank() >= 1)
      if (auto mv = VIInt(a_shape.Value()[0])) static_M = (int)*mv;

    if (static_M == 0)
      Error1(n.LOC(), "MMA on GCU requires statically known M dimension.");

    state.static_M = static_M;
    state.elem_type = a_sty->ElementType();
    state.method = method;

    // Determine output type from the accumulator fragment
    auto c_sty = GetSpannedType(c_frag->GetType());
    if (c_sty)
      state.out_type = c_sty->ElementType();
    else
      state.out_type = state.elem_type;

    auto stub = acore_mma.GetOrEmitStub(static_M, state.elem_type,
                                        state.out_type, method);
    has_acore_call = true;
    state.stub_name = stub;

    std::string a_addr = ResolveFragAddr(a_frag);
    std::string b_addr = ResolveFragAddr(b_frag);

    std::string k_dim, n_dim;
    if (method == AST::MMAOperation::ROW_COL) {
      k_dim = (a_shape.Rank() >= 2) ? STR(a_shape.Value()[1]) : "0";
      n_dim = (b_shape.Rank() >= 2) ? STR(b_shape.Value()[1]) : "0";
    } else {
      k_dim = (a_shape.Rank() >= 2) ? STR(a_shape.Value()[1]) : "0";
      n_dim = (b_shape.Rank() >= 2) ? STR(b_shape.Value()[0]) : "0";
    }

    int acc_flag = state.first_exec ? 0 : 1;
    int lt_flag = state.first_exec ? 0 : 1;

    // Flush any previously pending exec before recording a new one.
    // In a K-loop, each exec except the last emits with store_flag=0.
    if (state.pending.valid) {
      auto in_ptr = AcoreMMACodeGenState::PtrTypeStr(state.elem_type);
      auto out_ptr = AcoreMMACodeGenState::PtrTypeStr(state.out_type);
      ds << indent << state.stub_name << "(\n"
         << indent << "    (" << out_ptr << "*)" << a_addr << ",\n"
         << indent << "    (" << in_ptr << "*)" << state.pending.a_addr << ",\n"
         << indent << "    (" << in_ptr << "*)" << state.pending.b_addr << ",\n"
         << indent << "    " << state.ws_name << ",\n"
         << indent << "    " << state.pending.k_dim << ", "
         << state.pending.n_dim << ", " << state.pending.acc_flag << ", 0, "
         << state.vab_off << ", " << state.pending.lt_flag << ");\n";
    }

    // Record this exec as pending; it will be emitted by mma.store
    // (with store_flag=1) or by the next mma.exec (with store_flag=0).
    state.pending = {a_addr, b_addr, k_dim, n_dim, acc_flag, lt_flag, true};

    state.first_exec = false;
  } break;

  case AST::MMAOperation::Store: {
    auto frag = op.StoreFrom();
    auto frag_sym = AST::FragName(frag);
    auto scoped = InScopeName(frag_sym);

    if (!FCtx(fname).FragIsUKERNEL(scoped)) break;

    auto& state = acore_mma.accum_states[scoped];

    if (!state.pending.valid)
      Error1(n.LOC(), "mma.store without prior mma.exec on GCU target.");

    auto dest = op.StoreTo();

    // Resolve store destination address
    auto dest_ref = dest->RefSymbol();
    auto scoped_dest = InScopeName(dest_ref);
    std::string dest_name;
    if (ssm.HasDeviceName(scoped_dest))
      dest_name = ssm.DeviceName(scoped_dest);
    else
      dest_name = dest_ref;

    auto dest_offset = GenOffset(dest);
    std::string out_addr;
    if (!dest_offset.empty())
      out_addr = "(" + dest_name + " + (" + dest_offset + "))";
    else
      out_addr = dest_name;

    // Emit the pending exec with store_flag=1: compute + store in one call
    auto in_ptr = AcoreMMACodeGenState::PtrTypeStr(state.elem_type);
    auto out_ptr = AcoreMMACodeGenState::PtrTypeStr(state.out_type);
    ds << indent << state.stub_name << "(\n"
       << indent << "    (" << out_ptr << "*)" << out_addr << ",\n"
       << indent << "    (" << in_ptr << "*)" << state.pending.a_addr << ",\n"
       << indent << "    (" << in_ptr << "*)" << state.pending.b_addr << ",\n"
       << indent << "    " << state.ws_name << ",\n"
       << indent << "    " << state.pending.k_dim << ", " << state.pending.n_dim
       << ", " << state.pending.acc_flag << ", 1, " << state.vab_off << ", "
       << state.pending.lt_flag << ");\n";

    state.pending.valid = false;
  } break;

  case AST::MMAOperation::Commit:
  case AST::MMAOperation::Scale: break;

  default: break;
  }

  return true;
}

void TopsccCodeGen::EmitSource() {
  bool suppress_main = CCtx().DeviceOnly();
  if (suppress_main) {
    auto name = RemoveDirectoryPrefix(
        RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));
    outs() << "#define main __choreo_lib_suppressed_main_" << name << "\n";
  }
  for (size_t i = 0; i < code_segments.size(); ++i) {
    if (EnableLineDirective())
      outs() << PinLineDirectivePerGeneratedLine(code_segments[i]) << "\n";
    else
      outs() << code_segments[i] << "\n";
  }
  if (suppress_main) outs() << "#undef main\n";
}

void TopsccCodeGen::EmitScript(std::ostream& os, const std::string& exe_fn) {
  TopsccDeviceCodeGen dcg_instance;

  auto filename = RemoveDirectoryPrefix(
      RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));
  os << R"script(#!/usr/bin/env bash

# This is the choreo generated bash script to compile topscc code

if [[ -z ${TOPSCC_INSTALL} ]]; then
  if [[ \"$1\" == \"-st\" ]]; then
    TOPSCC_INSTALL=/opt/tops;
    shift 1;
  fi
)script";

  if (!use_system_toolchain)
    os << "  TOPSCC_INSTALL=" << STRINGIZE(__CHOREO_TOPSCC_DIR__) << "\n";

  os << "fi\n\n";

  // Pre-set gcu_arch from compile-time -arch flag so that SetupBuildEnv()
  // skips JIT device detection (and TOPS_VISIBLE_DEVICES side effects).
  if (CCtx().GetArch() == "gcu400" || CCtx().GetArch() == "gcu500")
    os << "export GCU_ARCH=" << ToLower(CCtx().GetArch()) << "\n";
  else if (CCtx().IsArchSet())
    os << "export GCU_ARCH=" << ToLower(CCtx().GetArch()) << "\n";

  dcg_instance.SetupBuildEnv(os);

  // Standalone extras: profiler and acore
  os << "TOPSPROF=${TOPSCC_INSTALL}/bin/topsprof\n";
#ifdef __CHOREO_GCU_ACORE_DIR__
  os << R"(if [[ -z "${ACORE_INSTALL}" ]]; then)"
     << "\n";
  os << "  ACORE_INSTALL=" << STRINGIZE(__CHOREO_GCU_ACORE_DIR__) << "\n";
  os << "fi\n";
#endif
  os << R"script(
if [[ ! -z "${ACORE_INSTALL}" ]]; then
  GCU_ACORE_INCLUDE=${ACORE_INSTALL}/include
  GCU_ACORE_LIB_PATH=${ACORE_INSTALL}/lib
  GCU_ACORE_LIB=libacoreop.bc
fi
)script";

  auto build_path = CreateUniquePath();
  auto cc_file = build_path + "/__choreo_topscc_" + filename + ".cpp";
  auto exe_file = exe_fn;
  if (exe_file.empty())
    exe_file = build_path + "/__choreo_topscc_" + filename + ".exe";
  auto fb_file = "__choreo_topscc_" + filename + ".topsfb";
  os << "rm -fr " << build_path << "\n";
  os << "mkdir -p " << build_path << "\n\n";

  // place the choreo header
  os << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  os << __choreo_header_as_string << "\nEOF\n";
  os << "cat <<'EOF' > " << build_path << "/choreo_types.h\n";
  os << __choreo_types_header_as_string << "\nEOF\n";

  // place the topscc header
  os << "cat <<'EOF' > " << build_path << "/private_target0_runtime.h\n";
  os << __topscc_header_as_string << "\nEOF\n";

  // place the library fallback runtime header (unified)
#ifdef __CHOREO_ACORE_RUNTIME_AVAILABLE__
  {
    os << "mkdir -p " << build_path << "/gcu\n";
    os << "cat <<'EOF' > " << build_path << "/gcu/lib_fallback.h\n";
    os << __lib_fallback_header_as_string << "\nEOF\n\n";
  }
#endif

  // place the target hack.
  // TODO: move all the target-specific to target runtime header
  os << "cat <<'EOF' > " << build_path << "/private_target0_defines.h\n";
  os << R"(#ifdef __TOPSCC__
#define __CHOREO_PRIVATE_TGT0__
#endif
#define __CHOREO_TGT0_ARCH__ __GCU_ARCH__
#define tgt0HostMalloc topsHostMalloc
#define tgt0HostFree topsHostFree
EOF

)";

  os << "cat <<'EOF' > " << cc_file << "\n";
  bool suppress_main = CCtx().DeviceOnly();
  if (suppress_main)
    os << "#define main __choreo_lib_suppressed_main_" << filename << "\n";
  for (size_t i = 0; i < code_segments.size(); ++i) {
    if (EnableLineDirective())
      os << PinLineDirectivePerGeneratedLine(code_segments[i]) << "\n";
    else
      os << code_segments[i] << "\n";
  }
  if (suppress_main) os << "#undef main\n";
  os << "\nEOF\n\n";

  // gcu_arch is set by SetupBuildEnv() -- either from GCU_ARCH env
  // (pre-set above for compile-time -arch) or via JIT lspci detection.

  os << R"script(
show_usage() {
  echo "  Usage: $0 <-st> <actions>"
  echo ""
  echo "  Options:"
  echo "   -st,                 Use default system path for target compilation"
  echo "   --execute,           Compile and execute"
  echo "   --profile,           Compile and execute with profiling"
  echo "   --compile-link,      Compile and link"
  echo "   --compile-module,    Compile and generate the module"
  echo "   --gen-fatbin,        Compile and generate the fatbin"
  echo "   --lib,               Compile and archive into a static library (.a)"
  echo "   --shared,            Compile into a shared library (.so)"
  echo ""
  echo "  Environment Variables:"
  echo "   EXTRA_TARGET_CFLAGS: Extra target compilation flags"
  echo "   TOPSCC_INSTALL:      Topscc compiler installation path"
  echo "   ACORE_INSTALL:       GCU Acore library installation path"
  exit 1
}

option_detect() {
  local tmpf=/tmp/$(date +"%Y%m%d_%H%M%S.%3N")__nasty_option_detect__.cpp
  echo "#include <krt/builtins.h>" > ${tmpf}
  echo "__device__ void foo() { tops::abort();  }" >> ${tmpf}

  ${TOPSCC} ${CFLAGS} -c ${tmpf} -o /dev/null 2>/dev/null

  if [[ $? -eq 0 ]]; then
    export CFLAGS="${CFLAGS} -D__CHOREO_USE_TOPS_ABORT__";
  fi

  rm -f ${tmpf}
}

# compile, execute
)script";

  os << R"(export CFLAGS="-arch ${gcu_arch} -std=c++17 -ltops -lm -O3)";
  if (CCtx().TargetDebugInfo()) os << " -g";
  if (!target_options.GetValue().empty())
    os << " " << target_options.GetValue();
  if (use_pic) os << " -fPIC";
  if (verbose) os << " -v"; // if it requires to be verbose
#ifdef __CHOREO_GCU_ACORE_DIR__
  os << R"( --tops-device-lib-path=${GCU_ACORE_LIB_PATH})";
  os << R"( --tops-device-lib=${GCU_ACORE_LIB})";
  os << R"( -I${GCU_ACORE_INCLUDE})";
  os << R"( -D__ACORE_OP__ -DTOPSCC_PRIVATE_DTE_AUTO_INIT -fPIC)";
  if (has_acore_call || has_lib_gemm_general || has_lib_fallback)
    os << " -I" << build_path;
#endif
  if (CCtx().GetArch() == "gcu500") { // enable gcusim5
    os << R"( -Wl,--disable-new-dtags -rpath "${TOPSCC_LIB}")";
  }
  if (CCtx().DMADiagnosis()) os << " -D__CHOREO_DMA_DIAGNOSIS__";
  // always enclose
  os << " ${EXTRA_TARGET_CFLAGS}";
  std::filesystem::path cwd = std::filesystem::current_path();
  auto input_file = OptionRegistry::GetInstance().GetInputFileName();
  auto input_abs_path = GetAbsPath(cwd.string(), input_file);
  os << " -I" << input_abs_path;
  for (auto inc_path : CCtx().GetIncPaths()) os << " -I" << inc_path;
  for (auto lib_path : CCtx().GetLibPaths()) os << " -L" << lib_path;
  for (auto lib : CCtx().GetLibs()) os << " -l" << lib;
  for (auto macro : CCtx().GetCLMacros())
    os << " -D" << macro.first
       << (macro.second.empty() ? "" : ("=" + macro.second));

  os << "\"";
  os << "\noption_detect";
  if (CCtx().IsSimArch()) {
    if (CCtx().ArchNum() == 400)
      os << "\nexport INTERNAL_GCU_SIM=LIBRA";
    else if (CCtx().ArchNum() == 500)
      os << "\nexport INTERNAL_GCU_SIM=DRACO";
  }

  os << "\n";

  os << R"(if [ "$1" == "--execute" ] || [ "$#" -eq 0 ]; then)";
  if (verbose)
    os << "\n  echo ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file;
  os << "\n  ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file;
  if (verbose) os << "\n  echo " << exe_file << "\n";
  os << "\n  " << exe_file << "\n";
  os << R"(elif [ "$1" == "--profile" ]; then)";
  os << "\n  if [ ! -f \"${TOPSPROF}\" ]; then";
  os << "\n    TOPSPROF=$(which topsprof)";
  os << "\n    if [ -z \"${TOPSPROF}\" ]; then";
  os << "\n      echo \"topsprof is not installed\" && exit 77";
  os << "\n    fi";
  os << "\n  fi";
  if (verbose)
    os << "\n    echo ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file;
  os << "\n    ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file;
  if (verbose)
    os << "\n  echo sudo ${TOPSPROF} ${PROF_OPTIONS} " << exe_file << "\n";
  os << "\n  sudo ${TOPSPROF} ${PROF_OPTIONS} " << exe_file << "\n";
  os << R"(elif [ "$1" == "--compile-module" ]; then)";
  if (verbose)
    os << "\n  echo ${TOPSCC} -c ${CFLAGS} " << cc_file << " -o " << exe_file
       << "\n";
  os << "\n  ${TOPSCC} -c ${CFLAGS} " << cc_file << " -o " << exe_file << "\n";
  os << R"(elif [ "$1" == "--compile-link" ]; then)";
  if (verbose)
    os << "\n  echo ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file
       << "\n";
  os << "\n  ${TOPSCC} ${CFLAGS} " << cc_file << " -o " << exe_file << "\n";
  os << R"(elif [ "$1" == "--gen-fatbin" ]; then)";
  os << "\n  __cur_dir=$(pwd)";
  os << "\n  cd " << build_path;
  if (verbose)
    os << "\n  echo ${TOPSCC} -save-temps -c ${CFLAGS} " << cc_file << " -o "
       << exe_file << "\n";
  os << "\n  ${TOPSCC} -save-temps -c ${CFLAGS} " << cc_file << " -o "
     << exe_file << "\n";
  if (verbose)
    os << "\n  echo cp " << cc_file
       << "-tops-dtu-enflame-tops.topsfb ${__cur_dir}/" << fb_file;
  os << "\n  cp " << cc_file << "-tops-dtu-enflame-tops.topsfb ${__cur_dir}/"
     << fb_file;
  os << "\n  cd ${__cur_dir}";
  os << "\n  echo \"Fatbin file generated: " << fb_file << "\"";

  auto obj_file = build_path + "/__choreo_topscc_" + filename + ".o";
  os << R"(
elif [ "$1" == "--lib" ]; then)";
  if (verbose)
    os << "\n  echo ${TOPSCC} -c -fPIC ${CFLAGS} " << cc_file << " -o "
       << obj_file << "\n";
  os << "\n  ${TOPSCC} -c -fPIC ${CFLAGS} " << cc_file << " -o " << obj_file;
  os << "\n  ar rcs " << exe_file << " " << obj_file;
  if (verbose) os << "\n  echo \"Library generated: " << exe_file << "\"";
  os << R"(
elif [ "$1" == "--shared" ]; then)";
  if (verbose)
    os << "\n  echo ${TOPSCC} -shared -fPIC ${CFLAGS} " << cc_file << " -o "
       << exe_file << "\n";
  os << "\n  ${TOPSCC} -shared -fPIC ${CFLAGS} " << cc_file << " -o "
     << exe_file;
  if (verbose)
    os << "\n  echo \"Shared library generated: " << exe_file << "\"";

  os << "\nelse show_usage";
  os << "\nfi";
}

bool TopsccCodeGen::CompileWithScript(const std::string& action) {
#ifdef __EMSCRIPTEN__
  (void)action;
  errs() << "CompileWithScript is not available in WebAssembly builds.\n";
  return false;
#else
  assert(!action.empty() && "no action is specified.");

  char tempFileName[] = "/tmp/choreo_topscc_script_XXXXXX";
  int fd = mkstemp(tempFileName);
  if (fd == -1) {
    errs() << "Failed to create temporary file.\n";
    return false;
  }
  close(fd);

  // Open the file for writing
  std::ofstream tempFile(tempFileName);
  if (!tempFile) {
    errs() << "Failed to open temporary file for writing.\n";
    return false;
  }

  auto outfile = OptionRegistry::GetInstance().GetOutputFileName();
  EmitScript(tempFile, outfile);
  tempFile.close(); // important: make sure the temp file is closed

  // Execute the file
  std::string command = "bash " + std::string(tempFileName) + " " + action;
  VST_DEBUG(dbgs() << "Compile " << outfile << ": " << command << "\n");
  int result = system(command.c_str());
  if (result == -1) {
    errs() << "Failed to execute the file.\n";
    return false;
  }

  // Remove the temporary file
  if (remove(tempFileName) != 0) {
    errs() << "Failed to remove the temporary file.\n";
    return false;
  }

  return true;
#endif
}

// Currently, it is guaranteed that ValueSTR can be used safely and directly.
const std::string TopsccCodeGen::ValueSTR(const ValueItem& vi,
                                          bool LL_suffix) const {
  return OpValueSTR(vi, "", true, LL_suffix);
}

const std::string TopsccCodeGen::ValueListSTR(const ValueList& vl,
                                              std::string sep,
                                              bool LL_suffix) const {
  std::ostringstream oss;
  if (!vl.empty()) {
    oss << ValueSTR(vl[0], LL_suffix);
    for (unsigned i = 1; i < vl.size(); ++i)
      oss << sep << ValueSTR(vl[i], LL_suffix);
  }
  return oss.str();
}

const std::string TopsccCodeGen::OpValueSTR(const ValueItem& vi,
                                            const std::string& parent_op,
                                            const bool is_left_child,
                                            bool LL_suffix) const {
  auto WrapParen = [&](const std::string& s, const std::string& cur_op) {
    if (Operator::NeedParen(cur_op, parent_op, is_left_child))
      return "(" + s + ")";
    // Used to ensure the above guarantee.
    if (parent_op == "") return "(" + s + ")";
    return s;
  };

  if (!IsValidValueItem(vi)) choreo_unreachable("invalid value item.");
  if (VIIsNil(vi)) {
    if (IsHost())
      return "choreo::__inf__";
    else
      return "-1"; // it looks the API requires -1
  } else if (auto iv = VIInt(vi)) {
    if (iv >= (int64_t)std::numeric_limits<int32_t>::max() ||
        iv <= (int64_t)std::numeric_limits<int32_t>::min())
      return PSTR(vi) + "LL";
    else if (LL_suffix)
      return PSTR(vi) + "LL";
    else
      return PSTR(vi);
  } else if (auto bv = VIBool(vi))
    return PSTR(vi);
  else if (auto sv = VISym(vi)) {
    auto res = UnScopedExpr(SSMName(sv.value(), IsHost()));
    if (LL_suffix) return "static_cast<long long>(" + res + ")";
    return res;
  } else if (auto uo = VIUop(vi)) {
    std::string op = STR(uo->GetOpCode());
    std::string res = op + OpValueSTR(uo->GetOperand(), op, false, LL_suffix);
    return WrapParen(res, op);
  } else if (auto bo = VIBop(vi)) {
    if (bo->GetOpCode() == OpCode::ADD) {
      if (auto rv = VIInt(bo->GetRight()); rv && rv.value() < 0) {
        std::string res = OpValueSTR(bo->GetLeft(), "-", true, LL_suffix) +
                          " - " + std::to_string(-rv.value());
        if (rv.value() >= (int64_t)std::numeric_limits<int32_t>::max() ||
            rv.value() <= (int64_t)std::numeric_limits<int32_t>::min())
          res += "LL";
        return WrapParen(res, "-");
      }
    }
    std::string op = STR(bo->GetOpCode());
    std::string res = OpValueSTR(bo->GetLeft(), op, true, LL_suffix) + " " +
                      op + " " +
                      OpValueSTR(bo->GetRight(), op, false, LL_suffix);
    return WrapParen(res, op);
  } else if (auto to = VITop(vi)) {
    std::string op = "?";
    std::string res = OpValueSTR(to->GetPred(), op, true) + " ? " +
                      OpValueSTR(to->GetLeft(), op, true, LL_suffix) + " : " +
                      OpValueSTR(to->GetRight(), op, false, LL_suffix);
    return WrapParen(res, op);
  } else
    choreo_unreachable("unsupported value.");
  return "";
}

// input is a `node` or `std::variant<int, float>`.
// If `val` is existed, use it first.
const std::string
TopsccCodeGen::ExprCastSTR(AST::ptr<AST::Node> n,
                           std::optional<std::variant<int, float>> val,
                           BaseType t, BaseType f, bool is_host,
                           size_t element_count, bool is_explicit) const {
  std::ostringstream res;
  std::string value;

  if (val.has_value()) {
    auto v = val.value();
    if (std::holds_alternative<int>(v))
      value = std::to_string(std::get<int>(v));
    else if (std::holds_alternative<float>(v))
      value = std::to_string(std::get<float>(v)) + "f";
    else
      choreo_unreachable("unexpect type of v");
  } else {
    assert(n);
    // using "" as op, cause `value` is always used inside `()`
    value = OpExprSTR(n, "", true, is_host);
  }

  if (f == t) return value;

  using BT = BaseType;
  // need to do casting or converting.
  if (!IsValuePreservingCast(f, t) && !is_explicit) {
    if (IsReinterpretiveCast(f, t))
      Warning(n->LOC(), "The implicit type conversion may lead to semantic "
                        "error(without data loss): '" +
                            STR(f) + "' to '" + STR(t) + "'");
    else if (IsLossyCast(f, t))
      Warning(n->LOC(), "The implicit type conversion may lose precision: '" +
                            STR(f) + "' to '" + STR(t) + "'");
    else
      choreo_unreachable("unexpect cast");
  }

  if (element_count > 1) { // vector type cast
    auto tty = MakeVectorType(t, element_count);
    auto fty = MakeVectorType(f, element_count);
    res << "tcle::cvt<" << VectorTypeSTR(tty) << ">(" << value << ")";
    return res.str();
  }

  if (t == BT::F8_E4M3 || f == BT::F8_E4M3 || t == BT::F8_E5M2 ||
      f == BT::F8_E5M2)
    choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) + "'");

  switch (t) {
  case BT::S64: [[fallthrough]];
  case BT::U64: [[fallthrough]];
  case BT::S32: [[fallthrough]];
  case BT::U32: [[fallthrough]];
  case BT::S16: [[fallthrough]];
  case BT::U16: [[fallthrough]];
  case BT::S8: [[fallthrough]];
  case BT::U8: {
    auto nbt = NameBaseType(t, is_host);
    if (IsIntegralType(f))
      res << "static_cast<" << nbt << ">(" << value << ")";
    else if (IsFloatType(f)) {
      if (f != BT::F32 && f != BT::F64)
        res << "static_cast<" << nbt << ">("
            << ExprCastSTR(n, val, BT::F32, f, is_host) << ")";
      else
        res << "static_cast<" << nbt << ">(" << value << ")";
    }
    break;
  }
  case BT::F64: {
    if (IsIntegralType(f))
      res << "static_cast<double>(" << value << ")";
    else
      res << "static_cast<double>(" << ExprCastSTR(n, val, BT::F32, f, is_host)
          << ")";
    break;
  }
  case BT::F32: {
    if (IsIntegralType(f))
      res << "static_cast<float>(" << value << ")";
    else {
      if (f == BT::F16)
        res << "f16_to_f32(" << value << ")";
      else if (f == BT::BF16)
        res << "bf16_to_f32(" << value << ")";
      else if (f == BT::F64)
        res << "static_cast<float>(" << value << ")";
      else
        choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) +
                           "'");
    }
    break;
  }
  case BT::F16:
    res << "f32_to_f16(" << ExprCastSTR(n, val, BT::F32, f, is_host) << ")";
    break;
  case BT::BF16:
    res << "f32_to_bf16(" << ExprCastSTR(n, val, BT::F32, f, is_host) << ")";
    break;
  default:
    choreo_unreachable("unsupport cast: '" + STR(f) + "' to '" + STR(t) + "'");
  }

  return res.str();
}

const std::string TopsccCodeGen::AddressOffset(const Shape& shape,
                                               const AST::DataAccess& da,
                                               bool is_host) const {
  size_t idx = 0;
  std::ostringstream oss;
  size_t operand_cnt = 0;
  auto AppendOffset = [this, &oss, &shape, &idx,
                       &operand_cnt](const ValueItem& op) {
    auto offset = op;
    assert(shape.Rank() >= idx + 1);
    if (shape.Rank() > idx + 1)
      offset = offset * shape.TrimDims(idx + 1).ElementCountValue();
    offset = SimplifyExpression(offset);
    if (!sbe::ceq(offset, sbe::nu(0))) {
      if (operand_cnt > 0) oss << " + ";
      oss << ValueSTR(offset);
      ++operand_cnt;
    }
    ++idx;
  };
  for (auto item : da.GetIndices()) {
    auto item_ty = item->GetType();
    if (auto id = AST::GetIdentifier(item)) {
      if (bv_map.count(InScopeNameForRef(id->name))) {
        auto ivs = bv_map.at(InScopeNameForRef(id->name));
        for (auto iv_itr = ivs.begin(); iv_itr != ivs.end(); ++iv_itr)
          AppendOffset(sbe::sym(*iv_itr));
      } else
        AppendOffset(sbe::sym(InScopeNameForRef(id->name)));
    } else if (auto il = AST::GetIntLiteral(*item)) {
      AppendOffset(sbe::nu(il->Val()));
    } else {
      if (idx > 0) oss << " + ";
      assert(shape.Rank() >= idx + 1);
      if (shape.Rank() > idx + 1)
        oss << OpExprSTR(item, "*", true, is_host) << "*"
            << ValueSTR(shape.TrimDims(idx + 1).ElementCountValue());
      else
        oss << OpExprSTR(item, "+", false, is_host);
      ++idx;
      ++operand_cnt;
    }
    if (IsActualVectorType(item_ty) && da.HasNote("VLDST")) { oss << "[0]"; }
  }
  if (operand_cnt == 0 && idx > 0) return "0";
  return oss.str();
}

const std::string TopsccCodeGen::ExprSTR(AST::ptr<AST::Node> e,
                                         bool is_host) const {
  // start with the lowest precedence op ""
  return OpExprSTR(e, "", true, is_host);
}

const std::string TopsccCodeGen::OpExprSTR(AST::ptr<AST::Node> e,
                                           const std::string& parent_op,
                                           bool is_left_child,
                                           bool is_host) const {
  std::ostringstream oss;

  // If output a expression with op to `oss`, then the expr maybe should
  // be wrapped with parentheses, e.g. `oss << a << "+" << b`,
  // should result in `WrapParen("a + b", "+")`
  // If `parent_op` is "*", then parentheses is necessary. If `parent_op` is
  // "-", for `is_left` == right, need parentheses: `xxx - (a + b)`
  auto WrapParen = [&](const std::string& s, const std::string& cur_op) {
    if (Operator::NeedParen(cur_op, parent_op, is_left_child))
      return "(" + s + ")";
    return s;
  };

  if (auto id = dyn_cast<AST::Identifier>(e)) {
    if (id->name == "__choreo_no_tiling__") {
      assert(!is_host);
      return id->name;
    }
    if (bv_map.count(InScopeNameForRef(id->name)) && !is_host) {
      size_t i = 0;
      for (auto iv_name : bv_map.at(InScopeNameForRef(id->name)))
        oss << ((i++ == 0) ? "" : ", ")
            << UnScopedName(ssm.DeviceName(iv_name));
    } else {
      oss << UnScopedName(SSMName(InScopeNameForRef(id->name), is_host));
    }
  } else if (auto np = dyn_cast<AST::Nullptr>(e)) {
    oss << "nullptr";
  } else if (auto il = dyn_cast<AST::IntLiteral>(e)) {
    oss << il->ValAsString();
  } else if (auto fl = dyn_cast<AST::FloatLiteral>(e)) {
    std::ostringstream fp_val;
    // std::fixed: the value should be in fixed-point notation
    // otherwise, 1.0f => 1f (error)
    if (fl->IsFloat32())
      fp_val << std::fixed << fl->Val_f32() << "f";
    else if (fl->IsFloat64())
      fp_val << std::fixed << fl->Val_f64();
    else
      choreo_unreachable("unsupported float literal.");
    oss << fp_val.str();
  } else if (auto sl = dyn_cast<AST::StringLiteral>(e)) {
    oss << sl->EscapedVal();
  } else if (auto b = dyn_cast<AST::BoolLiteral>(e)) {
    oss << (b->value ? "true" : "false");
  } else if (auto ii = dyn_cast<AST::IntIndex>(e)) {
    // currently, value of IntIndex is always IntLiteral or Identifier
    return OpExprSTR(ii->value, parent_op, true, is_host);
  } else if (auto da = dyn_cast<AST::DataAccess>(e)) {
    if (auto sty = GetSpannedType(GetSymbolType(da->data->name))) {
      if (auto da_ty = dyn_cast<VectorType>(da->GetType())) {
        oss << DASTR(da);
      } else {
        oss << "*((" << NameBaseType(sty->ElementType()) << "*)"
            << OpExprSTR(da->data, "+", true, is_host);
        auto shape = sty->GetShape();

        oss << " + " << AddressOffset(shape, *da, is_host) << ")";
      }
    } else {
      assert(!da->AccessElement());
      assert(!bv_map.count(InScopeNameForRef(da->data->name)));
      oss << UnScopedName(SSMName(InScopeNameForRef(da->data->name), is_host));
    }
  } else if (auto ce = dyn_cast<AST::CastExpr>(e)) {
    assert(ce->GetOp() == Op::Cast);
    if (ce->IsForeignCast())
      return "((" + ce->ForeignType() + ")" +
             OpExprSTR(ce->GetR(), "", true, is_host) + ")";
    return ExprCastSTR(ce->GetR(), std::nullopt, ce->ToType(), ce->FromType(),
                       is_host, ce->ElementCount(), ce->IsExplicit());
  } else if (auto expr = dyn_cast<AST::Expr>(e)) {
    // if this expr needs broadcasting
    bool rparen = false;
    if (expr->HasNote("broadcast")) {
      auto vector_width = cur_loop->GetVectorFactor();
      auto ety = expr->GetType();
      BaseType bty = ety->GetBaseType();
      if (IsActualBoundedIntegerType(ety)) bty = BaseType::S32;

      auto vty = MakeVectorType(bty, vector_width);
      oss << "(" << VectorTypeSTR(vty) << ")"
          << "(";
      rparen = true;
    }

    // utilize the optimize value whenever possible
    if (auto sym = expr->GetSymbol()) {
      auto sname = InScopeNameForRef(sym->name);
      if (FCtx(fname).HasSymbolValues(sname)) {
        auto svs = FCtx(fname).GetSymbolValues(sname);
        if (svs.HasVal()) return ValueSTR(svs.GetVal());
      }
    }

    if (ConvertibleToInt(NodeType(*e)))
      if (expr->Opts().HasVal()) return ValueSTR(expr->Opts().GetVal());

    if (expr->IsReference()) {
      if (PSTR(expr) == "_") return "0";
      if (auto ca = dyn_cast<AST::ChunkAt>(expr->GetR())) {
        auto caty = cast<SpannedType>(ca->GetType());
        std::string res;
        if (isa<FutureType>(NodeType(*ca->data))) {
          if (no_future_mode) {
            auto ca_id = dyn_cast<AST::Identifier>(ca->data);
            auto ca_sn = ca_id ? InScopeName(ca_id->name) : "";
            auto nf_it = nofuture_vars.find(ca_sn);
            if (nf_it != nofuture_vars.end())
              res = "((" + std::string(NameBaseType(caty->ElementType())) +
                    "*)" + nf_it->second.data_ptr + ") + " + GenOffset(ca);
            else
              res = OpExprSTR(ca->data, parent_op, true, is_host) +
                    ".data() + " + GenOffset(ca);
          } else {
            res = OpExprSTR(ca->data, parent_op, true, is_host) + ".data() + " +
                  GenOffset(ca);
          }
        } else
          res = OpExprSTR(ca->data, "+", true, is_host) + " + " + GenOffset(ca);
        return WrapParen(res, "+");
      } else {
        return OpExprSTR(expr->GetReference(), parent_op, is_left_child,
                         is_host);
      }
    } else if (expr->IsUnary()) {
      if (expr->GetOp() == Op::LogicNot) {
        oss << "!"
            << WrapParen(OpExprSTR(expr->GetR(), "!", false, is_host), "!");
      } else if (expr->GetOp() == Op::GetUBound) {
        auto rty = cast<BoundedType>(NodeType(*expr->GetR()));
        if (rty->Dims() == 1) oss << ValueSTR(rty->GetUpperBound());
      } else if (expr->GetOp() == Op::DataOf || expr->GetOp() == Op::MDataOf) {
        assert(isa<FutureType>(expr->GetR()->GetType()) &&
               "expect a future operand.");
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol()) {
          if (is_host) {
            oss << id->name << "__buf__";
          } else {
            auto sname = InScopeName(id->name);
            auto nf_data_it = no_future_mode ? nofuture_vars.find(sname)
                                             : nofuture_vars.end();
            if (nf_data_it != nofuture_vars.end()) {
              oss << nf_data_it->second.data_ptr;
            } else {
              oss << id->name
                  << (expr->GetOp() == Op::MDataOf ? ".mdata()" : ".data()");
            }
          }
        } else
          choreo_unreachable("Can not retrieve name of the future.");
      } else if (expr->GetOp() == Op::SizeOf) {
        auto se = expr->Opts().GetSize();
        if (IsValidValueItem(se))
          oss << ValueSTR(se);
        else {
          // TODO: deprecate this implementation
          auto var = RemoveSuffix(*AST::GetName(*expr->GetR()), ".span");
          auto shape = GetShape(GetSymbolType(var));
          assert(shape.IsValid() && "Invalid shape is found");
          oss << ValueSTR(shape.ElementCountValue());
        }
      } else if (expr->GetOp() == Op::PreInc) {
        oss << "++"
            << WrapParen(OpExprSTR(expr->GetR(), "++", false, is_host), "++");
      } else if (expr->GetOp() == Op::PreDec) {
        oss << "--"
            << WrapParen(OpExprSTR(expr->GetR(), "--", false, is_host), "--");
      } else if (expr->GetOp() == Op::PostInc) {
        oss << WrapParen(OpExprSTR(expr->GetR(), "post++", false, is_host),
                         "post++")
            << "++";
      } else if (expr->GetOp() == Op::PostDec) {
        oss << WrapParen(OpExprSTR(expr->GetR(), "post--", false, is_host),
                         "post--")
            << "--";
      } else if (expr->GetOp() == Op::AddrOf) {
        if (auto id = AST::GetIdentifier(expr->GetR()))
          oss << OpExprSTR(id, parent_op, is_left_child, is_host);
        else if (isa<AST::DataAccess>(expr->GetR()))
          oss << "&"
              << WrapParen(OpExprSTR(expr->GetR(), "&", false, is_host), "&");
        else
          choreo_unreachable("Can not retrieve name of the spanned data.");
      } else if (expr->GetOp() == Op::BitNot) {
        oss << "~"
            << WrapParen(OpExprSTR(expr->GetR(), "~", false, is_host), "~");
      } else
        choreo_unreachable("unsupported expression op: '" + STR(expr->GetOp()) +
                           "', expr: " + PSTR(expr) + ".");
    } else if (expr->IsBinary()) {
      if (expr->GetOp() == Op::CeilDiv) {
        std::string one = "1";
        auto L = OpExprSTR(expr->GetL(), "+", true, is_host);
        auto R0 = OpExprSTR(expr->GetR(), "+", false, is_host);
        auto R1 = OpExprSTR(expr->GetR(), "/", false, is_host);
        // (L + R0 - 1) / R1
        std::ostringstream res;
        res << "(" << L << " + " << R0 << " - " << one << ") / " << R1;
        oss << WrapParen(res.str(), "/");
      } else if (expr->GetOp() == Op::GetIth) {
        auto lty = cast<BoundedType>(NodeType(*expr->GetL()));
        auto r = expr->GetR();
        if (cast<AST::IntIndex>(r)->IsNegative()) {
          std::ostringstream res;
          // special case: str of R is always a negative integer.
          res << ValueSTR(lty->GetUpperBound()) << " + "
              << OpExprSTR(r, "+", false, is_host);
          oss << WrapParen(res.str(), "+");
        } else
          oss << OpExprSTR(r, parent_op, is_left_child, is_host);
      } else if (expr->GetOp() == Op::ElemOf) {
        if (AST::IsEventGenerationAt(*expr)) {
          auto event_array = cast<EventArrayType>(NodeType(*expr->GetL()));
          auto extent = VIInt(event_array->Dimension(0));
          assert(extent && *extent > 0);
          oss << OpExprSTR(expr->GetL(), "[]", true, is_host) << "[("
              << OpExprSTR(expr->GetR(), "", true, is_host) << ") % " << *extent
              << "]";
          return oss.str();
        }
        // When MemReuse is on and the base is an ArrayType of SpannedType,
        // the declaration is a flat pointer (or future data()), so arr[idx]
        // must become byte-level pointer arithmetic to handle void* base:
        //   (elem_type*)((char*)base + idx * sub_byte_size)
        bool use_ptr_arith = false;
        if (CCtx().MemReuse()) {
          if (auto base_id = dyn_cast<AST::Identifier>(expr->GetL())) {
            auto base_ty = GetSymbolType(base_id->name);
            if (auto arr_ty = dyn_cast<ArrayType>(base_ty)) {
              if (auto inner_sty = GetSpannedType(arr_ty)) {
                use_ptr_arith = true;
                std::string sub_bytes = inner_sty->ByteSizeExpression();
                std::string bts = std::string(
                    NameBaseType(inner_sty->ElementType(), is_host));
                oss << "((" << bts << "*)((char*)"
                    << OpExprSTR(expr->GetL(), "", true, is_host) << " + ("
                    << OpExprSTR(expr->GetR(), "*", true, is_host) << ") * "
                    << sub_bytes << "))";
              }
            }
          }
        }
        if (!use_ptr_arith)
          oss << OpExprSTR(expr->GetL(), "[]", true, is_host) << "["
              << OpExprSTR(expr->GetR(), "", true, is_host) << "]";
      } else if (expr->IsArith() || expr->IsLogical() || expr->IsCompare() ||
                 expr->isBitwise()) {
        auto& l = expr->GetL();
        auto& r = expr->GetR();
        const auto& op = expr->GetOp();
        const auto op_str = STR(op);
        if (op == Op::UBound && IsActualBoundedIntegerType(l->GetType()) &&
            IsActualBoundedIntegerType(r->GetType())) {
          auto rty = cast<BoundedType>(NodeType(*r));
          assert(rty->Dims() == 1);
          std::string r_upper_bound;
          if (PSTR(r) == "_")
            r_upper_bound = "1";
          else
            r_upper_bound = ValueSTR(rty->GetUpperBound());
          auto L = OpExprSTR(l, "*", true, is_host);
          auto R = OpExprSTR(r, "+", false, is_host);
          std::ostringstream res;
          res << L << " * " << r_upper_bound << " + " << R;
          oss << WrapParen(res.str(), "+");
        } else if ((op == Op::UBoundAdd || op == Op::UBoundSub) &&
                   IsActualBoundedIntegerType(l->GetType()) &&
                   isa<ScalarIntegerType>(r->GetType())) {
          oss << OpExprSTR(l, parent_op, is_left_child, is_host);
        } else if (op == Op::UBoundDiv || op == Op::UBoundScale ||
                   op == Op::UBoundMod) {
          choreo_unreachable("unsupported expression op: '" +
                             STR(expr->GetOp()) + "', expr: " + PSTR(expr) +
                             ".");
        } else {
          std::ostringstream res;
          res << OpExprSTR(l, op_str, true, is_host) << " " << op << " "
              << OpExprSTR(r, op_str, false, is_host);
          oss << WrapParen(res.str(), op_str);
        }
      }
    } else if (expr->IsTernary()) {
      std::ostringstream res;
      res << OpExprSTR(expr->GetC(), "?", true, is_host) << " ? "
          << OpExprSTR(expr->GetL(), "?", true, is_host) << " : "
          << OpExprSTR(expr->GetR(), "?", false, is_host);
      oss << WrapParen(res.str(), "?");
    } else
      choreo_unreachable("unsupported expression op: '" + STR(expr->GetOp()) +
                         "', expr: " + PSTR(expr) + ".");
    if (rparen) oss << ")";
  } else if (auto c = dyn_cast<AST::Call>(e)) {
    assert(!is_host);
    return CallSTR(*c);
  } else if (isa<AST::DataType>(e)) {
    return NameBaseType(e->GetType()->GetBaseType());
  } else
    choreo_unreachable("unsupported node type: " + e->TypeNameString() + ".");

  return oss.str();
}

const std::string TopsccCodeGen::CallSTR(AST::Call& n) const {
  std::ostringstream oss;
  auto func_name = [&n](const std::string& name) -> std::string {
    if (!n.IsArith()) return name;
    if (name == "__log")
      return "tcle::ln";
    else if (name == "__pow")
      return "tcle::power";
    else {
      const std::string prefix = "__";
      std::string func_name = name;
      if (auto res = RemovePrefixOrNull(prefix, name)) func_name = *res;
      return "tcle::" + func_name;
    }
  };

  oss << func_name(n.function->name);

  // emit template arguments
  if (n.template_args) {
    oss << "<";
    size_t i = 0;
    for (auto& ta : n.template_args->AllValues()) {
      // Namespace-qualified C++ name: emit verbatim; VALNO strips the prefix.
      std::string ta_str;
      if (auto id = dyn_cast<AST::Identifier>(ta))
        if (id->name.find("::") != std::string::npos) ta_str = id->name;
      if (ta_str.empty())
        if (auto expr = dyn_cast<AST::Expr>(ta))
          if (auto sym = expr->GetSymbol())
            if (sym->name.find("::") != std::string::npos) ta_str = sym->name;
      oss << ((i++ == 0) ? "" : ", ")
          << (ta_str.empty() ? OpExprSTR(ta, "", true, IsHost()) : ta_str);
    }
    oss << ">";
  }

  oss << "(";
  size_t i = 0;
  for (auto& a : n.GetArguments()) {
    oss << ((i++ == 0) ? "" : ", ");
    if (auto sty = GetSpannedType(NodeType(*a))) {
      auto elem = sty->ElementType();
      std::string bts = std::string(NameBaseType(elem, IsHost()));
      // show the detail type name to avoid link error
      if (!IsHost() && n.template_args && elem == BaseType::BF16 &&
          CCtx().ArchNum() >= 300)
        bts = "__bf16";
      auto m_ty = sty->GetStorage();
      auto mem_attr = TopsParamStorage(m_ty);
      if (a->HasNote("annotate_as") && !mem_attr.empty())
        bts = mem_attr + " " + bts;
      if (!no_decay_spanview || IsHost())
        oss << "(" << bts << "*)" << OpExprSTR(a, "", true, IsHost());
      else
        oss << "choreo::make_spanview<" << sty->Dims() << ">((" << bts << "*)"
            << OpExprSTR(a, "", true, IsHost()) << ", " << LSTR(sty->GetShape())
            << ")";
    } else if (n.IsArith())
      oss << OpExprSTR(a, "", true, IsHost());
    else
      oss << UnScopedExpr(OpExprSTR(a, "", true, IsHost()));
  }
  oss << ")";

  return oss.str();
}

const std::string TopsccCodeGen::BuildTcleLoad(const std::string& addr,
                                               const std::string& ty,
                                               const std::string& mask,
                                               const std::string& other) const {
  std::ostringstream oss;
  oss << "tcle::load<" << ty << ">(" << addr;
  if (mask.empty()) {
    oss << ")";
  } else {
    assert(!other.empty());
    oss << ", " << other << ", " << mask << ")";
  }
  return oss.str();
}

const std::string TopsccCodeGen::BuildTcleStore(const std::string& addr_str,
                                                const std::string& val_str,
                                                const std::string& mask) const {
  std::ostringstream oss;
  oss << "tcle::store(" << val_str << ", " << addr_str;
  if (mask.empty()) {
    oss << ")";
  } else {
    oss << ", " << mask << ")";
  }
  return oss.str();
}

const std::string TopsccCodeGen::BuildTcleGather(
    const std::string& base, const std::string& offset,
    const std::string& ty_str, const std::string& mask,
    const std::string& other) const {
  std::ostringstream oss;
  oss << "tcle::gather<" << ty_str << ">("
      << "(" << base << "), " << offset;
  if (mask.empty()) {
    oss << ")";
  } else {
    assert(!other.empty());
    oss << ", " << other << ", " << mask << ")";
  }
  return oss.str();
}

const std::string TopsccCodeGen::BuildTcleScatter(
    const std::string& value, const std::string& base,
    const std::string& offset, const std::string& mask) const {
  std::ostringstream oss;
  oss << "tcle::scatter(" << value << ", "
      << "(" << base << "), " << offset;
  if (mask.empty()) {
    oss << ")";
  } else {
    oss << ", " << mask << ")";
  }

  return oss.str();
}

const std::string TopsccCodeGen::DASTR(AST::ptr<AST::DataAccess>& da,
                                       const std::string& val_str, bool is_load,
                                       bool masking) const {
  std::ostringstream oss;
  if (auto sty = GetSpannedType(GetSymbolType(da->data->name))) {
    auto da_ty = da->GetType();
    auto elem_ty = sty->ElementType();
    if (auto vty = dyn_cast<VectorType>(da_ty)) {
      auto vty_str = VectorTypeSTR(vty);
      auto data_name = da->GetDataName();
      if (da->HasNote("VLDST")) {
        // continuous address, vector load/store
        std::string addr_str = "(" + std::string(NameBaseType(elem_ty)) +
                               " *)" + ssm.DeviceName(InScopeName(data_name)) +
                               " + " +
                               AddressOffset(sty->GetShape(), *da, false);
        if (CCtx().GetArch() == "gcu300")
          addr_str = "(__TCLE_AS__ char *)(" + addr_str + ")";
        if (is_load) {
          // load
          oss << BuildTcleLoad(addr_str, vty_str);
        } else {
          // store
          auto st_val = val_str;
          if (masking) {
            // for now, it use tcle::vsel to do masking store, the store val
            // 'st_val' is conditionally selected between 'val_str' and 'ld_val'
            // from the same memory location.
            if (CCtx().GetArch() == "gcu400") {
              oss << BuildTcleStore(addr_str, val_str, "exec");
            } else if (CCtx().GetArch() == "gcu300") {
              auto ld_val = symtab.GetAnonName();
              oss << VectorTypeSTR(vty) << " " << ld_val << " = "
                  << BuildTcleLoad(addr_str, vty_str) << ";\n";
              oss << (IsHost() ? h_indent : d_indent);
              oss << st_val << " = tcle::vsel(exec, " << val_str << ", "
                  << ld_val << ");\n";
              oss << (IsHost() ? h_indent : d_indent);
              oss << BuildTcleStore(addr_str, st_val);
            } else
              choreo_unreachable("unspported target arch");
          } else {
            oss << BuildTcleStore(addr_str, st_val);
          }
        }
      } else if (da->HasNote("VGZST")) {
        // random address, vector gather/scatter
        auto base_addr_str = std::string("(") + NameBaseType(elem_ty) + " *)" +
                             ssm.DeviceName(InScopeName(data_name));
        auto addr_offsets_str = AddressOffset(sty->GetShape(), *da, false);
        addr_offsets_str = "(" + addr_offsets_str + ")";
        if (SizeOf(elem_ty) != 1) {
          addr_offsets_str =
              addr_offsets_str + " * " + std::to_string(SizeOf(elem_ty));
        }
        if (is_load) {
          oss << BuildTcleGather(base_addr_str, addr_offsets_str, vty_str);
        } else {
          oss << BuildTcleScatter(val_str, base_addr_str, addr_offsets_str,
                                  masking ? "exec" : "");
        }
      }
    } else {
      oss << ExprSTR(da, false) << " = " << val_str;
    }
  }
  return oss.str();
}

void TopsccCodeGen::BuildSiteAssertionMap() {
  if (CCtx().DisableRuntimeCheck()) return;
  if (fname.empty()) return;

  for (const auto& ar : FCtx(fname).GetAssertions(AssessType::USE_SITE)) {
    if (!ar.enabled || !ar.EmitTarget()) continue;
    if (ar.emit_position == AssertionEmitPosition::BEFORE_NODE)
      pre_site_assertions[ar.EmitTarget()].push_back(ar);
    else
      post_site_assertions[ar.EmitTarget()].push_back(ar);
  }
  for (const auto& ar : FCtx(fname).GetAssertions(AssessType::HOIST_SITE)) {
    if (!ar.enabled || !ar.EmitTarget()) continue;
    if (ar.emit_position == AssertionEmitPosition::BEFORE_NODE)
      pre_site_assertions[ar.EmitTarget()].push_back(ar);
    else
      post_site_assertions[ar.EmitTarget()].push_back(ar);
  }
}

void TopsccCodeGen::EmitPreSiteAssertions(AST::Node& n) {
  if (CCtx().DisableRuntimeCheck()) return;
  auto it = pre_site_assertions.find(&n);
  if (it == pre_site_assertions.end()) return;

  for (const auto& ar : it->second) {
    IndStream() << "choreo::choreo_assert(" << ValueSTR(ar.expr, true) << ", \""
                << ar.message << ", " << ar.loc << "\");\n";
  }
}

void TopsccCodeGen::EmitPostSiteAssertions(AST::Node& n) {
  if (CCtx().DisableRuntimeCheck()) return;
  auto it = post_site_assertions.find(&n);
  if (it == post_site_assertions.end()) return;

  // HOIST_SITE and USE_SITE assertions are emitted in device code using
  // choreo_assert (printf-based) because std::cerr is not available on device.
  for (const auto& ar : it->second) {
    IndStream() << "choreo::choreo_assert(" << ValueSTR(ar.expr, true) << ", \""
                << ar.message << ", " << ar.loc << "\");\n";
  }
}
