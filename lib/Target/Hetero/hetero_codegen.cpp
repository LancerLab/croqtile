#include "hetero_codegen.hpp"
#include "codegen_utils.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "ast.hpp"
#include "choreo_header.inc"
#include "choreo_types_header.inc"
#include "codegen.hpp"
#include "context.hpp"
#include "pipeline.hpp"
#include "target.hpp"
#include "target_registry.hpp"
#include "types.hpp"

using namespace Choreo;
using namespace Choreo::Hetero;

extern AST::Program root;

namespace {
// Find the Nth device(target) ParallelBy inside a named function in the AST.
AST::ParallelBy* FindDevicePB(AST::Node& program, const std::string& func_name,
                              const std::string& target_name, int nth) {
  int count = 0;
  AST::ChoreoFunction* func = nullptr;
  if (auto p = dyn_cast<AST::Program>(&program)) {
    for (auto& child : p->stmts->values) {
      if (auto f = dyn_cast<AST::ChoreoFunction>(child.get())) {
        if (f->name == func_name) {
          func = f;
          break;
        }
      }
    }
  }
  if (!func || !func->stmts) return nullptr;
  for (auto& child : func->stmts->values) {
    if (auto pb = dyn_cast<AST::ParallelBy>(child.get())) {
      if (pb->HasDeviceTarget() && pb->DeviceTargetName() == target_name) {
        if (count == nth) return pb;
        count++;
      }
    }
  }
  return nullptr;
}
} // namespace

// ---- Preamble ----

void HeteroCodeGen::EmitPreamble() {
  // Device includes must come before choreo.h so that device SDK types
  // referenced by target-specific headers are available.
  for (auto& [name, dcg] : device_codegens) { dcg->EmitHostIncludes(os); }

  os << "#include \"choreo.h\"\n";
  os << "#include <cstring>\n";
  os << "#include <cstdlib>\n";
  os << "#include <future>\n";
  os << "#include <thread>\n\n";

  for (auto& [name, dcg] : device_codegens) { dcg->EmitHostPreamble(os); }
}

// ---- Before/InMid/After VisitImpl ----

bool HeteroCodeGen::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    for (auto& [name, target] : CCtx().DeviceTargets()) {
      auto dcg = target->MakeDeviceCodeGen();
      if (dcg)
        device_codegens[name] = std::move(dcg);
      else
        errs() << "warning: target '" << target->Name()
               << "' does not support heterogeneous device code generation.\n";
    }

    EmitPreamble();
    code_segments.push_back(os.str());
    os.str("");
    ssm.EnterScope();
    levels.push(ParallelLevel::NONE);
  } else if (auto f = dyn_cast<AST::ChoreoFunction>(&n)) {
    ResetFunctionStates();
    current_func_node_ = f;
    fty = cast<FunctionType>(GetSymbolType(fname));
    ssm.EnterScope();
    levels.push(ParallelLevel::SEQ);
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.push(pb->GetLevel());
    if (in_offload_device_block_ &&
        !(pb->HasDeviceTarget() && pb->GetLevel() == ParallelLevel::DEVICE))
      return true;
  } else if (in_offload_device_block_) {
    return true;
  } else if (isa<AST::WithBlock>(&n)) {
    IndStream() << "{\n";
    IncrIndent();
  }
  return true;
}

bool HeteroCodeGen::InMidVisitImpl(AST::Node& n) {
  if (in_offload_device_block_) return true;
  if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    if (!ie->HasElse()) return true;
    DecrIndent();
    IndStream() << "} else {\n";
    IncrIndent();
  }
  return true;
}

bool HeteroCodeGen::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    ssm.LeaveScope();

    if (!offload_functions_.empty() && !code_segments.empty()) {
      std::string fwd;
      for (auto& fi : offload_functions_) fwd += fi.host_fwd_decl + ";\n";
      fwd += "\n";
      code_segments.insert(code_segments.begin() + 1, fwd);
    }

    // Save output stream before in-process compilation may redirect it.
    auto& saved_outs = outs();
    // Compile offload functions in-process to get device source code.
    for (auto& fi : offload_functions_) {
      if (fi.offload_func && fi.device_source.empty()) {
        if (!CompileOffloadToSource(fi)) {
          // For source output mode, continue with .co source for display.
          // For compile/execute modes, this is fatal.
          if (CCtx().GetOutputKind() != OutputKind::TargetSourceCode) {
            error_count++;
            return false;
          }
        }
      }
    }

    // Restore output stream after in-process compilation.
    OptionRegistry::GetInstance().SetOutputStream(saved_outs);

    switch (CCtx().GetOutputKind()) {
    case OutputKind::TargetSourceCode: EmitHeteroSource(); break;
    case OutputKind::TargetModule:
    case OutputKind::TargetExecutable: {
      if (!CompileWithScript("--compile-link")) {
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
                         " is not supported by the hetero target.");
    }
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ssm.LeaveScope();
    code_segments.push_back(os.str());
    os.str("");
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.pop();

    if (pb->HasDeviceTarget() && pb->GetLevel() == ParallelLevel::DEVICE) {
      auto* dcg = ActiveDeviceCodeGen();
      if (dcg && dcg->IsHostDevice()) {
        auto pvs = pb->AllSubPVs();
        for (int i = pvs.size() - 1; i >= 0; --i) {
          DecrIndent();
          IndStream() << "}\n";
        }
        DecrIndent();
        IndStream() << "});\n";
      } else if (dcg && !dcg->IsHostDevice()) {
        EndOffloadFunction();
      }
      active_device_target = "";
    } else if (!in_offload_device_block_) {
      auto pvs = pb->AllSubPVs();
      for (int i = pvs.size() - 1; i >= 0; --i) {
        DecrIndent();
        IndStream() << "}\n";
      }
    }
  } else if (in_offload_device_block_) {
    return true;
  } else if (isa<AST::WithBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    const auto& ranges = fb->GetRangeNodes();
    for (int j = ranges->Count() - 1; j >= 0; --j) {
      auto rng = cast<AST::LoopRange>(ranges->ValueAt(j));
      auto cname = rng->IVName();
      auto ivs = within_map.at(InScopeName(cname));
      for (auto iv_itr = ivs.rbegin(); iv_itr != ivs.rend(); ++iv_itr) {
        DecrIndent();
        IndStream() << "}\n";
        os << indent << ssm.HostName(*iv_itr) << " = 0;\n";
      }
    }
  } else if (isa<AST::IfElseBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  } else if (isa<AST::WhileBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  }
  return true;
}

// ---- Visit methods (offload guard + delegate to CCCodeGen) ----

bool HeteroCodeGen::Visit(AST::WithIn& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::ForeachBlock& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::IfElseBlock& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::WhileBlock& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::Assignment& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::DMA& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::Break& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::Continue& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::Call& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::NamedVariableDecl& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::CppSourceCode& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::Return& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::Wait& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::MMA& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::Trigger& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

bool HeteroCodeGen::Visit(AST::Rotate& n) {
  if (in_offload_device_block_) return true;
  return CCCodeGen::Visit(n);
}

// ---- ParallelBy: hetero-specific dispatch ----

bool HeteroCodeGen::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  if (n.HasDeviceTarget() && n.GetLevel() == ParallelLevel::DEVICE) {
    active_device_target = n.DeviceTargetName();

    auto it = device_codegens.find(active_device_target);
    if (it == device_codegens.end()) {
      Error1(n.LOC(), "no codegen registered for device '" +
                          active_device_target + "'.");
      return false;
    }

    auto& dcg = *it->second;

    if (dcg.IsHostDevice()) {
      auto future_name =
          "__host_future_" + std::to_string(host_future_counter_++);
      pending_host_futures_.push_back(future_name);
      dcg.AddPendingFuture(future_name);
      IndStream() << "auto " << future_name
                  << " = __choreo_cpu_pool.submit([&]() {\n";
      IncrIndent();

      auto pvs = n.AllSubPVs();
      auto bvs = n.BoundValues();
      auto bpv_name = n.BPV()->name;
      for (size_t i = 0; i < pvs.size(); ++i) {
        auto pv_id = cast<AST::Identifier>(pvs[i]);
        auto pv_name = pv_id->name;
        auto bound = (i < bvs.size()) ? ValueSTR(bvs[i]) : std::string("1");
        ssm.MapHostSymbol(InScopeName(pv_name), pv_name);
        IndStream() << "for (int " << pv_name << " = 0; " << pv_name << " < "
                    << bound << "; ++" << pv_name << ") {\n";
        IncrIndent();
      }
      if (pvs.size() == 1) {
        auto pv_name = cast<AST::Identifier>(pvs[0])->name;
        ssm.MapHostSymbol(InScopeName(bpv_name), pv_name);
      }
    } else {
      in_offload_device_block_ = true;
      BeginOffloadFunction(n, dcg);
    }
    return true;
  }

  if (in_offload_device_block_) return true;

  return CCCodeGen::Visit(n);
}

// ---- Synchronize: hetero-specific sync ----

bool HeteroCodeGen::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);
  if (in_offload_device_block_) return true;

  if (n.Resource() == Storage::GLOBAL) {
    for (auto& f : pending_host_futures_) IndStream() << f << ".get();\n";
    pending_host_futures_.clear();
  }
  return true;
}

// ---- Source Emission ----

void HeteroCodeGen::EmitHeteroSource() {
  for (auto& code : code_segments) outs() << code << "\n";

  for (auto& fi : offload_functions_) {
    outs() << "\n// --- Offload device source: " << fi.co_func_name
           << " (target: " << fi.target_name << ") ---\n";
    if (!fi.device_source.empty())
      outs() << fi.device_source;
    else {
      outs() << "// [device compilation failed for " << fi.target_name << "]\n";
    }
  }
}

// ---- Offload Device Two-Step Delegation ----

void HeteroCodeGen::BeginOffloadFunction(AST::ParallelBy& n,
                                         DeviceCodeGen& dcg) {
  cur_offload_func_ = {};
  cur_offload_func_.co_func_name = "__hetero_" + dcg.DeviceName() + "_" +
                                   fname + "_" +
                                   std::to_string(offload_func_counter_++);
  cur_offload_func_.parent_fname = fname;
  cur_offload_func_.device_name = active_device_target;
  cur_offload_func_.target_name = dcg.TargetName();
  cur_offload_func_.source_ext = dcg.SourceExtension();

  for (auto& item : cgi.GetParameters(fname)) {
    auto ty = item.type;
    auto param_name = UnScopedName(item.name);
    if (auto sty = dyn_cast<SpannedType>(ty)) {
      OffloadFuncInfo::BufferInfo buf;
      buf.host_name = param_name;
      buf.device_name = param_name + "__device";
      buf.base_type = TypeSTR(*sty);
      buf.size_expr = UnScopedExpr(SizeExprOf(*sty, true));
      buf.is_output = cgi.IsReturnSymbol(fname, item.name);
      cur_offload_func_.buffers.push_back(buf);
    } else if (isa<ScalarType>(ty) || isa<BoundedType>(ty)) {
      cur_offload_func_.scalar_params.push_back(param_name);
      cur_offload_func_.scalar_types.push_back(TypeSTR(*ty));
    }
  }

  {
    std::ostringstream fwd;
    fwd << "extern void " << cur_offload_func_.co_func_name << "(";
    bool first = true;
    for (auto& item : cgi.GetParameters(fname)) {
      if (!first) fwd << ", ";
      fwd << HostTypeStringify(*item.type, false, item.IsReference()) << " "
          << UnScopedName(item.name);
      first = false;
    }
    fwd << ")";
    cur_offload_func_.host_fwd_decl = fwd.str();
  }

  // Build the offload function from the pre-sema AST clone. The pre-sema
  // clone is artifact-free (no implicit casts, no normalizer modifications,
  // no LateNorm __buf__ names), so the device pipeline can re-analyze it.
  auto pre_sema = CCtx().GetPreSemaRoot();
  if (!pre_sema)
    choreo_unreachable("pre-sema AST clone not available for hetero offload");
  if (!current_func_node_)
    choreo_unreachable("current function node not set in BeginOffloadFunction");

  auto target_name = n.DeviceTargetName();
  int nth = device_pb_counters_[target_name]++;
  auto* clean_pb = FindDevicePB(*pre_sema, fname, target_name, nth);
  if (!clean_pb)
    choreo_unreachable("failed to find device(" + target_name +
                       ") parallel-by #" + std::to_string(nth) +
                       " in pre-sema AST for function '" + fname + "'");

  auto func = AST::Make<AST::ChoreoFunction>(clean_pb->LOC());
  func->name = cur_offload_func_.co_func_name;
  func->f_decl.name = func->name;
  func->f_decl.ret_type =
      AST::Make<AST::DataType>(clean_pb->LOC(), BaseType::VOID);
  func->f_decl.params = CloneP(current_func_node_->f_decl.params);
  func->stmts = CloneP(clean_pb->stmts);
  cur_offload_func_.offload_func = func;
}

void HeteroCodeGen::EndOffloadFunction() {
  offload_functions_.push_back(cur_offload_func_);
  in_offload_device_block_ = false;

  EmitOffloadHostCall(cur_offload_func_);
}

void HeteroCodeGen::EmitOffloadHostCall(const OffloadFuncInfo& fi) {
  IndStream() << fi.co_func_name << "(";
  bool first = true;
  for (auto& item : cgi.GetParameters(fname)) {
    if (!first) os << ", ";
    os << UnScopedName(item.name);
    first = false;
  }
  os << ");\n";
}

// Run the device target's full pipeline on the offload function AST cloned
// from the pre-sema root. Returns true with fi.device_source populated.
bool HeteroCodeGen::CompileOffloadToSource(OffloadFuncInfo& fi) {
  if (!fi.offload_func) return false;

  auto saved = CCtx().SaveTargetState();
  auto saved_root_stmts = root.stmts;

  auto device_target = TargetRegistry::Create(fi.target_name);
  if (!device_target) {
    errs() << "Failed to create target '" << fi.target_name
           << "' for offload compilation.\n";
    CCtx().RestoreTargetState(std::move(saved));
    return false;
  }

  CCtx().SetTarget(std::move(device_target));
  CCtx().SetOutputKind(OutputKind::DeviceSourceOnly);
  CCtx().ClearArchs();
  CCtx().SetGlobalSymbolTable(std::make_shared<SymbolTable>());
  CodeGenInfo::instance = std::make_unique<CodeGenInfo>();

  // Build a program containing only the cloned offload function.
  location fresh_loc;
  root = AST::Program(fresh_loc);
  root.stmts->Append(fi.offload_func);

  ASTPipeline device_pl;
  device_pl.PlanSemanticRoutine().PlanCodeGenRoutine();

  std::ostringstream capture_os;
  auto& opt_reg = OptionRegistry::GetInstance();
  opt_reg.SetOutputStream(capture_os);

  bool ok = false;
  if (device_pl.RunOnProgram(root)) {
    fi.device_source = capture_os.str();
    ok = true;
  } else {
    errs() << "Device codegen pipeline failed for target '" << fi.target_name
           << "'.\n";
  }

  root.stmts = saved_root_stmts;
  CCtx().RestoreTargetState(std::move(saved));

  return ok;
}

void HeteroCodeGen::EmitScript(std::ostream& out, const std::string& exe_fn) {
  auto filename = RemoveDirectoryPrefix(
      RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));
  out << "#!/usr/bin/env bash\n\n";
  out << "# Choreo hetero target: two-step compile script\n\n";

  std::string build_path = CreateUniquePath();
  auto host_cc_file = build_path + "/__choreo_hetero_" + filename + ".cpp";
  auto host_obj_file = build_path + "/__choreo_hetero_" + filename + ".o";
  auto exe_file = exe_fn;
  if (exe_file.empty())
    exe_file = build_path + "/__choreo_hetero_" + filename + ".exe";

  out << "rm -fr " << build_path << "\n";
  out << "mkdir -p " << build_path << "\n\n";

  out << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  out << __choreo_header_as_string << "\nEOF\n\n";
  out << "cat <<'EOF' > " << build_path << "/choreo_types.h\n";
  out << __choreo_types_header_as_string << "\nEOF\n\n";

  // Emit target-specific setup files needed by device toolchains.
  for (auto& [name, dcg] : device_codegens) {
    if (!dcg->IsHostDevice()) dcg->EmitSetupFiles(out, build_path);
  }

  // Write offload device source files into per-target subdirectories.
  std::vector<std::string> offload_obj_files;
  for (size_t i = 0; i < offload_functions_.size(); ++i) {
    auto& fi = offload_functions_[i];
    auto it = device_codegens.find(fi.device_name);
    std::string tdir = (it != device_codegens.end())
                           ? it->second->TargetBuildDir(build_path)
                           : build_path;
    auto src_file = tdir + "/" + fi.co_func_name + fi.source_ext;
    auto obj_file = tdir + "/" + fi.co_func_name + ".o";
    offload_obj_files.push_back(obj_file);

    out << "cat <<'EOF' > " << src_file << "\n";
    out << fi.device_source;
    out << "\nEOF\n\n";
  }

  bool has_offload = !offload_functions_.empty();

  out << "cat <<'EOF' > " << host_cc_file << "\n";
  for (auto& code : code_segments) out << code << "\n";
  out << "\nEOF\n\n";

  // Emit build env setup for all offload devices
  for (auto& [name, dcg] : device_codegens) {
    if (!dcg->IsHostDevice()) dcg->SetupBuildEnv(out);
  }

  out << R"script(
show_usage() {
  echo "  Usage: $0 <actions>"
  echo ""
  echo "  Options:"
  echo "   --compile-link,     Compile and link to an executable"
  echo "   --compile-module,   Compile to an object file"
  echo "   --execute,          Compile and execute"
}

do_compile() {
)script";

  if (has_offload) {
    // Step 1: Compile each device source file with the device toolchain
    for (size_t i = 0; i < offload_functions_.size(); ++i) {
      auto& fi = offload_functions_[i];
      auto obj_file = offload_obj_files[i];
      out << "  echo \"Compiling offload function: " << fi.co_func_name
          << "\"\n";
      for (auto& [dname, dcg] : device_codegens) {
        if (dcg->TargetName() == fi.target_name) {
          auto tdir = dcg->TargetBuildDir(build_path);
          auto src_file = tdir + "/" + fi.co_func_name + fi.source_ext;
          dcg->EmitDeviceCompileCommand(out, build_path, src_file, obj_file);
          break;
        }
      }
    }

    // Step 2: Compile host C++ with the offload device's toolchain
    // (use the first offload device's toolchain for host compilation)
    DeviceCodeGen* offload_dcg = nullptr;
    for (auto& [name, dcg] : device_codegens) {
      if (!dcg->IsHostDevice()) {
        offload_dcg = dcg.get();
        break;
      }
    }

    out << "  echo \"Compiling host code\"\n";
    if (offload_dcg)
      offload_dcg->EmitHostCompileCommand(out, build_path, host_cc_file,
                                          host_obj_file);

    // Step 3: Link everything
    std::vector<std::string> all_objs;
    all_objs.push_back(host_obj_file);
    for (auto& obj : offload_obj_files) all_objs.push_back(obj);

    out << "\n  echo \"Linking\"\n";
    if (offload_dcg) offload_dcg->EmitLinkCommand(out, all_objs, exe_file);
  } else {
    // Host-only: compile with g++, use any host device's link flags
    out << "  ${CXX:-g++} -std=c++17 -O2 -pthread -I" << build_path << " "
        << host_cc_file << " -o " << exe_file;
    for (auto& [name, dcg] : device_codegens) out << " " << dcg->LinkFlags();
    out << " || { echo 'Compilation failed'; exit 1; }\n";
  }

  out << R"script(}

if [ "$1" = "--compile-link" ] || [ "$1" = "--compile-module" ]; then
  do_compile
elif [ "$1" = "--execute" ]; then
  do_compile && )script"
      << exe_file << R"script(
else
  show_usage
fi
)script";
}

bool HeteroCodeGen::CompileWithScript(const std::string& action) {
  assert(!action.empty() && "no action is specified.");

  char tempFileName[] = "/tmp/choreo_hetero_script_XXXXXX";
  int fd = mkstemp(tempFileName);
  if (fd == -1) {
    errs() << "Failed to create temporary script file.\n";
    return false;
  }
  close(fd);

  std::ofstream tempFile(tempFileName);
  if (!tempFile.is_open()) {
    errs() << "Failed to open temporary script file.\n";
    return false;
  }

  auto outfile = OptionRegistry::GetInstance().GetOutputFileName();
  EmitScript(tempFile, outfile);
  tempFile.close();

  std::string command = "bash " + std::string(tempFileName) + " " + action;
  int result = std::system(command.c_str());

  std::filesystem::remove(tempFileName);
  return result == 0;
}
