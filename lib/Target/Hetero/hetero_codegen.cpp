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
#include "target.hpp"
#include "types.hpp"

using namespace Choreo;
using namespace Choreo::Hetero;

// ---- Preamble ----

void HeteroCodeGen::EmitPreamble() {
  os << "#include \"choreo.h\"\n";
  os << "#include <cstring>\n";
  os << "#include <cstdlib>\n";
  os << "#include <future>\n";
  os << "#include <thread>\n\n";

  for (auto& [name, dcg] : device_codegens) {
    dcg->EmitHostIncludes(os);
    dcg->EmitHostPreamble(os);
  }
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
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ResetFunctionStates();
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
      for (auto& fi : offload_functions_)
        fwd += fi.host_fwd_decl + ";\n";
      fwd += "\n";
      code_segments.insert(code_segments.begin() + 1, fwd);
    }

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
    for (auto& f : pending_host_futures_)
      IndStream() << f << ".get();\n";
    pending_host_futures_.clear();
  }
  return true;
}

// ---- Source Emission ----

void HeteroCodeGen::EmitHeteroSource() {
  for (auto& code : code_segments) outs() << code << "\n";

  for (auto& fi : offload_functions_) {
    outs() << "\n// --- Offload .co source: " << fi.co_func_name
           << " (target: " << fi.target_name << ") ---\n";
    std::istringstream iss(fi.co_source);
    std::string line;
    while (std::getline(iss, line)) outs() << "// " << line << "\n";
  }
}

// ---- Offload Device Two-Step Delegation ----

std::string HeteroCodeGen::ChoreoTypeSTR(const Type& ty) const {
  if (auto sty = dyn_cast<SpannedType>(&ty)) {
    auto bt = sty->ElementType();
    std::string bts;
    switch (bt) {
    case BaseType::F32: bts = "f32"; break;
    case BaseType::F64: bts = "f64"; break;
    case BaseType::F16: bts = "f16"; break;
    case BaseType::BF16: bts = "bf16"; break;
    case BaseType::S32: bts = "s32"; break;
    case BaseType::U32: bts = "u32"; break;
    case BaseType::S64: bts = "s64"; break;
    case BaseType::U64: bts = "u64"; break;
    case BaseType::S16: bts = "s16"; break;
    case BaseType::U16: bts = "u16"; break;
    case BaseType::S8: bts = "s8"; break;
    case BaseType::U8: bts = "u8"; break;
    case BaseType::BOOL: bts = "bool"; break;
    default: choreo_unreachable("unsupported choreo base type"); break;
    }
    auto shape_str = UnScopedExpr(
        ValueSTR(sty->GetShape().ElementCountValue()));
    return bts + " [" + shape_str + "]";
  }
  if (isa<S32Type>(&ty)) return "s32";
  if (isa<U32Type>(&ty)) return "u32";
  if (isa<S64Type>(&ty)) return "s64";
  if (isa<U64Type>(&ty)) return "u64";
  if (isa<F32Type>(&ty)) return "f32";
  if (isa<F64Type>(&ty)) return "f64";
  if (isa<BooleanType>(&ty)) return "bool";
  choreo_unreachable("unsupported choreo type: " + STR(ty));
  return "";
}

void HeteroCodeGen::BeginOffloadFunction(AST::ParallelBy& n,
                                         DeviceCodeGen& dcg) {
  cur_offload_func_ = {};
  cur_offload_func_.co_func_name =
      "__hetero_" + dcg.DeviceName() + "_" + fname + "_" +
      std::to_string(offload_func_counter_++);
  cur_offload_func_.parent_fname = fname;
  cur_offload_func_.target_name = dcg.TargetName();

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

  auto input_fn = OptionRegistry::GetInstance().GetInputFileName();
  std::ifstream ifs(input_fn);
  if (!ifs.is_open())
    choreo_unreachable("cannot open input file: " + input_fn);
  std::string source((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
  ifs.close();

  int pb_start_line = n.LOC().begin.line;

  std::istringstream iss(source);
  std::string line;
  std::vector<std::string> all_lines;
  while (std::getline(iss, line)) all_lines.push_back(line);

  int brace_depth = 0;
  int body_start = -1;
  int body_end = -1;
  for (int i = pb_start_line - 1; i < (int)all_lines.size(); ++i) {
    for (char c : all_lines[i]) {
      if (c == '{') {
        if (brace_depth == 0) body_start = i + 1;
        brace_depth++;
      } else if (c == '}') {
        brace_depth--;
        if (brace_depth == 0) {
          body_end = i - 1;
          break;
        }
      }
    }
    if (body_end >= 0) break;
  }

  std::ostringstream body_text;
  if (body_start >= 0 && body_end >= body_start) {
    for (int i = body_start; i <= body_end; ++i)
      body_text << all_lines[i] << "\n";
  }

  std::ostringstream co;
  co << "#include \"choreo.h\"\n\n";
  co << "__co__ void " << cur_offload_func_.co_func_name << "(";
  bool first = true;
  for (auto& item : cgi.GetParameters(fname)) {
    if (!first) co << ", ";
    co << ChoreoTypeSTR(*item.type);
    if (item.IsReference()) co << " &";
    co << " " << UnScopedName(item.name);
    first = false;
  }
  co << ") {\n";
  co << body_text.str();
  co << "}\n";

  cur_offload_func_.co_source = co.str();
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

  // Write offload .co source files
  std::vector<std::string> offload_obj_files;
  for (size_t i = 0; i < offload_functions_.size(); ++i) {
    auto& fi = offload_functions_[i];
    auto co_file = build_path + "/" + fi.co_func_name + ".co";
    auto obj_file = build_path + "/" + fi.co_func_name + ".o";
    offload_obj_files.push_back(obj_file);

    out << "cat <<'EOF' > " << co_file << "\n";
    out << fi.co_source;
    out << "\nEOF\n\n";
  }

  bool has_offload = !offload_functions_.empty();

  out << "cat <<'EOF' > " << host_cc_file << "\n";
  for (auto& code : code_segments) out << code << "\n";
  out << "\nEOF\n\n";

  out << "CHOREO=\"${CHOREO:-choreo}\"\n";

  // Emit build env setup for all offload devices
  for (auto& [name, dcg] : device_codegens) {
    if (!dcg->IsHostDevice())
      dcg->SetupBuildEnv(out);
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
    // Step 1: Compile each offload .co file via choreo -target <target_name>
    for (size_t i = 0; i < offload_functions_.size(); ++i) {
      auto& fi = offload_functions_[i];
      auto co_file = build_path + "/" + fi.co_func_name + ".co";
      auto obj_file = offload_obj_files[i];
      out << "  echo \"Compiling offload function: " << fi.co_func_name
          << "\"\n";
      out << "  ${CHOREO} -t " << fi.target_name << " -c " << co_file
          << " -o " << obj_file << " -rtc=none"
          << " || { echo 'Offload compilation failed for "
          << fi.co_func_name << "'; exit 1; }\n\n";
    }

    // Step 2: Compile host C++ with the offload device's toolchain
    // (use the first offload device's toolchain for host compilation)
    DeviceCodeGen* offload_dcg = nullptr;
    for (auto& [name, dcg] : device_codegens) {
      if (!dcg->IsHostDevice()) { offload_dcg = dcg.get(); break; }
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
    if (offload_dcg)
      offload_dcg->EmitLinkCommand(out, all_objs, exe_file);
  } else {
    // Host-only: compile with g++, use any host device's link flags
    out << "  ${CXX:-g++} -std=c++17 -O2 -pthread -I" << build_path << " "
        << host_cc_file << " -o " << exe_file;
    for (auto& [name, dcg] : device_codegens)
      out << " " << dcg->LinkFlags();
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
