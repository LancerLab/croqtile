#include "mem_reuse.hpp"
#include "ast.hpp"
#include "aux.hpp"
#include "context.hpp"
#include "types.hpp"
#include "visitor.hpp"

using namespace Choreo;

bool MemAnalyzer::BeforeVisitImpl(AST::Node& n) {
  if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    for (const auto& param : cf->f_decl.params->values) {
      if (!param->HasSymbol()) continue;
      std::string sname = InScopeName(param->sym->name);
      auto sty = dyn_cast<SpannedType>(param->GetType());
      if (!sty) continue;
      VST_DEBUG(dbgs() << "[memanlz] BUFFER: " << sname << "\n");
      buf_sto.emplace(sname, sty->GetStorage());
      if (!sty->RuntimeShaped()) {
        // TODO: should we align the size to 512?!
        VST_DEBUG(dbgs() << "\tstatic  size:  " << sty->ByteSize() << "\n");
        buf_size.emplace(sname, sty->ByteSize());
      } else {
        auto size_expr = sty->ByteSizeExpression();
        // if (!sym_expr_map.count(size_expr)) {
        //   auto shape_expr = sty->ShapeSizeExpression();
        //   auto sym_size_expr = (GetSymExprFromSizeExpr(shape_expr) *
        //                         SymExpr(SizeOf(sty->f_type)))
        //                            .expand();
        //   sym_expr_map.emplace(size_expr, sym_size_expr);
        // }
        buf_size.emplace(sname, size_expr);
        VST_DEBUG({
          // dbgs() << "\tdynamic  size: " << size_expr
          //        << "\n\tsymbolic size: " << sym_expr_map.at(size_expr) <<
          //        "\n";
          dbgs() << "\tdynamic  size: " << size_expr << "\n";
        });
      }
    }
  }
  return true;
}

bool MemAnalyzer::Visit(AST::NamedVariableDecl& n) {
  auto ty = GetSymbolType(n.name_str);
  auto sname = InScopeName(n.name_str);

  if (auto et = dyn_cast<EventType>(ty)) {
    // need to consider the event type!
    event_vars.insert(sname);
    buf_sto.emplace(sname, n.mem->Get());
    buf_size.emplace(sname, n.ArraySize());
    return true;
  }

  if (auto sty = dyn_cast<SpannedType>(ty); sty && !IsRef(n)) {
    VST_DEBUG(dbgs() << "[memanlz] BUFFER: " << sname << "\n");
    buf_sto.emplace(sname, sty->GetStorage());
    if (!sty->RuntimeShaped()) {
      size_t total_size = sty->ByteSize() * n.ArraySize();
      buf_size.emplace(sname, total_size);
      VST_DEBUG(dbgs() << "\tstatic  size:  " << total_size << "\n");
    } else {
      have_dynamic_shape = true;
      auto size_expr = sty->ByteSizeExpression();
      if (n.IsArray())
        size_expr =
            "(" + size_expr + ") * (" + std::to_string(n.ArraySize()) + ")";
      // if (!sym_expr_map.count(size_expr)) {
      //   auto shape_expr = sty->ShapeSizeExpression();
      //   auto sym_size_expr =
      //       (GetSymExprFromSizeExpr(shape_expr) * SymExpr(n.ArraySize()) *
      //        SymExpr(SizeOf(sty->f_type)))
      //           .expand();
      //   sym_expr_map.emplace(size_expr, sym_size_expr);
      // }
      buf_size.emplace(sname, size_expr);
      VST_DEBUG({
        // dbgs() << "\tdynamic  size: " << size_expr
        //        << "\n\tsymbolic size: " << sym_expr_map.at(size_expr) <<
        //        "\n";
        dbgs() << "\tdynamic  size: " << size_expr << "\n";
      });
    }
    return true;
  }

  return true;
}

// MemAnalyzer::SymExpr
// MemAnalyzer::StringifyOpFromSymExpr(const SymExpr& sym_expr_l,
//                                     const std::string& op,
//                                     const SymExpr& sym_expr_r) {
//   std::string symbol_name;
//   std::string sym_expr_l_str = ExSTR(sym_expr_l.expand());
//   std::string sym_expr_r_str = ExSTR(sym_expr_r.expand());
//   symbol_name = "(" + sym_expr_l_str + op + sym_expr_r_str + ")";
//   return GetSymExprFromStr(symbol_name);
// }

// MemAnalyzer::SymExpr MemAnalyzer::GetSymExprFromStr(std::string str) {
//   if (symbol_map.count(str)) return SymExpr(symbol_map.at(str));
//   auto IsNumber = [](const std::string& str) {
//     return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit);
//   };
//   if (IsNumber(str)) { return SymExpr(std::stoi(str)); }
//   Symbol symbol(str, str);
//   symbol_map.emplace(str, symbol);
//   return SymExpr(symbol);
// }

// MemAnalyzer::SymExpr
// MemAnalyzer::GetSymExprFromSizeExpr(std::string size_expr) {
//   auto IsOperator = [](char c) -> bool {
//     return c == '+' || c == '-' || c == '*' || c == '/' || c == '%';
//   };

//   std::string temp = "";
//   for (auto c : size_expr)
//     if (c != ' ') temp += c;
//   size_expr = temp;

//   std::function<SymExpr(std::string)> HelperFunc = [&](std::string str) {
//     size_t size = str.length();
//     assert(!str.empty());
//     if (str[0] != '(') return GetSymExprFromStr(str);
//     size_t idx = 0;
//     size_t leftCount = 0;
//     do {
//       char c = str[idx];
//       if (c == '(')
//         ++leftCount;
//       else if (c == ')')
//         --leftCount;
//       if (leftCount == 0) break;
//       idx++;
//     } while (idx < size);

//     auto left_expr = HelperFunc(str.substr(1, idx - 1));

//     if (idx == size - 1) return left_expr;
//     char c = str[++idx];
//     if (!IsOperator(c))
//       choreo_unreachable("The operator(single char) " + std::string(1, c) +
//                          " is not supported in MemAnalyzer yet.");
//     std::string op = std::string(1, c);
//     auto right_expr = HelperFunc(str.substr(idx + 1));
//     SymExpr res;
//     if (op == "+")
//       res = SymExpr(left_expr + right_expr);
//     else if (op == "-")
//       res = SymExpr(left_expr - right_expr);
//     else if (op == "*")
//       res = SymExpr(left_expr * right_expr);
//     else if (op == "/" || op == "%")
//       res = StringifyOpFromSymExpr(left_expr, op, right_expr);
//     else
//       choreo_unreachable("The operator " + op +
//                          " is not supported in MemAnalyzer yet.");
//     return res;
//   };

//   return HelperFunc(size_expr);
// }

bool MemReuse::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    Initialize();
    AnalyzeMemOffset();
  } else if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    cur_func_name = cf->name;
    parallel_level = 0;
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    parallel_level++;
    max_parallel_level = std::max(parallel_level, max_parallel_level);
    // for now, we are allowed to decl different memory inside paraby level 1.
    // so generate all kinds of spm at level 1.
    if (parallel_level == 1) {
      size_t shared_spm_size = spm_size_map[cur_func_name].shared_spm_size;
      if (shared_spm_size != 0) {
        shared_spm_name = SymbolTable::GetAnonName();
        auto shared_spm =
            AST::Make<AST::NamedVariableDecl>(n.LOC(), shared_spm_name);
        assert(shared_spm_size > 0 &&
               "Shared scratch pad memory size is not set.");
        auto ssty =
            MakeSpannedType(BaseType::U8, Shape(1, Size_t2Int(shared_spm_size)),
                            Storage::SHARED);
        shared_spm->SetType(ssty);
        shared_spm->AppendNote("spm,");
        pb->stmts->values.insert(pb->stmts->values.begin(), shared_spm);
        SSTab().DefineSymbol(shared_spm_name, ssty);
        VST_DEBUG(dbgs() << "Defined shared scratch pad memory: "
                         << PSTR(shared_spm) << ", type: " << PSTR(ssty)
                         << ".\n");
      }
      size_t local_spm_size = spm_size_map[cur_func_name].local_spm_size;
      if (local_spm_size != 0) {
        local_spm_name = SymbolTable::GetAnonName();
        auto local_spm =
            AST::Make<AST::NamedVariableDecl>(n.LOC(), local_spm_name);
        assert(local_spm_size > 0 &&
               "Local scratch pad memory size is not set.");
        auto lsty = MakeSpannedType(
            BaseType::U8, Shape(1, Size_t2Int(local_spm_size)), Storage::LOCAL);
        local_spm->SetType(lsty);
        local_spm->AppendNote("spm,");
        pb->stmts->values.insert(pb->stmts->values.begin(), local_spm);
        SSTab().DefineSymbol(local_spm_name, lsty);
        VST_DEBUG(dbgs() << "Defined local scratch pad memory: "
                         << PSTR(local_spm) << ", type: " << PSTR(lsty)
                         << ".\n");
      }
      // TODO: subthread
    }
  }
  return true;
}

bool MemReuse::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::ParallelBy>(&n)) {
    if (parallel_level == 1) max_parallel_level = 0;
    parallel_level--;
  }
  return true;
}

bool MemReuse::Visit(AST::NamedVariableDecl& n) {
  if (isa<AST::Select>(n.init_expr)) return true;
  if (n.note.find("spm") != std::string::npos) return true;
  auto ty = GetSymbolType(n.name_str);
  if (auto sty = dyn_cast<SpannedType>(ty)) {
    auto sto = sty->GetStorage();
    if (sto == Storage::LOCAL || sto == Storage::SHARED) ApplyMemOffset(n, sto);
  }
  return true;
}

void MemReuse::Initialize() {
  const auto& var_ranges = la.VarRanges();

  for (const auto& [sname, size] : ma.buf_size) {
    // do not consider the event vars for now
    // cause shared events have `__volatile__` attribute
    if (ma.event_vars.count(sname)) {
      VST_DEBUG(dbgs() << "Ignore event buffer " << sname << ".\n");
      continue;
    }
    // TODO: local event?

    auto ranges = var_ranges.at(sname);
    if (ranges.Values().size() == 0) {
      VST_DEBUG(dbgs() << "Warning: buffer " << sname << " is never used!\n");
      continue;
    }
    // For now, there is no case that a var is used in multiple ranges.
    // Because there is no reassignment.
    if (ranges.Values().size() > 1) {
      VST_DEBUG({
        dbgs() << "Warning: buffer " << sname
               << " is used in multiple ranges:\n";
        for (const auto& r : ranges.Values())
          dbgs() << "\t[" << r.start << ", " << r.end << "]\n";
      });
      choreo_unreachable("multiple ranges for a buffer is not supported yet.");
    }
    if (std::holds_alternative<size_t>(size))
      buffers.push_back({std::get<size_t>(size), ranges.Values()[0].start,
                         ranges.Values()[0].end, sname});
    else
      dynamic_buffers.push_back({std::get<std::string>(size),
                                 ranges.Values()[0].start,
                                 ranges.Values()[0].end, sname});
  }

  VST_DEBUG({
    for (const auto& buffer : buffers) {
      dbgs() << "static  buffer: " << buffer.buffer_id << "\n\t"
             << STR(ma.buf_sto.at(buffer.buffer_id))
             << ", size: " << buffer.size
             << ", start_time: " << buffer.start_time
             << ", end_time: " << buffer.end_time << "\n";
    }
    for (const auto& buffer : dynamic_buffers) {
      dbgs() << "dynamic buffer: " << buffer.buffer_id << "\n\t"
             << STR(ma.buf_sto.at(buffer.buffer_id))
             << ", size: " << buffer.size
             << ", start_time: " << buffer.start_time
             << ", end_time: " << buffer.end_time << "\n";
    }
  });
}

void MemReuse::AnalyzeMemOffset() { ProtoType(); }

void MemReuse::ProtoType() {
  auto GetFuncNameFromScopedName = [](const std::string& name) -> std::string {
    if (!PrefixedWith(name, "::"))
      choreo_unreachable("The scopedname should contain '::'!");
    return SplitStringByDelimiter(name, "::", true)[0];
  };

  if (ma.have_dynamic_shape) {
    std::map<std::string, std::set<Storage>> required_storage_maps;
    std::map<std::string, std::vector<std::string>> mem_reuse_scripts;
    std::map<std::string, std::map<Storage, std::vector<std::string>>>
        offsets_arg_map;

    auto GenPushBackScript = [&](const auto& bs) -> void {
      for (const auto& buffer : bs) {
        auto sto = ma.buf_sto.at(buffer.buffer_id);
        // global buffer reuse is not supported yet
        if (sto == Storage::GLOBAL || sto == Storage::DEFAULT) continue;
        if (sto != Storage::LOCAL && sto != Storage::SHARED)
          choreo_unreachable("The storage type: " + STR(sto) +
                             " is not supported yet!");
        auto func_name = GetFuncNameFromScopedName(buffer.buffer_id);
        auto& required_storage_map = required_storage_maps[func_name];
        auto& script = mem_reuse_scripts[func_name];
        offsets_arg_map[func_name][sto].push_back("mr_offset" +
                                                  buffer.buffer_id);
        if (!required_storage_map.count(sto)) {
          required_storage_map.insert(sto);
          script.insert(script.begin(),
                        "HeapSimulator::Chunks __co__" + STR(sto) + "_chunks;");
        }
        std::string buffer_size;
        if constexpr (std::is_same_v<decltype(buffer.size), std::string>)
          buffer_size = UnScopedExpr(buffer.size);
        else if constexpr (std::is_same_v<decltype(buffer.size), size_t>)
          buffer_size = UnScopedExpr(std::to_string(buffer.size));
        else
          choreo_unreachable("Unexpected type of buffer.size: " +
                             std::string(typeid(buffer.size).name()) +
                             "\n\twith buffer " + buffer.buffer_id);
        script.push_back(
            "__co__" + STR(sto) + "_chunks.push_back({" + buffer_size + ", " +
            std::to_string(buffer.start_time) + ", " +
            std::to_string(buffer.end_time) + ", \"" +
            RegexReplaceAll(buffer.buffer_id, "::", "_") + "\"});");
      }
    };

    auto TotalEventSize = [&](const std::string& func_name,
                              Storage sto) -> size_t {
      size_t total_event_size = 0;
      for (const auto& event : ma.event_vars) {
        if (GetFuncNameFromScopedName(event) != func_name) continue;
        if (ma.buf_sto.at(event) != sto) continue;
        auto event_size = ma.buf_size.at(event);
        assert(std::holds_alternative<size_t>(event_size));
        total_event_size += std::get<size_t>(event_size);
      }
      return total_event_size;
    };

    GenPushBackScript(buffers);
    GenPushBackScript(dynamic_buffers);

    for (auto& [func_name, script] : mem_reuse_scripts) {
      script.push_back("HeapSimulator __co__heap_simulator;");
      for (const auto& sto : required_storage_maps.at(func_name)) {
        std::string stos = STR(sto);
        // TODO: Is shared alignment needed?
        script.push_back("HeapSimulator::Result __co__" + stos +
                         "_result = "
                         "__co__heap_simulator.Allocate(__co__" +
                         stos + "_chunks, 512);");
        std::string spm_size_var = "__co__" + stos + "_spm_size";
        script.push_back("unsigned " + spm_size_var + " = __co__" + stos +
                         "_result.heap_size;");
        // special case for RtCheck which emits after general RtCheck.
        size_t mem_capacity = CCtx().GetMemCapacity(sto);
        script.push_back("choreo::runtime_check(" + spm_size_var +
                         " <= (size_t)" + std::to_string(mem_capacity) +
                         ", \"In the memory reuse of dynamic shapes, the size "
                         "of the initial " +
                         stos +
                         " spm should not exceed the memory usage limit " +
                         std::to_string(mem_capacity) + "bytes.\");");
        size_t total_event_size = TotalEventSize(func_name, sto);
        if (sto == Storage::LOCAL)
          spm_size_map[func_name].local_spm_size =
              mem_capacity - AlignUp(total_event_size, 8);
        else if (sto == Storage::SHARED)
          spm_size_map[func_name].shared_spm_size =
              mem_capacity - AlignUp(total_event_size, 8);
        // generate offsets in array
        script.push_back(
            "unsigned long __co__" + stos + "_chunk_offsets[" +
            std::to_string(offsets_arg_map.at(func_name).at(sto).size()) +
            "];");
        // TODO: need validation?
        script.push_back("size_t __co__" + stos + "_chunk_idx = 0;");
        script.push_back("for (const auto& [buffer_id, offset] : __co__" +
                         stos + "_result.chunk_offsets)");
        script.push_back("  __co__" + stos + "_chunk_offsets[__co__" + stos +
                         "_chunk_idx++] = offset;");
      }
      FCtx(func_name).SetMemReuseScript(script);
      // record the offset args in sorted order
      auto& chunks = offsets_arg_map.at(func_name);
      std::map<Storage, std::vector<std::string>> offset_args;
      for (Storage sto : {Storage::LOCAL, Storage::SHARED})
        if (chunks.count(sto)) {
          auto& args = chunks.at(sto);
          std::sort(args.begin(), args.end());
          offset_args.emplace(sto, args);
        }
      FCtx(func_name).SetMemReuseOffsetArgs(offset_args);
    }
    return;
  }

  std::map<std::string, HeapSimulator::Chunks> local_chunks_map;
  std::map<std::string, HeapSimulator::Chunks> shared_chunks_map;

  for (const auto& buffer : buffers) {
    auto func_name = GetFuncNameFromScopedName(buffer.buffer_id);
    if (auto sto = ma.buf_sto.at(buffer.buffer_id); sto == Storage::LOCAL)
      local_chunks_map[func_name].push_back(buffer);
    else if (sto == Storage::SHARED)
      shared_chunks_map[func_name].push_back(buffer);
    else if (sto == Storage::GLOBAL || sto == Storage::DEFAULT)
      continue;
    else
      choreo_unreachable("The storage type " + STR(sto) +
                         " is not supported yet!");
  }

  HeapSimulator simulator;

  for (const auto& [func_name, local_chunks] : local_chunks_map) {
    if (!local_chunks.empty()) {
      HeapSimulator::Result local_result =
          simulator.Allocate(local_chunks, 512);
      assert(ValidateResult(local_result, local_chunks));
      spm_size_map[func_name].local_spm_size = local_result.heap_size;
      for (const auto& [buffer_id, offset] : local_result.chunk_offsets)
        mem_offset.emplace(buffer_id, offset);
      VST_DEBUG(dbgs() << "Function: " << func_name
                       << "\n\tLocal memory usage: " << local_result.heap_size
                       << " bytes\n");
    }
    if (const auto& shared_chunks = shared_chunks_map[func_name];
        !shared_chunks.empty()) {
      HeapSimulator::Result shared_result =
          simulator.Allocate(shared_chunks, 512);
      assert(ValidateResult(shared_result, shared_chunks));
      spm_size_map[func_name].shared_spm_size = shared_result.heap_size;
      for (const auto& [buffer_id, offset] : shared_result.chunk_offsets)
        mem_offset.emplace(buffer_id, offset);
      VST_DEBUG(dbgs() << "Function: " << func_name
                       << "\n\tShared memory usage: " << shared_result.heap_size
                       << " bytes\n");
    }
  }
}

bool MemReuse::ValidateResult(const HeapSimulator::Result& res,
                              const HeapSimulator::Chunks& chunks) {
  size_t size = chunks.size();
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      if (i == j) continue;
      const auto& c1 = chunks[i];
      const auto& c2 = chunks[j];
      if (c1.start_time <= c2.end_time && c2.start_time <= c1.end_time) {
        auto o1 = res.chunk_offsets.at(c1.buffer_id);
        auto o2 = res.chunk_offsets.at(c2.buffer_id);
        if ((o1 <= o2 && o1 + c1.size > o2) ||
            (o2 <= o1 && o2 + c2.size > o1)) {
          dbgs() << "Error: unexpect memory overlap detected between buffers "
                 << c1.buffer_id << " and " << c2.buffer_id
                 << " after applying memory reuse.\n";
          return false;
        }
      }
    }
  }
  return true;
}

void MemReuse::ApplyMemOffset(AST::NamedVariableDecl& n, Storage sto) {
  assert(sto == Storage::LOCAL || sto == Storage::SHARED);
  auto sname = InScopeName(n.name_str);
  auto spm_name = (sto == Storage::LOCAL ? local_spm_name : shared_spm_name);
  VST_DEBUG(dbgs() << STR(sto) << " buffer: " << sname << "\n\t";);
  if (!mem_offset.count(sname) && !ma.have_dynamic_shape) {
    VST_DEBUG(dbgs() << "has no valid reuse offset!\n");
    return;
  }
  std::string offset = ma.have_dynamic_shape
                           ? "mr_offset" + RegexReplaceAll(sname, "::", "_")
                           : std::to_string(mem_offset.at(sname));
  VST_DEBUG({
    dbgs() << "using spm:   " << spm_name << "\n\twith offset: " << offset
           << "\n";
  });
  n.note.append("reuse, " + spm_name + ", ");
  n.note.append("offset, " + offset + ", ");
}
