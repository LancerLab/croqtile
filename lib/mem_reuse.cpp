#include "mem_reuse.hpp"
#include "ast.hpp"
#include "aux.hpp"
#include "context.hpp"
#include "types.hpp"
#include "visitor.hpp"

using namespace Choreo;

bool MemAnalyzer::BeforeVisitImpl(AST::Node& n) {
  if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    parallel_level = 0;
    cur_dev_func_name = CurrentFunctionName();
    for (const auto& param : cf->f_decl.params->values) {
      if (!param->HasSymbol()) continue;
      std::string sname = InScopeName(param->sym->name);
      auto sty = dyn_cast<SpannedType>(param->GetType());
      if (!sty) continue;
      VST_DEBUG(dbgs() << "[memanlz] BUFFER: " << sname << "\n");
      buf_sto.emplace(sname, sty->GetStorage());
      // TODO: should we align the size to 512?!
      buf_size.emplace(sname, sty->ByteSizeValue());
      if (!sty->RuntimeShaped()) {
        VST_DEBUG(dbgs() << "\tstatic  size:  " << sty->ByteSizeValue()
                         << "\n");
      } else {
        VST_DEBUG(dbgs() << "\tdynamic  size: " << sty->ByteSizeValue()
                         << "\n";);
      }
      buf_dev_func_name.emplace(sname, cur_dev_func_name);
      VST_DEBUG(dbgs() << "\tdecl in dev func: " << cur_dev_func_name << "\n";);
    }
  } else if (isa<AST::ParallelBy>(&n)) {
    ++parallel_level;
    if (parallel_level == 1) cur_dev_func_name = SSTab().ScopeName();
  }
  return true;
}

bool MemAnalyzer::AfterVisitImpl(AST::Node& n) {
  if (isa<AST::ParallelBy>(&n)) {
    if (parallel_level == 1) cur_dev_func_name = CurrentFunctionName();
    parallel_level--;
  }
  return true;
}

bool MemAnalyzer::Visit(AST::NamedVariableDecl& n) {
  auto ty = GetSymbolType(n.name_str);
  auto sname = InScopeName(n.name_str);

  if (!have_dynamic_shape.count(cur_dev_func_name))
    have_dynamic_shape.emplace(cur_dev_func_name, false);

  if (auto et = dyn_cast<EventType>(ty)) {
    // need to consider the event type!
    event_vars.insert(sname);
    buf_sto.emplace(sname, n.mem->Get());
    buf_size.emplace(sname, sbe::nu(n.ArraySize()));
    buf_dev_func_name.emplace(sname, cur_dev_func_name);
    return true;
  }

  if (auto sty = dyn_cast<SpannedType>(ty); sty && !IsRef(n)) {
    VST_DEBUG(dbgs() << "[memanlz] BUFFER: " << sname << "\n");
    buf_sto.emplace(sname, sty->GetStorage());
    if (!sty->RuntimeShaped()) {
      auto total_size = sty->ByteSizeValue() * sbe::nu(n.ArraySize());
      buf_size.emplace(sname, total_size);
      VST_DEBUG(dbgs() << "\tstatic  size:  " << total_size << "\n");
    } else {
      have_dynamic_shape[cur_dev_func_name] = true;
      auto size_expr = sty->ByteSizeValue();
      if (n.IsArray()) size_expr = size_expr * sbe::nu(n.ArraySize());
      buf_size.emplace(sname, size_expr);
      VST_DEBUG(dbgs() << "\tdynamic  size: " << size_expr << "\n";);
    }
    buf_dev_func_name.emplace(sname, cur_dev_func_name);
    VST_DEBUG(dbgs() << "\tdecl in dev func: " << cur_dev_func_name << "\n";);
    return true;
  }

  return true;
}

bool MemReuse::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    Initialize();
    AnalyzeMemOffset();
  } else if (isa<AST::ChoreoFunction>(&n)) {
    parallel_level = 0;
    cur_dev_func_name = CurrentFunctionName();
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    parallel_level++;
    max_parallel_level = std::max(parallel_level, max_parallel_level);
    // for now, we are allowed to decl different memory inside paraby level 1.
    // so generate all kinds of spm at level 1.
    if (parallel_level == 1) {
      cur_dev_func_name = SSTab().ScopeName();
      if (DFCtx().shared_spm_size != 0) {
        DFCtx().shared_spm_name = SymbolTable::GetAnonName();
        auto shared_spm =
            AST::Make<AST::NamedVariableDecl>(n.LOC(), DFCtx().shared_spm_name);
        assert(DFCtx().shared_spm_size > 0 &&
               "Shared scratch pad memory size is not set.");
        auto ssty = MakeSpannedType(
            BaseType::U8, Shape(1, Size_t2Int(DFCtx().shared_spm_size)),
            Storage::SHARED);
        shared_spm->SetType(ssty);
        shared_spm->Note().insert_or_assign("spm", "true");
        pb->stmts->values.insert(pb->stmts->values.begin(), shared_spm);
        SSTab().DefineSymbol(DFCtx().shared_spm_name, ssty);
        VST_DEBUG(dbgs() << "Defined shared scratch pad memory: "
                         << PSTR(shared_spm) << ", type: " << PSTR(ssty)
                         << ".\n");
      }
      if (DFCtx().local_spm_size != 0) {
        DFCtx().local_spm_name = SymbolTable::GetAnonName();
        auto local_spm =
            AST::Make<AST::NamedVariableDecl>(n.LOC(), DFCtx().local_spm_name);
        assert(DFCtx().local_spm_size > 0 &&
               "Local scratch pad memory size is not set.");
        auto lsty = MakeSpannedType(
            BaseType::U8, Shape(1, Size_t2Int(DFCtx().local_spm_size)),
            Storage::LOCAL);
        local_spm->SetType(lsty);
        local_spm->Note().insert_or_assign("spm", "true");
        pb->stmts->values.insert(pb->stmts->values.begin(), local_spm);
        SSTab().DefineSymbol(DFCtx().local_spm_name, lsty);
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
    if (parallel_level == 1) {
      max_parallel_level = 0;
      cur_dev_func_name = CurrentFunctionName();
    }
    parallel_level--;
  }
  return true;
}

bool MemReuse::Visit(AST::NamedVariableDecl& n) {
  if (isa<AST::Select>(n.init_expr)) return true;
  if (n.Note().count("spm")) return true;
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
    std::string dev_func_name = GetDeclDevFuncOfBuffer(sname);
    if (auto sv = VIInt(size))
      DFCtx(dev_func_name)
          .buffers.push_back({.size = (size_t)sv.value(),
                              .start_time = ranges.Values()[0].start,
                              .end_time = ranges.Values()[0].end,
                              .buffer_id = sname});
    else
      DFCtx(dev_func_name)
          .dynamic_buffers.push_back({.size = STR(size),
                                      .start_time = ranges.Values()[0].start,
                                      .end_time = ranges.Values()[0].end,
                                      .buffer_id = sname});
  }

  VST_DEBUG({
    for (const auto& [df_name, ctx] : DFCtxs()) {
      dbgs() << "For '" << df_name << "'\n";
      for (const auto& buffer : ctx.buffers) {
        dbgs() << "static  buffer: " << buffer.buffer_id << "\n\t"
               << STR(ma.buf_sto.at(buffer.buffer_id))
               << ", size: " << buffer.size
               << ", start_time: " << buffer.start_time
               << ", end_time: " << buffer.end_time << "\n";
      }
      for (const auto& buffer : ctx.dynamic_buffers) {
        dbgs() << "dynamic buffer: " << buffer.buffer_id << "\n\t"
               << STR(ma.buf_sto.at(buffer.buffer_id))
               << ", size: " << buffer.size
               << ", start_time: " << buffer.start_time
               << ", end_time: " << buffer.end_time << "\n";
      }
    }
  });
}

void MemReuse::AnalyzeMemOffset() {
  for (auto& [df_name, ctx] : DFCtxs()) ProtoType(df_name, ctx);
}

void MemReuse::ProtoType(const std::string& df_name, DevFuncMemReuseCtx& ctx) {
  std::string co_func_name = GetFuncNameFromScopedName(df_name);
  if (ma.have_dynamic_shape.count(df_name) &&
      ma.have_dynamic_shape.at(df_name)) {
    std::set<Storage> required_storage;
    // JIT memory reuse script
    std::vector<std::string> script;
    // the args which are passed to device function
    FunctionContext::MemReuseOffsetMap offset_args;

    auto GenPushBackScript = [&](const auto& bs) -> void {
      for (const auto& buffer : bs) {
        auto sto = ma.buf_sto.at(buffer.buffer_id);
        // global buffer reuse is not supported yet
        if (sto == Storage::GLOBAL || sto == Storage::DEFAULT) continue;
        if (sto != Storage::LOCAL && sto != Storage::SHARED)
          choreo_unreachable("The storage type: " + STR(sto) +
                             " is not supported yet!");
        offset_args[sto].push_back("mr_offset" + buffer.buffer_id);
        if (!required_storage.count(sto)) {
          required_storage.insert(sto);
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

    auto TotalEventSize = [&](Storage sto) -> size_t {
      size_t total_event_size = 0;
      for (const auto& event : ma.event_vars) {
        if (GetDeclDevFuncOfBuffer(event) != df_name) continue;
        if (ma.buf_sto.at(event) != sto) continue;
        auto event_size = ma.buf_size.at(event);
        assert(VIIsInt(event_size));
        total_event_size += VIInt(event_size).value();
      }
      return total_event_size;
    };

    GenPushBackScript(ctx.buffers);
    GenPushBackScript(ctx.dynamic_buffers);

    script.push_back("HeapSimulator __co__heap_simulator;");
    for (const auto& sto : required_storage) {
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
                       stos + " spm should not exceed the memory usage limit " +
                       std::to_string(mem_capacity) + "bytes.\");");
      size_t total_event_size = TotalEventSize(sto);
      if (sto == Storage::LOCAL)
        ctx.local_spm_size = mem_capacity - AlignUp(total_event_size, 8);
      else if (sto == Storage::SHARED)
        ctx.shared_spm_size = mem_capacity - AlignUp(total_event_size, 8);
      // generate offsets in array
      script.push_back("unsigned long __co__" + stos + "_chunk_offsets[" +
                       std::to_string(offset_args.at(sto).size()) + "];");
      // TODO: need validation?
      script.push_back("size_t __co__" + stos + "_chunk_idx = 0;");
      script.push_back("for (const auto& [buffer_id, offset] : __co__" + stos +
                       "_result.chunk_offsets)");
      script.push_back("  __co__" + stos + "_chunk_offsets[__co__" + stos +
                       "_chunk_idx++] = offset;");
    }
    FCtx(co_func_name).SetMemReuseScript(df_name, script);

    // record the offset args in sorted order
    for (auto& [sto, args] : offset_args) std::sort(args.begin(), args.end());
    FCtx(co_func_name).SetMemReuseOffsetArgs(df_name, offset_args);

    return;
  }
  // All the buffers are static.
  HeapSimulator::Chunks local_chunks;
  HeapSimulator::Chunks shared_chunks;

  for (const auto& buffer : ctx.buffers) {
    if (auto sto = ma.buf_sto.at(buffer.buffer_id); sto == Storage::LOCAL)
      local_chunks.push_back(buffer);
    else if (sto == Storage::SHARED)
      shared_chunks.push_back(buffer);
    else if (sto == Storage::GLOBAL || sto == Storage::DEFAULT)
      continue;
    else
      choreo_unreachable("The storage type " + STR(sto) +
                         " is not supported yet!");
  }

  HeapSimulator simulator;

  if (!local_chunks.empty()) {
    HeapSimulator::Result local_result = simulator.Allocate(local_chunks, 512);
    assert(ValidateResult(local_result, local_chunks));
    ctx.local_spm_size = local_result.heap_size;
    for (const auto& [buffer_id, offset] : local_result.chunk_offsets)
      ctx.mem_offset.emplace(buffer_id, offset);
    VST_DEBUG(dbgs() << "For '" << df_name << "'\n\tLocal memory usage: "
                     << local_result.heap_size << " bytes\n");
  }
  if (!shared_chunks.empty()) {
    HeapSimulator::Result shared_result =
        simulator.Allocate(shared_chunks, 512);
    assert(ValidateResult(shared_result, shared_chunks));
    ctx.shared_spm_size = shared_result.heap_size;
    for (const auto& [buffer_id, offset] : shared_result.chunk_offsets)
      ctx.mem_offset.emplace(buffer_id, offset);
    VST_DEBUG(dbgs() << "For '" << df_name << "'\n\tShared memory usage: "
                     << shared_result.heap_size << " bytes\n");
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
  VST_DEBUG(dbgs() << STR(sto) << " buffer: " << sname << "\n\t";);

  bool dynamic = ma.have_dynamic_shape.at(cur_dev_func_name);
  if (!DFCtx().mem_offset.count(sname) && !dynamic) {
    VST_DEBUG(dbgs() << "has no valid reuse offset!\n");
    return;
  }

  std::string spm_name = (sto == Storage::LOCAL ? DFCtx().local_spm_name
                                                : DFCtx().shared_spm_name);
  std::string offset = dynamic ? "mr_offset" + RegexReplaceAll(sname, "::", "_")
                               : std::to_string(DFCtx().mem_offset.at(sname));
  VST_DEBUG({
    dbgs() << "using spm:   " << spm_name << "\n\twith offset: " << offset
           << "\n";
  });

  n.Note().emplace("reuse", spm_name);
  n.Note().emplace("offset", offset);
}
