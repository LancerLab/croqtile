#include "mem_reuse.hpp"
#include "ast.hpp"
#include "aux.hpp"
#include "context.hpp"
#include "dmaconf.hpp"
#include "types.hpp"
#include "visitor.hpp"

using namespace Choreo;

namespace {

struct SharedAlignmentCollector : public VisitorWithSymTab {
  std::map<std::string, size_t>& shared_alignment_reqs;
  std::string cur_dev_fname;

  SharedAlignmentCollector(std::map<std::string, size_t>& reqs)
      : VisitorWithSymTab("swiz_align"), shared_alignment_reqs(reqs) {}

private:
  static std::string GetCoFuncName(const std::string& scoped_name) {
    if (!PrefixedWith(scoped_name, "::")) return scoped_name;
    return SplitFirst(scoped_name, "::");
  }

  bool RunOnProgramImpl(AST::Node& root) override {
    root.accept(*this);
    return !HasError();
  }

  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      cur_dev_fname = CurrentFunctionName();
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      if (pb->IsDeviceEntry()) cur_dev_fname = SSTab().ScopeName();
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      if (pb->IsDeviceEntry()) cur_dev_fname = CurrentFunctionName();
    }
    return true;
  }

  bool Visit(AST::MMA& n) override {
    auto op = n.GetOperation();
    if (!op || !op->IsLoad()) return true;

    auto frag_sym = AST::FragName(op->LoadTo());
    auto scoped_frag_sym = InScopeName(frag_sym);
    auto co_func_name = GetCoFuncName(cur_dev_fname);
    if (!FCtx(co_func_name).FragIsWGMMA(scoped_frag_sym)) return true;

    size_t alignment = std::max(CCtx().GetMemoryAlignmentByte(Storage::SHARED),
                                SwizzleAlignmentByte(op->GetSwizzleMode()));
    shared_alignment_reqs[cur_dev_fname] =
        std::max(shared_alignment_reqs[cur_dev_fname], alignment);
    shared_alignment_reqs[co_func_name] =
        std::max(shared_alignment_reqs[co_func_name], alignment);
    return true;
  }
};

} // namespace

void MemReuse::CollectSharedAlignmentRequirements(AST::Node& root) {
  shared_alignment_reqs.clear();
  SharedAlignmentCollector collector(shared_alignment_reqs);
  collector.SSTab().UpdateGlobal(SymTab());
  collector.RunOnProgram(root);
}

size_t MemReuse::SharedAlignmentForDevFunc(const std::string& df_name) const {
  size_t alignment = CCtx().GetMemoryAlignmentByte(Storage::SHARED);
  if (auto it = shared_alignment_reqs.find(df_name);
      it != shared_alignment_reqs.end())
    alignment = std::max(alignment, it->second);
  std::string co_func_name = GetFuncNameFromScopedName(df_name);
  if (auto it = shared_alignment_reqs.find(co_func_name);
      it != shared_alignment_reqs.end())
    alignment = std::max(alignment, it->second);
  return alignment;
}

bool MemAnalyzer::BeforeVisitImpl(AST::Node& n) {
  if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
    parallel_level = 0;
    cur_dev_fname = CurrentFunctionName();
    for (const auto& param : cf->f_decl.params->values) {
      if (!param->HasSymbol()) continue;
      std::string sname = InScopeName(param->sym->name);
      auto sty = dyn_cast<SpannedType>(param->GetType());
      if (!sty) continue;
      VST_DEBUG(dbgs() << "[memanlz] BUFFER: " << sname << "\n");
      buf_sto.emplace(sname, sty->GetStorage());
      buf_size.emplace(sname, sty->ByteSizeValue());
      if (!sty->RuntimeShaped())
        VST_DEBUG(dbgs() << "\tstatic  size:  " << sty->ByteSizeValue()
                         << "\n");
      else
        VST_DEBUG(dbgs() << "\tdynamic  size: " << sty->ByteSizeValue()
                         << "\n";);
      buf_dev_func_name.emplace(sname, cur_dev_fname);
      VST_DEBUG(dbgs() << "\tdecl in dev func: " << cur_dev_fname << "\n";);
    }
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    ++parallel_level;
    if (pb->IsDeviceEntry()) cur_dev_fname = SSTab().ScopeName();
  }
  return true;
}

bool MemAnalyzer::AfterVisitImpl(AST::Node& n) {
  if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    if (pb->IsDeviceEntry()) cur_dev_fname = CurrentFunctionName();
    --parallel_level;
  }
  return true;
}

bool MemAnalyzer::Visit(AST::NamedVariableDecl& n) {
  auto ty = GetSymbolType(n.name_str);
  ValueItem elem_count = sbe::nu(1);
  auto aty = dyn_cast<ArrayType>(ty);
  if (aty) elem_count = aty->ElemCount();
  auto sname = InScopeName(n.name_str);

  if (auto et = dyn_cast<EventType>(ty)) {
    // need to consider the event type!
    event_vars.insert(sname);
    buf_sto.emplace(sname, n.mem->Get());
    // event is bool
    buf_size.emplace(sname, elem_count);
    buf_dev_func_name.emplace(sname, cur_dev_fname);
    return true;
  }

  if (auto sty = dyn_cast<SpannedType>(ty); sty && !IsRef(n)) {
    VST_DEBUG(dbgs() << "[memanlz] BUFFER: " << sname << "\n");
    auto sto = sty->GetStorage();
    buf_sto.emplace(sname, sto);
    if (!sty->RuntimeShaped()) {
      auto total_size = sty->ByteSizeValue() * elem_count;
      buf_size.emplace(sname, total_size);
      VST_DEBUG(dbgs() << "\tstatic  size:  " << total_size << "\n");
    } else {
      sto_have_dyn[cur_dev_fname][sto] = true;
      auto size_expr = sty->ByteSizeValue();
      if (n.IsArray()) size_expr = size_expr * elem_count;
      buf_size.emplace(sname, size_expr);
      VST_DEBUG(dbgs() << "\tdynamic  size: " << size_expr << "\n";);
    }
    buf_dev_func_name.emplace(sname, cur_dev_fname);
    VST_DEBUG(dbgs() << "\tdecl in dev func: " << cur_dev_fname << "\n";);
    return true;
  }

  return true;
}

bool MemReuse::BeforeVisitImpl(AST::Node& n) {
  if (isa<AST::Program>(&n)) {
    CollectSharedAlignmentRequirements(n);
    Initialize();
    AnalyzeMemOffset();
  } else if (isa<AST::ChoreoFunction>(&n)) {
    parallel_level = 0;
    cur_dev_fname = CurrentFunctionName();
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    parallel_level++;
    if (pb->IsDeviceEntry()) {
      cur_dev_fname = SSTab().ScopeName();
      if (DFCtx().shared_spm_size != 0) {
        size_t shared_alignment = SharedAlignmentForDevFunc(cur_dev_fname);
        DFCtx().shared_spm_name = SymbolTable::GetAnonName();
        auto shared_spm =
            AST::Make<AST::NamedVariableDecl>(n.LOC(), DFCtx().shared_spm_name);
        assert(DFCtx().shared_spm_size > 0 &&
               "Shared scratch pad memory size is not set.");
        auto ssty = MakeDenseSpannedType(
            BaseType::U8, Shape(1, Size_t2Int(DFCtx().shared_spm_size)),
            Storage::SHARED);
        shared_spm->SetType(ssty);
        shared_spm->AddNote("spm");
        shared_spm->AddNote("alignment", std::to_string(shared_alignment));
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
        auto lsty = MakeDenseSpannedType(
            BaseType::U8, Shape(1, Size_t2Int(DFCtx().local_spm_size)),
            Storage::LOCAL);
        local_spm->SetType(lsty);
        local_spm->AddNote("spm");
        local_spm->AddNote(
            "alignment",
            std::to_string(CCtx().GetMemoryAlignmentByte(Storage::LOCAL)));
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
  if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    if (pb->IsDeviceEntry()) cur_dev_fname = CurrentFunctionName();
    parallel_level--;
  }
  return true;
}

bool MemReuse::Visit(AST::NamedVariableDecl& n) {
  if (isa<AST::Select>(n.init_expr)) return true;
  if (n.HasNote("spm")) return true;
  if (n.HasNote("ref")) return true;
  auto ty = GetSymbolType(n.name_str);
  if (auto sty = dyn_cast<SpannedType>(ty)) {
    auto sto = sty->GetStorage();
    if (ShouldReuseStorage(sto, cur_dev_fname)) ApplyMemOffset(n, sto);
  }
  return true;
}

bool MemReuse::ShouldReuseStorage(Storage sto,
                                  const std::string& dev_func_name) const {
  if (sto == Storage::SHARED) return true;
  if (sto != Storage::LOCAL) return false;
  if (CCtx().TargetName() != "cute") return true;

  // For CUTE local buffers:
  // - static shape: skip reuse to preserve register/scalar promotion chances.
  // - dynamic shape: keep reuse because VLA-style local arrays cannot be
  //   reliably registerized and need runtime-managed storage.
  if (dev_func_name == "") return false;
  if (!ma.sto_have_dyn.count(dev_func_name)) return false;
  auto& dyn_flags = ma.sto_have_dyn.at(dev_func_name);
  if (!dyn_flags.count(Storage::LOCAL)) return false;
  return dyn_flags.at(Storage::LOCAL);
}

bool MemReuse::ShouldReuseBuffer(const std::string& buffer_id, Storage sto,
                                 const std::string& dev_func_name) const {
  if (!ShouldReuseStorage(sto, dev_func_name)) return false;
  if (sto != Storage::LOCAL || CCtx().TargetName() != "cute") return true;
  // In CUTE local storage, keep reuse only for dynamic-shape buffers.
  return !VIIsInt(ma.buf_size.at(buffer_id));
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

    const auto& ranges = var_ranges.at(sname);
    if (ranges.Values().size() == 1 &&
        ranges.front().start == ranges.front().end) {
      VST_DEBUG(dbgs() << "Warning: buffer " << sname << " is never used!\n");
    }
    std::string dev_func_name = GetDeclDevFuncOfBuffer(sname);
    if (auto sv = VIInt(size))
      DFCtx(dev_func_name)
          .buffers.push_back(Buffer{.size = (size_t)sv.value(),
                                    .ranges = ranges.Values(),
                                    .buffer_id = sname});
    else
      DFCtx(dev_func_name)
          .dynamic_buffers.push_back(DBuffer{.size = STR(size),
                                             .ranges = ranges.Values(),
                                             .buffer_id = sname});
  }
  for (auto& [df_name, ctx] : DFCtxs()) ctx.SortBuffers();

  {
    auto& ms = CCtx().GetMemReuseStats();
    for (const auto& [df_name, ctx] : DFCtxs()) {
      ++ms.device_functions;
      ms.static_buffers += ctx.buffers.size();
      ms.dynamic_buffers += ctx.dynamic_buffers.size();
      ms.buffers_analyzed += ctx.buffers.size() + ctx.dynamic_buffers.size();
      for (const auto& buffer : ctx.buffers)
        ms.total_buffer_bytes += buffer.size;
    }
  }

  VST_DEBUG({
    for (auto& [df_name, ctx] : DFCtxs()) {
      dbgs() << "For '" << df_name << "'\n";
      for (const auto& buffer : ctx.buffers) {
        dbgs() << "\tstatic  buffer: " << buffer.buffer_id << "\n\t\t"
               << STR(ma.buf_sto.at(buffer.buffer_id)) << ", " << buffer.size
               << " bytes, " << RangesSTR(buffer.ranges) << "\n";
      }
      for (const auto& buffer : ctx.dynamic_buffers) {
        dbgs() << "\tdynamic buffer: " << buffer.buffer_id << "\n\t\t"
               << STR(ma.buf_sto.at(buffer.buffer_id)) << ", " << buffer.size
               << " bytes, " << RangesSTR(buffer.ranges) << "\n";
      }
    }
  });
}

void MemReuse::AnalyzeMemOffset() {
  // df_name -> suffix of `__co__heap_simulator` used in codegen
  std::map<std::string, std::string> df_name_idx;
  // co_name -> #__co__heap_simulator in the co (updating)
  std::map<std::string, size_t> idx_count;
  for (const auto& [df_name, _] : DFCtxs()) {
    for (auto sto : {Storage::LOCAL, Storage::SHARED}) {
      if (!ShouldReuseStorage(sto, df_name)) continue;
      if (ma.sto_have_dyn[df_name][sto]) {
        std::string co_func_name = GetFuncNameFromScopedName(df_name);
        // TODO: check that no pb, but dynamic
        if (df_name == co_func_name) continue;
        // for local and shared, use the same suffix.
        if (df_name_idx.count(df_name)) continue;
        if (!idx_count.count(co_func_name))
          idx_count[co_func_name] = 0;
        else
          idx_count[co_func_name] += 1;
        df_name_idx[df_name] = std::to_string(idx_count[co_func_name]);
      }
    }
  }
  for (auto& [df_name, ctx] : DFCtxs()) {
    for (auto sto : {Storage::LOCAL, Storage::SHARED}) {
      if (!ShouldReuseStorage(sto, df_name)) continue;
      if (ma.sto_have_dyn[df_name][sto]) {
        std::string co_func_name = GetFuncNameFromScopedName(df_name);
        if (idx_count[co_func_name] == 0) df_name_idx[df_name] = "";
        continue;
      }
    }
    ProtoType(df_name, ctx,
              (df_name_idx.count(df_name) ? df_name_idx.at(df_name) : ""));
  }
}

void MemReuse::ProtoType(const std::string& df_name, DevFuncMemReuseCtx& ctx,
                         std::string idx_suffix) {
  std::string co_func_name = GetFuncNameFromScopedName(df_name);
  auto DoMemReuse = [&](Storage sto) -> void {
    if (sto != Storage::LOCAL && sto != Storage::SHARED)
      choreo_unreachable("The storage type: " + STR(sto) +
                         " is not supported yet!");
    size_t alignment = CCtx().GetMemoryAlignmentByte(sto);
    if (sto == Storage::SHARED) alignment = SharedAlignmentForDevFunc(df_name);
    if (ma.sto_have_dyn[df_name][sto]) {
      auto mri = FCtx(co_func_name).SetDynMemReuseInfo(df_name);
      std::string simulator =
          "__co_" + STR(sto) + "_heap_simulator" + idx_suffix;
      auto& infos = mri->infos;
      infos[sto].simulator = simulator;

      auto SetChunkInfo = [&](const auto& bs) -> void {
        for (const auto& buffer : bs) {
          if (ma.buf_sto.at(buffer.buffer_id) != sto) continue;

          infos[sto].offset_args.push_back("mr_offset" + buffer.buffer_id);
          auto chunks_name = "__co__" + STR(sto) + "_chunks" + idx_suffix;
          if (infos[sto].chunks_name == "")
            infos[sto].chunks_name =
                "__co__" + STR(sto) + "_chunks" + idx_suffix;
          std::string buffer_size;
          if constexpr (std::is_same_v<decltype(buffer.size), std::string>)
            buffer_size =
                "static_cast<size_t>(" + UnScopedExpr(buffer.size) + ")";
          else if constexpr (std::is_same_v<decltype(buffer.size), size_t>)
            buffer_size = UnScopedExpr(std::to_string(buffer.size));
          else
            choreo_unreachable("Unexpected type of buffer.size: " +
                               std::string(typeid(buffer.size).name()) +
                               "\n\twith buffer " + buffer.buffer_id);
          infos[sto].chunks.push_back(
              std::string("{") + buffer_size + ", " + "{" +
              RangesSTR(buffer.ranges, '{', '}') + "}" + ", \"" +
              RegexReplaceAll(buffer.buffer_id, "::", "_") + "\"}");
        }
      };

      auto TotalEventSize = [&]() -> size_t {
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

      // For CUTE local dynamic reuse, only keep dynamic local buffers in the
      // runtime heap simulator so static locals can still be plain arrays.
      if (!(sto == Storage::LOCAL && CCtx().TargetName() == "cute"))
        SetChunkInfo(ctx.buffers);
      SetChunkInfo(ctx.dynamic_buffers);

      std::string stos = STR(sto);
      std::string result = "__co__" + stos + "_result" + idx_suffix;
      std::string offsets_name =
          "__co__" + stos + "_chunk_offsets" + idx_suffix;
      infos[sto].result = result;
      infos[sto].offsets_name = offsets_name;
      std::string spm_size_var = "__co__" + stos + "_spm_size" + idx_suffix;
      infos[sto].spm_size = "__co__" + stos + "_spm_size" + idx_suffix;
      // special case for RtCheck which emits after general RtCheck.
      size_t mem_capacity = CCtx().GetMemCapacity(sto);
      size_t total_event_size = TotalEventSize();
      auto& ctx_spm_size =
          (sto == Storage::LOCAL ? ctx.local_spm_size : ctx.shared_spm_size);
      ctx_spm_size = mem_capacity - AlignUp(total_event_size, alignment);
      if (sto == Storage::LOCAL && CCtx().TargetName() == "cute") {
        size_t static_local_no_reuse_size = 0;
        for (const auto& buffer : ctx.buffers) {
          if (ma.buf_sto.at(buffer.buffer_id) != Storage::LOCAL) continue;
          if (ShouldReuseBuffer(buffer.buffer_id, Storage::LOCAL, df_name))
            continue;
          static_local_no_reuse_size += buffer.size;
        }
        if (ctx_spm_size >= static_local_no_reuse_size)
          ctx_spm_size -= static_local_no_reuse_size;
        else
          ctx_spm_size = 0;
      }

      // record the offset args in sorted order
      std::sort(infos[sto].offset_args.begin(), infos[sto].offset_args.end());

      // Build compile-time interference graph for parametric plan.
      // Liveness ranges are static even when buffer sizes are dynamic.
      {
        auto& ie = infos[sto];
        // Collect all buffers in the order chunks were added
        struct BufEntry {
          std::string size_expr;
          std::vector<Range> ranges;
          std::string buffer_id;
        };
        std::vector<BufEntry> all_bufs;

        auto CollectBufs = [&](const auto& bs) {
          for (const auto& buffer : bs) {
            if (ma.buf_sto.at(buffer.buffer_id) != sto) continue;
            if (sto == Storage::LOCAL && CCtx().TargetName() == "cute") {
              if constexpr (std::is_same_v<std::decay_t<decltype(buffer.size)>,
                                           size_t>) {
                if (!ShouldReuseBuffer(buffer.buffer_id, Storage::LOCAL,
                                       df_name))
                  continue;
              }
            }
            std::string sz;
            if constexpr (std::is_same_v<std::decay_t<decltype(buffer.size)>,
                                         std::string>)
              sz = "static_cast<size_t>(" + UnScopedExpr(buffer.size) + ")";
            else
              sz = UnScopedExpr(std::to_string(buffer.size));
            all_bufs.push_back({sz, buffer.ranges, buffer.buffer_id});
          }
        };
        if (!(sto == Storage::LOCAL && CCtx().TargetName() == "cute"))
          CollectBufs(ctx.buffers);
        CollectBufs(ctx.dynamic_buffers);

        size_t n = all_bufs.size();
        ie.n_buffers = n;
        ie.alignment = alignment;
        ie.size_exprs.clear();
        ie.buffer_ids.clear();
        ie.interference.assign(n * n, false);

        for (size_t i = 0; i < n; ++i) {
          ie.size_exprs.push_back(all_bufs[i].size_expr);
          ie.buffer_ids.push_back(
              RegexReplaceAll(all_bufs[i].buffer_id, "::", "_"));
        }

        const LivenessAnalyzer::HBGraph* hb_graph = nullptr;
        if (sto == Storage::SHARED) {
          auto it = la.HBGraphs().find(df_name);
          if (it != la.HBGraphs().end()) {
            hb_graph = &it->second;
          } else {
            // Find the most specific HB graph whose scope is a child
            // of df_name (scope starts with df_name). Use longest match
            // to avoid prefix-collision.
            size_t best_len = 0;
            for (const auto& [scope, graph] : la.HBGraphs()) {
              if (scope.find(df_name) == 0 && scope.size() > best_len) {
                best_len = scope.size();
                hb_graph = &graph;
              }
            }
          }
        }

        // Build interference from liveness ranges (static, known at compile
        // time)
        for (size_t i = 0; i < n; ++i) {
          for (size_t j = i + 1; j < n; ++j) {
            bool interfere = false;
            size_t ri = 0, rj = 0;
            const auto& ra = all_bufs[i].ranges;
            const auto& rb = all_bufs[j].ranges;
            while (ri < ra.size() && rj < rb.size()) {
              if (ra[ri].Overlaps(rb[rj])) {
                interfere = true;
                break;
              }
              if (ra[ri].end < rb[rj].end)
                ++ri;
              else
                ++rj;
            }
            if (interfere && hb_graph &&
                hb_graph->CanOverlap(all_bufs[i].buffer_id,
                                     all_bufs[j].buffer_id)) {
              interfere = false;
              errs() << "info: HB analysis: buffers "
                     << UnScopedExpr(all_bufs[i].buffer_id) << " and "
                     << UnScopedExpr(all_bufs[j].buffer_id)
                     << " can share memory (signal-ordered lifetimes).\n";
            }
            if (!interfere && hb_graph &&
                hb_graph->IsUnsafeMultiInstanceOverlap(all_bufs[i].buffer_id,
                                                       all_bufs[j].buffer_id)) {
              interfere = true;
              errs() << "info: HB analysis: buffers "
                     << UnScopedExpr(all_bufs[i].buffer_id) << " and "
                     << UnScopedExpr(all_bufs[j].buffer_id)
                     << " cannot share memory (multi-instance safety).\n";
            }
            ie.interference[i * n + j] = interfere;
            ie.interference[j * n + i] = interfere;
          }
        }
      }

      return;
    }

    // the buffers are of static shape.
    HeapSimulator::Chunks chunks;

    for (const auto& buffer : ctx.buffers) {
      if (ma.buf_sto.at(buffer.buffer_id) != sto) continue;
      if (!ShouldReuseBuffer(buffer.buffer_id, sto, df_name)) continue;
      chunks.push_back(buffer);
    }

    HeapSimulator simulator;
    HeapSimulator::HBOverride hb_override = nullptr;
    HeapSimulator::HBMustInterfere hb_must_interfere = nullptr;
    std::vector<std::pair<std::string, std::string>> hb_overlapped_pairs;
    if (sto == Storage::SHARED) {
      const auto& hb_graphs = la.HBGraphs();
      const LivenessAnalyzer::HBGraph* hb_graph_ptr = nullptr;
      auto it = hb_graphs.find(df_name);
      if (it != hb_graphs.end()) {
        hb_graph_ptr = &it->second;
      } else {
        size_t best_len = 0;
        for (const auto& [scope, graph] : hb_graphs) {
          if (scope.find(df_name) == 0 && scope.size() > best_len) {
            best_len = scope.size();
            hb_graph_ptr = &graph;
          }
        }
      }
      if (hb_graph_ptr) {
        hb_override = [hb_graph_ptr, &hb_overlapped_pairs](
                          const std::string& a, const std::string& b) {
          if (hb_graph_ptr->CanOverlap(a, b)) {
            hb_overlapped_pairs.emplace_back(a, b);
            return true;
          }
          return false;
        };
        hb_must_interfere = [hb_graph_ptr](const std::string& a,
                                           const std::string& b) {
          return hb_graph_ptr->IsUnsafeMultiInstanceOverlap(a, b);
        };
      }
    }

    // ptr<StaticMemReuseInfo>
    auto mri = FCtx(co_func_name).SetStaticMemReuseInfo(df_name);
    if (!chunks.empty()) {
      size_t baseline_size = simulator.Allocate(chunks, alignment).heap_size;
      HeapSimulator::Result result =
          simulator.Allocate(chunks, alignment, hb_override, hb_must_interfere);
      if (!ValidateResult(result, chunks, hb_override, hb_must_interfere)) {
        errs() << "  in device function: " << df_name << "\n";
        assert(false && "memory reuse validation failed");
      }
      auto& ctx_spm_size =
          (sto == Storage::LOCAL ? ctx.local_spm_size : ctx.shared_spm_size);
      ctx_spm_size = result.heap_size;
      mri->infos[sto].spm_size = result.heap_size;
      CCtx().GetMemReuseStats().total_static_heap_bytes += result.heap_size;
      for (const auto& [buffer_id, offset] : result.chunk_offsets)
        ctx.mem_offset.emplace(buffer_id, offset);
      VST_DEBUG(dbgs() << "For '" << df_name << "'\n\t" << STR(sto)
                       << " memory usage: " << result.heap_size << " bytes\n");
      if (!hb_overlapped_pairs.empty() && result.heap_size < baseline_size) {
        errs() << "info: HB analysis: shared memory reduced from "
               << baseline_size << " to " << result.heap_size << " bytes ("
               << (baseline_size - result.heap_size)
               << " saved). Overlapped buffers:";
        for (const auto& [a, b] : hb_overlapped_pairs) {
          errs() << " (" << UnScopedExpr(a) << ", " << UnScopedExpr(b) << ")";
        }
        errs() << "\n";
      }
    }
  };

  for (auto sto : {Storage::LOCAL, Storage::SHARED})
    if (ShouldReuseStorage(sto, df_name)) DoMemReuse(sto);
}

bool MemReuse::ValidateResult(
    const HeapSimulator::Result& res, const HeapSimulator::Chunks& chunks,
    HeapSimulator::HBOverride hb_override,
    HeapSimulator::HBMustInterfere hb_must_interfere) {
  size_t size = chunks.size();
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = i + 1; j < size; ++j) {
      const auto& c1 = chunks[i];
      const auto& c2 = chunks[j];
      bool interfere = c1.Interfere(c2);
      if (interfere && hb_override && hb_override(c1.buffer_id, c2.buffer_id))
        interfere = false;
      if (!interfere && hb_must_interfere &&
          hb_must_interfere(c1.buffer_id, c2.buffer_id))
        interfere = true;
      if (interfere) {
        auto o1 = res.chunk_offsets.at(c1.buffer_id);
        auto o2 = res.chunk_offsets.at(c2.buffer_id);
        if ((o1 <= o2 && o1 + c1.size > o2) ||
            (o2 <= o1 && o2 + c2.size > o1)) {
          errs() << "Error: unexpected memory overlap detected between "
                    "buffers "
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
  if (!ShouldReuseBuffer(sname, sto, cur_dev_fname)) {
    VST_DEBUG(dbgs() << STR(sto) << " buffer: " << sname
                     << "\n\tis not selected for memory reuse.\n";);
    return;
  }
  VST_DEBUG(dbgs() << STR(sto) << " buffer: " << sname << "\n\t";);

  bool dynamic = ma.sto_have_dyn[cur_dev_fname][sto];
  if (dynamic && sto == Storage::LOCAL && CCtx().TargetName() == "cute") {
    std::string co_func_name = GetFuncNameFromScopedName(cur_dev_fname);
    auto mri = FCtx(co_func_name).GetDynMemReuseInfo(cur_dev_fname);
    if (!mri || !mri->infos.count(Storage::LOCAL)) return;
    auto off_arg = "mr_offset" + sname;
    auto& off_args = mri->infos.at(Storage::LOCAL).offset_args;
    if (std::find(off_args.begin(), off_args.end(), off_arg) == off_args.end())
      return;
  }
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

  n.AddNote("reuse", spm_name);
  n.AddNote("offset", offset);
  size_t alignment = CCtx().GetMemoryAlignmentByte(sto);
  if (sto == Storage::SHARED)
    alignment = SharedAlignmentForDevFunc(cur_dev_fname);
  n.AddNote("alignment", std::to_string(alignment));
}

bool MemReuse::RunOnProgramImpl(AST::Node& root) {
  if (!CCtx().MemReuse()) return true;

  la.SetLevelPrefix("  ");
  la.SSTab().UpdateGlobal(SymTab());
  if (!la.RunOnProgram(root)) return la.Status();

  ma.SetLevelPrefix("  ");
  ma.SSTab().UpdateGlobal(SymTab());
  if (!ma.RunOnProgram(root)) return ma.Status();

  if (prt_visitor) dbgs() << LevelPrefix() << "|- " << GetName() << NewL;

  root.accept(*this);

  if (HasError()) return false;

  if (abend_after) return false;

  return true;
}
