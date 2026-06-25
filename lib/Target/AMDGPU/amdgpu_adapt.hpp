#ifndef __CHOREO_AMDGPU_ADAPTION_HPP__
#define __CHOREO_AMDGPU_ADAPTION_HPP__

#include "assess.hpp"
#include "ast.hpp"
#include "codegen.hpp"
#include "target_utils.hpp"

namespace Choreo {

struct AMDGPUParallelSymbols {
  std::map<ParallelLevel, std::set<std::string>> all_pvs;

  void AddLevelPV(ParallelLevel pl, const std::string& sym) {
    if (!all_pvs.count(pl)) all_pvs.emplace(pl, std::set<std::string>{});
    all_pvs[pl].insert(sym);
  }

  const std::set<std::string> GetLevelPVs(ParallelLevel pl) const {
    if (!all_pvs.count(pl)) return {};
    return all_pvs.at(pl);
  }

  const std::set<std::string> GetInnerPVs(ParallelLevel pl) const {
    std::set<std::string> inner;
    auto& pld = PlDepthMap::Get();
    for (auto& lvl_pvs : all_pvs) {
      if (pld.ToDepth(lvl_pvs.first) > pld.ToDepth(pl))
        inner.insert(lvl_pvs.second.begin(), lvl_pvs.second.end());
    }
    return inner;
  }

  ParallelLevel GetPVLevel(const std::string& sym) {
    for (auto& lvl_pvs : all_pvs)
      if (lvl_pvs.second.count(sym)) return lvl_pvs.first;
    return ParallelLevel::UNKNOWN;
  }

  void Reset() { all_pvs.clear(); }
};

struct AMDGPUAdaptor : public CodeGenerator {
private:
  std::unordered_map<std::string, AST::Parameter*> cur_params;
  std::stack<ParallelLevel> levels;
  AST::Node* cur_fnode;
  std::string cur_arch;

  AMDGPUParallelSymbols ps;

private:
  ParallelLevel Level() const { return levels.top(); }

  bool Assess(const ValueItem& pred, const std::string& message,
              AST::Node& node, AST::Node* emit_node = nullptr,
              UsageType uty = UsageType::HardwareConstraint) {
    return FCtx(fname)
        .GetAssessor(*this)
        .Assess(AssessPolicy::Error, pred, message, uty, AssessType::USE_SITE,
                node.LOC(), &node,
                ((emit_node == nullptr) ? cur_fnode : emit_node))
        .passed;
  }

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n)) {
      ps.Reset();
      cur_params.clear();
      cur_fnode = &n;
      levels.push(ParallelLevel::SEQ);
    } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      auto lvl = pb->GetLevel();
      levels.push(lvl);

      pb->SetMaxLevel(TargetMaxLevel());

      ps.AddLevelPV(lvl, InScopeName(pb->BPV()->name));
      for (auto id : pb->AllSubPVs())
        ps.AddLevelPV(lvl, InScopeName(cast<AST::Identifier>(id)->name));

      CheckPBSettings(pb);
      CheckStreamBinding(pb);
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ParallelBy>(&n)) {
      levels.pop();
    } else if (auto c = dyn_cast<AST::Call>(&n)) {
      if (!c->IsExpr()) n.SetLevel(Level());
    }
    return true;
  }

public:
  bool IsHost() const { return Level() == ParallelLevel::SEQ; }

  void CheckPBSettings(AST::ParallelBy* pb) {
    if (pb->GetLevel() != ParallelLevel::THREAD) return;

    auto total_threads = sbe::nu(1);
    for (auto bv : pb->BoundValues())
      total_threads = (total_threads * bv)->Normalize();

    auto ppb = cgi.GetPBTree(fname).GetParent(pb);
    assert(ppb->GetLevel() == ParallelLevel::GROUP);

    auto wavefront_size = CCtx().GetMinGroupDim();
    if (ppb->IsEnforced()) {
      auto msg = "The total thread dimension must be a multiple of " +
                 std::to_string(wavefront_size) +
                 " (wavefront size) when 'group' exists for " + cur_arch + ".";
      Assess(sbe::oc_eq(total_threads, sbe::nu((int64_t)wavefront_size)), msg,
             *pb, pb);
    }

    auto max_thr = CCtx().GetMaxThreadsPerBlock();
    if (max_thr > 0) {
      auto GetPBTotal = [](AST::ParallelBy* n) -> ValueItem {
        auto bvs = n->BoundValues();
        if (!bvs.empty()) {
          ValueItem t = sbe::nu(1);
          for (auto bv : bvs) t = (t * bv)->Normalize();
          return t;
        }
        return n->BoundValue();
      };

      auto physical = total_threads;
      auto group_bv = GetPBTotal(ppb);
      if (IsValidValueItem(group_bv))
        physical = (physical * group_bv)->Normalize();

      auto pred =
          sbe::cmp("<=", physical, sbe::nu((int64_t)max_thr))->Normalize();
      if (auto bv = VIBool(pred); !bv || !bv.value()) {
        auto msg = "Total threads per workgroup (thread x group dims) must "
                   "not exceed " +
                   std::to_string(max_thr) + " on " + cur_arch + ".";
        Assess(pred, msg, *pb, pb);
      }
    }

    // Occupancy warning: under-occupied workgroup
    if (auto thr_val = VIInt(total_threads); thr_val && *thr_val < 64) {
      Warning(pb->LOC(),
              "workgroup uses only " + std::to_string(*thr_val) +
                  " threads; RDNA CUs are most efficient with >= 64 threads "
                  "per workgroup.");
    }
  }

  void CheckStreamBinding(AST::ParallelBy* pb) {
    if (!pb->HasStream()) return;
    Error1(pb->StreamExpr()->LOC(),
           "stream-based kernel launch is not yet supported on HIP target. "
           "Remove the stream binding from parallel-by.");
  }

public:
  AMDGPUAdaptor()
      : CodeGenerator("amdgpu"), cur_arch(ToUpper(CCtx().GetArch())) {}
  ~AMDGPUAdaptor() {}

  bool Visit(AST::NamedVariableDecl& n) override {
    auto ty = GetSymbolType(n.name_str);

    if (!isa<SpannedType>(ty)) {
      auto mem = n.GetMemory();
      if (!mem) return true;

      auto st = mem->Get();
      if (n.init_expr != nullptr) {
        if (st == Storage::SHARED)
          Error1(n.LOC(), "initialization is not supported for " + STR(st) +
                              " variables");
        if (st == Storage::GLOBAL)
          Error1(n.LOC(), "'global' attribute only applies to functions ");
      }
      return true;
    }

    auto sty = cast<SpannedType>(ty);
    auto st = sty->GetStorage();
    switch (st) {
    case Storage::GLOBAL:
      if (Level() != ParallelLevel::SEQ)
        Error1(n.LOC(), "global variable '" + n.name_str +
                            "` mustn't be declared inside parallel-by.");
      break;
    case Storage::SHARED:
      if (Level() == ParallelLevel::SEQ)
        Error1(n.LOC(), "shared variable '" + n.name_str +
                            "` must be declared inside parallel-by.");
      if (sty->RuntimeShaped() && !CCtx().MemReuse())
        Error1(n.LOC(), "HIP forbids shared variable '" + n.name_str +
                            "` to be dynamically shaped (by " +
                            STR(sty->GetShape()) + ").");
      break;
    case Storage::LOCAL:
      if (Level() == ParallelLevel::SEQ)
        Error1(n.LOC(), "local variable '" + n.name_str +
                            "` must be declared inside parallel-by.");
      if (sty->RuntimeShaped() && !CCtx().MemReuse())
        Error1(n.LOC(), "HIP forbids local variable '" + n.name_str +
                            "` to be dynamically shaped (by " +
                            STR(sty->GetShape()) + ").");
      break;
    case Storage::REG:
      if (!n.HasNote("fragment_decl"))
        Error1(n.LOC(), "can not declare variable '" + n.name_str + "` as " +
                            STR(st) + " inside choreo function.");
      break;
    default:
      Error1(n.LOC(), "can not declare variable '" + n.name_str + "` as " +
                          STR(st) + " inside choreo function.");
      break;
    }
    return true;
  }

  bool Visit(AST::Parameter& n) override {
    auto ty = GetSymbolType(n.sym->name);
    if (isa<StreamType>(ty))
      Error1(n.LOC(), "stream parameters are not supported on HIP target.");
    if (n.sym) cur_params.emplace(InScopeName(n.sym->name), &n);
    return true;
  }

  bool Visit(AST::DMA& n) override {
    if (n.IsTMA()) {
      Error1(n.LOC(), "TMA is not supported on HIP target (arch: " +
                          CCtx().GetArch() + ").");
      return false;
    }

    if (n.IsAsync()) {
      Error1(n.LOC(), "async DMA is not supported on HIP target (arch: " +
                          CCtx().GetArch() +
                          "). RDNA does not have hardware async copy.");
      return false;
    }

    if (n.operation == ".any") return true;

    auto fty = GetSpannedType(n.from->GetType());
    auto tty = GetSpannedType(n.to->GetType());
    assert(fty && tty);
    auto fst = fty->GetStorage();
    auto tst = tty->GetStorage();

    if ((fst == Storage::GLOBAL) && (tst == Storage::GLOBAL))
      Error1(n.LOC(), "HIP does not allow sync " + n.operation.substr(1) +
                          " (global -> global). Use host memcpy instead.");

    n.SetLevel(ParallelLevel::THREAD);

    // DMA size validation: total transfer <= 4GB
    if (isa<AST::ChunkAt>(n.to)) {
      auto t_sty = GetSpannedType(GetSymbolType(
          cast<AST::ChunkAt>(n.to)->RefSymbol()));
      if (t_sty && !t_sty->RuntimeShaped() &&
          t_sty->ByteSize() >= (1ULL << 32))
        Error1(n.LOC(), "On " + cur_arch +
                            ", the size of data transferred by DMA cannot "
                            "exceed 4GB.");
    }

    // Parameter shadow annotation
    if (!isa<AST::ChunkAt>(n.from)) return true;
    auto f_name = cast<AST::ChunkAt>(n.from)->RefSymbol();
    auto sty = GetSpannedType(GetSymbolType(f_name));
    if (sty->GetStorage() == Storage::NONE) return false;
    if (!cur_params.count(InScopeName(f_name))) return false;

    auto annotate_by_storage = [this, &f_name](Storage st) {
      switch (st) {
      case Storage::GLOBAL:
      case Storage::SHARED: {
        auto p = cur_params[InScopeName(f_name)];
        if (p->attr == ParamAttr::NONE) p->attr = ParamAttr::SHADOW_TO_GLOBAL;
        break;
      }
      default: break;
      }
    };
    if (auto to = dyn_cast<AST::ChunkAt>(n.to)) {
      annotate_by_storage(
          cast<SpannedType>(GetSymbolType(to->RefSymbol()))->GetStorage());
    } else if (auto m = dyn_cast<AST::Memory>(n.to))
      annotate_by_storage(m->st);
    return true;
  }

  bool Visit(AST::MMA& n) override {
    Error1(n.LOC(), "MMA operations are not supported on HIP target (arch: " +
                        CCtx().GetArch() +
                        "). AMD matrix operations require CDNA architecture.");
    return false;
  }

  bool Visit(AST::Call&) override { return true; }

  bool Visit(AST::Synchronize& n) override {
    auto pl = Level();

    switch (n.Resource()) {
    case Storage::GLOBAL:
      if ((pl == ParallelLevel::BLOCK) || (pl == ParallelLevel::GROUP) ||
          (pl == ParallelLevel::THREAD) || (pl == ParallelLevel::TERM))
        Error1(n.LOC(), "unsupported: " + STR(n.Resource()) +
                            " synchronization in " + STR(pl) + " scope.");
      break;
    case Storage::SHARED:
    case Storage::LOCAL:
      if ((pl == ParallelLevel::SEQ) || (pl == ParallelLevel::TERM) ||
          (pl == ParallelLevel::DEVICE))
        Error1(n.LOC(), "unsupported: " + STR(n.Resource()) +
                            " synchronization in " + STR(pl) + " scope.");
      break;
    case Storage::NODE:
    default:
      Error1(n.LOC(),
             "unsupported synchronization: " + STR(n.Resource()) + ".");
    }
    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_AMDGPU_ADAPTION_HPP__
