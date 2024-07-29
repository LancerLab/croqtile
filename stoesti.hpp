#ifndef __CHOREO_STORAGE_ESTIMATE_INFO_HPP__
#define __CHOREO_STORAGE_ESTIMATE_INFO_HPP__

#include "ast.hpp"
#include "visitor.hpp"
#include <iomanip>

#define DBG if (trace) os << "[stoesti] "

namespace Choreo {

struct StoEstimate : public VisitorWithSymTab {
private:
  bool trace;
  std::ostream &os;
  bool trace_visit = false; // for debugging purpose only
  size_t error_count = 0;

  // memory usage in byte
  using ull = unsigned long long;

  // 'compile time' and 'runtime'

  // the top of stk_ct is the map of the current ast node!
  std::stack<std::map<Storage, ull>> stk_ct;
  // stack of memory allocate instruction(usage known at ct), do tracing
  std::map<Storage, std::stack<std::string>> mem_inst_stk;
  // total ct memory usage
  std::map<Storage, ull> mem_used_now_ct;
  // Only the maximum ct memory usage is recorded
  std::map<Storage, ull> mem_max;

  std::stack<std::map<Storage, std::vector<std::string>>> stk_rt;
  std::map<Storage, std::vector<std::string>> mem_used_now_rt;
  /*
  For rt memory usage checking. For example:
    __co__ foo(s32 [6, 12, ?] input, ...) {}
  After value numbering, input.span(2) will be given the signature ::foo::$0
  In Visit(AST::FunctionDecl &n), map the signature to the rt shape position
  That is, sig2pos["::foo::$0"] = "0-2"
  0 is the index of input, 2 is the index of dyn dim
  Then we can do codegen for rt memory usage check
  */
  std::map<std::string, std::string> sig2pos;
  // tuple<memory usages, code location, corresponding storage limit>
  std::vector<std::tuple<std::vector<std::string>, location, std::string>>
      rtshape_check;

  std::map<Storage, ull> mem_limit;
  // Indicates valid storage types
  std::map<Storage, std::string> sto2alloc;

  std::unordered_map<Storage, std::string> sto2str = {
      {Storage::LOCAL, "LOCAL"},   {Storage::SHARED, "SHARED"},
      {Storage::GLOBAL, "GLOBAL"}, {Storage::DEFAULT, "DEFAULT"},
      {Storage::NONE, "NONE"},
  };

private:
  bool BeforeVisitImpl(AST::Node &n) {
    if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
        isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
        isa<AST::ForeachBlock>(&n)) {
      DBG << "Entering Scope: " << SSTab().ScopeName() << "\n";
      // generate the map of current ast node that corresponding to the scope
      stk_ct.push(std::map<Storage, ull>{});
      stk_rt.push(std::map<Storage, std::vector<std::string>>{});
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node &n) {
    if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
        isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
        isa<AST::ForeachBlock>(&n)) {
      UpdateMemUsage();
      CheckOutOfBound(n);
      DBG << "Total mem used before leaving the scope:\n"
          << GetMemUsed(mem_used_now_ct);
      DBG << "Leaving Scope: " << SSTab().ScopeName() << "\n";

      // restore ct usage
      for (const auto &[sto, used_in_scop] : stk_ct.top()) {
        mem_used_now_ct[sto] -= used_in_scop;
        // if CheckOutOfBound processed, the stk_ct may be empty
        if (!mem_inst_stk[sto].empty())
          mem_inst_stk[sto].pop();
      }
      stk_ct.pop();

      // restore rt usage
      for (const auto &[sto, rt_used_in_scop] : stk_rt.top()) {
        for (size_t i = rt_used_in_scop.size(); i > 0; i--) {
          mem_used_now_rt[sto].pop_back();
        }
        // no inst trace when dealing with rt usage check
      }
      stk_rt.pop();
    }

    // the program is exiting, show the maximum ct mem usage
    if (isa<AST::Program>(&n)) {
      DBG << "The maximum memory usages at compile time for each level (Not at "
             "the same time):\n"
          << GetMemUsed(mem_max);
    }
    return true;
  }

  void TraceEachVisit(AST::Node &n, std::string sup = "") {
    if (trace_visit)
      os << n.TypeNameString() << sup << "\n";
  }

  // Check whether the ct memory usage at each level exceeds limits
  void CheckOutOfBound(AST::Node &n) {
    for (const auto &[sto, _] : sto2alloc) {
      if (mem_used_now_ct[sto] > mem_limit[sto]) {
        // get the variables which lead to out of bound
        std::ostringstream oss;
        while (!mem_inst_stk[sto].empty()) {
          oss << "\n\t\t" << mem_inst_stk[sto].top();
          mem_inst_stk[sto].pop();
        }
        Error(n.LOC(), "In the scope " + SSTab().ScopeName() +
                           ", compile time memory usage at " + sto2str[sto] +
                           " level is out of bound! \n\tUsed: " +
                           std::to_string(mem_used_now_ct[sto]) +
                           " bytes, Limit: " + std::to_string(mem_limit[sto]) +
                           " bytes. With variables:" + oss.str());
        error_count++;
      }
    }
  }

  // Update the maximum ct memory usage for each level
  void UpdateMemUsage() {
    for (const auto &[sto, v] : mem_used_now_ct)
      if (mem_max[sto] < v)
        mem_max[sto] = v;
  }

  // Return the detail memory usage of the given map
  std::string GetMemUsed(std::map<Storage, ull> &m) {
    std::ostringstream oss;
    for (const auto &[sto, v] : m) {
      oss << "\t" << std::setw(6) << sto2str[sto] << ":\t\t"
          << "0x" << std::setw(9) << std::setfill('0') << std::right << std::hex
          << v << " bytes" << SizeForHuman(v) << "\n";
    }
    return oss.str();
  }

  // Tranform size in byte to human readable format like 1KB 2GB etc in decimal.
  std::string SizeForHuman(ull size) {
    std::ostringstream oss;
    oss << std::fixed << " (" << std::setprecision(2);
    if (size >= (ull)1024 * 1024 * 1024) {
      oss << size / 1024.0 / 1024 / 1024 << " GB";
    } else if (size >= (ull)1024 * 1024) {
      oss << size / 1024.0 / 1024 << " MB";
    } else if (size >= (ull)1024) {
      oss << size / 1024.0 << " KB";
    } else if (size > 0) {
      oss << size << " B";
    }
    oss << ")";
    return oss.str();
  }

  // return the total memory usage corresponding to sto
  std::vector<std::string> GetCtRtUsage(Storage sto) {
    std::vector<std::string> res;
    // ct memory usage is always a single integer
    res.push_back(std::to_string(mem_used_now_ct[sto]));
    // rt memory usage may contain several expressions
    for (const auto used : mem_used_now_rt[sto])
      res.push_back(used);
    return res;
  }

public:
  StoEstimate(const ptr<SymbolTable> s_tab, Target t, bool trace = false,
              std::ostream &o = std::cout)
      : VisitorWithSymTab(s_tab), trace(trace), os(o),
        trace_visit(std::getenv("TRACE_STO")) {
    if (t == Target::Factor) {
      sto2alloc = {
          {Storage::LOCAL, "L1Type"},
          {Storage::SHARED, "SRAMType"},
          {Storage::GLOBAL, "DRAMType"},
      };
      // initialize with mem_used_now_ct
      for (const auto &[k, _] : sto2alloc)
        mem_used_now_ct[k] = 0;

      enum class HW { I20, c035 };
      HW hw = HW::I20;
      switch (hw) {
        case HW::c035:
          // The values obtained through testing on c035
          // TODO: All is different with Scorpio (1 Die) in the link below
          mem_limit[Storage::LOCAL] = 1.5 * 1024 * 1024ull; // 1.5MB
          mem_limit[Storage::SHARED] = 26165824ull;         // 24MB
          mem_limit[Storage::GLOBAL] = 4194304ull * 1024;   // 4GB
        case HW::I20:
        default:
          // initialize max memory we can allocate in byte
          // The values obtained through testing on I20
          /* TODO:
          L3 (gobal) is different with Dorado (3VG per Cluster) in
          http://wiki.enflame.cn/display/~james.zhu/Enflame+GCU+Programming+Model#EnflameGCUProgrammingModel-get_memory_space
          */
          mem_limit[Storage::LOCAL] = 0xfc000ull;         // 1008KB
          mem_limit[Storage::SHARED] = 26165824ull;       // 24MB
          mem_limit[Storage::GLOBAL] = 4194304ull * 1024; // 4GB
      }
    } else {
      choreo_unreachable("unsupported target in storage estimation.");
    }
  }
  ~StoEstimate() {}

  // return pair(rtshape_check, sig2pos) to 
  // do codegen for rt memory usage checking
  auto GetRtMemUsageInfo() { return std::make_pair(rtshape_check, sig2pos); }

  bool Visit(AST::MultiNodes &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::MultiValues &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::IntLiteral &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Boolean &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Expr &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::MultiDimSpans &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::NamedTypeDecl &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::NamedVariableDecl &n) {
    TraceEachVisit(n);
    // mem alloc may happends here
    auto ty = GetSymbolType(n.name_str);
    if (!isa<SpannedType>(ty))
      return true;
    auto sty = cast<SpannedType>(ty);
    auto sto = sty->GetStorage();
    assert(sto2alloc.count(sto) &&
           "Only support Storage types in `sto2alloc`!");
    if (sty->RuntimeShaped()) {
      // runtime usage
      DBG << sto2str[sto] << " runtime shape var `" << n.name_str
          << "` need : " << sty->ByteSizeExpression() << " bytes.\n";
      mem_used_now_rt[sto].push_back(sty->ByteSizeExpression());
      rtshape_check.push_back(std::make_tuple(GetCtRtUsage(sto), n.LOC(),
                                              std::to_string(mem_limit[sto])));
    } else {
      // compile time usage
      auto size = sty->ByteSize();
      DBG << sto2str[sto] << " `" << n.name_str << "` need " << size << " bytes"
          << SizeForHuman(size) << ".\n";
      stk_ct.top()[sto] += size;
      mem_used_now_ct[sto] += size;
      mem_inst_stk[sto].push(n.name_str);
    }
    return true;
  }
  bool Visit(AST::IntTuple &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Assignment &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::IntIndex &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::DataType &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Identifier &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Parameter &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ParamList &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ParallelBy &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WhereBind &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WithIn &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WithBlock &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Memory &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::DMA &d) {
    TraceEachVisit(d);
    // mem alloc happends here
    if (isa<AST::Memory>(d.to)) {
      auto dst_sto = cast<AST::Memory>(d.to)->Get();
      auto sty = dyn_cast<FutureType>(d.GetType())->GetSpannedType().get();
      if (sty->RuntimeShaped()) {
        DBG << sto2str[dst_sto] << " runtime shape dma `" << d.future
            << "` need : " << sty->ByteSizeExpression() << " bytes.\n";
        mem_used_now_rt[dst_sto].push_back(sty->ByteSizeExpression());
        rtshape_check.push_back(
            std::make_tuple(GetCtRtUsage(dst_sto), d.LOC(),
                            std::to_string(mem_limit[dst_sto])));
      } else {
        auto dst_size = sty->ByteSize();
        assert(sto2alloc.count(dst_sto) &&
               "Only support Storage types in `sto2alloc`!");
        DBG << sto2str[dst_sto] << " `" << d.future << "` need " << dst_size
            << " bytes" << SizeForHuman(dst_size) << ".\n";
        stk_ct.top()[dst_sto] += dst_size;
        mem_used_now_ct[dst_sto] += dst_size;
        mem_inst_stk[dst_sto].push(d.future);
      }
    }
    return true;
  }
  bool Visit(AST::ChunkAt &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Wait &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Call &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Select &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Return &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ForeachBlock &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::FunctionDecl &n) {
    TraceEachVisit(n);
    // Check the params in FunctionDecl for the presence of dynamic shape var
    // If so, maps the signature to (param index and dynamic dim)
    int param_idx = 0;
    for (const auto &p : n.params->values) {
      if (p->type->isSpanned()) {
        auto mds = cast<AST::MultiDimSpans>(p->type->getPartialType());
        auto &shape = mds->GetTypeDetail();
        if (shape.IsDynamic()) {
          for (const auto &[dyndim_idx, expr] : shape.GetDynamicDims()) {
            std::ostringstream oss;
            oss << param_idx << "-" << dyndim_idx;
            sig2pos[expr] = oss.str();
          }
        }
      }
      param_idx++;
    }
    return true;
  }
  bool Visit(AST::ChoreoFunction &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::CppSourceCode &n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Program &n) {
    TraceEachVisit(n);
    return true;
  }

  bool HasError() {
    if (error_count)
      os << "Totally " << error_count << " errors have been detected.\n";
    return error_count != 0;
  }
};

} // end namespace Choreo

#endif // __CHOREO_STORAGE_ESTIMATE_INFO_HPP__
