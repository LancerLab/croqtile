#ifndef __CHOREO_MEMORY_USAGE_CHECK_HPP__
#define __CHOREO_MEMORY_USAGE_CHECK_HPP__

#include <iomanip>

#include "ast.hpp"
#include "visitor.hpp"

namespace Choreo {

// tuple<runtime memory usages, code location, corresponding storage limit>
// to insert runtime memory usage check in codegen
typedef std::tuple<std::vector<std::string>, location, size_t>
    RtMemUsageCheckInfo;

// checking compile-time and runtime memory usage
struct MemUsageCheck : public VisitorWithSymTab {
private:
  std::ostream& os;
  size_t error_count = 0;

  // map from storage type to a integer
  typedef std::map<Storage, size_t> MemUsageMap;

  // Memory usage is measured in bytes
  // ct for compile time, rt for runtime

  // the top of ct_mem_usage_list is the MemUsageMap of the current ast node!
  std::stack<MemUsageMap> ct_mem_usage_list;
  // record memory allocations(usage known at ct), useful when tracing
  std::map<Storage, std::stack<std::string>> ct_mem_alloc_inst_set;
  // total ct memory usage
  MemUsageMap ct_tot_mem_usage;
  // Only the maximum ct memory usage is recorded
  MemUsageMap ct_max_mem_usage;

  std::stack<std::map<Storage, std::vector<std::string>>> rt_mem_usage_list;
  // only runtime usages are recorded
  std::map<Storage, std::vector<std::string>> rt_tot_mem_usage;
  std::vector<RtMemUsageCheckInfo> rt_mem_usage_check_list;

  MemUsageMap mem_usage_limit;
  std::unordered_set<Storage> valid_storage_type;

private:
  bool BeforeVisitImpl(AST::Node& n) {
    if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
        isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
        isa<AST::ForeachBlock>(&n)) {
      // generate the map of current ast node that corresponding to the scope
      ct_mem_usage_list.push(std::map<Storage, size_t>{});
      rt_mem_usage_list.push(std::map<Storage, std::vector<std::string>>{});
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) {
    if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
        isa<AST::ParallelBy>(&n) || isa<AST::WithBlock>(&n) ||
        isa<AST::ForeachBlock>(&n)) {
      UpdateCtMaxMemUsage();
      CheckCtMemUsage(n);
      VST_DEBUG(os << "[MemUsage] "
                   << "Total compile-time mem used before leaving scope "
                   << SSTab().ScopeName() << "\n"
                   << GetMemUsageMapDetail(ct_tot_mem_usage));
      RestoreMemUsage();
    }

    // special handling for AS::FunctionDecl
    if (isa<AST::ChoreoFunction>(&n)) RestoreMemUsage();

    // the program is exiting, show the maximum ct mem usage
    if (isa<AST::Program>(&n)) {
      VST_DEBUG(os << "[MemUsage] "
                   << "The maximum memory usages at compile time for each level"
                      "(Not at the same time):\n"
                   << GetMemUsageMapDetail(ct_max_mem_usage));
      assert(ct_mem_usage_list.empty() && rt_mem_usage_list.empty());
    }
    return true;
  }

  void TraceEachVisit(AST::Node& n, std::string sup = "") {
    if (trace_visit) os << n.TypeNameString() << sup << "\n";
  }

  void RestoreMemUsage() {
    // restore ct mem usage
    for (const auto& [sto, mem_used_in_scop] : ct_mem_usage_list.top()) {
      ct_tot_mem_usage[sto] -= mem_used_in_scop;
      ct_mem_alloc_inst_set[sto].pop();
    }
    ct_mem_usage_list.pop();

    // restore rt mem usage
    for (const auto& [sto, rt_mem_used_in_scop] : rt_mem_usage_list.top()) {
      for (size_t i = rt_mem_used_in_scop.size(); i > 0; i--)
        rt_tot_mem_usage[sto].pop_back();
      // no inst tracing when dealing with rt usage check yet
    }
    rt_mem_usage_list.pop();
  }

  // Check whether the ct memory usage at each level exceeds limits
  void CheckCtMemUsage(AST::Node& n) {
    for (const auto& sto : valid_storage_type) {
      if (ct_tot_mem_usage[sto] > mem_usage_limit[sto]) {
        // get the variables which lead to out of bound
        std::ostringstream oss;
        std::vector<std::string> inst_set;
        while (!ct_mem_alloc_inst_set[sto].empty()) {
          inst_set.push_back(ct_mem_alloc_inst_set[sto].top());
          oss << "\n\t\t" << inst_set.back();
          ct_mem_alloc_inst_set[sto].pop();
        }
        for (auto it = inst_set.rbegin(); it != inst_set.rend(); it++)
          ct_mem_alloc_inst_set[sto].push(*it);
        Error(n.LOC(),
              __internal__::GetStringFrom(sto) + " memory OUT OF BOUND!\n\t" +
                  "In the scope " + SSTab().ScopeName() + ", compile-time " +
                  __internal__::GetStringFrom(sto) +
                  " memory:\n\tUsed: " + std::to_string(ct_tot_mem_usage[sto]) +
                  " bytes, Limit: " + std::to_string(mem_usage_limit[sto]) +
                  " bytes. With variables:" + oss.str());
        error_count++;
      }
    }
  }

  // Update the maximum ct memory usage for each storage level
  void UpdateCtMaxMemUsage() {
    for (const auto& [sto, usage] : ct_tot_mem_usage)
      if (ct_max_mem_usage[sto] < usage) ct_max_mem_usage[sto] = usage;
  }

  // Return the detail memory usage of the given map
  std::string GetMemUsageMapDetail(MemUsageMap& m) {
    std::ostringstream oss;
    for (const auto& [sto, usage] : m) {
      oss << "\t" << std::setw(6) << __internal__::GetStringFrom(sto) << "("
          << std::setw(3) << std::setfill(' ') << GetCtMemOccupancyRate(sto)
          << "):\t\t"
          << "0x" << std::setw(9) << std::setfill('0') << std::right << std::hex
          << usage << " bytes" << SizeForHuman(usage) << "\n";
    }
    return oss.str();
  }

  // Tranform size in byte to human readable format like 1KB 2GB etc in decimal.
  std::string SizeForHuman(size_t size) {
    std::ostringstream oss;
    oss << std::defaultfloat << "(";
    if (size >= (size_t)1024 * 1024 * 1024) {
      oss << size / 1024.0 / 1024 / 1024 << " GB";
    } else if (size >= (size_t)1024 * 1024) {
      oss << size / 1024.0 / 1024 << " MB";
    } else if (size >= (size_t)1024) {
      oss << size / 1024.0 << " KB";
    } else {
      oss << size << " B";
    }
    oss << ")";
    return oss.str();
  }

  // return the total memory usage corresponding to sto
  std::vector<std::string> SumUpCtRtUsage(Storage sto) {
    std::vector<std::string> res;
    // ct memory usage is always a single integer
    res.push_back(std::to_string(ct_tot_mem_usage[sto]));
    // rt memory usage may contain several expressions
    for (const auto& usage : rt_tot_mem_usage[sto]) res.push_back(usage);
    return res;
  }

  // display memory occupancy as an integer percentage
  std::string GetCtMemOccupancyRate(Storage sto) {
    std::ostringstream oss;
    assert(mem_usage_limit[sto] > 0 &&
           "memory limitation should greater than 0!");
    oss << (size_t)(100.0 * ct_tot_mem_usage[sto] / mem_usage_limit[sto])
        << "%";
    return oss.str();
  }

public:
  MemUsageCheck(const ptr<SymbolTable> s_tab, Target t, std::string arch,
                std::ostream& o = std::cout)
      : VisitorWithSymTab("mucheck", s_tab), os(o) {
    if (t == Target::Factor) {
      valid_storage_type = {Storage::LOCAL, Storage::SHARED, Storage::GLOBAL};
      // initialize with ct_tot_mem_usage
      for (const auto& sto : valid_storage_type) ct_tot_mem_usage[sto] = 0;
      // initialize max memory we can allocate in byte
      if (arch == "gcu300") {
        // The values obtained through testing on c035
        // TODO: All is different with Scorpio (1 Die) in the link below
        // TODO: is S60G same with c035?
        mem_usage_limit[Storage::LOCAL] = (size_t)1.5 * 1024 * 1024; // 1.5MB
        mem_usage_limit[Storage::SHARED] = (size_t)24 * 1024 * 1024; // 24MB
        mem_usage_limit[Storage::GLOBAL] =
            (size_t)4 * 1024 * 1024 * 1024; // 4GB
      } else if (arch == "gcu210") {
        // The values obtained through testing on I20
        /* TODO:
        L3 (gobal) is different with Dorado (3VG per Cluster) in
        http://wiki.enflame.cn/display/~james.zhu/Enflame+GCU+Programming+Model#EnflameGCUProgrammingModel-get_memory_space
        */
        mem_usage_limit[Storage::LOCAL] = (size_t)0xfc000;           // 1008KB
        mem_usage_limit[Storage::SHARED] = (size_t)24 * 1024 * 1024; // 24MB
        mem_usage_limit[Storage::GLOBAL] =
            (size_t)4 * 1024 * 1024 * 1024; // 4GB
      } else {
        choreo_unreachable("unsupported gcu architecture " + arch +
                           " in memory usage check.");
      }

    } else {
      choreo_unreachable("unsupported target in memory usage check.");
    }
    VST_DEBUG(os << "[MemUsage] "
                 << "Memory usage limit of architecture " << arch << " is:\n"
                 << GetMemUsageMapDetail(mem_usage_limit));
  }
  ~MemUsageCheck() {}

  // return rt_mem_usage_check_list to
  // do codegen for rt memory usage checking
  auto GetRtMemUsageInfo() { return rt_mem_usage_check_list; }

  bool Visit(AST::MultiNodes& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::MultiValues& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::IntLiteral& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Boolean& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Expr& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::MultiDimSpans& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::NamedTypeDecl& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::NamedVariableDecl& n) {
    TraceEachVisit(n);
    // mem alloc may happends here
    auto ty = GetSymbolType(n.name_str);
    if (!isa<SpannedType>(ty)) return true;
    auto sty = cast<SpannedType>(ty);
    auto sto = sty->GetStorage();
    assert(valid_storage_type.count(sto) &&
           "Only support Storage types in `valid_storage_type`!");
    if (sty->RuntimeShaped()) {
      // runtime usage
      VST_DEBUG(os << "[MemUsage] " << __internal__::GetStringFrom(sto) << " `"
                   << SSTab().ScopedName(n.name_str)
                   << "` need : " << sty->ByteSizeExpression() << " bytes.\n");
      rt_mem_usage_list.top()[sto].push_back(sty->ByteSizeExpression());
      rt_tot_mem_usage[sto].push_back(sty->ByteSizeExpression());
      rt_mem_usage_check_list.push_back(
          std::make_tuple(SumUpCtRtUsage(sto), n.LOC(), mem_usage_limit[sto]));
    } else {
      // compile time usage
      auto size = sty->ByteSize();
      VST_DEBUG(os << "[MemUsage] " << __internal__::GetStringFrom(sto) << " `"
                   << SSTab().ScopedName(n.name_str) << "` need " << size
                   << " bytes" << SizeForHuman(size) << ".\n");
      ct_mem_usage_list.top()[sto] += size;
      ct_tot_mem_usage[sto] += size;
      ct_mem_alloc_inst_set[sto].push(SSTab().ScopedName(n.name_str));
    }
    return true;
  }
  bool Visit(AST::IntTuple& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Assignment& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::IntIndex& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::DataType& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Identifier& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Parameter& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ParamList& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ParallelBy& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WhereBind& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WithIn& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::WithBlock& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Memory& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::SpanAs& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::DMA& d) {
    TraceEachVisit(d);
    // mem alloc happends here
    if (isa<AST::Memory>(d.to)) {
      auto dst_sto = cast<AST::Memory>(d.to)->Get();
      assert(valid_storage_type.count(dst_sto) &&
             "Only support Storage types in `valid_storage_type`!");
      auto sty = dyn_cast<FutureType>(d.GetType())->GetSpannedType().get();
      if (sty->RuntimeShaped()) {
        VST_DEBUG(os << "[MemUsage] " << __internal__::GetStringFrom(dst_sto)
                     << " `" << SSTab().ScopedName(d.future) << "` need : "
                     << sty->ByteSizeExpression() << " bytes.\n");
        rt_mem_usage_list.top()[dst_sto].push_back(sty->ByteSizeExpression());
        rt_tot_mem_usage[dst_sto].push_back(sty->ByteSizeExpression());
        rt_mem_usage_check_list.push_back(std::make_tuple(
            SumUpCtRtUsage(dst_sto), d.LOC(), mem_usage_limit[dst_sto]));
      } else {
        auto dst_size = sty->ByteSize();
        assert(valid_storage_type.count(dst_sto) &&
               "Only support Storage types in `valid_storage_type`!");
        VST_DEBUG(os << "[MemUsage] " << __internal__::GetStringFrom(dst_sto)
                     << " `" << SSTab().ScopedName(d.future) << "` need "
                     << dst_size << " bytes" << SizeForHuman(dst_size)
                     << ".\n");
        ct_mem_usage_list.top()[dst_sto] += dst_size;
        ct_tot_mem_usage[dst_sto] += dst_size;
        ct_mem_alloc_inst_set[dst_sto].push(SSTab().ScopedName(d.future));
      }
    }
    return true;
  }
  bool Visit(AST::ChunkAt& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Wait& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Call& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Rotate& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Select& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Return& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::LoopRange& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::ForeachBlock& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::FunctionDecl& n) {
    TraceEachVisit(n);
    // special handling
    // Because AST::FunctionDecl.accept doesn't call BeforeVisit)
    ct_mem_usage_list.push(std::map<Storage, size_t>{});
    rt_mem_usage_list.push(std::map<Storage, std::vector<std::string>>{});

    int param_idx = 0;
    Storage func_param_sto = Storage::GLOBAL;
    for (const auto& p : n.params->values) {
      if (auto sty = dyn_cast<SpannedType>(p->GetType())) {
        std::string name = "::" + n.name + "::";
        name +=
            (p->HasSymbol() ? p->sym->name : ("#" + std::to_string(param_idx)));
        if (sty->RuntimeShaped()) {
          rt_mem_usage_list.top()[func_param_sto].push_back(
              sty->ByteSizeExpression());
          rt_tot_mem_usage[func_param_sto].push_back(sty->ByteSizeExpression());
          rt_mem_usage_check_list.push_back(
              std::make_tuple(SumUpCtRtUsage(func_param_sto), p->LOC(),
                              mem_usage_limit[func_param_sto]));
          VST_DEBUG(os << "[MemUsage] "
                       << "Function parameter `" << name << "`("
                       << __internal__::GetStringFrom(func_param_sto)
                       << ") need " << sty->ByteSizeExpression()
                       << " bytes.\n");
        } else {
          ct_mem_usage_list.top()[func_param_sto] += sty->ByteSize();
          ct_tot_mem_usage[func_param_sto] += sty->ByteSize();
          std::ostringstream oss;
          p->Print(oss);
          ct_mem_alloc_inst_set[func_param_sto].push("parameter of function " +
                                                     n.name + ": " + oss.str());
          VST_DEBUG(os << "[MemUsage] "
                       << "Function parameter `" << name << "`("
                       << __internal__::GetStringFrom(func_param_sto)
                       << ") need " << sty->ByteSize() << " bytes.\n");
        }
      }
      param_idx++;
    }
    // special handling
    // Because AST::FunctionDecl.accept doesn't call AfterVisit
    UpdateCtMaxMemUsage();
    CheckCtMemUsage(n);
    VST_DEBUG(os << "[MemUsage] "
                 << "Total compile-time mem used of parameters of function "
                 << n.name << ":\n"
                 << GetMemUsageMapDetail(ct_tot_mem_usage));
    return true;
  }
  bool Visit(AST::ChoreoFunction& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::CppSourceCode& n) {
    TraceEachVisit(n);
    return true;
  }
  bool Visit(AST::Program& n) {
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

#endif // __CHOREO_MEMORY_USAGE_CHECK_HPP__
