#ifndef __CHOREO_MEMORY_USAGE_CHECK_HPP__
#define __CHOREO_MEMORY_USAGE_CHECK_HPP__

#include <iomanip>
#include <numeric>

#include "ast.hpp"
#include "context.hpp"
#include "visitor.hpp"

namespace Choreo {

// tuple<runtime memory usages, code location, corresponding storage limit>
// to insert runtime memory usage check in codegen
using RtMemUsageCheckInfo =
    std::tuple<std::vector<std::string>, location, size_t, Storage>;

// checking compile-time and runtime memory usage
struct MemUsageCheck : public VisitorWithSymTab {
private:
  // map from storage type to a integer
  typedef std::map<Storage, size_t> CtMemUsageMap;
  typedef std::map<Storage, std::vector<std::string>> RtMemUsageMap;

  // Memory usage is measured in bytes
  // ct for compile time, rt for runtime

  // the top of ct_mem_usage_list is the CtMemUsageMap of the current ast node!
  std::stack<CtMemUsageMap> ct_mem_usage_list;
  // record memory allocations(usage known at ct), useful when tracing
  std::vector<std::map<Storage, std::vector<std::string>>>
      ct_mem_alloc_inst_sets;
  // total ct memory usage
  CtMemUsageMap ct_tot_mem_usage;
  // Only the maximum ct memory usage is recorded
  CtMemUsageMap ct_max_mem_usage;

  std::stack<RtMemUsageMap> rt_mem_usage_list;
  // only runtime usages are recorded
  RtMemUsageMap rt_tot_mem_usage;
  std::vector<RtMemUsageCheckInfo> rt_mem_usage_check_list;

  CtMemUsageMap mem_usage_limit;
  std::unordered_set<Storage> tocheck_storage;

private:
  bool BeforeVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
        isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n)) {
      // generate the map of current ast node that corresponding to the scope
      ct_mem_usage_list.push(CtMemUsageMap{});
      rt_mem_usage_list.push(RtMemUsageMap{});
      ct_mem_alloc_inst_sets.push_back(/*empty map*/ {});
    }
    return true;
  }

  bool AfterVisitImpl(AST::Node& n) override {
    if (isa<AST::ChoreoFunction>(&n) || isa<AST::ParallelBy>(&n) ||
        isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n)) {
      UpdateCtMaxMemUsage();
      CheckCtMemUsage(n);
      VST_DEBUG(dbgs() << "[MemUsage] "
                       << "Total compile-time mem used before leaving scope "
                       << SSTab().ScopeName() << "\n"
                       << GetMemUsageMapDetail(ct_tot_mem_usage));
      RestoreMemUsage();
    }

    // the choreo function is exiting, show the maximum ct mem usage
    if (auto cf = dyn_cast<AST::ChoreoFunction>(&n)) {
      RestoreMemUsage();
      VST_DEBUG(
          dbgs() << "[MemUsage] "
                 << "The maximum memory usages at compile time for each level"
                    "(Not at the same time):\n"
                 << GetMemUsageMapDetail(ct_max_mem_usage));
      assert(ct_mem_usage_list.empty() && rt_mem_usage_list.empty());
      AppendRuntimeCheck(cf->name);
    }
    return true;
  }

  void TraceEachVisit(AST::Node& n, std::string sup = "") {
    if (trace_visit) dbgs() << n.TypeNameString() << sup << "\n";
  }

  void RestoreMemUsage() {
    // restore ct mem usage
    for (const auto& [sto, mem_used_in_scop] : ct_mem_usage_list.top())
      ct_tot_mem_usage[sto] -= mem_used_in_scop;
    ct_mem_alloc_inst_sets.pop_back();
    ct_mem_usage_list.pop();

    // restore rt mem usage
    for (const auto& [sto, rt_mem_used_in_scop] : rt_mem_usage_list.top()) {
      for (size_t i = 0; i < rt_mem_used_in_scop.size(); ++i)
        rt_tot_mem_usage[sto].pop_back();
      // no inst tracing when dealing with rt usage check yet
    }
    rt_mem_usage_list.pop();
  }

  // Check whether the ct memory usage at each level exceeds limits
  void CheckCtMemUsage(AST::Node& n) {
    for (const auto& sto : tocheck_storage) {
      if (ct_tot_mem_usage[sto] > mem_usage_limit[sto]) {
        // get the variables which lead to out of bound
        std::ostringstream oss;
        for (const auto& inst_set : ct_mem_alloc_inst_sets)
          if (inst_set.count(sto))
            for (const auto& inst : inst_set.at(sto)) oss << "\n\t\t" << inst;
        std::string error_msg =
            __internal__::GetStringFrom(sto) + " memory OUT OF BOUND!\n\t" +
            "In the scope " + SSTab().ScopeName() + ", compile-time " +
            __internal__::GetStringFrom(sto) +
            " memory:\n\tUsed: " + std::to_string(ct_tot_mem_usage[sto]) +
            " bytes, Limit: " + std::to_string(mem_usage_limit[sto]) +
            " bytes. With variables:" + oss.str();
        if (sto == Storage::LOCAL && CCtx().HasFeature(ChoreoFeature::SLML))
          error_msg += "\n\tNote: For " + CCtx().TargetName() +
                       " target, the local memory limits can be set via "
                       "`--max-local-mem-capacity` option.";
        Error1(n.LOC(), error_msg);
      }
    }
  }

  // Update the maximum ct memory usage for each storage level
  void UpdateCtMaxMemUsage() {
    for (const auto& [sto, usage] : ct_tot_mem_usage)
      if (ct_max_mem_usage[sto] < usage) ct_max_mem_usage[sto] = usage;
  }

  // Return the detail memory usage of the given map
  std::string GetMemUsageMapDetail(CtMemUsageMap& m) {
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

  // Transform size in byte to human readable format like 1KB 2GB etc in
  // decimal.
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
    if (mem_usage_limit.count(sto) == 0) return "N/A";
    assert(mem_usage_limit[sto] > 0 &&
           "memory limitation should greater than 0!");
    oss << (size_t)(100.0 * ct_tot_mem_usage[sto] / mem_usage_limit[sto])
        << "%";
    return oss.str();
  }

  void AppendRuntimeCheck(const std::string& fname) {
    for (const auto& [useds, loc, limit, sto] : rt_mem_usage_check_list) {
      std::string lhs, op, rhs, message;

      lhs = "(size_t)";

      op = "<=";

      rhs = "(size_t)";
      rhs += std::to_string(limit);

      message = "total memory usage at ";
      message += __internal__::GetStringFrom(sto);
      message += " level (compile time and runtime) should not exceed ";
      message += std::to_string(limit);
      message += " bytes";

      for (auto& used : useds) {
        if (used.find(":") == std::string::npos) {
          // `used` is compile time memory usage
          lhs += (lhs.back() == ')' ? "" : " + ") + used;
          continue;
        }
        // `used` is runtime memory usage
        auto operands = SplitStringByDelimiter(used, "*");

        lhs += (lhs.back() == ')' ? "" : " + ");
        lhs += DelimitedString(operands, "*");
      }
      FCtx(fname).AppendRtCheck({lhs, op, rhs, loc, message, {}});
    }
    rt_mem_usage_check_list.clear();
  }

public:
  MemUsageCheck() : VisitorWithSymTab("muchk") {
    // Don not manage global memory cap when target requires
    if (!CCtx().HasFeature(ChoreoFeature::MGM))
      tocheck_storage = {Storage::LOCAL, Storage::SHARED};
    else
      tocheck_storage = {Storage::LOCAL, Storage::SHARED, Storage::GLOBAL};
    // initialize with ct_tot_mem_usage
    for (const auto& sto : tocheck_storage) {
      ct_tot_mem_usage[sto] = 0;
      // initialize max memory we can allocate in byte
      mem_usage_limit[sto] = CCtx().GetMemCapacity(sto);
    }
    VST_DEBUG(dbgs() << "[MemUsage] "
                     << "Memory usage limit of architecture "
                     << ToUpper(CCtx().GetArch()) << " is:\n"
                     << GetMemUsageMapDetail(mem_usage_limit));
  }
  ~MemUsageCheck() {}

  bool Visit(AST::NamedVariableDecl& n) override {
    TraceEachVisit(n);
    // mem alloc could happen here
    auto sty = dyn_cast<SpannedType>(GetSymbolType(n.name_str));
    if (!sty) return true;
    auto sto = sty->GetStorage();
    if (tocheck_storage.count(sto) == 0)
      return true; // only check valid storage types
    assert(tocheck_storage.count(sto) &&
           "Only support Storage types in `tocheck_storage`!");
    if (n.HasNote("offset")) {
      VST_DEBUG({
        dbgs() << "[MemUsage] The mem space of buffer " << n.name_str
               << " reuses the space of self-defined SPM!\n";
      });
      return true;
    }
    size_t array_dim_product = 1;
    if (n.IsArray())
      array_dim_product = std::accumulate(n.ArrayDimensions().begin(),
                                          n.ArrayDimensions().end(), 1,
                                          std::multiplies<size_t>());
    if (sty->RuntimeShaped()) {
      // runtime usage
      std::string byte_size = sty->ByteSizeExpression(true);
      if (array_dim_product != 1)
        byte_size =
            std::to_string(array_dim_product) + " * (" + byte_size + ")";
      VST_DEBUG(dbgs() << "[MemUsage] " << __internal__::GetStringFrom(sto)
                       << " `" << SSTab().ScopedName(n.name_str) << "` need "
                       << sty->ByteSizeExpression(false) << " bytes.\n");
      rt_mem_usage_list.top()[sto].push_back(byte_size);
      rt_tot_mem_usage[sto].push_back(byte_size);
      if (mem_usage_limit.count(sto))
        rt_mem_usage_check_list.push_back(std::make_tuple(
            SumUpCtRtUsage(sto), n.LOC(), mem_usage_limit.at(sto), sto));
    } else {
      // compile time usage
      auto size = sty->ByteSize() * array_dim_product;
      VST_DEBUG(dbgs() << "[MemUsage] " << __internal__::GetStringFrom(sto)
                       << " `" << SSTab().ScopedName(n.name_str) << "` need "
                       << size << " bytes" << SizeForHuman(size) << ".\n");
      ct_mem_usage_list.top()[sto] += size;
      ct_tot_mem_usage[sto] += size;
      ct_mem_alloc_inst_sets.back()[sto].push_back(
          SSTab().ScopedName(n.name_str));
    }

    return true;
  }
  bool Visit(AST::FunctionDecl& n) override {
    TraceEachVisit(n);
    // special handling
    // Because AST::FunctionDecl.accept doesn't call BeforeVisit)
    ct_mem_usage_list.push(CtMemUsageMap{});
    rt_mem_usage_list.push(RtMemUsageMap{});
    ct_mem_alloc_inst_sets.push_back(/*empty map*/ {});

    int param_idx = 0;
    Storage func_param_sto = Storage::GLOBAL;
    for (const auto& p : n.params->values) {
      if (p->attr == ParamAttr::GLOBAL_INPUT) continue; // skip global input
      auto sty = dyn_cast<SpannedType>(p->GetType());
      if (!sty) continue;

      std::string name = "::" + n.name + "::";
      name +=
          (p->HasSymbol() ? p->sym->name : ("#" + std::to_string(param_idx)));
      if (sty->RuntimeShaped()) {
        std::string byte_size = sty->ByteSizeExpression(true);
        rt_mem_usage_list.top()[func_param_sto].push_back(byte_size);
        rt_tot_mem_usage[func_param_sto].push_back(byte_size);
        if (mem_usage_limit.count(func_param_sto))
          rt_mem_usage_check_list.push_back(std::make_tuple(
              SumUpCtRtUsage(func_param_sto), p->LOC(),
              mem_usage_limit.at(func_param_sto), func_param_sto));
        VST_DEBUG(dbgs() << "[MemUsage] "
                         << "Function parameter `" << name << "`("
                         << __internal__::GetStringFrom(func_param_sto)
                         << ") need " << sty->ByteSizeExpression(false)
                         << " bytes.\n");
      } else {
        size_t size = sty->ByteSize();
        ct_mem_usage_list.top()[func_param_sto] += size;
        ct_tot_mem_usage[func_param_sto] += size;
        std::ostringstream oss;
        p->Print(oss);
        std::string func_param_inst =
            "parameter of function " + n.name + ": " + oss.str();
        ct_mem_alloc_inst_sets.back()[func_param_sto].push_back(
            func_param_inst);
        VST_DEBUG(dbgs() << "[MemUsage] "
                         << "Function parameter `" << name << "`("
                         << __internal__::GetStringFrom(func_param_sto)
                         << ") need " << size << " bytes" << SizeForHuman(size)
                         << ".\n");
      }
      param_idx++;
    }
    // special handling
    // Because AST::FunctionDecl.accept doesn't call AfterVisit
    UpdateCtMaxMemUsage();
    CheckCtMemUsage(n);
    VST_DEBUG(dbgs() << "[MemUsage] "
                     << "Total compile-time mem used of parameters of function "
                     << n.name << ":\n"
                     << GetMemUsageMapDetail(ct_tot_mem_usage));
    return true;
  }
};

} // end namespace Choreo

#endif // __CHOREO_MEMORY_USAGE_CHECK_HPP__
