#ifndef __CHOREO_DMA_COMPLETION_ANALYSIS_HPP__
#define __CHOREO_DMA_COMPLETION_ANALYSIS_HPP__

#include "visitor.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace Choreo {

// Completion occupancy is independent of buffer and future.data liveness.
// Identities denote persistent context bindings, not source spellings or DMA
// sites: rotate moves bindings, and a single site may use several bindings.
struct DMACompletionFinding {
  const AST::Node* source;
  std::string resource;
  std::string reason;
};

struct BufferSafetyFinding {
  const AST::Node* access;
  const AST::Node* pending_issue;
  std::string buffer;
  std::string code;
  bool operator<(const BufferSafetyFinding& other) const {
    return std::tie(access, pending_issue, buffer, code) <
           std::tie(other.access, other.pending_issue, other.buffer,
                    other.code);
  }
};

struct DMACompletionInfo {
  const AST::Node* root = nullptr;
  std::string function;
  size_t scope_ordinal = 0;
  bool async_launch = false;
  bool analyzed = false;
  std::vector<DMACompletionFinding> findings;
  std::set<BufferSafetyFinding> buffer_findings;
  std::set<std::string> buffer_limitations;
  size_t buffer_accesses = 0;
  std::set<std::string> resources;
  std::map<const AST::DMA*, std::string> anonymous;
  std::set<std::pair<std::string, std::string>> conflicts;
  std::set<std::string> diagnostics;
  std::string fallback;
  bool dedicated_fallback = false;
  size_t pending_bound = 0;

  bool interferes(const std::string& a, const std::string& b) const {
    return !fallback.empty() || conflicts.count(std::minmax(a, b));
  }
};

// Structured CFG with a may-pending fixed point, partitioned by the exact
// permutation of context bindings. Branches are conservative; waits kill only
// their current binding. Unknown completion/participant protocols never prove
// reuse. This analysis does not change buffer liveness or the DMA schedule.
class DMACompletionAnalysis : public VisitorWithSymTab {
public:
  // Participant-local checking is report-only. Allocation keeps its existing
  // conservative handling of asynchronous launches.
  explicit DMACompletionAnalysis(bool participant_local = false)
      : VisitorWithSymTab("dma-completion"),
        participant_local_(participant_local) {}
  using ResultsMap = std::map<std::string, DMACompletionInfo>;

  const ResultsMap& results() const { return results_; }

private:
  enum class Kind { Empty, Issue, Wait, Rotate };
  struct Action {
    Kind kind;
    std::vector<std::string> names;
    bool async = false;
    bool declaration = false;
  };
  struct Device {
    AST::Node* root = nullptr;
    DMACompletionInfo info;
    std::map<const AST::Node*, Action> actions;
    std::set<std::string> declarations;
    std::map<std::string, bool> private_contexts;
    std::set<ParallelLevel> participant_levels;
    std::set<const AST::ParallelBy*> participant_scopes;
  };
  std::map<std::string, Device> devices_;
  std::map<std::string, DMACompletionInfo> results_;
  Device* device_ = nullptr;
  std::string device_name_;
  std::vector<ParallelLevel> levels_;
  std::vector<const AST::ParallelBy*> participants_;
  bool participant_local_;
  std::map<std::string, size_t> scope_ordinals_;
  std::map<const AST::Node*, const AST::Node*> parents_;
  std::vector<const AST::Node*> nodes_;

  void recordParticipant(Device& device);

  bool RunOnProgramImpl(AST::Node& root) override;
  bool BeforeVisitImpl(AST::Node& n) override;
  bool AfterVisitImpl(AST::Node& n) override;
  void analyze(Device& device);
};

} // namespace Choreo

#endif
