#include "dma_completion_analysis.hpp"

#include "ast.hpp"
#include "aux.hpp"
#include "types.hpp"

#include <algorithm>
#include <deque>
#include <functional>
#include <numeric>
#include <sstream>

using namespace Choreo;

bool DMACompletionAnalysis::RunOnProgramImpl(AST::Node& root) {
  devices_.clear();
  results_.clear();
  device_ = nullptr;
  levels_.clear();
  root.accept(*this);
  for (auto& [name, device] : devices_) {
    analyze(device);
    results_[name] = std::move(device.info);
  }
  return !HasError();
}

bool DMACompletionAnalysis::BeforeVisitImpl(AST::Node& n) {
  if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels_.push_back(pb->GetLevel());
    if (pb->IsDeviceEntry()) {
      device_name_ = SSTab().ScopeName();
      device_ = &devices_[device_name_];
      device_->root = &n;
    }
    if (device_ && pb->IsAsync())
      device_->info.fallback = "asynchronous parallel participants";
  }
  if (!device_) return true;
  auto& device = *device_;
  if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
    if (it->async) device.info.fallback = "asynchronous inthreads participants";
  }
  if (auto dma = dyn_cast<AST::DMA>(&n)) {
    std::string name;
    if (!dma->future.empty()) name = InScopeName(dma->future);
    if (dma->operation == ".any") {
      device.declarations.insert(name);
      device.info.resources.insert(name);
      device.actions[&n] = {Kind::Empty, {name}, false, true};
      return true;
    }
    if (dma->chained || dma->HasEvent())
      device.info.fallback = "DMA chain or native completion publication";
    auto storage = [&](const ptr<AST::Node>& operand) {
      if (auto m = dyn_cast<AST::Memory>(operand)) return m->Get();
      if (auto ca = dyn_cast<AST::ChunkAt>(operand)) {
        if (auto ty = GetSpannedType(GetSymbolType(ca->data->name)))
          return ty->GetStorage();
      }
      return Storage::NONE;
    };
    auto from = storage(dma->from), to = storage(dma->to);
    if (from == Storage::NONE || to == Storage::NONE)
      device.info.fallback = "unclassified DMA context family";
    bool private_context = std::min(from, to) != Storage::SHARED;
    if (!name.empty()) {
      auto [it, inserted] =
          device.private_contexts.emplace(name, private_context);
      if (!inserted && it->second != private_context)
        device.info.fallback = "future crosses context families";
    }
    if (!private_context) {
      // Collective contexts have a different ownership contract. They are
      // not part of the private pool, including placeholders later bound here.
      device.info.resources.erase(name);
      return true;
    }
    if (name.empty()) {
      name =
          device_name_ + "$sync" + std::to_string(device.info.anonymous.size());
      device.info.anonymous[dma] = name;
      if (dma->IsAsync())
        device.info.fallback = "anonymous asynchronous completion";
    }
    bool declaration = device.declarations.insert(name).second;
    device.info.resources.insert(name);
    device.actions[&n] = {Kind::Issue, {name}, dma->IsAsync(), declaration};
    device.participant_levels.insert(levels_.back());
  } else if (auto wait = dyn_cast<AST::Wait>(&n)) {
    Action action{Kind::Wait, {}};
    for (auto& target : wait->GetTargets()) {
      if (!isa<FutureType>(NodeType(*target))) continue;
      auto id = cast<AST::Expr>(target)->GetSymbol();
      if (id) action.names.push_back(InScopeName(id->name));
    }
    device.actions[&n] = std::move(action);
    if (!wait->GetTargets().empty())
      device.participant_levels.insert(levels_.back());
  } else if (auto rotate = dyn_cast<AST::Rotate>(&n)) {
    Action action{Kind::Rotate, {}};
    for (auto& value : rotate->GetIds())
      action.names.push_back(InScopeName(cast<AST::Identifier>(value)->name));
    device.actions[&n] = std::move(action);
  } else if (auto select = dyn_cast<AST::Select>(&n)) {
    for (auto& value : select->expr_list->AllValues()) {
      if (isa<FutureType>(NodeType(*value)))
        device.info.fallback = "selected future reference";
    }
  } else if (auto trigger = dyn_cast<AST::Trigger>(&n)) {
    if (trigger->HasDependencies())
      device.info.fallback = "native completion publication";
  } else if (auto decl = dyn_cast<AST::NamedVariableDecl>(&n)) {
    if (isa<FutureType>(NodeType(n)) && !isa<AST::DMA>(decl->init_expr))
      device.info.fallback = "aliased future binding";
  } else if (auto assignment = dyn_cast<AST::Assignment>(&n)) {
    if (isa<FutureType>(NodeType(n)) && !isa<AST::DMA>(assignment->value))
      device.info.fallback = "aliased future binding";
  }
  return true;
}

bool DMACompletionAnalysis::AfterVisitImpl(AST::Node& n) {
  if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    if (pb->IsDeviceEntry()) device_ = nullptr;
    levels_.pop_back();
  }
  return true;
}

void DMACompletionAnalysis::analyze(Device& device) {
  auto& info = device.info;
  if (device.participant_levels.size() > 1)
    info.fallback = "completion uses different participant levels";
  if (!info.fallback.empty() || info.resources.empty()) return;
  std::vector<std::string> names(info.resources.begin(), info.resources.end());
  std::map<std::string, size_t> indices;
  for (size_t i = 0; i < names.size(); ++i) indices[names[i]] = i;

  struct Node {
    const AST::Node* source = nullptr;
    std::vector<size_t> successors;
  };
  std::vector<Node> cfg(1); // index zero is the common device exit
  auto add = [&](const AST::Node* source, std::vector<size_t> successors) {
    size_t index = cfg.size();
    cfg.push_back({source, std::move(successors)});
    return index;
  };
  std::function<size_t(AST::Node*, size_t, size_t, size_t)> build;
  build = [&](AST::Node* n, size_t next, size_t break_to,
              size_t continue_to) -> size_t {
    if (!n) return next;
    if (auto seq = dyn_cast<AST::MultiNodes>(n)) {
      for (auto i = seq->values.rbegin(); i != seq->values.rend(); ++i)
        next = build(i->get(), next, break_to, continue_to);
      return next;
    }
    if (isa<AST::Return>(n)) return add(n, {0});
    if (isa<AST::Break>(n)) return add(n, {break_to});
    if (isa<AST::Continue>(n)) return add(n, {continue_to});
    if (auto branch = dyn_cast<AST::IfElseBlock>(n))
      return add(
          n, {build(branch->GetThenBody().get(), next, break_to, continue_to),
              build(branch->GetElseBody().get(), next, break_to, continue_to)});
    if (isa<AST::WhileBlock>(n) || isa<AST::ForeachBlock>(n)) {
      size_t head = add(n, {});
      size_t body = build(n->GetBody().get(), head, next, head);
      cfg[head].successors = {body, next}; // includes zero iterations
      return head;
    }
    if (n->HasBody()) {
      size_t body = build(n->GetBody().get(), next, break_to, continue_to);
      if (isa<AST::InThreadsBlock>(n)) return add(n, {body, next});
      return body;
    }
    return device.actions.count(n) ? add(n, {next}) : next;
  };
  size_t entry = build(device.root->GetBody().get(), 0, 0, 0);

  // Merge pending sets only within an identical binding permutation. This
  // bounds ordinary loops independently of trip count without confusing a
  // rotated pending operation with the next operation at the same DMA site.
  using Bindings = std::vector<size_t>;
  using Pending = std::vector<bool>;
  std::vector<std::map<Bindings, Pending>> incoming(cfg.size());
  std::deque<std::pair<size_t, Bindings>> work;
  Bindings initial(names.size());
  std::iota(initial.begin(), initial.end(), 0);
  incoming[entry][initial] = Pending(names.size(), false);
  work.emplace_back(entry, initial);
  size_t partitions = 1;
  constexpr size_t partition_limit = 32768;
  std::set<size_t> unproven;
  auto diagnose = [&](const AST::Node* source, size_t owner,
                      const std::string& reason) {
    std::ostringstream message;
    if (source) message << source->LOC() << ": ";
    message << names[owner] << ": " << reason;
    info.diagnostics.insert(message.str());
    unproven.insert(owner);
  };
  while (!work.empty()) {
    auto [pc, binding] = std::move(work.front());
    work.pop_front();
    Pending pending = incoming[pc].at(binding);
    if (pc == 0) {
      for (size_t i = 0; i < pending.size(); ++i)
        if (pending[i]) diagnose(device.root, i, "may exit with pending DMA");
      continue;
    }
    auto action_it = device.actions.find(cfg[pc].source);
    if (action_it != device.actions.end()) {
      const auto& action = action_it->second;
      std::vector<size_t> handles;
      for (const auto& name : action.names) {
        auto i = indices.find(name);
        if (i != indices.end()) handles.push_back(i->second);
      }
      if (action.kind == Kind::Rotate) {
        if (handles.size() != action.names.size()) {
          info.fallback = "rotation across context families";
          return;
        }
        if (handles.size() > 1) {
          auto first = binding[handles.front()];
          for (size_t i = 1; i < handles.size(); ++i)
            binding[handles[i - 1]] = binding[handles[i]];
          binding[handles.back()] = first;
        }
      } else if (action.kind == Kind::Wait) {
        for (auto handle : handles) pending[binding[handle]] = false;
      } else if (!handles.empty()) {
        auto handle = handles.front();
        if (action.declaration && binding[handle] != handle) {
          // Re-entering a lexical declaration after its original context was
          // rotated elsewhere needs a dynamic instance model beyond a ring.
          info.fallback = "reinitialized rotating context binding";
          info.dedicated_fallback = true;
          return;
        }
        auto owner = binding[handle];
        if (pending[owner])
          diagnose(cfg[pc].source, owner, "may overwrite pending DMA");
        if (action.kind == Kind::Issue) {
          for (size_t i = 0; i < pending.size(); ++i)
            if (pending[i] && i != owner)
              info.conflicts.insert(std::minmax(names[owner], names[i]));
          size_t demand = std::count(pending.begin(), pending.end(), true);
          info.pending_bound =
              std::max(info.pending_bound, demand + !pending[owner]);
          pending[owner] = action.async;
        } else {
          pending[owner] = false;
        }
      }
    }
    for (size_t next : cfg[pc].successors) {
      auto [it, inserted] = incoming[next].emplace(binding, pending);
      bool changed = inserted;
      if (inserted && ++partitions > partition_limit) {
        info.fallback = "completion binding analysis budget exceeded";
        info.dedicated_fallback = true;
        return;
      }
      if (!inserted) {
        for (size_t i = 0; i < pending.size(); ++i) {
          if (pending[i] && !it->second[i]) {
            it->second[i] = true;
            changed = true;
          }
        }
      }
      if (changed) work.emplace_back(next, binding);
    }
  }
  // Do not make a suspect lifecycle less safe through sharing. Diagnostics
  // are may-path facts (scalar predicates are not correlated), not errors.
  for (size_t owner : unproven)
    for (size_t other = 0; other < names.size(); ++other)
      if (owner != other)
        info.conflicts.insert(std::minmax(names[owner], names[other]));
}
