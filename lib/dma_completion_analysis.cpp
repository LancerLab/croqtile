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

void DMACompletionAnalysis::recordParticipant(Device& device) {
  device.participant_levels.insert(levels_.back());
  device.participant_scopes.insert(participants_.back());
}

bool DMACompletionAnalysis::RunOnProgramImpl(AST::Node& root) {
  parents_.clear();
  nodes_.clear();
  devices_.clear();
  results_.clear();
  device_ = nullptr;
  levels_.clear();
  participants_.clear();
  scope_ordinals_.clear();
  root.accept(*this);
  for (auto& [name, device] : devices_) {
    analyze(device);
    results_[name] = std::move(device.info);
  }
  return !HasError();
}

bool DMACompletionAnalysis::BeforeVisitImpl(AST::Node& n) {
  if (participant_local_) {
    parents_[&n] = nodes_.empty() ? nullptr : nodes_.back();
    // Expressions and device prototypes have no AfterVisit callback.
    // They cannot own a statement effect; retain their parent but do not
    // leave them on the structural statement stack.
    if (!isa<AST::Expr>(&n) && !isa<AST::DeviceFunctionDecl>(&n))
      nodes_.push_back(&n);
  }
  if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels_.push_back(pb->GetLevel());
    participants_.push_back(pb);
    if (pb->IsDeviceEntry()) {
      device_name_ = SSTab().ScopeName();
      device_ = &devices_[device_name_];
      device_->root = &n;
      device_->info.root = &n;
      device_->info.function = CurrentFunctionName();
      device_->info.scope_ordinal = scope_ordinals_[CurrentFunctionName()]++;
      device_->info.async_launch = pb->IsAsync();
    }
    if (device_ && pb->IsAsync() &&
        !(participant_local_ && pb->IsDeviceEntry()))
      device_->info.fallback = "asynchronous parallel participants";
  }
  if (!device_) return true;
  auto& device = *device_;
  if (participant_local_) {
    if (auto call = dyn_cast<AST::Call>(&n);
        call && !call->IsArith() && !call->CompileTimeEval() && !call->IsAnno())
      device.info.buffer_limitations.insert("opaque_call_effects");
    if (isa<AST::CppSourceCode>(&n))
      device.info.buffer_limitations.insert("embedded_code_effects");
    if (auto mma = dyn_cast<AST::MMA>(&n)) {
      auto op = mma->GetOperation();
      if ((op->IsLoad() && op->IsAsync()) ||
          (op->IsKind(AST::MMAOperation::Store) && op->StoreIsAsync()))
        device.info.buffer_limitations.insert("asynchronous_mma_effects");
    }
  }
  if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
    // At leaf THREAD level this is a per-participant conditional in the
    // emitted kernel. It does not fork another execution stream in that SIP.
    if (it->async &&
        !(participant_local_ && levels_.back() == ParallelLevel::THREAD))
      device.info.fallback = "asynchronous inthreads participants";
  }
  if (auto dma = dyn_cast<AST::DMA>(&n)) {
    std::string name;
    if (!dma->future.empty()) name = InScopeName(dma->future);
    if (participant_local_) recordParticipant(device);
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
      if (participant_local_)
        device.info.fallback = "collective completion ownership not analyzed";
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
    recordParticipant(device);
  } else if (auto wait = dyn_cast<AST::Wait>(&n)) {
    Action action{Kind::Wait, {}};
    for (auto& target : wait->GetTargets()) {
      if (!isa<FutureType>(NodeType(*target))) continue;
      auto id = cast<AST::Expr>(target)->GetSymbol();
      if (id) action.names.push_back(InScopeName(id->name));
    }
    device.actions[&n] = std::move(action);
    if (!wait->GetTargets().empty()) recordParticipant(device);
  } else if (auto rotate = dyn_cast<AST::Rotate>(&n)) {
    Action action{Kind::Rotate, {}};
    for (auto& value : rotate->GetIds())
      action.names.push_back(InScopeName(cast<AST::Identifier>(value)->name));
    device.actions[&n] = std::move(action);
    if (participant_local_) recordParticipant(device);
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
  if (participant_local_) nodes_.pop_back();
  if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    if (pb->IsDeviceEntry()) device_ = nullptr;
    levels_.pop_back();
    participants_.pop_back();
  }
  return true;
}

void DMACompletionAnalysis::analyze(Device& device) {
  auto& info = device.info;
  if (device.participant_levels.size() > 1)
    info.fallback = "completion uses different participant levels";
  if (participant_local_ && !info.resources.empty() &&
      (device.participant_scopes.size() != 1 ||
       device.participant_levels !=
           std::set<ParallelLevel>{ParallelLevel::THREAD}))
    info.fallback = "completion requires one lexical THREAD participant scope";
  if (!info.fallback.empty() || (!participant_local_ && info.resources.empty()))
    return;
  if (participant_local_) {
    for (const auto& [node, action] : device.actions)
      for (const auto& name : action.names)
        if (!info.resources.count(name)) {
          info.fallback = "completion references an external context";
          return;
        }
  }
  std::vector<std::string> names(info.resources.begin(), info.resources.end());
  std::map<std::string, size_t> indices;
  for (size_t i = 0; i < names.size(); ++i) indices[names[i]] = i;

  std::map<const AST::Node*, std::vector<BufferAccessEvent>> effects;
  if (participant_local_) {
    for (const auto& event : CCtx().GetBufferAccessLog().events) {
      const AST::Node* current = event.stmt;
      const AST::Node* statement = current;
      bool belongs = false;
      while (current) {
        if (current == device.root) {
          belongs = true;
          break;
        }
        auto parent = parents_.find(current);
        if (parent == parents_.end()) break;
        if (isa<AST::MultiNodes>(parent->second)) statement = current;
        current = parent->second;
      }
      if (!belongs) continue;
      // Keep effects on the innermost statement, not the enclosing loop.
      current = event.stmt;
      while (current && current != device.root) {
        auto parent = parents_.find(current);
        if (parent == parents_.end()) break;
        statement = current;
        if (isa<AST::MultiNodes>(parent->second)) break;
        current = parent->second;
      }
      effects[statement].push_back(event);
      ++info.buffer_accesses;
      if (event.opaque_effect)
        info.buffer_limitations.insert("opaque_call_effects");
      if (event.storage == Storage::SHARED ||
          event.storage == Storage::GROUP_SHARED)
        info.buffer_limitations.insert("cross_participant_storage");
      if (!event.allocation && event.level != ParallelLevel::THREAD)
        info.buffer_limitations.insert("non_thread_buffer_access");
    }
  }

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
    return device.actions.count(n) || effects.count(n) ? add(n, {next}) : next;
  };
  size_t entry = build(device.root->GetBody().get(), 0, 0, 0);

  // Merge pending sets only within an identical binding permutation. This
  // bounds ordinary loops independently of trip count without confusing a
  // rotated pending operation with the next operation at the same DMA site.
  using Bindings = std::vector<size_t>;
  using Pending = std::vector<bool>;
  struct State {
    Pending pending;
    Pending must_pending;
    std::vector<std::set<const AST::Node*>> transfers;
    std::set<std::string> initialized;
  };
  std::vector<std::map<Bindings, State>> incoming(cfg.size());
  std::deque<std::pair<size_t, Bindings>> work;
  Bindings initial(names.size());
  std::iota(initial.begin(), initial.end(), 0);
  incoming[entry][initial] = {
      Pending(names.size(), false),
      Pending(names.size(), false),
      std::vector<std::set<const AST::Node*>>(names.size()),
      {}};
  work.emplace_back(entry, initial);
  size_t partitions = 1;
  constexpr size_t partition_limit = 32768;
  std::set<size_t> unproven;
  auto diagnose = [&](const AST::Node* source, size_t owner,
                      const std::string& reason) {
    std::ostringstream message;
    if (source) message << source->LOC() << ": ";
    message << names[owner] << ": " << reason;
    if (info.diagnostics.insert(message.str()).second)
      info.findings.push_back({source, names[owner], reason});
    unproven.insert(owner);
  };
  while (!work.empty()) {
    auto [pc, binding] = std::move(work.front());
    work.pop_front();
    State state = incoming[pc].at(binding);
    auto& pending = state.pending;
    if (pc == 0) {
      for (size_t i = 0; i < pending.size(); ++i)
        if (pending[i]) diagnose(device.root, i, "may exit with pending DMA");
      continue;
    }
    const auto* source = cfg[pc].source;
    for (const auto& access : effects[source]) {
      for (size_t owner = 0; owner < pending.size(); ++owner) {
        if (!pending[owner]) continue;
        for (const auto* issue : state.transfers[owner]) {
          for (const auto& transfer : effects[issue]) {
            if (transfer.buffer != access.buffer ||
                (transfer.kind == AccessKind::READ &&
                 access.kind == AccessKind::READ))
              continue;
            auto code = access.kind == AccessKind::READ
                            ? "BUFFER_READ_BEFORE_DMA_COMPLETE"
                            : "BUFFER_WRITE_WHILE_DMA_PENDING";
            info.buffer_findings.insert(
                {access.stmt, issue, access.buffer, code});
          }
        }
      }
      if (access.allocation) {
        state.initialized.erase(access.buffer);
        if (access.initializes_whole_buffer)
          state.initialized.insert(access.buffer);
      } else if (access.kind == AccessKind::READ &&
                 access.storage == Storage::LOCAL) {
        if (!state.initialized.count(access.buffer))
          info.buffer_findings.insert({access.stmt, nullptr, access.buffer,
                                       "BUFFER_INITIALIZATION_UNPROVEN"});
      }
    }
    auto complete = [&](size_t owner) {
      // Initialization must hold for every issue that can reach this wait.
      std::set<std::string> initialized;
      bool first = true;
      for (const auto* issue : state.transfers[owner]) {
        std::set<std::string> written;
        for (const auto& access : effects[issue])
          if (access.kind == AccessKind::WRITE &&
              access.initializes_whole_buffer)
            written.insert(access.buffer);
        if (first)
          initialized = written;
        else {
          std::set<std::string> intersection;
          std::set_intersection(
              initialized.begin(), initialized.end(), written.begin(),
              written.end(), std::inserter(intersection, intersection.end()));
          initialized = std::move(intersection);
        }
        first = false;
      }
      if (state.must_pending[owner])
        state.initialized.insert(initialized.begin(), initialized.end());
      state.must_pending[owner] = false;
      state.transfers[owner].clear();
      pending[owner] = false;
    };
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
        for (auto handle : handles) complete(binding[handle]);
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
          if (participant_local_) {
            state.transfers[owner].insert(source);
            state.must_pending[owner] = true;
          }
          pending[owner] = action.async;
          if (!action.async) complete(owner);
        } else {
          complete(owner);
        }
      }
    }
    for (size_t next : cfg[pc].successors) {
      auto [it, inserted] = incoming[next].emplace(binding, state);
      bool changed = inserted;
      if (inserted && ++partitions > partition_limit) {
        info.fallback = "completion binding analysis budget exceeded";
        info.dedicated_fallback = true;
        return;
      }
      if (!inserted) {
        for (size_t i = 0; i < pending.size(); ++i) {
          if (pending[i] && !it->second.pending[i]) {
            it->second.pending[i] = true;
            changed = true;
          }
        }
      }
      if (!inserted && participant_local_) {
        for (size_t i = 0; i < state.transfers.size(); ++i) {
          bool must = it->second.must_pending[i] && state.must_pending[i];
          changed |= must != it->second.must_pending[i];
          it->second.must_pending[i] = must;
          auto& target = it->second.transfers[i];
          auto before = target.size();
          target.insert(state.transfers[i].begin(), state.transfers[i].end());
          changed |= target.size() != before;
        }
        std::set<std::string> initialized;
        std::set_intersection(
            it->second.initialized.begin(), it->second.initialized.end(),
            state.initialized.begin(), state.initialized.end(),
            std::inserter(initialized, initialized.end()));
        changed |= initialized != it->second.initialized;
        it->second.initialized = std::move(initialized);
      }
      if (changed) work.emplace_back(next, binding);
    }
  }
  info.analyzed = true;
  if (participant_local_ && !info.findings.empty())
    info.buffer_limitations.insert("completion_not_proved");
  // Do not make a suspect lifecycle less safe through sharing. Diagnostics
  // are may-path facts (scalar predicates are not correlated), not errors.
  for (size_t owner : unproven)
    for (size_t other = 0; other < names.size(); ++other)
      if (owner != other)
        info.conflicts.insert(std::minmax(names[owner], names[other]));
}
