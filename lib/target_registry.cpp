#include "target_registry.hpp"
#include <mutex>

using namespace Choreo;

namespace {

struct Entry {
  std::string name;
  std::string desc;
  TargetCreateFn create;
};

std::unordered_map<TargetID, Entry>& registry() {
  static std::unordered_map<TargetID, Entry> map;
  return map;
}

std::mutex& registryMutex() {
  static std::mutex m;
  return m;
}

} // end anonymous namespace

void TargetRegistry::Register(TargetID id, const std::string& name,
                              const std::string& desc, TargetCreateFn fn) {
  std::lock_guard<std::mutex> lock(registryMutex());
  if (registry().count(id))
    choreo_unreachable("The target " + name + " has been registered twice.");
  registry().emplace(id, Entry{name, desc, fn});
}

std::unique_ptr<Target> TargetRegistry::Create(const std::string& name) {
  std::lock_guard<std::mutex> lock(registryMutex());
  auto& regs = registry();
  for (auto& reg : regs)
    if (reg.second.name == name) return reg.second.create();
  return nullptr;
}

std::vector<TargetInfo> TargetRegistry::List() {
  std::lock_guard<std::mutex> lock(registryMutex());
  std::vector<TargetInfo> out;
  for (auto& [id, entry] : registry())
    out.push_back({id, entry.name, entry.desc});
  return out;
}
