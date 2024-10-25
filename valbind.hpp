#ifndef __CHOREO_VALUE_BINDS_HPP__
#define __CHOREO_VALUE_BINDS_HPP__

// bind information: some value are bound to each other when applying statements
// that set where-binding, like `with ..., where a <-> b`, we need to trace the
// values.
//
// For example, in value numbering, we have different value number 'a' and 'b':
//
//  a - {a, b, c}
//  d - {}
//
// Note both 'a' and 'b' may have en non-empty bind-set. Thus binding 'a' and
// 'b' results in:
//
//  a - {a, b, c, d}
//  d _/
//

#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Choreo {

namespace ValBind {

template <typename T>
class Binds {
public:
  using Set = std::unordered_set<T>;
  // Add a bind relationship between a and b
  void AddBind(const T& a, const T& b) {
    // Find or create bind sets for a and b
    auto set_a = GetOrCreateSet(a);
    auto set_b = GetOrCreateSet(b);

    // If a and b are already in the same set, nothing to do
    if (set_a == set_b) return;

    // Merge the two sets
    if (set_a->size() < set_b->size()) {
      MergeSets(set_b, set_a);
    } else {
      MergeSets(set_a, set_b);
    }
  }

  // Get the bind set for a given value
  const Set& GetBinds(const T& value) const {
    static Set ret;
    ret = GetSet(value);
    ret.erase(value);
    return ret;
  }

  // Get the bind set for a given value
  const Set& GetSet(const T& value) const {
    auto it = bind_map.find(value);
    if (it != bind_map.end()) {
      return *(it->second);
    } else {
      static Set ret;
      return ret;
    }
  }

  void Clear() { bind_map.clear(); }

  // Print all bind sets for debugging
  void Print(std::ostream& os) const {
    std::unordered_set<const std::unordered_set<int>*> printed;
    for (const auto& pair : bind_map) {
      if (printed.insert(pair.second).second) {
        os << "{ ";
        for (int val : *(pair.second)) { os << val << " "; }
        os << "}\n";
      }
    }
  }

private:
  // Helper to get or create a bind set
  std::shared_ptr<Set> GetOrCreateSet(const T& value) {
    auto it = bind_map.find(value);
    if (it == bind_map.end()) {
      bind_map[value] = std::make_shared<Set>();
      bind_map[value]->emplace(value); // always bind to itself
    }
    return bind_map[value];
  }

  // Helper to merge two bind sets
  void MergeSets(std::shared_ptr<Set>& set_a, std::shared_ptr<Set>& set_b) {
    for (const T& value : *set_b) {
      set_a->insert(value);
      bind_map[value] = set_a;
    }
  }

  // Map from value to its bind set
  std::unordered_map<T, std::shared_ptr<Set>> bind_map;
};

template <typename T>
class BindInfo {
public:
  // Add an alias relationship between a and b
  void AddBind(const T& a, const T& b) { bind_sets.AddBind(a, b); }

  // Get the alias set for a given value
  const typename Binds<T>::Set& GetBinds(const T& value) const {
    return bind_sets.GetBinds(value);
  }

  // Get the alias set for a given value
  const typename Binds<T>::Set& GetSet(const T& value) const {
    return bind_sets.GetSet(value);
  }

  // Print all alias sets for debugging
  void PrintBinds(std::ostream& os) const { bind_sets.Print(os); }

  void Clear() { bind_sets.Clear(); }

private:
  Binds<T> bind_sets;
};

} // end namespace ValBind

} // end namespace Choreo

#endif // __CHOREO_VALUE_BINDS_HPP__
