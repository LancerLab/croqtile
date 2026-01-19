#ifndef __CHOREO_TARAGET_REGISTRY_HPP__
#define __CHOREO_TARAGET_REGISTRY_HPP__

#include "target.hpp"
#include <memory>
#include <string>
#include <vector>

namespace Choreo {

using TargetCreateFn = std::unique_ptr<Target> (*)();

class TargetRegistry {
public:
  static void Register(TargetID id, const std::string& name,
                       const std::string& desc, TargetCreateFn fn);

  static std::unique_ptr<Target> Create(const std::string&);
  static std::vector<TargetInfo> List();
}; // TargetRegistry

} // namespace Choreo

#endif //__CHOREO_TARAGET_REGISTRY_HPP__
