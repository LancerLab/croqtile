#ifndef __CHOREO_TARGET_HPP__
#define __CHOREO_TARGET_HPP__

#include "aux.hpp"
#include "preprocess.hpp"
#include <set>

namespace Choreo {

enum class Storage;
enum class BaseType;
enum class ParallelLevel;
class ASTPipeline;
class Preprocess;

// see the Description
enum class ChoreoFeature {
  EVENT,
  MMA,
  WGMMA,
  TMA,
  MEMALLOC,
  DGMA,
  DSDMA,
  NSVR,
  SLML,
  MGM,
  VECTORIZE,
  RSTM0,
  HDRPARSE,
};

inline static const std::string STR(ChoreoFeature cf) {
  switch (cf) {
  case ChoreoFeature::EVENT: return "event";
  case ChoreoFeature::MMA: return "mma";
  case ChoreoFeature::WGMMA: return "wgmma";
  case ChoreoFeature::TMA: return "tma";
  case ChoreoFeature::MEMALLOC: return "memalloc";
  case ChoreoFeature::DGMA: return "dgma";
  case ChoreoFeature::DSDMA: return "dsdma";
  case ChoreoFeature::NSVR: return "nsvr";
  case ChoreoFeature::SLML: return "slml";
  case ChoreoFeature::MGM: return "mgm";
  case ChoreoFeature::VECTORIZE: return "vectorize";
  case ChoreoFeature::RSTM0: return "rstm0";
  case ChoreoFeature::HDRPARSE: return "hdrparse";
  default: choreo_unreachable("unsupported feature kind.");
  }
}

inline static const std::string Description(ChoreoFeature cf) {
  switch (cf) {
  case ChoreoFeature::EVENT:
    return "Choreo Event for Asynchronized Computation.";
  case ChoreoFeature::MMA: return "Choreo MMA for Matrix Computation.";
  case ChoreoFeature::WGMMA: return "Choreo MMA for Wide Matrix Computation.";
  case ChoreoFeature::TMA:
    return "Hardware TMA for Tensor Asynchronous Data Movement across Memory "
           "Hierarchy.";
  case ChoreoFeature::MEMALLOC: return "Chroeo Memory Analysis and Allocation.";
  case ChoreoFeature::DGMA: return "Hardware with Direct Global Memory Access.";
  case ChoreoFeature::DSDMA:
    return "Single DMA/TMA with Both Slice and DeSlice.";
  case ChoreoFeature::NSVR: return "No Scalar Value Return Support.";
  case ChoreoFeature::SLML: return "Allow User to Set Local Memory Limit.";
  case ChoreoFeature::MGM: return "Choreo to Manage Global Memory.";
  case ChoreoFeature::VECTORIZE: return "Choreo Scalar Code Vectorization.";
  case ChoreoFeature::RSTM0: return "Choreo Target Restricted Mode 0.";
  case ChoreoFeature::HDRPARSE:
    return "Parse C++ Header Files Included by Choreo Source.";
  default: choreo_unreachable("unsupported feature kind.");
  }
}

using TargetID = uintptr_t;
using ArchId = std::string;
using Feature = std::string;

struct FeatureToggle {
  Feature name;
  std::string descption;
  bool operator==(const FeatureToggle& other) const {
    return name == other.name; // only compare name
  }
};

struct ArchInfo {
  ArchId id;
  std::string description;
  bool operator==(const ArchInfo& other) const {
    return id == other.id; // Only compare id
  }
};

struct TargetInfo {
  TargetID id;
  std::string name;
  std::string description;
};

class Target {
public:
  virtual ~Target() {};

  // Target hooks — keep these abstract
  virtual const std::string Name() const = 0;
  virtual const std::vector<ArchInfo> SupportedArchs() const { return {}; }
  virtual const std::unordered_map<std::string, std::string>
  ChoreoMacros(const ArchId&) const {
    return {};
  }
  virtual const ArchId DefaultArch() const {
    auto& archs = SupportedArchs();
    if (archs.empty())
      choreo_unreachable("the target '" + Name() +
                         "' supports no architecture.");
    return archs.begin()->id;
  }
  virtual const std::vector<FeatureToggle>
  SupportedFeatures(const ArchId&) const {
    return {};
  }

public:
  virtual size_t GetMemCapacity(const Storage&, const ArchId&) const = 0;
  virtual size_t GetMemAlignment(const Storage&, const ArchId&) const = 0;
  virtual size_t GetMinGroupDim(const ArchId& arch) const {
    choreo_unreachable("unsupported target '" + Name() + "(" + arch + ")'.");
  }
  virtual size_t GetVectorLength(const ArchId& arch) const {
    choreo_unreachable("unsupported target '" + Name() + "(" + arch + ")'.");
    return 0;
  }
  virtual bool IsEventSupported() const { return false; }
  virtual bool IsMMASupported() const { return false; }
  virtual bool IsArchSupported(const ArchId& arch) const {
    for (auto& ai : SupportedArchs())
      if (ai.id == arch) return true;
    return false;
  }
  virtual bool IsFeatureSupported(const ArchId& arch,
                                  const Feature& feat) const {
    for (auto& fi : SupportedFeatures(arch))
      if (fi.name == feat) return true;
    return false;
  }
  virtual bool IsFeatureSupported(const Feature& feat) const {
    if (SupportedArchs().empty()) return false;
    for (auto arch : SupportedArchs())
      if (!IsFeatureSupported(arch.id, feat)) return false;
    return true;
  }
  virtual int DefaultOptLevel(const ArchId&) const { return 0; }

  virtual size_t VectorizeLimit(const ArchId& arch) const {
    choreo_unreachable("unexpected architecture: " + arch + ".");
  }

  virtual const std::set<BaseType> VectorizableTypes(const ArchId&) const {
    return {};
  }

  virtual bool EnforceVectorAlignment(const ArchId&) const { return false; }

  virtual bool PlanCodeGenStages(ASTPipeline&) const = 0;

  virtual const std::vector<ParallelLevel>
  GetParallelLevels(const ArchId&) const = 0;

  virtual const std::unique_ptr<Preprocess> MakePP(std::ostream& os) const;

public:
  virtual int ArchNum(std::string arch) const {
    if (arch.empty())
      choreo_unreachable("unexpected architecture: " + arch + ".");

    // Find the last non-digit character from the end
    auto it = arch.rbegin();
    // skip any tailing characters
    auto rb = it;
    while (!std::isdigit(static_cast<unsigned char>(*it))) {
      ++it;
      rb = it;
    }
    while (it != arch.rend() && std::isdigit(static_cast<unsigned char>(*it)))
      ++it;

    if (it == rb) choreo_unreachable("incorrect arch: " + arch + ".");

    // Keep the trailing digits
    arch.erase(arch.begin(), it.base());
    return std::stoi(arch);
  }
}; // class Target

} // end namespace Choreo

#endif //__CHOREO_TARGET_HPP__
