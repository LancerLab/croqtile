#include "target.hpp"
#include "codegen.hpp"
#include "codegen_prepare.hpp"
#include "pipeline.hpp"
#include "preprocess.hpp"
#include "types.hpp"

using namespace Choreo;

const std::unique_ptr<Preprocess> Target::MakePP(std::ostream& os) const {
  return std::make_unique<Preprocess>(os);
}

std::unique_ptr<DeviceCodeGen> Target::MakeDeviceCodeGen() const {
  return nullptr;
}

bool Target::PlanPreCodegenStages(ASTPipeline& p) const {
  p.AddStage<CodegenPrepare>();
  return true;
}

bool Target::HasDeviceCodeGen() const { return MakeDeviceCodeGen() != nullptr; }

Storage Target::GetDefaultFenceMemory(const ArchId&,
                                      ParallelLevel visibility) const {
  switch (visibility) {
  case ParallelLevel::THREAD: return Storage::LOCAL;
  case ParallelLevel::GROUP:
  case ParallelLevel::GROUPx4: return Storage::SHARED;
  case ParallelLevel::BLOCK:
  case ParallelLevel::DEVICE: return Storage::GLOBAL;
  default: return Storage::NONE;
  }
}
