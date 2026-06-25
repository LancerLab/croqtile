#include "target.hpp"
#include "codegen.hpp"
#include "codegen_prepare.hpp"
#include "pipeline.hpp"
#include "preprocess.hpp"

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
