#include "codegen.hpp"

using namespace Choreo;

std::once_flag CodeGenInfo::init_flag;
std::unique_ptr<CodeGenInfo> CodeGenInfo::instance;
int TMADesc::index = 0;

Choreo::CodeGenInfo& CodeGenInfo::Get() {
  std::call_once(init_flag,
                 []() { instance = std::make_unique<CodeGenInfo>(); });
  return *instance;
}
