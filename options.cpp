#include "options.hpp"

using namespace Choreo;

std::unique_ptr<OptionRegistry> OptionRegistry::instance;
std::once_flag OptionRegistry::initFlag;
std::mutex OptionRegistry::regMutex;
