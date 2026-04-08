#include "options.hpp"

namespace Choreo {
namespace SDK {

const char* Version() {
#ifdef CHOREO_SDK_VERSION
  return CHOREO_SDK_VERSION;
#else
  return "unknown";
#endif
}

} // namespace SDK
} // namespace Choreo
