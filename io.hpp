#ifndef __CHOREO_IO_HPP__
#define __CHOREO_IO_HPP__

#include "options.hpp"

namespace Choreo {

inline std::ostream& outs() {
  return OptionRegistry::GetInstance().GetOutputStream();
}
inline std::ostream& dbgs() { return std::cout; }
inline std::ostream& errs() { return std::cerr; }

} // end namespace Choreo

#endif //__CHOREO_IO_HPP__
