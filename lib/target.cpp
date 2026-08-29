#include "target.hpp"
#include "preprocess.hpp"

using namespace Choreo;

const std::unique_ptr<Preprocess> Target::MakePP(std::ostream& os) const {
  return std::make_unique<Preprocess>(os);
}
