#ifndef __CHOREO_SEMANTIC_CHECK_HPP__
#define __CHOREO_SEMANTIC_CHECK_HPP__

#include "visitor.hpp"

namespace Choreo {

struct SemanticChecker : public Visitor {
  bool Visit(AST::Node*) override { return true; };
};

} // end namespace Choreo
  //
#endif // __CHOREO_SEMANTIC_CHECK_HPP__
