#ifndef __CHOREO_SEMANTIC_CHECK_HPP__
#define __CHOREO_SEMANTIC_CHECK_HPP__

#include "visitor.hpp"

struct SemanticChecker : public Visitor {
  void visit(AST::Node*) override {};
};

#endif // __CHOREO_SEMANTIC_CHECK_HPP__
