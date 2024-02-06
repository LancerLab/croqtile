#ifndef __CHOREO_SEMANTIC_CHECK_HPP__
#define __CHOREO_SEMANTIC_CHECK_HPP__

namespace AST { class Node; }

struct Visitor {
  virtual void visit(AST::Node*) = 0;
};

struct SemanticChecker : public Visitor {
  void visit(AST::Node*) override {};
};

#endif // __CHOREO_SEMANTIC_CHECK_HPP__
