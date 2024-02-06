#ifndef __CHOREO_VISITOR_HPP__
#define __CHOREO_VISITOR_HPP__

namespace AST { class Node; }

struct Visitor {
  virtual void visit(AST::Node*) = 0;
};

#endif // __CHOREO_VISITOR_HPP__

