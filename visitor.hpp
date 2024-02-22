#ifndef __CHOREO_VISITOR_HPP__
#define __CHOREO_VISITOR_HPP__

namespace AST { class Node; }

namespace Choreo {

struct Visitor {
  virtual bool Visit(AST::Node*) = 0;
};

} // end namespace Choreo

#endif // __CHOREO_VISITOR_HPP__

