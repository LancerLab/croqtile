#ifndef __CHOREO_CODEGEN_HPP__
#define __CHOREO_CODEGEN_HPP__

#include "visitor.hpp"

namespace Choreo {

struct CodeGenerator : public Visitor {
  bool Visit(AST::Node*) override;
};

} // end namespace Choreo

#endif // __CHOREO_CODEGEN_HPP__
