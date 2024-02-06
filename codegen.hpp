#ifndef __CHOREO_CODEGEN_HPP__
#define __CHOREO_CODEGEN_HPP__

#include "visitor.hpp"

struct CodeGenerator : public Visitor {
  void visit(AST::Node*) override {};
};

#endif // __CHOREO_CODEGEN_HPP__
