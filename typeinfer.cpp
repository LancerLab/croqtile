#include <iostream>

#include "ast.hpp"
#include "typeinfer.hpp"

using namespace AST;
using namespace Choreo;

bool TypeInference::Visit(AST::NamedVariableDecl&) {
  return true;
}
