#include "typeinfer.hpp"

#include <iostream>

#include "ast.hpp"
#include "types.hpp"

using namespace Choreo;
using namespace Choreo::AST;

bool TypeInference::Visit(AST::DataType& n) {
  assert((current_tc == TypeCategory::UNKNOWN) && "Expecting no type.");

  switch (n.getBaseType()) {
    case BaseType::ITUPLE:
      current_tc = TypeCategory::ITUPLE;
      break;
    case BaseType::INT:
      current_tc = TypeCategory::INT;
      break;
    case BaseType::BOOL:
      current_tc = TypeCategory::BOOL;
      break;
    default:
      break;
  }

  if (!n.isSpanned())
    Error(n.LOC(), "Unexpected spanned type without mdspan partial type.");

  current_tc = TypeCategory::SPANNED;

  return true;
}

bool TypeInference::Visit(AST::NamedVariableDecl& n) {
  if (current_tc == TypeCategory::UNKNOWN) {
    Error(n.LOC(), "Can not inference the type of variable declaration.");
    return false;
  }

  // n.SetTypeCategory(current_tc);
  current_tc = TypeCategory::UNKNOWN;

  return true;
}

bool TypeInference::Visit(AST::NamedTypeDecl&) { return true; }

// ituple override operator "=" for definition
bool TypeInference::Visit(AST::Assignment& n) { return true; }

bool TypeInference::Visit(AST::FunctionDecl&) {
  current_tc = TypeCategory::UNKNOWN;
  return true;
};
