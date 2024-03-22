#include "typeinfer.hpp"

#include <iostream>

#include "ast.hpp"
#include "types.hpp"

using namespace Choreo;
using namespace Choreo::AST;

bool TypeInference::Visit(AST::DataType& n) {
  assert((cur_type == nullptr) && "Expecting no type.");

  switch (n.getBaseType()) {
    case BaseType::ITUPLE:
      cur_type = MakeUninitITupleType();
      break;
    case BaseType::INT:
      cur_type = MakeIntegerType();
      break;
    case BaseType::BOOL:
      cur_type = MakeBooleanType();
      break;
    default:
      if (!n.isSpanned())
        Error(n.LOC(), "Unexpected spanned type without mdspan partial type.");
      cur_type = MakeSpannedType(n.getBaseType(), GenUninitMDSpanValue());
      break;
  }

  return true;
}

bool TypeInference::Visit(AST::NamedVariableDecl& n) {
  if (isa<UnknownType>(cur_type.get())) {
    Error(n.LOC(), "can not infer the type of `" + n.name_str + "'.");
    cur_type.reset();
    return false;
  }

  if (!cur_type->HasSufficientInfo()) {
    Error(n.LOC(), "can not infer '" + cur_type->Name() + "' type detail of `" + n.name_str + "'.");
    cur_type.reset();
    return false;
  }

  n.SetType(cur_type);
  cur_type.reset();

  if (Dump) {
    os << "Symbol: " << n.name_str << ", Type: ";
    n.GetType()->Print(os);
    os << "\n";
  }

  return true;
}

bool TypeInference::Visit(AST::NamedTypeDecl&) { return true; }

// ituple override operator "=" for definition
bool TypeInference::Visit(AST::Assignment& n) { return true; }

bool TypeInference::Visit(AST::FunctionDecl&) {
  cur_type.reset();
  return true;
};
