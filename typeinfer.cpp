#include "typeinfer.hpp"

#include <iostream>

#include "ast.hpp"
#include "types.hpp"

using namespace Choreo;
using namespace Choreo::AST;

bool TypeInference::Visit(AST::DataType& n) {
  assert((cur_type == nullptr) && "Expecting null type.");

  cur_type = n.MakeSemaType();

  return true;
}

bool TypeInference::SetCurrentType(AST::Node& nd, const std::string& n) {
  const auto ty = nd.GetType();
  if (ty->HasSufficientInfo()) {
    // already has a type with sufficient info, check for consistence.
    if (cur_type->HasSufficientInfo() && !(*cur_type == *ty)) {
      Error(nd.LOC(), "can not infer the type of `" + n + "'.");
      return false;
    } else
      return true;
  }

  // Or else we need to set the type with current
  // Check for inference failures
  if (isa<UnknownType>(cur_type.get())) {
    Error(nd.LOC(), "can not infer the type of `" + n + "'.");
    return false;
  }

  if (!cur_type->HasSufficientInfo()) {
    Error(nd.LOC(), "can not infer '" + cur_type->Name() +
                        "' type detail of symbol `" + n + "'.");
    return false;
  }

  // The type is successfully inferred, set the node
  nd.SetType(cur_type);

  return true;
}

bool TypeInference::Visit(AST::NamedVariableDecl& n) {
  if (!SetCurrentType(n, n.name_str)) {
    cur_type.reset();
    return false;
  }

  cur_type.reset();

  if (Dump) {
    os << "Symbol: " << n.name_str << ", Type: ";
    n.PrintType(os);
    os << "\n";
  }

  return true;
}

bool TypeInference::Visit(AST::NamedTypeDecl& ntd) {
  if (Dump) {
    os << "[Partial Type] " << ntd.name_str << ": ";
    ntd.GetType()->Print(os);
    os << "\n";
  }
  return true;
}

// ituple override operator "=" for definition
bool TypeInference::Visit(AST::Assignment&) { return true; }

bool TypeInference::Visit(AST::FunctionDecl&) {
  cur_type.reset();
  return true;
};

bool TypeInference::Visit(AST::Parameter& p) {
  p.SetType(cur_type);
  // collect the parameter types
  cur_param_types.push_back(cur_type);

  if (Dump) {
    os << "[Parameter] ";
    if (p.HasSymbol()) os << "Symbol: " << p.sym->name << ",";
    os << " Type: ";
    p.GetType()->Print(os);
    os << "\n";
  }

  cur_type.reset();
  return true;
}

bool TypeInference::Visit(AST::ParamList&) { return true; }

bool TypeInference::Visit(AST::MultiDimSpans&) {
  //  assert(!cur_mdspan_value.IsValid() && "Expecting null mdspan value.");
  //  cur_mdspan_value = mds.MakeValueList();
  return true;
}

bool TypeInference::Visit(AST::Expr&) { return true; }

bool TypeInference::Visit(AST::IntTuple& n) {
  cur_type = n.GetType();
  return true;
}

bool TypeInference::Visit(AST::SValList&) { return true; }
