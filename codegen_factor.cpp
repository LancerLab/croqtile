#include <iostream>

#include "ast.hpp"
#include "codegen.hpp"
#include "types.hpp"

using namespace Choreo;

extern StringifyTable strtab;

namespace {

static inline void print_fixed_header(std::ostream &os) {
  os << "#include \"../../utils/utils.h\"\n"
     << "#include \"dtu/factor/factor.h\"\n"
     << "#include \"dtu/factor/program_experimental.h\"\n"
     << "#include \"llvm/ADT/ArrayRef.h\"\n"
     << "#include \"logging_api.h\"\n"
     << "#include \"tests/factor/api/base/fixture.h\"\n"
     << "#include \"choreo.h\"\n";
}

static inline void print_wrapper_begin(std::ostream &os, std::string name) {
  name = "choreo_" + name;
  os << "TEST(DoradoBasicTest, " << name << "_test) {\n";
  os << "  using namespace factor;\n";
  os << "  FACTOR_PROGRAM(" << name << ");\n\n";
  os << "  " << name << "([&](auto target_name) {\n";
}

static inline void print_wrapper_end(std::ostream &os, std::string name) {
  os << "  });\n\n";
  os << "  choreo_" << name << ".Compile(\"dorado\");\n";
  os << "  choreo_" << name << ".Run();\n";
  os << "};\n";
}

static inline std::string factor_typestr(Choreo::BaseType t) {
  switch (t) {
    case AST::BaseType::F32:
      return "FloatType(32)";
      break;
    case AST::BaseType::F16:
      return "FloatType(16)";
      break;
    case AST::BaseType::BF16:
      return "BFloatType(16)";
      break;
    case AST::BaseType::U32:
    case AST::BaseType::S32:
      return "IntType(32)";
      break;
    case AST::BaseType::U16:
    case AST::BaseType::S16:
      return "IntType(16)";
      break;
    case AST::BaseType::U8:
    case AST::BaseType::S8:
      return "IntType(8)";
      break;
    // should it be passed in?
    case AST::BaseType::INT:
      return "IntType(32)";
      break;
    case AST::BaseType::BOOL:
      return "BoolType(32)";
      break;
    default:
      choreo_unreachable();
  }
}

}  // end anonymous namespace

bool FactorCodeGen::BeforeVisit(AST::Node &n) {
  if (isa<AST::Program>(&n)) {
    print_fixed_header(os);
  }
  return 0;
}

bool FactorCodeGen::AfterVisit(AST::Node &n) {
  if (auto p = dyn_cast<AST::ChoreoFunction>(&n)) {
    print_wrapper_end(os, p->name);
  } else if (isa<AST::ParallelBy>(&n)) {
    os << "    }); // end of choreo-factor kernel function\n";
  }
  return 0;
}

bool FactorCodeGen::Visit(AST::MultiNodes &) { return true; }
bool FactorCodeGen::Visit(AST::IntLiteral &) { return true; };
bool FactorCodeGen::Visit(AST::SValList &) { return true; };
bool FactorCodeGen::Visit(AST::Expr &) { return true; };
bool FactorCodeGen::Visit(AST::MultiDimSpans &) { return true; };
bool FactorCodeGen::Visit(AST::NamedTypeDecl &) { return true; };

// handle stmts like:
//   f32 [a.span] g_buffer;
//   local f32[f1.span] l_buffer;
//
// ast like:
//   NamedVariableDecl
bool FactorCodeGen::Visit(AST::NamedVariableDecl &node) { 
  // TODO(albert): 'a.span' will be replace to the type-decl related to 'a'
  // auto dtype = AST::dyn_cast<AST::DataType>(node.type.get());
  // auto ptype = dtype->getPartialType();
  //
  // auto spantype = AST::dyn_cast<AST::MultiDimSpans>(ptype);
  // os << dtype->isSpanned();
  // 
  // dtype->Print(os);
  // ptype->Print(os);
  // os << spantype->ref_name;
  // TODO: hardcode 'a', wait for expr eval
  if (strtab.Exists("a")) {
    os << "      auto " << node.name_str << " = alloc_(";
    os << strtab.GetTypeSymbol("a");
    os << ");\n";
  }

  return true; 
};
bool FactorCodeGen::Visit(AST::IntTuple &) { return true; };
bool FactorCodeGen::Visit(AST::Assignment &) { return true; };
bool FactorCodeGen::Visit(AST::IntIndex &) { return true; };
bool FactorCodeGen::Visit(AST::NthBound &) { return true; };
bool FactorCodeGen::Visit(AST::IntIndexList &) { return true; };
bool FactorCodeGen::Visit(AST::DataType &) { return true; };

bool FactorCodeGen::Visit(AST::Identifier &n) {
  //os << n.name;
  return true;
}

bool FactorCodeGen::Visit(AST::Parameter &p) {
  (void)p;
  return true;
}

bool FactorCodeGen::Visit(AST::ParamList &pl) {
  cur_params = &pl.values;
  return true;
}

bool FactorCodeGen::Visit(AST::ParallelBy &by) {
  os << "      Dim3 grid_dim(1);\n";
  os << "      Dim3 block_dim(" << by.bound << ");\n";
  os << "      Value stream = alloc_stream_();\n";
  os << "      create_stream_(stream);\n";
  os << "      auto ts = launch_kernel_(\"" << current_fn
     << "\", grid_dim, block_dim, stream, {";
  if (cur_params->size() > 0) {
    os << "args[0]";
    for (size_t i = 1; i < cur_params->size(); ++i) {
      os << ", "
         << "args[" << i << "]";
    }
  }
  os << "}, {output});\n";
  os << "      destroy_stream_(stream);\n";
  os << "      return std::vector<Value>{output};\n";
  os << "    }); // end of choreo-factor dataflow program\n";
  os << "\n";
  os << "    D(func_)\n";
  os << "    (\"" << current_fn << "\", {";
  if (cur_params->size() > 0) {
    os << (*cur_params)[0]->sym->name << "_type";
    for (unsigned i = 1; i < cur_params->size(); ++i)
      os << ", " << (*cur_params)[i]->sym->name << "_type";
  }
  os << "}, {choreo_output_type}, [&](auto args, auto results) {\n";
  int i = 0;
  for (auto &param : *cur_params) {
    os << "      "
       << "auto k_" << param->sym->name << " = args[" << i++ << "];\n";
  }

  return true;
}

bool FactorCodeGen::Visit(AST::RequireBind &) { return true; };
bool FactorCodeGen::Visit(AST::WithIn &) { return true; };

bool FactorCodeGen::Visit(AST::WithBlock &n) { return true; }

bool FactorCodeGen::Visit(AST::Memory &n) {
  // n.Print(os);
  return true;
}

bool FactorCodeGen::Visit(AST::DMA &d) {
  auto future_name = d.future->name;
  os << "      auto " << future_name << "_dma = alloc_dma_(SDMAType());\n";
  return true;
}

bool FactorCodeGen::Visit(AST::ChunkAt &) { return true; };
bool FactorCodeGen::Visit(AST::Wait &) { return true; };
bool FactorCodeGen::Visit(AST::Call &) { return true; };

bool FactorCodeGen::Visit(AST::ForeachBlock &n) { return true; }

bool FactorCodeGen::Visit(AST::FunctionDecl &d) {
  current_output = d.ret_type;

  for (auto &param : *cur_params) {
    auto name = param->sym->name;
    if (!AST::typeof<IntegerType>(param.get())) {
      // define spanned type
      auto type_symbol = name+"_type";
      auto type_string = "DRAMType(" + factor_typestr(param->type->getBaseType()) + ", (1));";
      strtab.AddSymbol(name, type_symbol, type_string);
      os << "    auto " << strtab.GetTypeSymbol(name) << " = " << strtab.GetTypeString(name) << "\n";
    } else {
      auto type_symbol = name+"_type";
      auto type_string = "DRAMType(" + factor_typestr(param->type->getBaseType()) + ", (1));";
      strtab.AddSymbol(name, type_symbol, type_string);
      os << "    auto " << strtab.GetTypeSymbol(name) << " = " << strtab.GetTypeString(name) << "\n";
    }
  }

  os << "    auto choreo_output_type = DRAMType(";
  os << factor_typestr(current_output->getBaseType());
  os << ", (1));\n";  // todo

  os << "\n";
  os << "    D(main_) // choreo-factor dataflow function\n";
  os << "    ({";

  bool first_param = true;
  for (auto &param : *cur_params) {
    auto name = param->sym->name;

    if (first_param) {
      os << name + "_type";
      first_param = false;
    } else
      os << ", " << name + "_type";
  }

  os << "}, [&](auto args) {\n";

  os << "      auto choreo_output = alloc_(choreo_output_type);\n";

  return true;
}

bool FactorCodeGen::Visit(AST::ChoreoFunction &n) {
  print_wrapper_begin(os, n.name);
  current_fn = n.name;
  return true;
}

bool FactorCodeGen::Visit(AST::CppSourceCode &n) {
  os << n.GetCode();
  return true;
}

bool FactorCodeGen::Visit(AST::Program &) { return true; }
