#include <iostream>

#include "ast.hpp"
#include "codegen.hpp"

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

static inline std::string factor_typestr(AST::BaseType t) {
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
    default:
      choreo_unreachable();
  }
}

}  // end anonymous namespace

using namespace AST;

namespace Choreo {

bool FactorCodeGen::BeforeVisit(AST::Node &n) {
  if (isa<Program>(&n)) {
    print_fixed_header(os);
  }
  return 0;
}

bool FactorCodeGen::AfterVisit(AST::Node &n) {
  if (auto p = dyn_cast<ChoreoFunction>(&n)) {
    print_wrapper_end(os, p->name);
  } else if (isa<ParallelBy>(&n)) {
    os << "    }); // end of choreo-factor kernel function\n";
  }
  return 0;
}

bool FactorCodeGen::Visit(AST::MultiNodes &) { return true; }
bool FactorCodeGen::Visit(AST::IntLiteral &) { return true; };
bool FactorCodeGen::Visit(AST::IntList &) { return true; };
bool FactorCodeGen::Visit(AST::SValList &) { return true; };
bool FactorCodeGen::Visit(AST::Expr &) { return true; };
bool FactorCodeGen::Visit(AST::MultiDimSpans &) { return true; };
bool FactorCodeGen::Visit(AST::NamedTypeDecl &) { return true; };
bool FactorCodeGen::Visit(AST::NamedVariableDecl &) { return true; };
bool FactorCodeGen::Visit(AST::IntTuple &) { return true; };
bool FactorCodeGen::Visit(AST::Assignment &) { return true; };
bool FactorCodeGen::Visit(AST::IntIndex &) { return true; };
bool FactorCodeGen::Visit(AST::NthBound &) { return true; };
bool FactorCodeGen::Visit(AST::IntIndexList &) { return true; };
bool FactorCodeGen::Visit(AST::DataType &) { return true; };

bool FactorCodeGen::Visit(AST::Identifier &n) {
  os << n.name;
  return true;
}

bool FactorCodeGen::Visit(AST::ParamList &pl) {
  current_parameters = &pl.values;
  return true;
}

bool FactorCodeGen::Visit(AST::ParallelBy &by) {
  os << "      Dim3 grid_dim(1);\n";
  os << "      Dim3 block_dim(" << by.bound << ");\n";
  os << "      Value stream = alloc_stream_();\n";
  os << "      create_stream_(stream);\n";
  os << "      auto ts = launch_kernel_(\"" << current_fn
     << "\", grid_dim, block_dim, stream, {";
  if (current_parameters->size() > 0) {
    os << "args[0]";
    for (size_t i = 1; i < current_parameters->size(); ++i) {
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
  if (current_parameters->size() > 0) {
    os << (*current_parameters)[0]->second->name << "_type";
    for (unsigned i = 1; i < current_parameters->size(); ++i)
      os << ", " << (*current_parameters)[i]->second->name << "_type";
  }
  os << "}, {choreo_output_type}, [&](auto args, auto results) {\n";
  int i = 0;
  for (auto &param : *current_parameters) {
    os << "      "
       << "auto k_" << param->second->name << " = args[" << i++ << "];\n";
  }

  return true;
}

bool FactorCodeGen::Visit(AST::RequireBind &) { return true; };
bool FactorCodeGen::Visit(AST::WithIn &) { return true; };

bool FactorCodeGen::Visit(AST::WithBlock &n) { return true; }

bool FactorCodeGen::Visit(AST::Memory &n) {
  n.Print(os);
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

  for (auto &param : *current_parameters) {
    auto name = param->second->name;
    if (!param->first->isScalar()) {
      os << "    auto " << name << "_type = DRAMType(";
      os << factor_typestr(param->first->getBaseType());
      os << ", (1));\n";  // todo
    } else {
      os << "    auto " << name << "_type = DRAMType(";
      os << factor_typestr(param->first->getBaseType());
      os << ", (1));\n";
    }
  }

  os << "    auto choreo_output_type = DRAMType(";
  os << factor_typestr(current_output->getBaseType());
  os << ", (1));\n";  // todo

  os << "\n";
  os << "    D(main_) // choreo-factor dataflow function\n";
  os << "    ({";

  bool first_param = true;
  for (auto &param : *current_parameters) {
    auto name = param->second->name;

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

}  // end namespace Choreo
