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
  os << "using namespace factor;\n";
  os << "FACTOR_PROGRAM(" << name << ");\n\n";
  os << "" << name << "([&](auto target_name) {\n";
}

static inline void print_wrapper_end(std::ostream &os, std::string name) {
  os << "});\n\n";
  os << "choreo_" << name << ".Compile(\"dorado\");\n";
  os << "choreo_" << name << ".Run();\n";
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
  } else if (auto c = dyn_cast<AST::ChoreoFunction>(&n)) {
    print_wrapper_begin(os, c->name);
    current_fn = c->name;
    this->incrementIndent();
  } else if (isa<AST::ParallelBy>(&n)) {
    this->incrementIndent();
  } else if (isa<AST::ForeachBlock>(&n)) {
    this->incrementIndent();
  }
  CodeGenerator::BeforeVisit(n);
  return 0;
}

bool FactorCodeGen::AfterVisit(AST::Node &n) {
  CodeGenerator::AfterVisit(n);
  if (auto p = dyn_cast<AST::ChoreoFunction>(&n)) {
    print_wrapper_end(os, p->name);
  } else if (isa<AST::ParallelBy>(&n)) {
    this->decrementIndent();
    os << this->indent << "}); // end of choreo-factor kernel function\n";
  } else if (isa<AST::ForeachBlock>(&n)) {
    this->decrementIndent();
    os << this->indent << "}); // end of choreo-foreach block\n";
  }
  return 0;
}

bool FactorCodeGen::Visit(AST::MultiNodes &) { return true; }
bool FactorCodeGen::Visit(AST::MultiValues &) { return true; }
bool FactorCodeGen::Visit(AST::IntLiteral &) { return true; };
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
  // TODO(albert): refine this function with TYPE_STR new API
  // os << AST::TYPE_STR(node);
  auto dtype = dyn_cast<AST::DataType>(node.type.get());
  auto ptype = dtype->getPartialType();
  if (ptype) {
    // get full name of mdspan type
    // NOTE: full ref name may use "a.span" to ref to a var's span partial type
    // the decl of new var will need this symbol "a", not "a.span"
    // we do a hardcode workaround here
    auto full_ref_name = ptype->getRefName();
    auto ref_symbol = full_ref_name.substr(0, full_ref_name.find('.'));
    if (strtab.Exists(ref_symbol)) {
      os << this->indent;
      os << "auto " << node.name_str << " = alloc_(";
      os << strtab.GetTypeSymbol("a");
      os << ");\n";
    }
  } else {
    // TODO(albert): handle anon case
    os << this->indent;
    os << "auto " << node.name_str << " = alloc_(?";
    os << ");\n";
  }
  //
  // auto spantype = AST::dyn_cast<AST::MultiDimSpans>(ptype);
  // os << dtype->isSpanned();
  // 
  // ptype->Print(os);
  // os << spantype->ref_name;
  // TODO: hardcode 'a', wait for expr eval

  return true; 
};
bool FactorCodeGen::Visit(AST::IntTuple &) { return true; };
bool FactorCodeGen::Visit(AST::Assignment &) { return true; };
bool FactorCodeGen::Visit(AST::IntIndex &) { return true; };
bool FactorCodeGen::Visit(AST::DataType &) { return true; };

bool FactorCodeGen::Visit(AST::Identifier &n) {
  (void)n;
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
  os << this->indent << "Dim3 grid_dim(1);\n";
  os << this->indent << "Dim3 block_dim(" << by.bound << ");\n";
  os << this->indent << "Value stream = alloc_stream_();\n";
  os << this->indent << "create_stream_(stream);\n";
  os << this->indent << "auto ts = launch_kernel_(\"" << current_fn
     << "\", grid_dim, block_dim, stream, {";
  if (cur_params->size() > 0) {
    os << "args[0]";
    for (size_t i = 1; i < cur_params->size(); ++i) {
      os << ", "
         << "args[" << i << "]";
    }
  }
  os << "}, {output});\n";
  os << this->indent << "destroy_stream_(stream);\n";
  os << this->indent << "dealloc_stream_(stream);\n";
  os << this->indent << "return std::vector<Value>{output};\n";
  os << this->indent << "}); // end of choreo-factor dataflow program\n";
  os << "\n";

  this->decrementIndent();
  os << this->indent << "D(func_)\n";
  os << this->indent << "(\"" << current_fn << "\", {";
  if (cur_params->size() > 0) {
    os << (*cur_params)[0]->sym->name << "_type";
    for (unsigned i = 1; i < cur_params->size(); ++i)
      os << ", " << (*cur_params)[i]->sym->name << "_type";
  }
  os << "}, {choreo_output_type}, [&](auto args, auto results) {\n";
  this->incrementIndent();
  int i = 0;
  // NOTE: remove unused aliasing 'auto k_a = args[0];'
  // for (auto &param : *cur_params) {
  //   os << "      "
  //      << "auto k_" << param->sym->name << " = args[" << i++ << "];\n";
  // }

  return true;
}

bool FactorCodeGen::Visit(AST::RequireBind &) { return true; };
bool FactorCodeGen::Visit(AST::WithIn &) { return true; };

bool FactorCodeGen::Visit(AST::WithBlock &) { return true; }

bool FactorCodeGen::Visit(AST::Memory &n) {
  (void)n;
  return true;
}

// TODO(albert): handle indent
// TODO(albert): fix offset calculation after foreach/with stmt resolved
bool FactorCodeGen::Visit(AST::DMA &d) {
  // handle .to  in AST::Memory
  // d.to->Print(os); // shared
  auto future_name = d.future->name;
  auto to_node_name = future_name + "_buffer";
  std::string from_node_name = "";
  std::string offset_string = "0";
  if (auto mem_node = dyn_cast<AST::Memory>(d.to)) {
    // TODO(albert): generate 'local_buffer' with more smart naming way by valno support
    switch(mem_node->getStorageLevel()) {
      case Storage::LOCAL:
        os << this->indent << "auto " + to_node_name + " = alloc_(L1Type());\n";
        break;
      case Storage::SHARED:
        os << this->indent << "auto " + to_node_name + " = alloc_(SRAMType());\n";
        break;
      case Storage::GLOBAL:
        os << this->indent << "auto " + to_node_name + " = alloc_(DRAMType());\n";
        break;
      default:
        assert(false && "Unexpected storage type.");
    }
  }

  // AST::ChunkAt
  // print as a.ChunkAt(p, l1_tile)
  // d.from->Print(os) => a.ChunkAt(p, l2_tile)
  // TODO(albert): resolve hardcode
  if (auto chunkat_node = dyn_cast<AST::ChunkAt>(d.from)) {
    from_node_name = STR(chunkat_node->data);
    auto tile_factors = chunkat_node->positions;
    int dim_cursor = 0;
    std::vector<int> dim = {12, 1024, 1};
    for (const auto tile_factor : tile_factors->getValues()) {
      auto tf_symbol = STR(tile_factor);
      auto tf_bounds = dyn_cast<BoundedITupleType>(this->GetSymbolType(tf_symbol))->GetBounds().Value();
      auto tf_bound = *(std::get_if<int>(&tf_bounds[0]));
      offset_string = offset_string + " + " + std::to_string(dim[dim_cursor] / tf_bound * dim[dim_cursor+1]) + " * " + STR(tile_factor);
      // os << offset_string;
      ++dim_cursor;
    }
  }
  os << this->indent << "auto " << future_name << " = alloc_dma_(SDMAType());\n";
  os << this->indent << "async_load_(" << future_name 
     << ", " << from_node_name << ", " 
     << to_node_name << ", " << offset_string <<  ");\n";
  os << this->indent << "wait_dma_(" << future_name << ");\n";

  return true;
}

bool FactorCodeGen::Visit(AST::ChunkAt &) { return true; };
bool FactorCodeGen::Visit(AST::Wait &) { return true; };
bool FactorCodeGen::Visit(AST::Call &) { return true; };
bool FactorCodeGen::Visit(AST::Return &) { return true; };

bool FactorCodeGen::Visit(AST::ForeachBlock &forNode) { 
  // auto ty = this->GetSymbolType("l2_tile");
  // ty->Print(os);
  // auto l2_tile_idx = itervars->getValueAt(0);
  // auto l1_tile_idx = itervars->getValueAt(1);
  //
  auto itervars = forNode.getIterationVars();
  for (auto idx = 0; idx != itervars->Count(); ++idx) {
    // TODO(albert): support non-unit stride in loop
    std::ostringstream _os;
    itervars->getValueAt(idx)->Print(_os);
    auto iv_str = _os.str();

    // get the lower/upper and stride for spanned iter var
    auto iv_type = this->GetSymbolType(iv_str);
    auto iv_bounds = dyn_cast<BoundedITupleType>(iv_type)->GetBounds();
    auto iv_values = iv_bounds.Value();
    // for (const auto &value : iv_values) {
    //   if (auto intValue = std::get_if<int>(&value)) os << *intValue;
    // }
    // NOTES: bounds always has one integer indicating the upperbound value
    // we can certainly use idx=0 directly
    auto upper_bound = *(std::get_if<int>(&iv_values[0]));

    // synthesise the emitting string
    os << this->indent << "for_(0, " << std::to_string(upper_bound) 
       << ", " << 1 /* TODO(albert): need fix, unit stride is hardcoded for now*/ 
       << ", " << "[&](auto " 
       << iv_str << ") {\n";
  }
  return true; 
}

bool FactorCodeGen::Visit(AST::FunctionDecl &d) {
  current_output = d.ret_type;

  for (auto &param : *cur_params) {
    auto name = param->sym->name;
    if (AST::typeof<SpannedType>(param.get())) {
      // define spanned type
      auto type_symbol = name+"_type";
      std::ostringstream _os;
      // param->type->Print(os, "");
      if(param->type->getPartialType()) {
        _os << param->type->getPartialType()->EmitTo("", Target::Factor);
      } else {
        // TODO: this guard code may not needed
        _os << "{?}";
      }
      auto type_string = "DRAMType(" + 
                         factor_typestr(param->type->getBaseType()) + 
                         ", " + _os.str(); 

      strtab.AddSymbol(name, type_symbol, type_string);
      os << this->indent << "auto " << strtab.GetTypeSymbol(name) << " = " << strtab.GetTypeString(name);
      os << ");\n";
    } else {
      auto type_symbol = name+"_type";
      auto type_string = "DRAMType(" + factor_typestr(param->type->getBaseType()) + ", (1));";
      strtab.AddSymbol(name, type_symbol, type_string);
      os << this->indent << "auto " << strtab.GetTypeSymbol(name) << " = " << strtab.GetTypeString(name) << "\n";
    }
  }

  if (AST::typeof<SpannedType>(current_output.get())) {
    auto type_symbol = "choreo_output_type";
    std::ostringstream _os;
    // param->type->Print(os, "");
    if(current_output->getPartialType()) {
      _os << current_output->getPartialType()->EmitTo("", Target::Factor);
    } else {
      // TODO: this guard code may not needed
      _os << "{?}";
    }
    auto type_string = "DRAMType(" + 
                       factor_typestr(current_output->getBaseType()) + 
                       ", " + _os.str(); 

    strtab.AddSymbol(type_symbol, type_symbol, type_string);
    os << this->indent << "auto " << type_symbol << " = " << strtab.GetTypeString(type_symbol);
    os << ");\n";
  } else {
    auto type_symbol = "choreo_output_type";
    auto type_string = "DRAMType(" + factor_typestr(current_output->getBaseType()) + ", (1));";
    strtab.AddSymbol(type_symbol, type_symbol, type_string);
    os << this->indent << "auto " << strtab.GetTypeSymbol(type_symbol) << " = " << strtab.GetTypeString(type_symbol) << "\n";
  }

  // os << "    auto choreo_output_type = DRAMType(";
  // os << factor_typestr(current_output->getBaseType());
  // os << ", (1));\n";  // todo

  os << "\n";
  os << this->indent << "D(main_) // choreo-factor dataflow function\n";
  os << this->indent << "({";

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

  // os << "      auto choreo_output = alloc_(choreo_output_type);\n";

  return true;
}

bool FactorCodeGen::Visit(AST::ChoreoFunction &) {
  return true;
}

bool FactorCodeGen::Visit(AST::CppSourceCode &n) {
  os << n.GetCode();
  return true;
}

bool FactorCodeGen::Visit(AST::Program &) { return true; }
