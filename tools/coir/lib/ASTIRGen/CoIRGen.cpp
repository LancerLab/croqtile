//===- CoIRGen.cpp - co2ir: Choreo AST -> CoIR MLIR -----------------------===//
//
// Translates Choreo .co source into CoIR MLIR.
//
//===----------------------------------------------------------------------===//

#include "ast.hpp"
#include "ASTCoIRGen.hpp"
#include "choreo_api.hpp"
#include "command_line.hpp"
#include "context.hpp"
#include "io.hpp"
#include "options.hpp"

#include "Dialect/CoIR/CoIRDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>

using namespace Choreo;

extern Option<bool> dump_ast;

int main(int argc, char *argv[]) {
  CommandLine cl;
  if (!cl.Parse(argc, argv)) return cl.ReturnCode();

  auto &reg = OptionRegistry::GetInstance();
  std::string input_file = reg.GetInputFileName();

  CompilerAPI api;
  int fe_status = api.RunFrontend(input_file);
  if (fe_status != 0) return fe_status;

  if (dump_ast) {
    CompilerAPI::GetAST().Print(dbgs());
    return 0;
  }

  CoIR::IRSession::Get().ResetModule(input_file);
  CoIR::ASTCoIRGen translator;
  if (!translator.RunOnProgram(CompilerAPI::GetAST())) {
    errs() << "co2ir: IR generation failed\n";
    return 1;
  }

  return 0;
}
