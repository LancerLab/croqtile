#include "codegen_topscc.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>

#include "ast.hpp"
// #include "choreo_topscc_header.inc"
#include "codegen.hpp"
// #include "topscc_script.inc"
#include "types.hpp"

#ifndef __CHOREO_TOPSCC_DIR__
#error "missing macro definition of __CHOREO_TOPSCC_DIR__"
#endif

using namespace Choreo;
using namespace Choreo::Topscc;

extern Option<std::string> output;

bool TopsccCodeGen::BeforeVisitImpl(AST::Node& n) {
  TraceEachVisit(n);

  return 0;
}

bool TopsccCodeGen::AfterVisitImpl(AST::Node& n) {
  TraceEachVisit(n);
  return 0;
}
