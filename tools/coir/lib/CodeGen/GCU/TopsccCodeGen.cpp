// CoIR GCU codegen driver -- placeholder for future GCU/topscc emission.
//
// This will implement CoIR MLIR -> topscc C++ source emission,
// analogous to codegen/gpu/coir_codegen_main.cpp for CUDA.
//
// TODO: Implement EmitGCU pass and wire it here.

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "CoIR GCU codegen -- emit topscc C++ from CoIR IR\n");
  llvm::errs() << "coir-codegen-gcu: not yet implemented\n";
  return 1;
}
