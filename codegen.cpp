#include "codegen.hpp"
#include <iostream>

static inline void print_fixed_header() {
  std::cout << "#include \"../../utils/utils.h\"\n"
            << "#include \"dtu/factor/factor.h\"\n"
            << "#include \"dtu/factor/program_experimental.h\"\n"
            << "#include \"llvm/ADT/ArrayRef.h\"\n"
            << "#include \"logging_api.h\"\n"
            << "#include \"tests/factor/api/base/fixture.h\"\n";
}

static inline void print_wrapper_begin() {
  std::cout << "TEST(DoradoBasicTest, SimpleAddtest) {\n";
  std::cout << "  using namespace factor;\n";
  std::cout << "  FACTOR_PROGRAM(p);\n\n";
  std::cout << "  p([&](auto target_name) {\n";
}

static inline void print_wrapper_end() {
  std::cout << "TEST(DoradoBasicTest, SimpleAddtest) {\n";
  std::cout << "  };\n";
  std::cout << "};\n";
}

