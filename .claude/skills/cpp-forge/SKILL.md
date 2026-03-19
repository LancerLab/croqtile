---
name: cpp-forge
description: |
  C++ development standards with make/cmake, g++/clang++, static analysis (clang-tidy, cppcheck),
  and best practices. Use when compiling, debugging, building, or analyzing C++ code.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
---

# C++ Forge

C++ development standards, build system configuration, and static analysis.

## When to Use This Skill

- Compiling C++ code with cmake/make
- Debugging C++ applications
- Setting up build configurations
- Running static analysis
- CI/CD pipeline setup

## Build Systems

### CMake Best Practices

```cmake
cmake_minimum_required(VERSION 3.16)
project(MyProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Sanitizers for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    add_link_options(-fsanitize=address,undefined)
endif()
```

### Makefile Patterns

```makefile
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -O2
TARGET = myprogram

.PHONY: all clean test

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f $(TARGET) *.o

test: $(TARGET)
	./$(TARGET) --test
```

## Compiler Flags

### Recommended Warning Flags
```
-Wall -Wextra -Wpedantic
-Werror        # Treat warnings as errors
-Wconversion   # Implicit conversions
-Wshadow       # Shadowing variables
-Wold-style-cast
-Wuseless-cast
```

### Optimization Flags
```
-O2           # Standard optimization
-O3           # Aggressive optimization
-Ofast        # Fastest (may relax standards)
-Os           # Size optimization
```

### Debug Flags
```
-g             # Debug symbols
-fno-omit-frame-pointer
-fsanitize=address,undefined,thread
```

## Static Analysis

### clang-tidy

```bash
# Run on specific files
clang-tidy src/*.cpp -- -std=c++17

# Run with checks enabled
clang-tidy src/main.cpp -checks='-*,readability-*,modernize-*'

# Fix suggestions automatically
clang-tidy src/main.cpp -fix

# Export compile_commands.json first
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
```

### cppcheck

```bash
# Basic check
cppcheck --std=c++17 src/

# Enable all checks
cppcheck --enable=all --std=c++17 src/

# Suppress specific warnings
cppcheck --suppress=unusedFunction src/
```

### Include What You Use (IWYU)

```bash
include-what-you-use main.cpp -- -std=c++17 -I./include
```

## Sanitizers

### AddressSanitizer (ASan)
Detects: memory leaks, use-after-free, buffer overflow
```bash
clang++ -fsanitize=address -fno-omit-frame-pointer -g main.cpp -o main
./main
```

### UndefinedBehaviorSanitizer (UBSan)
Detects: null dereference, signed integer overflow, invalid casts
```bash
clang++ -fsanitize=undefined -fno-omit-frame-pointer -g main.cpp -o main
```

### ThreadSanitizer (TSan)
Detects: data races
```bash
clang++ -fsanitize=thread -fno-omit-frame-pointer -g main.cpp -o main
```

### MemorySanitizer (MSan)
Detects: use-of-uninitialized-memory
```bash
clang++ -fsanitize=memory -fno-omit-frame-pointer -g main.cpp -o main
```

## Code Formatting

### clang-format

```yaml
# .clang-format
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
PointerAlignment: Left
```

```bash
# Format files
clang-format -i src/*.cpp include/*.h

# Check formatting
clang-format --dry-run -Werror src/*.cpp
```

## Cross-Compilation

### ARM64 (aarch64-linux-gnu)

```bash
cmake .. \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DCMAKE_SYSTEM_NAME=Linux

# Or with clang
cmake .. \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_FLAGS="--target=aarch64-linux-gnu"
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: C++ CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure
        run: cmake -B build -DCMAKE_BUILD_TYPE=Debug
      - name: Build
        run: cmake --build build
      - name: Test
        run: ctest --output-on-failure
      - name: Analyze
        run: |
          clang-tidy build/compile_commands.json || true
```

## Dependencies

### Conan (Package Manager)

```bash
conan install . -if build -s build_type=Release
```

### vcpkg

```bash
./vcpkg install boost:x64-linux
cmake -B build -DCMAKE_TOOLCHAIN_FILE=[vcpkg]/scripts/buildsystems/vcpkg.cmake
```

## Performance Profiling

### perf

```bash
perf record -g ./myprogram
perf report
perf stat ./myprogram
```

### gprof

```bash
g++ -pg -g main.cpp -o main
./main
gprof ./main gmon.out > analysis.txt
```
