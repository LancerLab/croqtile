SHELL:=/bin/bash

WORK_DIR:=$(PWD)
TOOLCHAIN_DIR=$(WORK_DIR)/extern
TOOLS_DIR=$(WORK_DIR)/tools
SCRIPT_DIR=$(WORK_DIR)/scripts
RT_DIR=$(WORK_DIR)/runtime

# Targets
SRC_DIR = $(WORK_DIR)/lib
BUILD_DIR = $(WORK_DIR)/build
DBG_BUILD_DIR = $(WORK_DIR)/build-debug
REL_BUILD_DIR = $(WORK_DIR)/build-release
LGY_BUILD_DIR = $(WORK_DIR)/build-legacy
CHOREO_BIN = $(LGY_BUILD_DIR)/choreo
COPP_BIN =  $(LGY_BUILD_DIR)/copp
TARGET = $(CHOREO_BIN) $(COPP_BIN)
LEX_SRC = $(SRC_DIR)/scanner.l
PARSER_SRC = $(SRC_DIR)/parser.yy
BISON_FLAGS = -t -d  # Generates both parser.tab.cpp and parser.tab.h
MDPP = $(SCRIPT_DIR)/mdpp.awk # Markdown Pre-process

# lit max-jobs config
JOBS ?= 1

# toolchains
FLEX = $(TOOLCHAIN_DIR)/bin/flex
BISON_BIN = $(TOOLCHAIN_DIR)/bin/bison
LIT:=$(WORK_DIR)/tests/lit.sh -j$(JOBS)
BISON?=/usr/bin/bison

# Test targets
TEST_FILES :=  $(shell find tests -name '*_test.co')
TEST_TARGETS := $(TEST_FILES:.co=.test)

# headers
HEADER_FILES :=  $(shell find $(SRC_DIR) -name '*.hpp') $(LGY_BUILD_DIR)/choreo_header.inc

CHOREO_DEFAULT_TARGET ?= cute
CLANG_FORMAT ?= /usr/bin/clang-format

CC = g++
CFLAGS += -MMD -MP -std=c++17 -Wall -Wextra -g

TARGET_DIRS := $(sort $(wildcard lib/Target/*/))
TARGET_MK   := $(addsuffix target.mk,$(TARGET_DIRS))
TOOLS_MK    := $(sort $(wildcard tools/*/target.mk))

all: build docs

build: build-with-cmake-ninja

build-all: coir

-include $(TARGET_MK)
-include $(TOOLS_MK)
-include .local.mk

CC = g++
CFLAGS += -D__CHOREO_DEFAULT_TARGET__=\"$(CHOREO_DEFAULT_TARGET)\"

# For gtest
GTEST_DIR = extern/gtest
GTEST_LIBS = $(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a

# For CMAKE config
CMAKE_BUILD_DIR = $(BUILD_DIR)
CMAKE = /usr/bin/cmake
CMAKE_BUILD_TYPE = Release
STANDALONE = OFF

PUBLIC_PACKAGE=OFF
FAST_COMPILE_DEFAULT=OFF

# Build rules
# Ensure running `make` with no target invokes the `build` target by default.
# This overrides any `.DEFAULT_GOAL` set in included makefragments.
.DEFAULT_GOAL := build

all: build


build: build-with-cmake-ninja

# Specific Release/debug build
release: CMAKE_BUILD_TYPE=Release
release: CMAKE_BUILD_DIR=$(REL_BUILD_DIR)
release: STANDALONE=OFF
release: build-with-cmake-ninja

release-full: CMAKE_BUILD_TYPE=Release
release-full: CMAKE_BUILD_DIR=$(REL_BUILD_DIR)
release-full: STANDALONE=ON
release-full: build-with-cmake-ninja

package: PUBLIC_PACKAGE=ON
package: release
	@cmake --build $(REL_BUILD_DIR) --target package-compiler

package-full: PUBLIC_PACKAGE=ON
package-full: release-full
	@cmake --build $(REL_BUILD_DIR) --target package-compiler

sdk-package: release
	@cmake --build $(REL_BUILD_DIR) --target package-sdk

SDK_INSTALL_PREFIX ?= $(WORK_DIR)/sdk-install

sdk: build
	@ninja -C $(CMAKE_BUILD_DIR) choreo-dev
	@echo "SDK library built: $(CMAKE_BUILD_DIR)/libchoreo-dev.a"

sdk-install: sdk
	@cmake --install $(CMAKE_BUILD_DIR) --prefix $(SDK_INSTALL_PREFIX) --component sdk_headers
	@cmake --install $(CMAKE_BUILD_DIR) --prefix $(SDK_INSTALL_PREFIX) --component libraries
	@cmake --install $(CMAKE_BUILD_DIR) --prefix $(SDK_INSTALL_PREFIX) --component rt_headers
	@echo "SDK installed to $(SDK_INSTALL_PREFIX)"

sdk-test: sdk-install
	@$(MAKE) -C tests/sdk test SDK_PREFIX=$(SDK_INSTALL_PREFIX)

debug: CMAKE_BUILD_TYPE=Debug
debug: CMAKE_BUILD_DIR=$(DBG_BUILD_DIR)
debug: build-with-cmake-ninja

legacy: $(TARGET)
	ln -sf $(CHOREO_BIN) $(WORK_DIR)/choreo
	ln -sf $(COPP_BIN) $(WORK_DIR)/copp

test-legacy: legacy
	$(LIT) tests && $(MAKE) standalone_test

test: build
	$(LIT) -l tests && $(MAKE) standalone-test-with-cmake

test-debug: debug
	$(LIT) -l tests && $(MAKE) standalone-test-with-cmake

test-release: release
	$(LIT) tests && $(MAKE) standalone-test-with-cmake

# Unified test targets: choreo + coir
test-all: coir
	$(LIT) -l tests && $(MAKE) standalone-test-with-cmake && $(LIT) tools/coir/tests/

test-all-debug: coir-debug
	$(LIT) -l tests && $(MAKE) standalone-test-with-cmake && $(LIT) tools/coir/tests/

test-all-release: coir-release
	$(LIT) tests && $(MAKE) standalone-test-with-cmake && $(LIT) tools/coir/tests/

ci-test:
	$(LIT) tests && $(MAKE) standalone-test-with-cmake

# ---- CoIR IR tooling ----
COIR_DEP_CONF = $(WORK_DIR)/tools/coir/llvm-dep.conf
COIR_LLVM_DIR = $(WORK_DIR)/extern/llvm-project
COIR_LLVM_SHASH := $(shell grep '^LLVM_SHASH=' $(COIR_DEP_CONF) | cut -d= -f2-)
COIR_LLVM_TAR := $(shell grep '^LLVM_TAR=' $(COIR_DEP_CONF) | cut -d= -f2-)
COIR_LLVM_URL := $(shell grep '^LLVM_URL=' $(COIR_DEP_CONF) | cut -d= -f2-)
COIR_LLVM_MD5 := $(shell grep '^LLVM_MD5=' $(COIR_DEP_CONF) | cut -d= -f2-)

COIR_BUILD_DIR ?= $(CMAKE_BUILD_DIR)

.PHONY: setup-coir-deps
setup-coir-deps:
	@echo "=== Setting up LLVM/MLIR for CoIR tooling ==="
	@mkdir -p $(COIR_LLVM_DIR)
	@if [ ! -f "$(COIR_LLVM_DIR)/lib/cmake/mlir/MLIRConfig.cmake" ]; then \
		echo "Downloading LLVM/MLIR ($(COIR_LLVM_SHASH))..."; \
		cd $(WORK_DIR)/extern && \
		wget -q --show-progress -O $(COIR_LLVM_TAR) $(COIR_LLVM_URL) && \
		echo "Extracting into extern/llvm-project/..." && \
		tar -xzf $(COIR_LLVM_TAR) --strip-components=1 -C llvm-project && \
		rm -f $(COIR_LLVM_TAR) && \
		echo "LLVM/MLIR ready at $(COIR_LLVM_DIR)"; \
	else \
		echo "LLVM/MLIR already present at $(COIR_LLVM_DIR)"; \
	fi

.PHONY: symlink-coir
symlink-coir:
	@test -f $(COIR_BUILD_DIR)/tools/coir/co2ir && ln -sf $(COIR_BUILD_DIR)/tools/coir/co2ir $(WORK_DIR)/co2ir || true
	@test -f $(COIR_BUILD_DIR)/tools/coir/coir-opt && ln -sf $(COIR_BUILD_DIR)/tools/coir/coir-opt $(WORK_DIR)/coir-opt || true
	@test -f $(COIR_BUILD_DIR)/tools/coir/cocc && ln -sf $(COIR_BUILD_DIR)/tools/coir/cocc $(WORK_DIR)/cocc || true

define build-coir
	@echo "=== Building CoIR tools ($(1)) ==="
	cmake -S $(WORK_DIR) -B $(COIR_BUILD_DIR) \
		-G Ninja \
		-DCMAKE_BUILD_TYPE=$(1) \
		-DCHOREO_BUILD_COIR=ON \
		-DCHOREO_DEFAULT_TARGET=$(CHOREO_DEFAULT_TARGET)
	ninja -C $(COIR_BUILD_DIR) co2ir coir-opt cocc
	@$(MAKE) --no-print-directory COIR_BUILD_DIR=$(COIR_BUILD_DIR) symlink-coir
endef

.PHONY: coir
coir: build
	$(call build-coir,$(CMAKE_BUILD_TYPE))

.PHONY: coir-debug
coir-debug: debug
	$(eval COIR_BUILD_DIR := $(DBG_BUILD_DIR))
	$(call build-coir,Debug)

.PHONY: coir-release
coir-release: release
	$(eval COIR_BUILD_DIR := $(REL_BUILD_DIR))
	$(call build-coir,Release)

.PHONY: coir-test
coir-test: coir
	@echo "=== Running CoIR tests ==="
	$(LIT) tools/coir/tests/

.PHONY: coir-clean
coir-clean:
	@for d in $(BUILD_DIR) $(DBG_BUILD_DIR) $(REL_BUILD_DIR); do \
		if [ -d "$$d/tools/coir" ]; then \
			rm -rf "$$d/tools/coir"; \
			echo "Removed $$d/tools/coir"; \
		fi; \
	done

# ---- co-mock (standalone mock interpreter) ----
co-mock: build-co-mock-with-cmake-ninja

build-co-mock-with-cmake-ninja:
	@echo "Building co-mock..."
	@if [ ! -d $(CMAKE_BUILD_DIR) ]; then mkdir -p $(CMAKE_BUILD_DIR); fi
	$(CMAKE) -S . -B $(CMAKE_BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCHOREO_DEFAULT_TARGET=$(CHOREO_DEFAULT_TARGET)
	ninja -C $(CMAKE_BUILD_DIR) co-mock
	@ln -sf $(CMAKE_BUILD_DIR)/co-mock $(WORK_DIR)/co-mock

mock-test: co-mock
	$(LIT) tools/co-mock/tests/

standalone-test-with-cmake: build-with-cmake-ninja
	cd tests/standalone/ && $(MAKE) test

clean:
	@rm -rf $(BUILD_DIR) $(DBG_BUILD_DIR) $(REL_BUILD_DIR) $(LGY_BUILD_DIR) $(WASM_BUILD_DIR) $(TEST_TARGETS) tests/*.result
	@cd tests/standalone/ && $(MAKE) clean

clean-all: clean coir-clean

build-with-cmake:
	@echo "Starting build with CMake..."
	@if [ ! -d $(CMAKE_BUILD_DIR) ]; then mkdir $(CMAKE_BUILD_DIR); fi
	$(CMAKE) -S . -B $(CMAKE_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCHOREO_DEFAULT_TARGET=$(CHOREO_DEFAULT_TARGET)
	time $(MAKE) -C $(CMAKE_BUILD_DIR)

build-with-cmake-ninja:
	@echo "Starting build with CMake..."
	@if [ ! -d $(CMAKE_BUILD_DIR) ]; then mkdir -p $(CMAKE_BUILD_DIR); fi
	$(CMAKE) -S . -B $(CMAKE_BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DPUBLIC_PACKAGE=$(PUBLIC_PACKAGE) -DSTANDALONE=$(STANDALONE) -DCHOREO_DEFAULT_TARGET=$(CHOREO_DEFAULT_TARGET) -DCHOREO_FAST_COMPILE_DEFAULT=$(FAST_COMPILE_DEFAULT)
	time ninja -C $(CMAKE_BUILD_DIR)
	ln -sf $(CMAKE_BUILD_DIR)/choreo $(WORK_DIR)/choreo
	ln -sf $(CMAKE_BUILD_DIR)/copp $(WORK_DIR)/copp
	@test -f $(CMAKE_BUILD_DIR)/co-mock && ln -sf $(CMAKE_BUILD_DIR)/co-mock $(WORK_DIR)/co-mock || true
	@$(MAKE) --no-print-directory COIR_BUILD_DIR=$(CMAKE_BUILD_DIR) symlink-coir

config-with-cmake-ninja:
	@echo "Starting build with CMake..."
	@if [ ! -d $(CMAKE_BUILD_DIR) ]; then mkdir -p $(CMAKE_BUILD_DIR); fi
	$(CMAKE) -S . -B $(CMAKE_BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCHOREO_DEFAULT_TARGET=$(CHOREO_DEFAULT_TARGET)


# Legacy Makefile
SRC_CPP := $(shell find $(SRC_DIR) -type f -name '*.cpp')
LGY_BUILD_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(LGY_BUILD_DIR)/%.o,$(SRC_CPP))

$(LGY_BUILD_DIR):
	mkdir -p $(LGY_BUILD_DIR)

$(CHOREO_BIN): $(TOOLS_DIR)/choreo/choreo_main.cpp $(LGY_BUILD_DIR)/parser.tab.o $(LGY_BUILD_DIR)/scanner.yy.o $(LGY_BUILD_OBJECTS) | $(LGY_BUILD_DIR)
	$(CC) $(CFLAGS) $^ -I$(WORK_DIR) -I$(SRC_DIR) -lpthread -o $@

$(LGY_BUILD_DIR)/scanner.yy.cc: $(LEX_SRC) | $(LGY_BUILD_DIR)
	$(FLEX) -o $@ $(LEX_SRC)

$(LGY_BUILD_DIR)/parser.tab.hh $(LGY_BUILD_DIR)/parser.tab.cc: $(PARSER_SRC) | $(LGY_BUILD_DIR)
	$(BISON) $(BISON_FLAGS) $(PARSER_SRC) --defines=$(LGY_BUILD_DIR)/parser.tab.hh -o $(LGY_BUILD_DIR)/parser.tab.cc

$(LGY_BUILD_DIR)/parser.tab.o: $(LGY_BUILD_DIR)/parser.tab.cc $(HEADER_FILES) parser.tab.hh | $(LGY_BUILD_DIR)
	$(CC) -I$(LGY_BUILD_DIR) -I$(SRC_DIR) $(CFLAGS) $< -c -o $@

$(LGY_BUILD_DIR)/%.o: $(LGY_BUILD_DIR)/%.cc $(HEADER_FILES) parser.tab.hh | $(LGY_BUILD_DIR)
	$(CC) -I$(LGY_BUILD_DIR) -I$(SRC_DIR) $(CFLAGS) $< -c -o $@

$(LGY_BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp | $(LGY_BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) -I$(LGY_BUILD_DIR) -I$(SRC_DIR) $(CFLAGS) $< -c  -o $@

$(COPP_BIN): $(TOOLS_DIR)/copp/choreo_preprocess.cpp $(LGY_BUILD_DIR)/parser.tab.o $(LGY_BUILD_DIR)/scanner.yy.o $(LGY_BUILD_OBJECTS)
	$(CC) $(CFLAGS) $^ -I$(LGY_BUILD_DIR) -I$(SRC_DIR) -static-libstdc++ -lpthread -o $@

-include $(OBJ:.o=.d)

$(LGY_BUILD_DIR)/choreo_header.inc : $(RT_DIR)/choreo.h
	echo "#ifndef __CHOREO_RUNTIME_HEADER_H__" > $@
	echo "#define __CHOREO_RUNTIME_HEADER_H__" >> $@
	echo -n "static const char* __choreo_header_as_string = R\"(" >> $@
	cat $< >> $@
	echo ")\";" >> $@
	echo "#endif // __CHOREO_RUNTIME_HEADER_H__" >> $@

clean-legacy:
	@rm $(LGY_BUILD_DIR)/* $(TEST_TARGETS)
	@cd $(WORK_DIR) && find tests -name '*.result' -exec rm -f {} \;
	@cd $(WORK_DIR)/tests/standalone/ && $(MAKE) clean

clobber: clean
	find $(TOOLCHAIN_DIR) -mindepth 1 ! -name 'Makefile' -print0 | xargs -0 rm -rf

# =============================================================================
# Auto-tune helper

# find all scripts in the scripts directory whose name begins with
# "auto_tune_" and run them one by one.  the order is deterministic
# (sorted lexicographically) so repeated invocations behave the same.
AUTOTUNE_SCRIPTS := $(sort $(wildcard $(SCRIPT_DIR)/auto_tune_*.sh))

.PHONY: auto_tune
auto_tune:
	@echo "running auto-tune scripts ($(AUTOTUNE_SCRIPTS))"
	@for s in $(AUTOTUNE_SCRIPTS); do \
		echo "---- $$s ----"; \
		bash "$$s" || exit 1; \
	done

# =============================================================================
# Precompiled CuTe Runtime (for --fast-compile mode)
# =============================================================================
# Build a precompiled CuTe runtime object for faster nvcc compilation.
# Usage:
#   make precompile-cute ARCH=sm_90a
#   make precompile-cute ARCH=sm_90a CUTE_HOME=/path/to/cutlass
#   make precompile-cute-clean   (remove all cached precompiled objects)
#
# The precompiled object is cached in ~/.cache/choreo/ and is used
# automatically when compiling with 'choreo --fast-compile'.

ARCH ?= sm_90a
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CHOREO_CACHE_DIR ?= $(if $(XDG_CACHE_HOME),$(XDG_CACHE_HOME),$(HOME)/.cache)/choreo

# Determine compute arch from SM arch (e.g. sm_90a -> compute_90a)
COMPUTE_ARCH = $(subst sm_,compute_,$(ARCH))

PRECOMPILE_CFLAGS = -gencode arch=$(COMPUTE_ARCH),code=$(ARCH) \
	-std=c++17 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
	-D__CHOREO_TARGET_CUTE__ -D__USE_CUDA_TYPE__ \
	--expt-relaxed-constexpr -O3

.PHONY: precompile-cute precompile-cute-clean

precompile-cute:
	@if [ -z "$(CUTE_HOME)" ]; then \
		echo "Error: CUTE_HOME is not set. Set it to the CUTLASS installation directory."; \
		echo "  Example: make precompile-cute ARCH=$(ARCH) CUTE_HOME=extern/cutlass"; \
		exit 1; \
	fi
	@if [ ! -x "$(NVCC)" ]; then \
		echo "Error: nvcc not found at $(NVCC). Set CUDA_HOME correctly."; \
		exit 1; \
	fi
	@mkdir -p $(CHOREO_CACHE_DIR)
	@if [ ! -w "$(CHOREO_CACHE_DIR)" ]; then \
		echo "Error: cannot write to $(CHOREO_CACHE_DIR)"; \
		exit 1; \
	fi
	$(eval CUDA_VER := $(shell $(NVCC) --version | grep -oP 'release \K[0-9]+\.[0-9]+'))
	$(eval FP := $(shell cat $(RT_DIR)/precompiled/choreo_precompiled.cu $(RT_DIR)/choreo.h $(RT_DIR)/choreo_cute.h | md5sum | cut -c1-8))
	$(eval PRECOMP_OUT := $(CHOREO_CACHE_DIR)/choreo_precompiled_$(ARCH)_cuda$(CUDA_VER)_$(FP).o)
	@echo "[precompile-cute] Building precompiled CuTe runtime for $(ARCH)..."
	@echo "[precompile-cute] CUDA_HOME=$(CUDA_HOME) (CUDA $(CUDA_VER))"
	@echo "[precompile-cute] CUTE_HOME=$(CUTE_HOME)"
	@echo "[precompile-cute] Fingerprint: $(FP)"
	$(NVCC) -dc $(PRECOMPILE_CFLAGS) \
		-I$(CUTE_HOME)/include -I$(RT_DIR) -I$(RT_DIR)/precompiled \
		$(RT_DIR)/precompiled/choreo_precompiled.cu \
		-o $(PRECOMP_OUT)
	@echo "[precompile-cute] Cached at $(PRECOMP_OUT)"
	@ls -lh $(PRECOMP_OUT)

precompile-cute-clean:
	@echo "Removing precompiled CuTe runtime objects..."
	@rm -fv $(CHOREO_CACHE_DIR)/choreo_precompiled_*.o
	@rm -fv $(CHOREO_CACHE_DIR)/.choreo_precompile.lock

# =============================================================================
# Help and Documentation
# =============================================================================

help:
	@echo "Choreo Build System"
	@echo "==================="
	@echo ""
	@echo "Build Targets:"
	@echo "  all, build          - Build choreo, copp, and co-mock"
	@echo "  co-mock             - Build co-mock standalone (no choreo/copp)"
	@echo "  release             - Build release version"
	@echo "  debug               - Build debug version"
	@echo "  clean               - Clean build artifacts"
	@echo ""
	@echo "Test Targets:"
	@echo "  test                - Run choreo compiler tests"
	@echo "  test-all            - Run all tests (choreo + co-mock)"
	@echo "  mock-test           - Run co-mock tests only"
	@echo "  sample-test         - Test all elementwise operators"
	@echo "  sample-test-operator OPERATOR=name"
	@echo "                      - Test specific operator"
	@echo "                        Available operators: $(OPERATOR_NAMES)"
	@echo ""
	@echo "Code Quality:"
	@echo "  format              - Format C++ code with clang-format"
	@echo "  oss-help            - Show open-source sync commands"
	@echo ""
	@echo "Fast-Compile (Precompiled CuTe Runtime):"
	@echo "  precompile-cute ARCH=sm_90a CUTE_HOME=extern/cutlass"
	@echo "                      - Build precompiled CuTe runtime for arch"
	@echo "                        Cached in \$$XDG_CACHE_HOME/choreo/ or ~/.cache/choreo/"
	@echo "                        Cache key includes arch, CUDA version, and content hash"
	@echo "  precompile-cute-clean"
	@echo "                      - Remove cached precompiled objects"

CODE_DIRS := $(SRC_DIR) $(RT_DIR) tools
TEST_DIRS := tests benchmark samples

DOC_DIR := Documents/Documentation

.PHONY: docs

docs:
	@test -f $(DOC_DIR)/index.md || { echo "ERROR: $(DOC_DIR)/index.md not found"; exit 1; }
	@echo "Docs directory OK ($(DOC_DIR))"

lines:
	@source_files="$$(find $(CODE_DIRS) -type f \( \
		-name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o \
		-name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' -o \
		-name '*.cu' -o -name '*.cuh' -o -name Makefile -o \
		-name 'CMakeList.txt' -o -name '*.mk' \
	\) -print)"; \
	if [ -z "$$source_files" ]; then \
	  echo "No source files."; \
	else \
	  echo "- library, runtime & tools: $$(wc -l $$source_files | grep total)"; \
	fi; \
	test_files="$$(find $(TEST_DIRS) -type f \( \
		-name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o \
		-name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' -o \
		-name '*.cu' -o -name '*.cuh' -o -name '*.co' \
	\) -print)"; \
	if [ -z "$$test_files" ]; then \
	  echo "No test files."; \
	else \
	  echo "- tests, samples & benchmarks: $$(wc -l $$test_files | grep total)"; \
	fi;

FORMAT_DIRS := $(SRC_DIR) $(RT_DIR) tests/standalone
format:
	files="$$(find $(FORMAT_DIRS) -type f \( \
		-name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o \
		-name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' -o \
		-name '*.cu' -o -name '*.cuh' \
	\) -print)"; \
	if [ -z "$$files" ]; then \
	  echo "No files to format."; \
	else \
	  $(CLANG_FORMAT) -i -Werror $$files; \
	fi

standalone_test: $(TARGET)
	cd tests/standalone/ && $(MAKE) test

%.test: $(TEST_FILES)
	filecheck $< > $@.result
	@echo "Tested $<"

.PHONY: all clean lines test test-all co-mock mock-test

setup-core: $(SETUP_TARGET_DEPENDS)
	git submodule update --init --recursive;
	@if [ -d scripts/hooks ]; then \
	  git config core.hooksPath scripts/hooks; \
	  echo "[setup-core] Git hooks: core.hooksPath -> scripts/hooks/"; \
	fi
