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

all: build docs

build: build-with-cmake-ninja

-include $(TARGET_MK)
-include .local.mk

CC = g++
CFLAGS += -D__CHOREO_DEFAULT_TARGET__=\"$(CHOREO_DEFAULT_TARGET)\"

# For gtest
GTEST_DIR = extern/gtest
GTEST_LIBS = $(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a

# For CMAKE config
CMAKE_BUILD_DIR = $(BUILD_DIR)
CMAKE = cmake
CMAKE_BUILD_TYPE = Release
STANDALONE = OFF

PUBLIC_PACKAGE=OFF

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

ci-test:
	$(LIT) tests && $(MAKE) standalone-test-with-cmake

standalone-test-with-cmake: build-with-cmake-ninja
	cd tests/standalone/ && $(MAKE) test

clean:
	@rm -rf $(BUILD_DIR) $(DBG_BUILD_DIR) $(REL_BUILD_DIR) $(LGY_BUILD_DIR) $(TEST_TARGETS) tests/*.result
	@cd tests/standalone/ && $(MAKE) clean

build-with-cmake:
	@echo "Starting build with CMake..."
	@if [ ! -d $(CMAKE_BUILD_DIR) ]; then mkdir $(CMAKE_BUILD_DIR); fi
	$(CMAKE) -S . -B $(CMAKE_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCHOREO_DEFAULT_TARGET=$(CHOREO_DEFAULT_TARGET)
	time $(MAKE) -C $(CMAKE_BUILD_DIR)

build-with-cmake-ninja:
	@echo "Starting build with CMake..."
	@if [ ! -d $(CMAKE_BUILD_DIR) ]; then mkdir -p $(CMAKE_BUILD_DIR); fi
	$(CMAKE) -S . -B $(CMAKE_BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DPUBLIC_PACKAGE=$(PUBLIC_PACKAGE) -DSTANDALONE=$(STANDALONE) -DCHOREO_DEFAULT_TARGET=$(CHOREO_DEFAULT_TARGET)
	time ninja -C $(CMAKE_BUILD_DIR)
	ln -sf $(CMAKE_BUILD_DIR)/choreo $(WORK_DIR)/choreo
	ln -sf $(CMAKE_BUILD_DIR)/copp $(WORK_DIR)/copp

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
# Help and Documentation
# =============================================================================

help:
	@echo "Choreo Build System"
	@echo "==================="
	@echo ""
	@echo "Build Targets:"
	@echo "  all, build          - Build choreo and copp"
	@echo "  release             - Build release version"
	@echo "  debug               - Build debug version"
	@echo "  clean               - Clean build artifacts"
	@echo ""
	@echo "Test Targets:"
	@echo "  test                - Run all tests"
	@echo "  sample-test         - Test all elementwise operators"
	@echo "  sample-test-operator OPERATOR=name"
	@echo "                      - Test specific operator"
	@echo "                        Available operators: $(OPERATOR_NAMES)"

CODE_DIRS := $(SRC_DIR) $(RT_DIR) tools
TEST_DIRS := tests benchmark samples

DOC_DIR := Documents/Documentation/
DOC_FILES := $(DOC_DIR)/getting-started-with-choreo.md \
						 $(DOC_DIR)/call-in-choreo.md

.PHONY: docs

docs: $(DOC_FILES)

Documents/Documentation/%.md: Documents/Documentation/%.src.md
	@base="$(@D)"; gawk -v base="$$base" -f $(MDPP) $< > $@

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

.PHONY: all clean lines test

setup-core: $(SETUP_TARGET_DEPENDS)
	git submodule update --init --recursive;
