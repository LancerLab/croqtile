SHELL:=/bin/bash

WORK_DIR:=$(PWD)
TOOLCHAIN_DIR=$(WORK_DIR)/extern
TOOLS_DIR=$(WORK_DIR)/tools
RT_DIR=$(WORK_DIR)/runtime

FTP_SERVER:=172.16.11.18

# Targets
CHOREO_BIN = build/choreo
COPP_BIN = build/copp
TARGET = $(CHOREO_BIN) $(COPP_BIN)
SRC_DIR = $(WORK_DIR)/lib
BUILD_DIR = $(WORK_DIR)/build
DBG_BUILD_DIR = $(WORK_DIR)/build-debug
REL_BUILD_DIR = $(WORK_DIR)/build-release
LEX_SRC = $(SRC_DIR)/scanner.l
PARSER_SRC = $(SRC_DIR)/parser.yy
#BISON_FLAGS = --language=c++ --skeleton=lalr1.cc -t -d  # Generates both parser.tab.c and parser.tab.h
#BISON_FLAGS = --report=all -t -d  # Generates both parser.tab.cpp and parser.tab.h
BISON_FLAGS = -t -d  # Generates both parser.tab.cpp and parser.tab.h

# Test targets
TEST_FILES :=  $(shell find tests -name '*_test.co')
TEST_TARGETS := $(TEST_FILES:.co=.test)
#$(info TEST_FILES is $(TEST_FILES))
#$(info TEST_TARGETS is $(TEST_TARGETS))

# headers
HEADER_FILES :=  $(shell find $(SRC_DIR) -name '*.hpp') choreo_header.inc choreo_cuda_header.inc factor_script.inc cuda_script.inc

CC = g++
CFLAGS += -std=c++17 -Wall -Wextra -g -D__CHOREO_FACTOR_DIR__="$(TOOLCHAIN_DIR)" -D__CHOREO_CUDA_DIR__="$(TOOLCHAIN_DIR)" -D__CHOREO_TOPSCC_DIR__="$(TOOLCHAIN_DIR)"

# fix version of clang-format
CLANG_FORMAT:=$(WORK_DIR)/extern/clang-format-19-1-2

# For gtest
GTEST_DIR = extern/gtest
GTEST_LIBS = $(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a

# For GiNaC
SYMBOLIC_DIR = $(WORK_DIR)/extern/ginac
CLN_DIR = $(SYMBOLIC_DIR)/cln-1.3.7
GINAC_DIR = $(SYMBOLIC_DIR)/ginac-1.8.7

SYMBOLIC_LIB_FLAGS = -L$(CLN_DIR)/install/lib -L$(GINAC_DIR)/install/lib -lginac -Wl,-rpath -Wl,$(GINAC_DIR)/install/lib -lcln -lgmp
SYMBOLIC_INCLUDE_FLAGS = -I$(CLN_DIR)/install/include -I$(GINAC_DIR)/install/include

# For CMAKE config
CMAKE_BUILD_DIR = $(BUILD_DIR)
CMAKE = cmake
CMAKE_BUILD_TYPE = Release
STANDALONE = OFF

PUBLIC_PACKAGE=OFF

# Build rules
all: build

# lit max-jobs config
JOBS ?= 1

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

test: build-with-cmake-ninja
	$(LIT) -l tests && $(MAKE) standalone-test-with-cmake

test-debug: debug
	$(LIT) -l tests && $(MAKE) standalone_test

test-release: release
	$(LIT) tests && $(MAKE) standalone_test

test-libra: release
	$(LIT) tests/libra && $(MAKE) standalone_test

ci-test:
	$(LIT) tests && $(MAKE) standalone-test-with-cmake

standalone-test-with-cmake: build-with-cmake-ninja
	cd tests/standalone/ && $(MAKE) test

clean:
	@rm -rf $(BUILD_DIR) $(DBG_BUILD_DIR) $(REL_BUILD_DIR) $(TEST_TARGETS) tests/*.result
	@cd tests/standalone/ && $(MAKE) clean

build-with-cmake:
	@echo "Starting build with CMake..."
	@if [ ! -d $(CMAKE_BUILD_DIR) ]; then mkdir $(CMAKE_BUILD_DIR); fi
	$(CMAKE) -S . -B $(CMAKE_BUILD_DIR) -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	time $(MAKE) -C $(CMAKE_BUILD_DIR)

build-with-cmake-ninja:
	@echo "Starting build with CMake..."
	@if [ ! -d $(CMAKE_BUILD_DIR) ]; then mkdir -p $(CMAKE_BUILD_DIR); fi
	$(CMAKE) -S . -B $(CMAKE_BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DPUBLIC_PACKAGE=$(PUBLIC_PACKAGE) -DSTANDALONE=$(STANDALONE)
	time ninja -C $(CMAKE_BUILD_DIR)
	ln -sf $(CMAKE_BUILD_DIR)/choreo $(WORK_DIR)/choreo
	ln -sf $(CMAKE_BUILD_DIR)/copp $(WORK_DIR)/copp

config-with-cmake-ninja:
	@echo "Starting build with CMake..."
	@if [ ! -d $(CMAKE_BUILD_DIR) ]; then mkdir -p $(CMAKE_BUILD_DIR); fi
	$(CMAKE) -S . -B $(CMAKE_BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)


# Legacy Makefile
BUILD_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(wildcard $(SRC_DIR)/*.cpp))

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(CHOREO_BIN): $(TOOLS_DIR)/choreo/choreo_main.cpp $(BUILD_DIR)/parser.tab.o $(BUILD_DIR)/scanner.yy.o $(BUILD_OBJECTS)
	$(CC) $(CFLAGS) $^ -I$(WORK_DIR) -I$(SRC_DIR) $(SYMBOLIC_INCLUDE_FLAGS) $(SYMBOLIC_LIB_FLAGS) -o $@

scanner.yy.cc: $(LEX_SRC)
	$(FLEX) -o $@ $(LEX_SRC)

parser.tab.cc parser.tab.hh: $(PARSER_SRC)
	$(BISON) $(BISON_FLAGS) $(PARSER_SRC)

$(BUILD_DIR)/%.o : %.cc $(HEADER_FILES) parser.tab.hh | $(BUILD_DIR)
	$(CC) -I$(WORK_DIR) -I$(SRC_DIR) $(CFLAGS) $< -c -o $@

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp $(HEADER_FILES) | $(BUILD_DIR)
	$(CC) -I$(WORK_DIR) -I$(SRC_DIR) $(CFLAGS) $(SYMBOLIC_INCLUDE_FLAGS) $< -c  -o $@

$(COPP_BIN): $(TOOLS_DIR)/copp/choreo_preprocess.cpp $(BUILD_DIR)/parser.tab.o $(BUILD_DIR)/scanner.yy.o $(BUILD_OBJECTS)
	$(CC) $(CFLAGS) $^ -I$(WORK_DIR) -I$(SRC_DIR) $(SYMBOLIC_INCLUDE_FLAGS) $(SYMBOLIC_LIB_FLAGS) -o $@

choreo_header.inc : $(RT_DIR)/choreo.h
	echo "#ifndef __CHOREO_RUNTIME_HEADER_H__" > $@
	echo "#define __CHOREO_RUNTIME_HEADER_H__" >> $@
	echo -n "static const char* __choreo_header_as_string = R\"(" >> $@
	cat $< >> $@
	echo ")\";" >> $@
	echo "#endif // __CHOREO_RUNTIME_HEADER_H__" >> $@

choreo_cuda_header.inc : $(RT_DIR)/choreo_cuda.h
	echo "#ifndef __CHOREO_RUNTIME_HEADER_H__" > $@
	echo "#define __CHOREO_RUNTIME_HEADER_H__" >> $@
	echo -n "static const char* __choreo_header_as_string = R\"(" >> $@
	cat $< >> $@
	echo ")\";" >> $@
	echo "#endif // __CHOREO_RUNTIME_HEADER_H__" >> $@

factor_script.inc : scripts/factor_script.sh
	echo "#ifndef __CHOREO_FACTOR_SCRIPT_H__" > $@
	echo "#define __CHOREO_FACTOR_SCRIPT_H__" >> $@
	echo -n "static const char* __factor_script_as_string = R\"__co_factor__(" >> $@
	cat $< >> $@
	echo ")__co_factor__\";" >> $@
	echo "#endif // __CHOREO_FACTOR_SCRIPT_H__" >> $@

cuda_script.inc : scripts/cuda_script.sh
	echo "#ifndef __CHOREO_CUDA_SCRIPT_H__" > $@
	echo "#define __CHOREO_CUDA_SCRIPT_H__" >> $@
	echo -n "static const char* __cuda_script_as_string = R\"__co_cuda__(" >> $@
	cat $< >> $@
	echo ")__co_cuda__\";" >> $@
	echo "#endif // __CHOREO_CUDA_SCRIPT_H__" >> $@

clean-legacy:
	@rm -f *.cc *.hh *.inc *.o $(TEST_TARGETS) tests/*.result
	@cd tests/standalone/ && $(MAKE) clean

clobber: clean
	find $(TOOLCHAIN_DIR) -mindepth 1 ! -name 'Makefile' -print0 | xargs -0 rm -rf

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

# =============================================================================
# Sample Tests for topscc/elementwise
# =============================================================================

ELEMENTWISE_DIR = samples/topscc/elementwise
OPERATOR_NAMES = $(notdir $(basename $(wildcard $(ELEMENTWISE_DIR)/*.co)))
CHOREO_FLAGS = -gs -t topscc

sample-test: $(OPERATOR_NAMES:%=sample-test-%)

sample-test-%: $(ELEMENTWISE_DIR)/%.co
	@TMPDIR=$$(mktemp -d) && \
	echo -n "Testing $*... " && \
	if choreo $(CHOREO_FLAGS) $< -o $$TMPDIR/test.result > /dev/null 2>&1 && \
	   bash $$TMPDIR/test.result --execute > /dev/null 2>&1; then \
		echo "PASSED"; \
		ret=0; \
	else \
		echo "FAILED"; \
		ret=1; \
	fi; \
	rm -rf $$TMPDIR; \
	exit $$ret

sample-test-operator:
	@if [ -z "$(OPERATOR)" ]; then \
		echo "Usage: make sample-test-operator OPERATOR=operator_name"; \
		echo "Available operators: $(OPERATOR_NAMES)"; \
		exit 1; \
	fi
	@TMPDIR=$$(mktemp -d) && \
	echo -n "Testing $(OPERATOR)... " && \
	if choreo $(CHOREO_FLAGS) $(ELEMENTWISE_DIR)/$(OPERATOR).co -o $$TMPDIR/test.result > /dev/null 2>&1 && \
	   bash $$TMPDIR/test.result --execute > /dev/null 2>&1; then \
		echo "PASSED"; \
		ret=0; \
	else \
		echo "FAILED"; \
		ret=1; \
	fi; \
	rm -rf $$TMPDIR; \
	exit $$ret

lines:
	@echo "source code:"; wc -l lib/*.cpp lib/*.yy lib/*.l lib/*.hpp Makefile $(RT_DIR)/*.h | grep total;
	@echo "test code"; wc -l $$(find tests/ -type f |grep -v "\.test"|grep -v "\.result") | grep total;

format:
	$(CLANG_FORMAT) -i -Werror $(SRC_DIR)/*.cpp $(SRC_DIR)/*.hpp $(RT_DIR)/*.h $(RT_DIR)/*.cpp tests/standalone/*.cu tests/standalone/*.cpp

standalone_test: $(TARGET)
	cd tests/standalone/ && $(MAKE) test

%.test: $(TEST_FILES)
	filecheck $< > $@.result
	@echo "Tested $<"

.PHONY: all clean lines test

# toolchains
FLEX = $(TOOLCHAIN_DIR)/bin/flex
BISON_BIN = $(TOOLCHAIN_DIR)/bin/bison
LIT:=$(WORK_DIR)/tests/lit.sh -j$(JOBS)
FILECHECK:=$(TOOLCHAIN_DIR)/bin/FileCheck
PACKAGE_NAME=choreo_toolchain_250930.tgz
SUPPORT_PKG =$(TOOLCHAIN_DIR)/$(PACKAGE_NAME)
PACKAGE_MD5:=a4297fca634dcdda3d08c467550d4b22
CUR_PKG_MD5:=$(shell md5sum $(SUPPORT_PKG) 2>/dev/null| cut -d ' ' -f 1)
BISON_ENV:=BISON_PKGDATADIR=$(TOOLCHAIN_DIR)/shared/bison/
BISON:=$(BISON_ENV) $(BISON_BIN)

check-choreo-kit:
	@if [ "$(CUR_PKG_MD5)" != "$(PACKAGE_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the supporting package..."; \
		$(MAKE) download-choreo-kit; \
	else \
		echo "$(SUPPORT_PKG) MD5 hash matches. No need to download."; \
	fi;

download-choreo-kit:
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(PACKAGE_NAME) -o $(SUPPORT_PKG);\

install-choreo-kit: check-choreo-kit
	cd $(TOOLCHAIN_DIR) && tar -zvxf $(SUPPORT_PKG); \
	chmod +x $(BISON_BIN)

setup-choreo-kit: check-choreo-kit
	@if [ "$(CUR_PKG_MD5)" != "$(PACKAGE_MD5)"  ]; then \
	  $(MAKE) install-choreo-kit; \
	fi;

setup-git-hooks:
	@mkdir .git/hooks; \
	cp ./scripts/hooks/pre-commit-check.sh .git/hooks/pre-commit; \
	chmod +x .git/hooks/pre-commit

setup-core: setup-choreo-kit setup-ginac setup-clang-format setup-git-hooks setup-gcu-acore
	git submodule update --init --recursive;\
	ln -sf extern/bin/not.sh tests

setup-cuda:
	cd $(TOOLCHAIN_DIR) && $(MAKE) setup-cuda FTP_SERVER=$(FTP_SERVER)

setup-gcu2: setup-core
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu2-kit FTP_SERVER=$(FTP_SERVER)

setup-gcu3: setup-core
	git submodule update --init --recursive;\
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu3-kit FTP_SERVER=$(FTP_SERVER)

setup-gcu4: setup-gcu3
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu-sim

resetup-gcu2: install-choreo-kit
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu2-install FTP_SERVER=$(FTP_SERVER)

resetup-gcu3: install-choreo-kit
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu3-install FTP_SERVER=$(FTP_SERVER)

gcu2-kmd:
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu2-kmd FTP_SERVER=$(FTP_SERVER)

gcu3-kmd:
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu3-kmd FTP_SERVER=$(FTP_SERVER)

setup-gcu-acore:
	cd $(TOOLCHAIN_DIR) && $(MAKE) setup-acore

GINAC_MD5=9d0eaa439c7b825311e99a9aa9b15f9a
GINAC_PACKAGE_NAME=ginac-cln-251014.tgz
GINAC_PACKAGE=$(TOOLCHAIN_DIR)/$(GINAC_PACKAGE_NAME)
CUR_GINAC_MD5:=$(shell md5sum $(GINAC_PACKAGE) 2>/dev/null| cut -d ' ' -f 1)

download-ginac:
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GINAC_PACKAGE_NAME) -o $(GINAC_PACKAGE);\

check-ginac:
	@if [ "$(CUR_GINAC_MD5)" != "$(GINAC_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the ginac package..."; \
		$(MAKE) download-ginac; \
	else \
		echo "$(SUPPORT_PKG) MD5 hash matches. No need to download."; \
	fi;

setup-ginac: check-ginac
	@if [ ! -f "$(TOOLCHAIN_DIR)/ginac/ginac-1.8.7/install/lib/libginac.a" ]; then \
	    tar -zxvf $(GINAC_PACKAGE) -C $(TOOLCHAIN_DIR);\
	fi;

CFORMAT_MD5=6ee59eba63782b362bc9ba1138911f3a
CFORMAT_NAME=clang-format-19-1-2
CUR_CFORMAT_MD5:=$(shell md5sum $(CLANG_FORMAT) 2>/dev/null| cut -d ' ' -f 1)

download-clang-format:
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(CFORMAT_NAME) -o $(CLANG_FORMAT);\

check-clang-format:
	@if [ "$(CUR_CFORMAT_MD5)" != "$(CFORMAT_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the clang-format..."; \
		$(MAKE) download-clang-format; \
	else \
		echo "$(SUPPORT_PKG) MD5 hash matches. No need to download."; \
	fi;

setup-clang-format: check-clang-format
	chmod +x $(CLANG_FORMAT)

# utils to serve Choreo Documents
MKDOCS_CMD = mkdocs serve --dev-addr=0.0.0.0:8000

serve-doc: stop-doc start-doc

stop-doc:
	@echo "Stopping existing mkdocs serve processes..."
	@ps aux | grep 'mkdocs serve' | grep -v grep | awk '{print $$2}' | xargs -r kill
	@echo "Old mkdocs serve processes stopped."

start-doc:
	@echo "Starting mkdocs serve in the background..."
	nohup $(MKDOCS_CMD) &>/dev/null &

status-doc:
	@echo "Checking mkdocs serve process..."
	@ps aux | grep 'mkdocs serve' | grep -v grep || echo "No mkdocs serve process is running."

# utils to publish packages to releases or package registry
publish-package: package
	@bash scripts/publish-package.sh

publish-release: package
	@bash scripts/publish-release.sh

publish-to-apex: package
	@bash scripts/publish-choreo-for-apex.sh

publish-to-topsop: package
	@bash scripts/publish-choreo-for-topsop.sh

publish-sdk: sdk-package
	pkg_name=$$(find $(REL_BUILD_DIR)/package/_CPack_Packages/Linux/DEB/ -name 'choreo-dev*.deb'); \
	sdk_name=$$(basename $$pkg_name); \
	md5sum $$pkg_name; \
	curl -T $$pkg_name ftp://$(FTP_SERVER)/\%2fdev/choreo-sdk/$$sdk_name --user ftp_era:Enflame@321

prepare: setup-ginac
run-samples: $(OPERATOR_NAMES:%=test-%)

