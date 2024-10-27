SHELL:=/bin/bash

WORK_DIR:=$(PWD)
TOOLCHAIN_DIR=$(WORK_DIR)/tools

FTP_SERVER:=172.16.11.18

# Targets
TARGET = choreo
LEX_SRC = scanner.l
PARSER_SRC = parser.yy
#BISON_FLAGS = --language=c++ --skeleton=lalr1.cc -t -d  # Generates both parser.tab.c and parser.tab.h
#BISON_FLAGS = --report=all -t -d  # Generates both parser.tab.cpp and parser.tab.h
BISON_FLAGS = -t -d  # Generates both parser.tab.cpp and parser.tab.h

# Test targets
TEST_FILES :=  $(shell find tests -name '*_test.co')
TEST_TARGETS := $(TEST_FILES:.co=.test)
#$(info TEST_FILES is $(TEST_FILES))
#$(info TEST_TARGETS is $(TEST_TARGETS))

# headers
HEADER_FILES :=  $(shell find . -name '*.hpp') choreo_header.inc choreo_cuda_header.inc factor_script.inc cuda_script.inc

CC = g++
CFLAGS = -std=c++17 -Wall -Wextra -g -D__CHOREO_FACTOR_DIR__="$(TOOLCHAIN_DIR)" -D__CHOREO_CUDA_DIR__="$(TOOLCHAIN_DIR)" -D__CHOREO_TOPSCC_DIR__="$(TOOLCHAIN_DIR)"

# fix version of clang-format
CLANG_FORMAT:=$(WORK_DIR)/extern/clang-format-19-1-2

# For gtest
GTEST_DIR = extern/gtest
GTEST_LIBS = $(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a

# For GiNaC
SYMBOLIC_DIR = $(WORK_DIR)/extern/ginac
CLN_TAR = $(SYMBOLIC_DIR)/cln-1.3.7.tar.bz2
CLN_DIR = $(SYMBOLIC_DIR)/cln-1.3.7
GINAC_TAR = $(SYMBOLIC_DIR)/ginac-1.8.7.tar.bz2
GINAC_DIR = $(SYMBOLIC_DIR)/ginac-1.8.7

SYMBOLIC_LIB_FLAGS = -L$(CLN_DIR)/install/lib -lcln -L$(GINAC_DIR)/install/lib -lginac -Wl,-rpath -Wl,$(GINAC_DIR)/install/lib
SYMBOLIC_INCLUDE_FLAGS = -I$(CLN_DIR)/install/include -I$(GINAC_DIR)/install/include

# Build rules
all: $(TARGET)

test: $(TARGET) standalone_test
	$(LIT) tests

$(TARGET): scanner.yy.o parser.tab.o choreo_main.o codegen_factor.o codegen_cuda.o codegen_topscc.o earlysema.o typeinfer.o typecheck.o ast.o types.o codegen_factor_types.o codegen_cuda_types.o valno.o visitor.o sym_replace.o
	$(CC) $(CFLAGS) $^ $(SYMBOLIC_LIB_FLAGS) -o $(TARGET)

scanner.yy.cc: $(LEX_SRC)
	$(FLEX) -o $@ $(LEX_SRC)

parser.tab.cc parser.tab.hh location.hh: $(PARSER_SRC)
	$(BISON) $(BISON_FLAGS) $(PARSER_SRC)

%.o : %.cc types.hpp aux.hpp ast.hpp codegen_factor_types.hpp codegen_cuda_types.hpp scanner.hpp symtab.hpp parser.tab.hh location.hh
	$(CC) $(CFLAGS) $< -c -o $@

%.o : %.cpp $(HEADER_FILES) location.hh
	$(CC) $(CFLAGS) $(SYMBOLIC_INCLUDE_FLAGS) $< -c  -o $@

choreo_header.inc : utils/choreo.h
	echo "#ifndef __CHOREO_RUNTIME_HEADER_H__" > $@
	echo "#define __CHOREO_RUNTIME_HEADER_H__" >> $@
	echo -n "static const char* __choreo_header_as_string = R\"(" >> $@
	cat $< >> $@
	echo ")\";" >> $@
	echo "#endif // __CHOREO_RUNTIME_HEADER_H__" >> $@

choreo_cuda_header.inc : utils/choreo_cuda.h
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

clean:
	@rm -f *.cc *.hh *.inc *.o $(TEST_TARGETS) tests/*.result

clobber: clean
	find $(TOOLCHAIN_DIR) -mindepth 1 ! -name 'Makefile' -print0 | xargs -0 rm -rf

lines:
	echo "source files:"; wc -l *.cpp *.yy *.l *.hpp Makefile utils/*.h; \
	echo "test files"; wc -l $$(find tests/ -type f |grep -v "\.test")

format:
	$(CLANG_FORMAT) -i -Werror *.cpp *.hpp utils/*.h tests/standalone/*.cu

standalone_test: $(TARGET)
	cd tests/standalone/ && $(MAKE) test

%.test: $(TEST_FILES)
	filecheck $< > $@.result
	@echo "Tested $<"

.PHONY: all clean lines test

# toolchains
FLEX = $(TOOLCHAIN_DIR)/bin/flex
BISON_BIN = $(TOOLCHAIN_DIR)/bin/bison
LIT:=$(WORK_DIR)/tests/lit.sh
FILECHECK:=$(TOOLCHAIN_DIR)/bin/FileCheck
PACKAGE_NAME=choreo_toolchain_240703.tgz
SUPPORT_PKG =$(TOOLCHAIN_DIR)/$(PACKAGE_NAME)
PACKAGE_MD5:=2b630f549063d8fbfbe31ac23984ea6d
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

setup: setup-choreo-kit ginac-setup
	git submodule update --init --recursive;\

setup-gcu2: setup
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu2-kit FTP_SERVER=$(FTP_SERVER)

setup-gcu3: setup
	git submodule update --init --recursive;\
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu3-kit FTP_SERVER=$(FTP_SERVER)

resetup-gcu2: install-choreo-kit
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu2-install FTP_SERVER=$(FTP_SERVER)

resetup-gcu3: install-choreo-kit
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu3-install FTP_SERVER=$(FTP_SERVER)

gcu2-kmd:
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu2-kmd FTP_SERVER=$(FTP_SERVER)

gcu3-kmd:
	cd $(TOOLCHAIN_DIR) && $(MAKE) gcu3-kmd FTP_SERVER=$(FTP_SERVER)

cln-setup: $(CLN_TAR)
	tar -xvf $(CLN_TAR) -C $(SYMBOLIC_DIR); \
	cd $(CLN_DIR); \
	./configure --prefix=$(CLN_DIR)/install; \
	$(MAKE) -j && $(MAKE) install

ginac-setup: $(GINAC_TAR) cln-setup
	tar -xvf $(GINAC_TAR) -C $(SYMBOLIC_DIR); \
	cd $(GINAC_DIR); \
	PKG_CONFIG_PATH=$(CLN_DIR) ./configure --prefix=$(GINAC_DIR)/install; \
	$(MAKE) -j && $(MAKE) install
