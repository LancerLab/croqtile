SHELL:=/bin/bash

WORK_DIR:=$(PWD)
TOOLCHAIN=$(WORK_DIR)/tools

FTP_SERVER:=172.16.11.50

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
HEADER_FILES :=  $(shell find . -name '*.hpp') choreo_header.inc

CC = g++
CFLAGS = -std=c++17 -Wall -Wextra -g

# For gtest
GTEST_DIR = extern/gtest
GTEST_LIBS = $(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a

# Build rules
all: $(TARGET)

test: $(TARGET) standalone_test
	$(LIT) tests

$(TARGET): scanner.yy.o parser.tab.o choreo_main.o codegen_factor.o symvalid.o typeinfer.o typecheck.o ast.o types.o valno.o
	$(CC) $(CFLAGS) $^ -o $(TARGET)

scanner.yy.cc: $(LEX_SRC)
	$(FLEX) -o $@ $(LEX_SRC)

parser.tab.cc parser.tab.hh location.hh: $(PARSER_SRC)
	$(BISON) $(BISON_FLAGS) $(PARSER_SRC)

%.o : %.cc types.hpp aux.hpp ast.hpp scanner.hpp symtab.hpp parser.tab.hh location.hh
	$(CC) $(CFLAGS) $< -c -o $@

%.o : %.cpp $(HEADER_FILES) location.hh
	$(CC) $(CFLAGS) $< -c -o $@

choreo_header.inc : utils/choreo.h
	echo "#ifndef __CHOREO_RUNTIME_HEADER_H__" > $@
	echo "#define __CHOREO_RUNTIME_HEADER_H__" >> $@
	echo "static const char* __choreo_header_as_string = R\"(" >> $@
	cat $< >> $@
	echo ")\";" >> $@
	echo "#endif // __CHOREO_RUNTIME_HEADER_H__" >> $@

clean:
	rm -f *.cc *.hh *.inc *.o $(TEST_TARGETS) tests/*.result

lines:
	wc -l *.cpp *.yy *.l *.hpp Makefile

standalone_test: $(TARGET)
	cd tests/standalone/ && $(MAKE) test

%.test: $(TEST_FILES)
	filecheck $< > $@.result
	@echo "Tested $<"

.PHONY: all clean lines test

# Toolchains
FLEX = flex
BISON_BIN = $(TOOLCHAIN)/bin/bison
LIT:=$(WORK_DIR)/tests/lit.sh
FILECHECK:=$(TOOLCHAIN)/bin/FileCheck
PACKAGE_NAME=choreo_toolchain_240511.tgz
SUPPORT_PKG = $(TOOLCHAIN)/$(PACKAGE_NAME)
PACKAGE_MD5:=1f77ae0083922fa94ed6c84c5f9cad24
BISON_ENV:=BISON_PKGDATADIR=$(TOOLCHAIN)/shared/bison/
BISON:=$(BISON_ENV) $(BISON_BIN)

support-pkg:
	@if [ "$(shell md5sum $(SUPPORT_PKG) | cut -d ' ' -f 1)" != "$(PACKAGE_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the supporting package..."; \
		rm -fr $(TOOLCHAIN)/*; mkdir -p $(TOOLCHAIN); \
		curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(PACKAGE_NAME) -o $(SUPPORT_PKG);\
		cd $(TOOLCHAIN) && tar -zvxf $(SUPPORT_PKG); \
		chmod +x $(BISON_BIN); \
	else \
		echo "$(SUPPORT_PKG) MD5 hash matches. No need to download."; \
	fi

setup: support-pkg
