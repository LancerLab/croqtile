HELL:=/bin/bash

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
HEADER_FILES :=  $(shell find tests -name '*.hpp')

CC = g++
CFLAGS = -std=c++17 -Wall -Wextra -g

# Build rules
all: $(TARGET)

test: $(TARGET)
	$(LIT) tests

$(TARGET): scanner.yy.o parser.tab.o choreo_main.o codegen.o
	$(CC) $(CFLAGS) $^ -o $(TARGET)

scanner.yy.cc: $(LEX_SRC)
	$(FLEX) -o $@ $(LEX_SRC)

parser.tab.cc parser.tab.hh: $(PARSER_SRC)
	$(BISON) $(BISON_FLAGS) $(PARSER_SRC)

%.o : %.cc ast.hpp scanner.hpp symtab.hpp parser.tab.hh
	$(CC) $(CFLAGS) $< -c -o $@

%.o : %.cpp $(HEADER_FILES)
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	rm -f *.cc *.hh *.o $(TEST_TARGETS) tests/*.result

lines:
	wc -l *.cpp *.yy *.l *.hpp Makefile

%.test: $(TEST_FILES)
	filecheck $< > $@.result
	@echo "Tested $<"

.PHONY: all clean lines test

# Toolchains
FLEX = flex
BISON_BIN = $(TOOLCHAIN)/bin/bison
LIT:=$(WORK_DIR)/tests/lit.sh
FILECHECK:=$(TOOLCHAIN)/bin/FileCheck
PACKAGE_NAME=choreo_toolchain.tgz
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
