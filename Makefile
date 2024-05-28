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
HEADER_FILES :=  $(shell find . -name '*.hpp') choreo_header.inc factor_script.inc

CC = g++
CFLAGS = -std=c++17 -Wall -Wextra -g -D__CHOREO_FACTOR_DIR__="$(TOOLCHAIN)"

# For gtest
GTEST_DIR = extern/gtest
GTEST_LIBS = $(GTEST_DIR)/libgtest.a $(GTEST_DIR)/libgtest_main.a

# Build rules
all: $(TARGET)

test: $(TARGET) standalone_test
	$(LIT) tests

$(TARGET): scanner.yy.o parser.tab.o choreo_main.o codegen_factor.o earlysema.o typeinfer.o typecheck.o ast.o types.o valno.o
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

clean:
	rm -f *.cc *.hh *.inc *.o $(TEST_TARGETS) tests/*.result

clobber: clean
	rm -fr $(TOOLCHAIN)/*

lines:
	echo "source files:"; wc -l *.cpp *.yy *.l *.hpp Makefile utils/*.h; \
	echo "test files"; wc -l $$(find tests/ -type f |grep -v "\.test")

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
SUPPORT_PKG =$(TOOLCHAIN)/$(PACKAGE_NAME)
PACKAGE_MD5:=1f77ae0083922fa94ed6c84c5f9cad24
BISON_ENV:=BISON_PKGDATADIR=$(TOOLCHAIN)/shared/bison/
BISON:=$(BISON_ENV) $(BISON_BIN)

support-pkg:
	@if [ "$(shell md5sum $(SUPPORT_PKG) | cut -d ' ' -f 1)" != "$(PACKAGE_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the supporting package..."; \
		mkdir -p $(TOOLCHAIN); \
		curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(PACKAGE_NAME) -o $(SUPPORT_PKG);\
		cd $(TOOLCHAIN) && tar -zvxf $(SUPPORT_PKG); \
		chmod +x $(BISON_BIN); \
	else \
		echo "$(SUPPORT_PKG) MD5 hash matches. No need to download."; \
	fi

GCU_CMP_NAME=240524-gcu-compiler.tgz
GCU_CMP_PKG = $(TOOLCHAIN)/$(GCU_CMP_NAME)
GCU_CMP_PKG_MD5:=f6e0b029763acd1f66a192542a068ae5

gcu-sfc-pkg:
	@if [ "$(shell md5sum $(GCU_CMP_PKG) | cut -d ' ' -f 1)" != "$(GCU_CMP_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		mkdir -p $(TOOLCHAIN); \
		curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU_CMP_NAME) -o $(GCU_CMP_PKG);\
		cd $(TOOLCHAIN) && tar -zvxf $(GCU_CMP_PKG); \
	else \
		echo "$(GCU_CMP_PKG) MD5 hash matches. No need to download."; \
	fi

GCU_PLATFORM_NAME=TopsPlatform_1.0.1.6-a1e560_deb_amd64.run
GCU_PLATFORM_PKG = $(TOOLCHAIN)/$(GCU_PLATFORM_NAME)
GCU_PLATFORM_PKG_MD5:=216051f60566227b6b95bf7502a70178

gcu-platform-pkg:
	@if [ "$(shell md5sum $(GCU_PLATFORM_PKG) | cut -d ' ' -f 1)" != "$(GCU_PLATFORM_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		mkdir -p $(TOOLCHAIN); \
		curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU_PLATFORM_NAME) -o $(GCU_PLATFORM_PKG);\
		chmod +x $(GCU_PLATFORM_PKG); \
		$(GCU_PLATFORM_PKG) -y -C topsruntime --install-dir $(TOOLCHAIN); \
		$(GCU_PLATFORM_PKG) -y -C topscc --install-dir $(TOOLCHAIN); \
		rsync -av $(TOOLCHAIN)/opt/tops/* $(TOOLCHAIN); \
		rm -fr $(TOOLCHAIN)/opt; \
	else \
		echo "$(GCU_PLATFORM_PKG) MD5 hash matches. No need to download."; \
	fi

GCU_FACTOR_NAME=topsfactor_3.0.1-1_amd64.deb
GCU_FACTOR_PKG = $(TOOLCHAIN)/$(GCU_FACTOR_NAME)
GCU_FACTOR_PKG_MD5:=03a257e7069cc4bb42270103e3616438

gcu-factor-pkg:
	@if [ "$(shell md5sum $(GCU_FACTOR_PKG) | cut -d ' ' -f 1)" != "$(GCU_FACTOR_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		mkdir -p $(TOOLCHAIN); \
		curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU_FACTOR_NAME) -o $(GCU_FACTOR_PKG);\
		fakeroot sudo dpkg --instdir=$(TOOLCHAIN) -i $(GCU_FACTOR_PKG); \
		rsync -av $(TOOLCHAIN)/usr/* $(TOOLCHAIN); \
		fakeroot sudo rm -fr $(TOOLCHAIN)/usr/; \
	else \
		echo "$(GCU_FACTOR_PKG) MD5 hash matches. No need to download."; \
	fi

GCU_KMD_NAME=240524-enflame-x86_64-gcc-1.0.1.6.run
GCU_KMD_PKG = $(TOOLCHAIN)/$(GCU_KMD_NAME)
GCU_KMD_PKG_MD5:=efe16643457b9a0e3a02de3198590ec0

gcu-kmd:
	@if [ "$(shell md5sum $(GCU_KMD_PKG) | cut -d ' ' -f 1)" != "$(PACKAGE_MD5)"  ]; then \
		sudo echo "MD5 hash does not match. Downloading the GCU kmd package..."; \
		mkdir -p $(TOOLCHAIN); \
		curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU_KMD_NAME) -o $(GCU_KMD_PKG);\
		sudo bash $(GCU_KMD_PKG);\
	else \
		echo "$(GCU_KMD_PKG) MD5 hash matches. No need to download."; \
	fi

gcu-pkg: gcu-sfc-pkg gcu-platform-pkg gcu-factor-pkg

setup: support-pkg gcu-pkg
