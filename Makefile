SHELL:=/bin/bash

WORK_DIR:=$(PWD)
TOOLCHAIN_DIR=$(WORK_DIR)/tools

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
CFLAGS = -std=c++17 -Wall -Wextra -g -D__CHOREO_FACTOR_DIR__="$(TOOLCHAIN_DIR)"

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
	rm -fr $(TOOLCHAIN_DIR)/*

lines:
	echo "source files:"; wc -l *.cpp *.yy *.l *.hpp Makefile utils/*.h; \
	echo "test files"; wc -l $$(find tests/ -type f |grep -v "\.test")

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
BISON_ENV:=BISON_PKGDATADIR=$(TOOLCHAIN_DIR)/shared/bison/
BISON:=$(BISON_ENV) $(BISON_BIN)

support-pkg:
	@if [ "$(shell md5sum $(SUPPORT_PKG) | cut -d ' ' -f 1)" != "$(PACKAGE_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the supporting package..."; \
		curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(PACKAGE_NAME) -o $(SUPPORT_PKG);\
		mkdir -p $(TOOLCHAIN_DIR) ;\
		cd $(TOOLCHAIN_DIR) && tar -zvxf $(SUPPORT_PKG); \
		chmod +x $(BISON_BIN); \
	else \
		echo "$(SUPPORT_PKG) MD5 hash matches. No need to download."; \
	fi;

install-support-pkg:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	cd $(TOOLCHAIN_DIR) && tar -zvxf $(SUPPORT_PKG); \
	chmod +x $(BISON_BIN)

GCU_REL_PATH=.
GCU_CMP_NAME=240524-gcu-compiler.tgz
GCU_CMP_PKG = $(TOOLCHAIN_DIR)/$(GCU_CMP_NAME)
GCU_CMP_PKG_MD5:=f6e0b029763acd1f66a192542a068ae5
CUR_GCU_CMP_PKG_MD5:=$(shell md5sum $(GCU_CMP_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-gcu-sfc-pkg:
	@if [ "$(CUR_GCU_CMP_PKG_MD5)" != "$(GCU_CMP_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		$(MAKE) download-gcu-sfc-pkg; \
	else \
		echo "$(GCU_CMP_PKG) MD5 hash matches. No need to download."; \
	fi

download-gcu-sfc-pkg:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU_REL_PATH)/$(GCU_CMP_NAME) -o $(GCU_CMP_PKG);\

install-gcu-sfc-pkg:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	cd $(TOOLCHAIN_DIR) && tar -zvxf $(GCU_CMP_PKG);

setup-gcu-sfc-pkg: check-gcu-sfc-pkg
	@if [ "$(CUR_GCU_CMP_PKG_MD5)" != "$(GCU_CMP_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu-sfc-pkg; \
	fi

GCU_PLATFORM_NAME=TopsPlatform_1.0.1.6-a1e560_deb_amd64.run
GCU_PLATFORM_PKG=$(TOOLCHAIN_DIR)/$(GCU_PLATFORM_NAME)
GCU_PLATFORM_PKG_MD5:=216051f60566227b6b95bf7502a70178
CUR_GCU_PLATFORM_PKG_MD5:=$(shell md5sum $(GCU_PLATFORM_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-gcu-platform-pkg:
	@if [ "$(CUR_GCU_PLATFORM_PKG_MD5)" != "$(GCU_PLATFORM_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		$(MAKE) download-gcu-platform-pkg; \
	else \
		echo "$(GCU_PLATFORM_PKG) MD5 hash matches. No need to download."; \
	fi

download-gcu-platform-pkg:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU_REL_PATH)/$(GCU_PLATFORM_NAME) -o $(GCU_PLATFORM_PKG);\
	chmod +x $(GCU_PLATFORM_PKG);

install-gcu-platform-pkg:
	mkdir -p $(TOOLCHAIN_DIR) ;\
	$(GCU_PLATFORM_PKG) -y -C topsruntime --install-dir $(TOOLCHAIN_DIR); \
	$(GCU_PLATFORM_PKG) -y -C topscc --install-dir $(TOOLCHAIN_DIR); \
	rsync -av $(TOOLCHAIN_DIR)/opt/tops/* $(TOOLCHAIN_DIR); \
	rm -fr $(TOOLCHAIN_DIR)/opt;

setup-gcu-platform-pkg: check-gcu-platform-pkg
	@if [ "$(CUR_GCU_PLATFORM_PKG_MD5)" != "$(GCU_PLATFORM_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu-platform-pkg; \
	fi

GCU_FACTOR_NAME=topsfactor_3.0.1-1_amd64.deb
GCU_FACTOR_PKG = $(TOOLCHAIN_DIR)/$(GCU_FACTOR_NAME)
GCU_FACTOR_PKG_MD5:=03a257e7069cc4bb42270103e3616438
CUR_GCU_FACTOR_PKG_MD5:=$(shell md5sum $(GCU_FACTOR_PKG) 2>/dev/null| cut -d ' ' -f 1)

check-gcu-factor-pkg:
	@if [ "$(CUR_GCU_FACTOR_PKG_MD5)" != "$(GCU_FACTOR_PKG_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the GCU compiler package..."; \
		$(MAKE) download-gcu-factor-pkg; \
	else \
		echo "$(GCU_FACTOR_PKG) MD5 hash matches. No need to download."; \
	fi

download-gcu-factor-pkg:
	curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/$(GCU_REL_PATH)/$(GCU_FACTOR_NAME) -o $(GCU_FACTOR_PKG);\

install-gcu-factor-pkg:
	dpkg-deb -x $(GCU_FACTOR_PKG) $(TOOLCHAIN_DIR)/; \
	rsync -av $(TOOLCHAIN_DIR)/usr/* $(TOOLCHAIN_DIR); \
	rsync -av $(TOOLCHAIN_DIR)/local/* $(TOOLCHAIN_DIR); \
	rm -fr $(TOOLCHAIN_DIR)/usr/;
	rm -fr $(TOOLCHAIN_DIR)/local/;

setup-gcu-factor-pkg: check-gcu-factor-pkg
	@if [ "$(CUR_GCU_FACTOR_PKG_MD5)" != "$(GCU_FACTOR_PKG_MD5)"  ]; then \
	    $(MAKE) install-gcu-factor-pkg; \
	fi

gcu-kmd: check-gcu-platform-pkg
	sudo $(GCU_PLATFORM_PKG) -y -C enflame

gcu-pkg: setup-gcu-sfc-pkg setup-gcu-platform-pkg setup-gcu-factor-pkg

setup: support-pkg gcu-pkg
	git submodule update --init --recursive

resetup: install-support-pkg install-gcu-sfc-pkg install-gcu-platform-pkg install-gcu-factor-pkg
