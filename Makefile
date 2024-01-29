HELL:=/bin/bash

WORK_DIR:=$(PWD)
TOOLCHAIN=$(WORK_DIR)/tools

FTP_SERVER:=172.16.11.50

# Targets
TARGET = choreo
LEX_SRC = scanner.l
BISON_SRC = parser.yy
#BISON_FLAGS = --language=c++ --skeleton=lalr1.cc -t -d  # Generates both parser.tab.c and parser.tab.h
BISON_FLAGS = -t -d  # Generates both parser.tab.cpp and parser.tab.h

CC = g++
CFLAGS = -std=c++17 -Wall -Wextra -g

# Build rules
all: $(TARGET)

$(TARGET): scanner.yy.o parser.tab.o
	$(CC) $(CFLAGS) $^ -o $(TARGET)

scanner.yy.cc: $(LEX_SRC)
	$(FLEX) -o $@ $(LEX_SRC)

parser.tab.cc parser.tab.hh: $(BISON_SRC)
	$(BISON) $(BISON_FLAGS) $(BISON_SRC)

%.o : %.cc ast.hpp scanner.hpp symtab.hpp parser.tab.hh
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	rm -f *.cc *.hh *.o $(TARGET)

.PHONY: all clean

# Toolchains
FLEX = flex
BISON_BIN = $(TOOLCHAIN)/bin/bison
BISON_PKG = $(TOOLCHAIN)/bison.tgz
BISON_MD5:=ad632d8c8dcab7033353dd1a923f6128
BISON_ENV:=BISON_PKGDATADIR=$(TOOLCHAIN)/shared/bison/
BISON:=$(BISON_ENV) $(BISON_BIN)

bison-pkg:
	@mkdir -p $(TOOLCHAIN); \
	if [ "$(shell md5sum $(BISON_PKG) | cut -d ' ' -f 1)" != "$(BISON_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the bison..."; \
		rm -f $(BISON_BIN); \
		curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/bison.tgz -o $(BISON_PKG);\
		cd $(TOOLCHAIN) && tar -zvxf $(BISON_PKG); \
		chmod +x $(BISON_BIN); \
	else \
		echo "$(BISON_PKG) MD5 hash matches. No need to download."; \
	fi

setup: bison-pkg

