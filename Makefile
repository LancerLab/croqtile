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
BISON = $(TOOLCHAIN)/bison
BISON_MD5:=fbe6a2c8ec7c69ee8ee7a67e3dbc9bc8

bison-bin:
	@mkdir -p $(TOOLCHAIN); \
	if [ "$(shell md5sum $(BISON) | cut -d ' ' -f 1)" != "$(BISON_MD5)"  ]; then \
		echo "MD5 hash does not match. Downloading the bison..."; \
		rm -f $(BISON); \
		curl -u ftp_era:Enflame@321 ftp://$(FTP_SERVER)/\%2fdev/choreo-toolchain/bison -o $(BISON);\
		chmod +x $(BISON); \
	else \
		echo "$(BISON) MD5 hash matches. No need to download."; \
	fi

setup: bison-bin

