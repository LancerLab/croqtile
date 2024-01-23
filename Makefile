# Targets
TARGET = choreo
LEX_SRC = scanner.l
BISON_SRC = parser.yy
#BISON_FLAGS = --language=c++ --skeleton=lalr1.cc -t -d  # Generates both parser.tab.c and parser.tab.h
BISON_FLAGS = -t -d  # Generates both parser.tab.cpp and parser.tab.h

# Tools
FLEX = flex
BISON = /home/gxf/software/bin/bison
########BISON = bison
CC = g++
CFLAGS = -std=c++2a -Wall -Wextra -g

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

