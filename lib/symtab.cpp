#include "symtab.hpp"

using namespace Choreo;

unsigned SymbolTable::anonymous_count = 0;
unsigned SymbolTable::anon_type_count = 0;
unsigned SymbolTable::anon_pb_count = 0;

SymbolTable symtab;
