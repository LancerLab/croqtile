#ifndef __CHOREO_SYMTAB_H__
#define __CHOREO_SYMTAB_H__

#include <string>
#include <unordered_map>

namespace AST {

enum class SymbolType {
    MDSPANS, MINDS,
    F32, F16, BF16, U16, S16, U8, S8, U32, S32,
    INT,
};

class Symbol {
public:
    std::string name;    // The identifier's name
    SymbolType type;     // The identifier's type
    
    // Constructor
    Symbol(const std::string& n, SymbolType t) : name(n), type(t) {}
    Symbol(){}
    // ... You might also add other attributes like scope, memory location, etc. as needed
};

class SymbolTable {
private:
    std::unordered_map<std::string, Symbol> table;

public:
    // Add a symbol to the symbol table
    void addSymbol(const std::string& name, SymbolType type) {
        table[name] = Symbol(name, type);
    }

    // Retrieve a symbol from the symbol table
    Symbol* getSymbol(const std::string& name) {
        if(table.find(name) != table.end()) {
            return &table[name];
        }
        return nullptr;
    }

    // Check if a symbol with the given name exists in the symbol table
    bool exists(const std::string& name) {
        return table.find(name) != table.end();
    }

    // ... Add more methods as needed, e.g., removeSymbol, updateSymbolType, etc.
};

}  // end of namespace AST

#endif // __CHOREO_SYMTAB_H__
