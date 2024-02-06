#ifndef __CHOREO_AST_HPP__
#define __CHOREO_AST_HPP__

#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "symtab.hpp"

namespace AST {

template <typename T>
using ptr = std::shared_ptr<T>;

// Base class for all AST nodes
struct Node {
  virtual ~Node() {}
  virtual void Print(std::ostream& os, const std::string& prefix = {}) const {
    (void)os;
    (void)prefix;
  }
};

// Single node with reference to another
struct NodeRef {
  ptr<Node> value;
  NodeRef(ptr<Node> n) : value(n) {}
  virtual void Print(std::ostream& os, const std::string& prefix = {}) const {
    value->Print(os, prefix);
  }
};

// General cluster of nodes
struct MultiNodes : public Node {
  std::vector<ptr<NodeRef>> values;
  explicit MultiNodes(){};
  void Append(ptr<NodeRef>& m) { values.push_back(m); }
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (auto& v : values) v->Print(os, prefix);
  }
};

// For storage specifiers like local, global, shared
enum class StorageSpec { LOCAL, GLOBAL, SHARED };

class Identifier;
class DataType;
using ParamType = std::pair<ptr<DataType>, ptr<Identifier>>;

inline static BaseType getTypeFromString(const std::string& input) {
  static const std::map<std::string, BaseType> typeMap = {
      {"f32", BaseType::F32}, {"f16", BaseType::F16}, {"bf16", BaseType::BF16},
      {"u32", BaseType::U32}, {"s32", BaseType::S32}, {"u16", BaseType::U16},
      {"s16", BaseType::S16}, {"u8", BaseType::U8},   {"s8", BaseType::S8},
      {"int", BaseType::INT}};

  auto it = typeMap.find(input);
  if (it != typeMap.end()) return it->second;

  assert(0 && "incorrect type string");
}

inline static std::string getStringFrom(BaseType dataType) {
  static const std::map<BaseType, std::string> enumToString = {
      {BaseType::F32, "f32"}, {BaseType::F16, "f16"}, {BaseType::BF16, "bf16"},
      {BaseType::U32, "u32"}, {BaseType::S32, "s32"}, {BaseType::U16, "u16"},
      {BaseType::S16, "s16"}, {BaseType::U8, "u8"},   {BaseType::S8, "s8"},
      {BaseType::INT, "int"}};

  auto it = enumToString.find(dataType);
  if (it != enumToString.end()) return it->second;

  assert(0 && "unsupported type.");
}

struct IntLiteral : public Node {
  int value;
  IntLiteral(int v) : value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << value;
  }
};

struct IntList : public Node {
  std::vector<ptr<IntLiteral>> values;

  void Append(ptr<IntLiteral> v) { values.push_back(v); }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "[";
    for (size_t i = 0; i < values.size() - 1; ++i)
      os << values[i]->value << ", ";
    os << values.back()->value << "]";
  }
};

struct SValList : public Node {
  std::vector<ptr<Node>> values;

  void Append(ptr<Node> v) { values.push_back(v); }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "[";
    for (size_t i = 0; i < values.size() - 1; ++i) {
      values[i]->Print(os);
      os << ", ";
    }
    values.back()->Print(os);
    os << "]";
  }
};

// Represents both dimensions and s like {3, 4, 5} or {1, 2, 1}
struct MultiSpans : public Node {
  std::string name = "";      // could be anonymous
  std::string ref_name = "";  // syntax delight
  ptr<NodeRef> list;
  explicit MultiSpans(const std::string& n, const ptr<NodeRef>& l)
      : name(n), list(l) {}
  explicit MultiSpans(const std::string& n, const std::string& rn,
                      const ptr<NodeRef>& l)
      : name(n), ref_name(rn), list(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Decl (span): ";
    if (name.size() > 0)
      os << name << " - ";
    else
      os << "(anon) - ";
    if (ref_name != "") {
      os << "{";
      list->Print(os, " " + ref_name);
      os << "}";
    } else
      list->Print(os);
  }
};

// Represents declarations like: ituple t = {3, 4, 5};
struct IntTuple : public Node {
  std::string name;  // could be anonymous
  ptr<SValList> value;
  explicit IntTuple(const std::string& n, ptr<SValList> l) : name(n), value(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Decl (tuple): ";
    if (name.size() > 0)
      os << name << " ";
    else
      os << "(anonymous) ";
    value->Print(os);
  }
};

class ITupleSymbolTable {
 private:
  std::unordered_map<std::string, ptr<IntTuple>> table;

 public:
  // Add a symbol to the symbol table
  void addITupleSymbol(const std::string& name, ptr<IntTuple>& ituple) {
    table[name] = ituple;
  }

  // Retrieve a symbol from the symbol table
  ptr<IntTuple> getSymbol(const std::string& name) {
    if (table.find(name) != table.end()) {
      return table[name];
    }
    return nullptr;
  }

  // Check if a symbol with the given name exists in the symbol table
  bool exists(const std::string& name) {
    return table.find(name) != table.end();
  }

};

struct IntVal : public Node {
  std::string name;  // could be anonymous
  ptr<Node> value;
  explicit IntVal(std::string& n, ptr<Node>& v) : name(n), value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Decl (int): ";
    if (name.size() > 0)
      os << name << " ";
    else
      os << "(anonymous) ";
    os << "= ";
    value->Print(os);
  }
};

struct IntIndex : public Node {
  int value;
  explicit IntIndex(int v) : value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "(" << value << ")";
    (void)prefix;
  }
};

struct IntIndexList : public Node {
  std::vector<ptr<IntIndex>> indices;

  void Append(ptr<IntIndex> v) { indices.push_back(v); }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "[";
    for (size_t i = 0; i < indices.size() - 1; ++i)
      os << indices[i]->value << ", ";
    os << indices.back()->value << "]";
  }
};


#if 0
// Represents data declarations like: global f32 data{d};
struct DataDecl : public Node {
  StorageSpec storage;
  BaseType type;
  std::string name;
  ptr<MultiSpans> mdspans;
  DataDecl(StorageSpec s, BaseType t, const std::string& n,
           const ptr<MultiSpans>& spans)
      : storage(s), type(t), name(n), mdspans(spans) {}
};
#endif

struct DataType : public Node {
 private:
  bool scalar;

 private:
  BaseType type;
  ptr<MultiSpans> mdspans = nullptr;

 public:
  DataType(BaseType t, bool s = true) : scalar(s), type(t) {}

  DataType(BaseType t, const ptr<MultiSpans>& spans)
      : scalar(false), type(t), mdspans(spans) {
    assert(namedSpans.count(spans->name) != 0 &&
           "Unexpected: symbol already existed.");
    namedSpans.emplace(spans->name, spans);
  }

  BaseType getBaseType() const { return type; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (scalar)
      os << prefix << getStringFrom(type);
    else if (mdspans == nullptr)
      os << prefix << getStringFrom(type) << "<>";
    else {
      os << prefix << getStringFrom(type);
      mdspans->Print(os);
    }
  }

  static ptr<MultiSpans> getSpanType(const std::string& name) {
    assert(namedSpans.count(name) && "symbol does not exist.");
    return namedSpans[name];
  }

  static std::unordered_map<std::string, ptr<MultiSpans>> namedSpans;
};

struct Identifier : public Node {
  std::string name;
  Identifier(const std::string& n) : name(n) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << name;
  }
};

struct ParamList : public Node {
  std::vector<ptr<ParamType>> values;
  explicit ParamList() {}
  ParamList(std::vector<ptr<ParamType>>& v) : values(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "Parameters";
    for (auto& item : values) {
      os << "\n  " << prefix;
      item->first->Print(os, " type: ");
      item->second->Print(os, ", symbol: ");
    }
  }
};

struct FunctionDecl : public Node {
  std::string name;
  ptr<DataType> ret_type;
  ptr<ParamList> params;

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "Name: " << name;
    os << "\n" << prefix << "Return type: ";
    ret_type->Print(os);
    params->Print(os, prefix);
  }
};

struct ChoreoFunction : public Node {
  std::string name;
  FunctionDecl f_decl;
  ptr<MultiNodes> statms;
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "ChoreoFunction";
    f_decl.Print(os, prefix + " `- ");
    if (statms) statms->Print(os, prefix + " ");
  }
};

struct CppSourceCode : public Node {
  std::string code;
  CppSourceCode(const std::string& c) : code(c) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << code;
    (void)prefix;
  }
};

struct Expression : public Node {
  std::vector<ptr<Node>> exprs;
};

// Top-level program structure
struct Program : public Node {
  std::vector<ptr<Node>> nodes;
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (auto& node : nodes) node->Print(os, "");
    (void)prefix;
  }
};

}  // end of namespace AST

#endif  // __CHOREO_AST_HPP__
