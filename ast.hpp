#ifndef __CHOREO_AST_HPP__
#define __CHOREO_AST_HPP__

#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

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
  ptr<Node> val;
  NodeRef(ptr<Node> n) : val(n) {}
  virtual void Print(std::ostream& os, const std::string& prefix = {}) const {
    val->Print(os, prefix);
  }
};

// General cluster of nodes
struct MultiNodes : public Node {
  std::vector<ptr<NodeRef>> vals;
  explicit MultiNodes(){};
  void Append(ptr<NodeRef>& m) { vals.push_back(m); }
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (auto& v : vals) v->Print(os, prefix);
  }
};

// For types like f32, f16, etc.
enum class BaseType { F32, F16, BF16, U32, S32, U16, S16, U8, S8, INT };

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

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "[";
    for (size_t i = 0; i < values.size() - 1; ++i)
      os << values[i]->value << ", ";
    os << values.back()->value << "]";
  }
};

// Represents both dimensions and s like {3, 4, 5} or {1, 2, 1}
struct MultiSpans : public Node {
  ptr<IntList> list;
  std::string name;  // could be anonymous
  explicit MultiSpans(ptr<IntList>& l) : list(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Decl (span): ";
    if (name.size() > 0)
      os << name << " ";
    else
      os << "(anonymous) ";
    list->Print(os);
  }
};

// Represents declarations like: mdimspans d{3, 4, 5};
struct MdimSpansDecl : public Node {
  std::string name;
  ptr<MultiSpans> spans;
  MdimSpansDecl(const std::string& n, const ptr<MultiSpans>& s)
      : name(n), spans(s) {}
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
  bool scalar;
  BaseType type;
  ptr<MultiSpans> mdspans;
  DataType(BaseType t) : scalar(true), type(t) {}
  DataType(BaseType t, ptr<MultiSpans>& spans)
      : scalar(false), type(t), mdspans(spans) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (scalar) os << prefix << getStringFrom(type);
  }
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
