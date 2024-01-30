#ifndef __CHOREO_AST_HPP__
#define __CHOREO_AST_HPP__

#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace AST {

// Base class for all AST nodes
struct Node {
  virtual ~Node() {}
  virtual void Print(std::ostream& os, const std::string& prefix = {}) const {
    (void)os;
    (void)prefix;
  }
};

// For types like f32, f16, etc.
enum class BaseType { F32, F16, BF16, U32, S32, U16, S16, U8, S8, INT };

// For storage specifiers like local, global, shared
enum class StorageSpec { LOCAL, GLOBAL, SHARED };

class Identifier;
class DataType;
using ParamType =
    std::pair<std::shared_ptr<DataType>, std::shared_ptr<Identifier>>;

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

// Represents both dimensions and s like {3, 4, 5} or {1, 2, 1}
struct MdimSpans : public Node {
  std::vector<std::shared_ptr<Node>> values;
  explicit MdimSpans() {}
  MdimSpans(const std::vector<std::shared_ptr<Node>>& v) : values(v) {}
};

// Represents declarations like: mdimspans d{3, 4, 5};
struct MdimSpansDecl : public Node {
  std::string name;
  std::shared_ptr<MdimSpans> spans;
  MdimSpansDecl(const std::string& n, const std::shared_ptr<MdimSpans>& s)
      : name(n), spans(s) {}
};

// Represents data declarations like: global f32 data{d};
struct DataDecl : public Node {
  StorageSpec storage;
  BaseType type;
  std::string name;
  std::shared_ptr<MdimSpans>
      mdspans;  // Refers to a previously declared MdimSpans
  DataDecl(StorageSpec s, BaseType t, const std::string& n,
           const std::shared_ptr<MdimSpans>& spans)
      : storage(s), type(t), name(n), mdspans(spans) {}
};

struct DataType : public Node {
  bool scalar;
  BaseType type;
  std::shared_ptr<MdimSpans> mdspans;
  DataType(BaseType t) : scalar(true), type(t) {}
  DataType(BaseType t, const std::shared_ptr<MdimSpans>& spans)
      : scalar(false), type(t), mdspans(spans) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (scalar) os << prefix << getStringFrom(type);
  }
};

struct IntLiteral : public Node {
  int value;
  IntLiteral(int v) : value(v) {}
};

struct Identifier : public Node {
  std::string name;
  Identifier(const std::string& n) : name(n) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << name;
  }
};

struct ParamList : public Node {
  std::vector<std::shared_ptr<ParamType>> values;
  explicit ParamList() {}
  ParamList(std::vector<std::shared_ptr<ParamType>>& v) : values(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "Parameters";
    for (auto& item : values) {
      os << "\n  " << prefix;
      item->first->Print(os, " type: ");
      item->second->Print(os, ", symbol: ");
    }
  }
};

struct Statements : public Node {
  std::vector<std::shared_ptr<Node>> subs;
};

struct FunctionDecl : public Node {
  std::shared_ptr<DataType> ret_type;
  std::string name;
  std::shared_ptr<ParamList> params;

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "Name: " << name;
    os << "\n" << prefix << "Return type: ";
    ret_type->Print(os);
    params->Print(os, prefix);
  }
};

struct Declaration : public Node {};

struct ChoreoFunction : public Node {
  std::string name;
  FunctionDecl decls;
  std::shared_ptr<Statements> states;
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "ChoreoFunction";
    decls.Print(os, prefix + " `- ");
    states->Print(os, prefix + " ");
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
  std::vector<std::shared_ptr<Node>> exprs;
};

// Top-level program structure
struct Program : public Node {
  std::vector<std::shared_ptr<Node>> nodes;
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (auto& node : nodes) node->Print(os, "");
    (void)prefix;
  }
};

}  // end of namespace AST

#endif  // __CHOREO_AST_HPP__
