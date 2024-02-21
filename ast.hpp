#ifndef __CHOREO_AST_HPP__
#define __CHOREO_AST_HPP__

#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "symtab.hpp"
#include "visitor.hpp"

namespace AST {

template <typename T>
using ptr = std::shared_ptr<T>;

#define __NODE_TYPE_STRING__ \
  const std::string TypeString() override { return __PRETTY_FUNCTION__; }

// interface class for all AST nodes
struct Node {
  virtual ~Node() {}

  virtual void Print(std::ostream& os, const std::string& prefix = {}) const {
    (void)os;
    (void)prefix;
  }

  virtual const std::string TypeString() = 0;
  // TODO: implementation for each derived-type
  virtual void accept(Visitor& visitor) { visitor.visit(this); }
};

// A node with the reference to another node.
//
// It is normally used for a non-terminal node that referring an union of other
// nodes, i.e.:
//
//   non-term : term_1 | term_2
//
struct NodeRef : public Node {
  ptr<Node> value;

  NodeRef(ptr<Node> n) : value(n) {}

  virtual void Print(std::ostream& os, const std::string& prefix = {}) const {
    value->Print(os, prefix);
  }

  void accept(Visitor& visitor) override { visitor.visit(&*value); }

  __NODE_TYPE_STRING__
};

// A general cluster of nodes
//
// It is normally used for a non-terminal node that comprises multiple nodes,
// i.e.:
//
//   non-term : non-term term_1 | term_2
//
struct MultiNodes : public Node {
  std::vector<ptr<Node>> values;

  explicit MultiNodes(){};

  void Append(const ptr<Node>& m) {
    assert(m != nullptr && "Unexpected: null pointer.");
    values.push_back(m);
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (auto& v : values) v->Print(os, prefix);
  }

  void accept(Visitor& visitor) override {
    for (auto& v : values) visitor.visit(&*v);
  }

  __NODE_TYPE_STRING__
};

// For storage specifiers like local, global, shared
enum class Storage { LOCAL, SHARED, GLOBAL };

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

  __NODE_TYPE_STRING__
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

  __NODE_TYPE_STRING__
};

struct SValList : public Node {
  std::vector<ptr<Node>> values;

  void Append(ptr<Node> v) { values.push_back(v); }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (size_t i = 0; i < values.size() - 1; ++i) {
      values[i]->Print(os);
      os << ", ";
    }
    values.back()->Print(os);
    (void)prefix;
  }

  __NODE_TYPE_STRING__
};

struct Expr : public Node {
  std::string op;
  ptr<Expr> value_l;
  ptr<Node> value_r;
  explicit Expr(const std::string& o, const ptr<Expr>& v1, const ptr<Node>& v2)
      : op(o), value_l(v1), value_r(v2) {}
  explicit Expr(const ptr<Node>& v) : value_r(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (op.size() > 0) {
      os << " (";
      value_l->Print(os);
      os << " " << op << " ";
      value_r->Print(os);
      os << ") ";
    } else
      value_r->Print(os);
    (void)prefix;
  }

  __NODE_TYPE_STRING__
};

// Represents both dimensions and s like {3, 4, 5} or {1, 2, 1}
struct MultiSpans : public Node {
  std::string ref_name;  // syntax suger, could be empty
  ptr<Node> list;

  explicit MultiSpans(const std::string& n, const ptr<Node>& l)
      : ref_name(n), list(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "[";

    if (!ref_name.empty()) {
      list->Print(os, " " + ref_name);
    } else if (list != nullptr)
      list->Print(os, " ");

    os << " ]";

    (void)prefix;
  }

  __NODE_TYPE_STRING__
};

struct NamedDecl : public Node {
  const std::string name_str;
  const std::string type_str;
  const std::string disp_str;
  const ptr<Node> value;

  explicit NamedDecl(const std::string& n, const std::string& t,
                     const ptr<Node>& v, const std::string& d = "=")
      : name_str(n), type_str(t), disp_str(d), value(v) {
    assert(name_str.size() > 0 && "Invalid name string.");
    assert(type_str.size() > 0 && "Invalid type string.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Decl (" << type_str << "): ";
    os << name_str << " " << disp_str << " ";
    value->Print(os);
  }

  __NODE_TYPE_STRING__
};

// Represents declarations like: ituple t = {3, 4, 5};
struct IntTuple : public Node {
  std::string ref_name;  // could be anonymous
  ptr<Node> list;

  explicit IntTuple(const std::string& n, ptr<Node> l) : ref_name(n), list(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (ref_name.size() > 0) os << ref_name << " ";
    os << "{";
    list->Print(os);
    os << "}";

    (void)prefix;
  }

  __NODE_TYPE_STRING__
};

class ITupleTable {
 private:
  std::unordered_map<std::string, ptr<IntTuple>> table;
  SymbolTable& st;

 public:
  ITupleTable(SymbolTable& symtab) : st(symtab) {}

  // Add a symbol to the symbol table
  void addITupleSymbol(const std::string& name, ptr<IntTuple>& ituple) {
    st.addSymbol(name, BaseType::INT);
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

struct Assignment : public Node {
  std::string name;
  ptr<Node> value;
  explicit Assignment(std::string& n, const ptr<Node>& v) : name(n), value(v) {
    assert(n.size() > 0 && "invalid assignment to the un-named value.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Assign: " << name << " = ";
    value->Print(os);
  }

  __NODE_TYPE_STRING__
};

struct IntIndex : public Node {
  ptr<Node> value;
  explicit IntIndex(const ptr<Node>& v) : value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "(";
    value->Print(os);
    os << ")";
  }

  __NODE_TYPE_STRING__
};

struct NthBound : public Node {
  ptr<Node> mdarray;
  ptr<IntIndex> index;

  explicit NthBound(const ptr<Node>& a, const ptr<IntIndex>& i)
      : mdarray(a), index(i) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix;
    mdarray->Print(os);
    index->Print(os);
  }

  __NODE_TYPE_STRING__
};

struct IntIndexList : public Node {
  std::vector<ptr<IntIndex>> indices;

  void Append(ptr<IntIndex> v) { indices.push_back(v); }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (indices.empty()) return;

    size_t i = 0;
    for (; i < indices.size() - 1; ++i) {
      indices[i]->Print(os);
      os << ", ";
    }
    indices[i]->Print(os);

    (void)prefix;
  }

  __NODE_TYPE_STRING__
};

// A data type could either be
//
// 1. a simple type, including `int`, `bool`.
// 2. a composited type, including the base type and the span type.
//
struct DataType : public Node {
 private:
  bool scalar;

 private:
  BaseType base_type;
  ptr<Node> span_type = nullptr;

 public:
  DataType(BaseType t, bool s = true) : scalar(s), base_type(t) {}

  DataType(BaseType bt, const ptr<Node>& st)
      : scalar(false), base_type(bt), span_type(st) {}

  BaseType getBaseType() const { return base_type; }
  bool isAggregate() const { return !scalar; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << getStringFrom(base_type);
    if (!scalar) {
      if (span_type)
        os << " " << span_type;
      else
        os << "<>";
    }
  }

  __NODE_TYPE_STRING__
};

struct Identifier : public Node {
  std::string name;
  Identifier(const std::string& n) : name(n) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << name;
  }

  __NODE_TYPE_STRING__
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

  __NODE_TYPE_STRING__
};

struct ParallelBy : public Node {
  std::string iv;
  int bound;
  ptr<MultiNodes> statms;

  ParallelBy(const std::string v, int b) : iv(v), bound(b) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Parellelization: ";
    os << " IV symbol: " << iv << ", bound: " << bound;
    if (!statms)
      os << std::endl;
    else
      statms->Print(os, prefix + " ");
  }

  __NODE_TYPE_STRING__
};

// `require_bind` parsing "idx_1 <-> idx_2"
struct RequireBind : public Node {
  ptr<Node> lhs;
  ptr<Node> rhs;

  RequireBind(const ptr<Node>& lhs, const ptr<Node>& rhs)
      : lhs(lhs), rhs(rhs) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "`- ";
    lhs->Print(os);
    os << " bind-to ";
    rhs->Print(os);
    os << "\n";
  }

  __NODE_TYPE_STRING__
};

struct WithIn : public Node {
  ptr<Node> with;
  ptr<Node> in;

  WithIn(const ptr<Node>& w, const ptr<Node>& i) : with(w), in(i) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "`- ";
    with->Print(os);
    os << " in ";
    in->Print(os);
    os << "\n";
  }

  __NODE_TYPE_STRING__
};

struct WithBlock : public Node {
  ptr<MultiNodes> withins;
  ptr<MultiNodes> reqs;    // optional requirements
  ptr<MultiNodes> statms;  // may be empty

  explicit WithBlock() {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- With Block:\n";
    os << prefix << "  (within constraints)\n";
    withins->Print(os, prefix + "  ");
    if (reqs) {
      os << prefix << "  (require clause)\n";
      reqs->Print(os, prefix + "  ");
    }
    if (statms) {
      if (statms->values.size() == 0) {
        os << prefix << "  (with empty statements)\n";
        return;
      }
      os << prefix << "  (with statements)";
      statms->Print(os, prefix + "  ");
    }
  }

  __NODE_TYPE_STRING__
};

struct Memory : public Node {
  Storage st;
  Memory(const Storage s) : st(s) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    (void)prefix;
    switch (st) {
      case Storage::LOCAL:
        os << "local";
        break;
      case Storage::SHARED:
        os << "shared";
        break;
      case Storage::GLOBAL:
        os << "global";
        break;
    }
  }

  __NODE_TYPE_STRING__
};

struct DMA : public Node {
  std::string operation;
  ptr<Node> future;
  ptr<Node> from;
  ptr<Node> to;

  DMA(const std::string& o, const ptr<Node>& r, const ptr<Node>& f,
      const ptr<Node>& t)
      : operation(o), future(r), from(f), to(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- DMA" << operation;
    os << "\n" << prefix << "  `- furture: ";
    future->Print(os);
    os << "\n" << prefix << "  `- from: ";
    from->Print(os);
    os << prefix << "  `- to: ";
    to->Print(os);
  }

  __NODE_TYPE_STRING__
};

struct ChunkAt : public Node {
  ptr<Node> data;
  ptr<MultiNodes> positions;

  ChunkAt(const ptr<Node>& d, const ptr<MultiNodes>& p)
      : data(d), positions(p) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    data->Print(os);
    os << ".ChunkAt(";
    positions->Print(os);
    os << ")\n";

    (void)prefix;
  }

  __NODE_TYPE_STRING__
};

struct Wait : public Node {
  ptr<Node> target;

  Wait(const ptr<Node>& t) : target(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- WAIT: ";
    target->Print(os);
  }

  __NODE_TYPE_STRING__
};

struct Call : public Node {
  ptr<Node> function;
  ptr<Node> arguments;

  Call(const ptr<Node>& f, const ptr<Node>& a) : function(f), arguments(a) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Call: ";
    function->Print(os);
    os << "\n" << prefix << "  `- with arguements:";
    arguments->Print(os);
  }

  __NODE_TYPE_STRING__
};

struct ForeachBlock : public Node {
  ptr<MultiNodes> ivs;
  ptr<MultiNodes> statms;

  explicit ForeachBlock(const ptr<MultiNodes>& i, const ptr<MultiNodes>& s)
      : ivs(i), statms(s) {
    assert(i != nullptr && "missing iteration variables for the statement.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Foreach Block:";
    os << "\n" << prefix << " `- Iteration variables: ";
    ivs->Print(os);
    if (statms) {
      statms->Print(os, prefix + " ");
    }
  }

  __NODE_TYPE_STRING__
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

  __NODE_TYPE_STRING__
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

  __NODE_TYPE_STRING__
};

struct CppSourceCode : public Node {
  std::string code;
  CppSourceCode(const std::string& c) : code(c) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << code;
    (void)prefix;
  }

  __NODE_TYPE_STRING__
};

// Top-level program structure
struct Program : public Node {
  std::vector<ptr<Node>> nodes;
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (auto& node : nodes) node->Print(os, "");
    (void)prefix;
  }

  __NODE_TYPE_STRING__
};

}  // end of namespace AST

#endif  // __CHOREO_AST_HPP__
