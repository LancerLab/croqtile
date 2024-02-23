#ifndef __CHOREO_AST_HPP__
#define __CHOREO_AST_HPP__

#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "symtab.hpp"

namespace Choreo { struct Visitor; }

[[noreturn]] inline void choreo_unreachable(const char* msg = "Unreachable code reached", const char* file = __FILE__, int line = __LINE__) {
  std::cerr << "Assertion failed: " << msg << ", file " << file << ", line " << line << std::endl;
  std::abort();
}

namespace AST {

//------------------------- AST Node Fundamentals ----------------------------//

template <typename T>
using ptr = std::shared_ptr<T>;

// For storage specifiers like local, global, shared
enum class Storage { LOCAL, SHARED, GLOBAL };

class Identifier;
class DataType;
using ParamType = std::pair<ptr<DataType>, ptr<Identifier>>;

// smart typeid provider suggested by GPT
template <typename T>
struct TypeIDProvider {
  static int unique;
};

template <typename T>
int TypeIDProvider<T>::unique;

#define __NODE_TYPE_INFO__                                                \
  const std::string TypeString() override { return __PRETTY_FUNCTION__; } \
  uint64_t TypeID() const override {                                      \
    return reinterpret_cast<uint64_t>(                                    \
        &TypeIDProvider<decltype(*this)>::unique);                        \
  }

// interface class for all AST nodes
struct Node {
  virtual ~Node() = default;

  virtual void Print(std::ostream& os,
                     const std::string& prefix = {}) const = 0;

  // for the runtime type disambiguition
  virtual const std::string TypeString() = 0;
  virtual uint64_t TypeID() const = 0;

  virtual void accept(Choreo::Visitor&) = 0;
};

// LLVM-style type utility functions for AST::Node
//
// Note:
// To be simple, we do not handle any relationship about inheritance but only
// the extact (most-derived) type
//

template <typename T>
bool isa(Node* n) {
  T t;
  return t.TypeID() == n->TypeID();
}

template <typename T>
T* dyn_cast(Node* n) {
  if (isa<T>(n))
    return (T*)n;
  else
    return nullptr;
}

template <typename T>
T* cast(Node* n) {
  if (isa<T>(n))
    return (T*)n;
  else {
    std::cerr << "Cast failure for the type inconsistence." << std::endl;
    abort();
  }
}

//---------------------------------------------------------------------------//

// A node with the reference to another node.
//
// TODO: deprecated it
struct NodeRef : public Node {
  ptr<Node> value;

  NodeRef(ptr<Node> n) : value(n) {}

  virtual void Print(std::ostream& os, const std::string& prefix = {}) const {
    value->Print(os, prefix);
  }

  void accept(Choreo::Visitor& visitor) override;

  __NODE_TYPE_INFO__
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
    for (auto& v : values) {
      v->Print(os, prefix);
    }
  }

  // TODO: workaround for "x, y" like print, we may need typeid to merge this
  // print logic into trivial Print()
  void InlinePrint(std::ostream& os, const std::string& prefix = {}) const {
    for (auto& v : values) {
      v->Print(os, prefix);
      if (&v != &values.back()) os << ", ";
    }
  }

  void accept(Choreo::Visitor& visitor) override;

  __NODE_TYPE_INFO__
};

struct Boolean : public Node {
  std::string value;
  Boolean(std::string v) : value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << value;
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct IntLiteral : public Node {
  int value;
  IntLiteral(int v) : value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << value;
  }


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct Expr : public Node {
  std::string op;
  ptr<Expr> value_l;
  ptr<Node> value_r;
  explicit Expr(const std::string& o, const ptr<Expr>& v1, const ptr<Node>& v2)
      : op(o), value_l(v1), value_r(v2) {}
  explicit Expr(const ptr<Node>& v) : value_r(v) {}
  explicit Expr(const std::string& o, const ptr<Node>& v2)
      : op(o), value_r(v2) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (op.size() > 0) {
      os << " (";
      if(op != "!") {
        value_l->Print(os);
        os << " ";
      }
      os << op << " ";
      value_r->Print(os);
      os << ") ";
    } else
      value_r->Print(os);
    (void)prefix;
  }


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct IntIndex : public Node {
  ptr<Node> value;
  explicit IntIndex(const ptr<Node>& v) : value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "(";
    value->Print(os);
    os << ")";
  }


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct Identifier : public Node {
  std::string name;
  Identifier(const std::string& n) : name(n) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << name;
  }


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct WithIn : public Node {
  ptr<Node> with;
  ptr<Node> in;
  ptr<MultiNodes> with_matchers;  // optional requirements

  WithIn(const ptr<Node>& w, const ptr<Node>& i) : with(w), in(i) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "`- ";
    with->Print(os);
    if (with_matchers) {
      os << " = {";
      with_matchers->InlinePrint(os);
      os << "}";
    }
    os << " in ";
    in->Print(os);
    os << "\n";
  }


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct Wait : public Node {
  ptr<Node> target;

  Wait(const ptr<Node>& t) : target(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- WAIT: ";
    target->Print(os);
  }


  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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
  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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
  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
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
  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct CppSourceCode : public Node {
  std::string code;
  CppSourceCode(const std::string& c) : code(c) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << code;
    (void)prefix;
  }

  std::string GetCode() { return code; }
  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

// Top-level program structure
struct Program : public Node {
  std::vector<ptr<Node>> nodes;
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (auto& node : nodes) node->Print(os, "");
    (void)prefix;
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

}  // end of namespace AST

#endif  // __CHOREO_AST_HPP__
