#ifndef __CHOREO_AST_HPP__
#define __CHOREO_AST_HPP__

#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "location.hh"
#include "symtab.hpp"

namespace Choreo {
struct Visitor;
}

[[noreturn]] inline void choreo_unreachable(
    const char* msg = "Unreachable code reached", const char* file = __FILE__,
    int line = __LINE__) {
  std::cerr << "Assertion failed: " << msg << ", file " << file << ", line "
            << line << std::endl;
  std::abort();
}

namespace AST {

//------------------------- AST Node Fundamentals ----------------------------//

template <typename T>
using ptr = std::shared_ptr<T>;

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

#define __NODE_TYPE_INFO__                                                    \
  const std::string NodeTypeString() override { return __PRETTY_FUNCTION__; } \
  uint64_t NodeTypeID() const override {                                      \
    return reinterpret_cast<uint64_t>(                                        \
        &TypeIDProvider<decltype(*this)>::unique);                            \
  }

// interface class for all AST nodes
struct Node {
  Choreo::location loc;
  BaseType b_type;

  Node(const Choreo::location& l) : loc(l) {}

  virtual void SetBaseType(BaseType b) { b_type = b; }
  virtual BaseType GetBaseType() { return b_type; }
  virtual const Choreo::location& LOC() { return loc; }

  virtual ~Node() = default;

  virtual void Print(std::ostream& os,
                     const std::string& prefix = {}) const = 0;

  // for the runtime type disambiguition
  virtual const std::string NodeTypeString() = 0;
  virtual uint64_t NodeTypeID() const = 0;

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
  T t(n->LOC());
  return t.NodeTypeID() == n->NodeTypeID();
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

// A general cluster of nodes
//
// It is normally used for a non-terminal node that comprises multiple nodes,
// i.e.:
//
//   non-term : non-term term_1 | term_2
//
struct MultiNodes : public Node {
  std::vector<ptr<Node>> values;

  explicit MultiNodes(const Choreo::location& l) : Node(l){};

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
  Boolean(const Choreo::location& l, std::string v) : Node(l), value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << value;
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct IntLiteral : public Node {
  int value;
  IntLiteral(const Choreo::location& l, int v) : Node(l), value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << value;
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct IntList : public Node {
  std::vector<ptr<IntLiteral>> values;

  IntList(const Choreo::location& l) : Node(l) {}

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

  SValList(const Choreo::location& l) : Node(l) {}

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
  ptr<Expr> value_c;
  ptr<Expr> value_l;
  ptr<Node> value_r;

  Expr(const Choreo::location& l) : Node(l) {}

  explicit Expr(const Choreo::location& l, const ptr<Node>& v)
      : Node(l), value_r(v) {}
  explicit Expr(const Choreo::location& l, const std::string& o,
                const ptr<Node>& v2)
      : Node(l), op(o), value_r(v2) {}
  explicit Expr(const Choreo::location& l, const std::string& o,
                const ptr<Expr>& v1, const ptr<Node>& v2)
      : Node(l), op(o), value_l(v1), value_r(v2) {}
  explicit Expr(const Choreo::location& l, const std::string& o,
                const ptr<Expr>& c, const ptr<Expr>& v1, const ptr<Node>& v2)
      : Node(l), op(o), value_c(c), value_l(v1), value_r(v2) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (op.size() > 0) {
      os << " (";
      if (op == "$") {
        value_c->Print(os);
        os << " ? ";
      }
      if (op != "!" && op != "sizeof" && op != ".data") {
        value_l->Print(os);
        os << " ";
      }
      if (op == "$")
        os << ": ";
      else
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
struct MultiDimSpans : public Node {
  std::string ref_name;  // syntax suger, could be empty
  ptr<Node> list;        // null if the span is a dynamic value
  int dim_count = 0;     // dynamic value with known dimension count

  // If the mdspan is known
  explicit MultiDimSpans(const Choreo::location& l, const std::string& n,
                         const ptr<Node>& lst)
      : Node(l), ref_name(n), list(lst), dim_count(0) {
    assert(list && "Unexpected: span list is not provided");
  }

  // mdspan is unknown - for parameter passing
  explicit MultiDimSpans(const Choreo::location& l, const std::string& n, int c)
      : Node(l), ref_name(n), list(nullptr), dim_count(c) {
    assert(dim_count > 0 && "Invalid dimensions.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (!list)
      os << "<" << dim_count << ">";
    else {
      os << "[";
      list->Print(os, " " + ref_name);
      os << " ]";
    }

    (void)prefix;
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct NamedTypeDecl : public Node {
  const std::string name_str;
  const std::string disp_str;
  const ptr<Node> init_expr;  // associated init_expr

  explicit NamedTypeDecl(const Choreo::location& l, const std::string& n,
                         const ptr<Node>& v, const std::string& d = "-")
      : Node(l), name_str(n), disp_str(d), init_expr(v) {
    assert(name_str.size() > 0 && "Invalid name string.");
    assert(init_expr && "Invalid value.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Type Decl: ";
    os << name_str << " " << disp_str << " ";
    init_expr->Print(os);
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct Memory : public Node {
  Storage st;
  Memory(const Choreo::location& l, const Storage s = Storage::DEFAULT)
      : Node(l), st(s) {}

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
      case Storage::DEFAULT:
        os << "default";
        break;
      default:
        assert(false && "Unexpected storage type.");
    }
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct NamedVariableDecl : public Node {
  const std::string name_str;
  const std::string disp_str;
  const ptr<Memory> mem = nullptr;  // storage location
  const ptr<Node> type = nullptr;
  const ptr<Node> initializer = nullptr;  // associated initializer

  explicit NamedVariableDecl(const Choreo::location& l, const std::string& n,
                             const ptr<Node>& t, const ptr<Memory>& s = nullptr,
                             const ptr<Node>& v = nullptr,
                             const std::string& d = "=")
      : Node(l), name_str(n), disp_str(d), mem(s), type(t), initializer(v) {
    assert(name_str.size() > 0 && "Invalid name string.");
    assert(type && "Invalid type.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Var Decl (";
    type->Print(os);
    if (mem) {
      os << ", ";
      mem->Print(os);
    }
    os << "): " << name_str;
    if (initializer) {
      os << " " << disp_str << " ";
      initializer->Print(os);
    }
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

// Represents declarations like: ituple t = {3, 4, 5};
struct IntTuple : public Node {
  std::string ref_name;  // could be anonymous
  ptr<Node> list;

  explicit IntTuple(const Choreo::location& l, const std::string& n,
                    ptr<Node> lst)
      : Node(l), ref_name(n), list(lst) {}

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
  explicit Assignment(const Choreo::location& l, std::string& n,
                      const ptr<Node>& v)
      : Node(l), name(n), value(v) {
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
  explicit IntIndex(const Choreo::location& l, const ptr<Node>& v)
      : Node(l), value(v) {}

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

  explicit NthBound(const Choreo::location& l, const ptr<Node>& a,
                    const ptr<IntIndex>& i)
      : Node(l), mdarray(a), index(i) {}

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

  IntIndexList(const Choreo::location& l) : Node(l) {}

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
// 1. A scalar type, including `int`, `bool`.
// 2. A composited type, including the fundamental type and the mdspan type.
// 3. An 'ituple' type.
//
struct DataType : public Node {
 private:
  BaseType base_type;
  ptr<Node> mdspan_type = nullptr;

 public:
  DataType(const Choreo::location& l, BaseType t)
      : Node(l), base_type(t), mdspan_type(nullptr) {}

  DataType(const Choreo::location& l, BaseType bt, const ptr<Node>& st)
      : Node(l), base_type(bt), mdspan_type(st) {
    assert(bt != BaseType::ITUPLE && "Unexpected type!");
    assert(bt != BaseType::INT && "Unexpected type!");
    assert(bt != BaseType::BOOL && "Unexpected type!");
  }

  BaseType getBaseType() const { return base_type; }

  bool isScalar() const { return !mdspan_type; }
  bool isITuple() const { return base_type == BaseType::ITUPLE; }
  bool isSpanned() const { return mdspan_type.get() != nullptr; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << getStringFrom(base_type);
    if (isSpanned()) {
      os << " ";
      mdspan_type->Print(os);
    }
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct Identifier : public Node {
  std::string name;
  Identifier(const Choreo::location& l, const std::string& n)
      : Node(l), name(n) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << name;
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct ParamList : public Node {
  std::vector<ptr<ParamType>> values;
  explicit ParamList(const Choreo::location& l) : Node(l) {}
  ParamList(const Choreo::location& l, std::vector<ptr<ParamType>>& v)
      : Node(l), values(v) {}

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

struct IfElse : public Node {
  ptr<Node> cond;
  ptr<MultiNodes> if_stmts;
  ptr<MultiNodes> else_stmts;  // optional requirements

  IfElse(const Choreo::location& l, const ptr<Node>& c,
         const ptr<MultiNodes>& if_s)
      : Node(l), cond(c), if_stmts(if_s) {}
  IfElse(const Choreo::location& l, const ptr<Node>& c,
         const ptr<MultiNodes>& if_s, const ptr<MultiNodes>& else_s)
      : Node(l), cond(c), if_stmts(if_s), else_stmts(else_s) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "\n`- IF: ";
    cond->Print(os);
    os << "\n`---- THEN: ";
    if_stmts->Print(os);
    if (!else_stmts->values.empty()) {
      os << "\n`---- ELSE: ";
      else_stmts->Print(os);
    }
    os << "\n";
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct ParallelBy : public Node {
  std::string iv;
  int bound;
  ptr<MultiNodes> statms;

  ParallelBy(const Choreo::location& l, const std::string v, int b)
      : Node(l), iv(v), bound(b) {}

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

  RequireBind(const Choreo::location& l, const ptr<Node>& lhs,
              const ptr<Node>& rhs)
      : Node(l), lhs(lhs), rhs(rhs) {}

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

  WithIn(const Choreo::location& l, const ptr<Node>& w, const ptr<Node>& i)
      : Node(l), with(w), in(i) {}

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

  explicit WithBlock(const Choreo::location& l) : Node(l) {}

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

struct DMA : public Node {
  std::string operation;
  ptr<Node> future;
  ptr<Node> from;
  ptr<Node> to;

  DMA(const Choreo::location& l, const std::string& o, const ptr<Node>& r,
      const ptr<Node>& f, const ptr<Node>& t)
      : Node(l), operation(o), future(r), from(f), to(t) {}

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

  ChunkAt(const Choreo::location& l, const ptr<Node>& d,
          const ptr<MultiNodes>& p)
      : Node(l), data(d), positions(p) {}

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

  Wait(const Choreo::location& l, const ptr<Node>& t) : Node(l), target(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- WAIT: ";
    target->Print(os);
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct Return : public Node {
  ptr<Node> value = nullptr;

  Return(const Choreo::location& l) : Node(l) {}
  Return(const Choreo::location& l, const ptr<Node>& t) : Node(l), value(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Return: ";
    if (!value)
      os << "void";
    else
      value->Print(os);
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

struct Call : public Node {
  ptr<Node> function;
  ptr<Node> arguments;

  Call(const Choreo::location& l, const ptr<Node>& f, const ptr<Node>& a)
      : Node(l), function(f), arguments(a) {}

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

  explicit ForeachBlock(const Choreo::location& l, const ptr<MultiNodes>& i,
                        const ptr<MultiNodes>& s)
      : Node(l), ivs(i), statms(s) {
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

  FunctionDecl(const Choreo::location& l) : Node(l) {}

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

  ChoreoFunction(const Choreo::location& l) : Node(l), f_decl(l) {}

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
  CppSourceCode(const Choreo::location& l, const std::string& c)
      : Node(l), code(c) {}
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

  Program(const Choreo::location& l) : Node(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (auto& node : nodes) node->Print(os, "");
    (void)prefix;
  }

  void accept(Choreo::Visitor&) override;

  __NODE_TYPE_INFO__
};

}  // end of namespace AST

#endif  // __CHOREO_AST_HPP__
