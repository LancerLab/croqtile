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

[[noreturn]] inline void choreo_unreachable(
    const char* msg = "Unreachable code reached", const char* file = __FILE__,
    int line = __LINE__) {
  std::cerr << "Assertion failed: " << msg << ", file " << file << ", line "
            << line << std::endl;
  std::abort();
}

namespace Choreo {
struct Visitor;

namespace AST {

// short hands
using BaseType = Choreo::BaseType;
using Storage = Choreo::Storage;
using SymbolTable = Choreo::SymbolTable;
template <typename T>
using ptr = Choreo::ptr<T>;

//------------------------- AST Node Fundamentals ----------------------------//

class Identifier;
class DataType;

// interface class for all AST nodes
struct Node {
  location loc;
  ptr<Type> pty = MakeUnknownType();

  Node(const location& l, const ptr<Type>& p = MakeUnknownType()) : loc(l), pty(p) {}

  virtual bool TypeUnknown() const { return isa<UnknownType>(pty.get()); }

  virtual void SetType(const ptr<Type>& t) { pty = t; }
  virtual const ptr<Type>& GetType() const { return pty; }
  virtual const location& LOC() const { return loc; }

  virtual ~Node() = default;

  virtual void Print(std::ostream& os,
                     const std::string& prefix = {}) const = 0;

  virtual void PrintType(std::ostream& os,
                         const std::string& prefix = {}) const {
    os << prefix;
    pty->Print(os);
  };

  virtual void accept(Visitor&) = 0;

  // for runtime type disambiguition
  virtual const std::string NodeTypeString() = 0;
  virtual uint64_t RuntimeID() const { return 0ULL; }
  static uint64_t TypeID() { return 0ULL; }
};

//---------------------------------------------------------------------------//

// A group of nodes
//
// It is normally used for a non-terminal node that comprises multiple nodes,
// i.e.:
//
//   non-term : non-term term_1 | term_2
//
struct MultiNodes : public Node, public TypeIDProvider<MultiNodes> {
  std::vector<ptr<Node>> values;
  std::string delimiter;

  explicit MultiNodes(const location& l, std::string d = "")
      : Node(l), delimiter(d){};

  void Append(const ptr<Node>& m) {
    assert(m != nullptr && "Unexpected: null pointer.");
    values.push_back(m);
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (delimiter != "" && values.size() > 1) {
      auto i = values.begin();
      auto e = values.end();
      (*i)->Print(os, prefix);
      ++i;
      for (; i != e; ++i) {
        os << delimiter;
        (*i)->Print(os, prefix);
      }
    } else {
      for (auto& v : values) v->Print(os, prefix);
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

  void accept(Visitor& visitor) override;

  __UDT_TYPE_INFO__
};

struct Boolean : public Node, public TypeIDProvider<Boolean> {
  std::string value;
  Boolean(const location& l, std::string v) : Node(l, MakeBooleanType()), value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << value;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct IntLiteral : public Node, public TypeIDProvider<IntLiteral> {
  int value;
  IntLiteral(const location& l, int v = __UNKNOWN_INTVAL__)
      : Node(l, MakeIntegerType()), value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (value == __UNKNOWN_INTVAL__)
      os << prefix << "?";
    else
      os << prefix << value;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct SValList : public Node, public TypeIDProvider<SValList> {
  std::vector<ptr<Node>> values;

  SValList(const location& l) : Node(l) {}

  void Append(ptr<Node> v) { values.push_back(v); }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (size_t i = 0; i < values.size() - 1; ++i) {
      values[i]->Print(os);
      os << ", ";
    }
    values.back()->Print(os);
    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Expr : public Node, public TypeIDProvider<Expr> {
  std::string op;
  ptr<Expr> value_c = nullptr;
  ptr<Expr> value_l = nullptr;
  ptr<Node> value_r = nullptr;

  Expr(const location& l) : Node(l) {}

  explicit Expr(const location& l, const ptr<Node>& v) : Node(l), value_r(v) {}
  explicit Expr(const location& l, const std::string& o, const ptr<Node>& v2)
      : Node(l), op(o), value_r(v2) {}
  explicit Expr(const location& l, const std::string& o, const ptr<Expr>& v1,
                const ptr<Node>& v2)
      : Node(l), op(o), value_l(v1), value_r(v2) {}
  explicit Expr(const location& l, const std::string& o, const ptr<Expr>& c,
                const ptr<Expr>& v1, const ptr<Node>& v2)
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

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

// Represents both dimensions and s like {3, 4, 5} or {1, 2, 1}
struct MultiDimSpans : public Node, public TypeIDProvider<MultiDimSpans> {
  std::string ref_name;  // syntax suger, could be empty
  ptr<Node> list;        // null if the span is a dynamic value
  size_t dim_count =
      __INVALID_VALUE__;  // dynamic value with known dimension count

  // If the mdspan is known
  explicit MultiDimSpans(const location& l, const std::string& n,
                         const ptr<Node>& lst)
      : Node(l, MakeUninitMDSpanType()), ref_name(n), list(lst), dim_count(__INVALID_VALUE__) {
    assert(list && "Unexpected: span list is not provided");
  }

  // set both the mdspan and dim count
  explicit MultiDimSpans(const location& l, const std::string& n,
                         const ptr<Node>& lst, size_t dc)
      : Node(l, MakeUninitMDSpanType()), ref_name(n), list(lst), dim_count(dc) {
    assert(list && "Unexpected: span list is not provided");
    // check the consistent between dim_count and span list in semantic time
  }

  // mdspan is unknown - for parameter passing
  explicit MultiDimSpans(const location& l, const std::string& n, size_t c)
      : Node(l, MakeUninitMDSpanType()), ref_name(n), list(nullptr), dim_count(c) {
    assert(dim_count != __INVALID_VALUE__ && "Invalid dimensions.");
  }

  MDSpanValue MakeValueList() {
    if (!list) {
      // dynamically valued
      return {MDSpanValue(dim_count)};
    } else
      return {MDSpanValue(0) /*TODO: make Type from the list*/};
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

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct NamedTypeDecl : public Node, public TypeIDProvider<NamedTypeDecl> {
  const std::string name_str;
  const std::string init_str;
  const ptr<Node> init_expr;  // associated init_expr

  explicit NamedTypeDecl(const location& l, const std::string& n,
                         const ptr<Node>& v, const std::string& d = "-")
      : Node(l), name_str(n), init_str(d), init_expr(v) {
    assert(name_str.size() > 0 && "Invalid name string.");
    assert(init_expr && "Invalid value.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Type Decl: ";
    os << name_str << " " << init_str << " ";
    init_expr->Print(os);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Memory : public Node, public TypeIDProvider<Memory> {
  Storage st;
  Memory(const location& l, const Storage s = Storage::DEFAULT)
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

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct NamedVariableDecl : public Node,
                           public TypeIDProvider<NamedVariableDecl> {
  const std::string name_str;
  const std::string init_str;
  const ptr<Memory> mem = nullptr;  // storage location
  const ptr<Node> type = nullptr;
  const ptr<Node> initializer = nullptr;  // associated initializer

  explicit NamedVariableDecl(const location& l, const std::string& n,
                             const ptr<Node>& t, const ptr<Memory>& s = nullptr,
                             const ptr<Node>& v = nullptr,
                             const std::string& d = "=")
      : Node(l), name_str(n), init_str(d), mem(s), type(t), initializer(v) {
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
      os << " " << init_str << " ";
      initializer->Print(os);
    }
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

// Represents declarations like: ituple t = {3, 4, 5};
struct IntTuple : public Node, public TypeIDProvider<IntTuple> {
  std::string ref_name;  // could be anonymous
  ptr<Node> list;

  explicit IntTuple(const location& l, const std::string& n, ptr<Node> lst)
      : Node(l, MakeUninitITupleType()), ref_name(n), list(lst) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (ref_name.size() > 0) os << ref_name << " ";
    os << "{";
    list->Print(os);
    os << "}";

    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Assignment : public Node, public TypeIDProvider<Assignment> {
  std::string name;
  ptr<Node> value;
  explicit Assignment(const location& l, std::string& n, const ptr<Node>& v)
      : Node(l), name(n), value(v) {
    assert(n.size() > 0 && "invalid assignment to the un-named value.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Assign: " << name << " = ";
    value->Print(os);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct IntIndex : public Node, public TypeIDProvider<IntIndex> {
  ptr<Node> value;
  explicit IntIndex(const location& l, const ptr<Node>& v)
      : Node(l), value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "(";
    value->Print(os);
    os << ")";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct NthBound : public Node, public TypeIDProvider<NthBound> {
  ptr<Node> mdarray;
  ptr<IntIndex> index;

  explicit NthBound(const location& l, const ptr<Node>& a,
                    const ptr<IntIndex>& i)
      : Node(l, MakeIntegerType()), mdarray(a), index(i) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix;
    mdarray->Print(os);
    index->Print(os);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct IntIndexList : public Node, public TypeIDProvider<IntIndexList> {
  std::vector<ptr<IntIndex>> indices;

  IntIndexList(const location& l) : Node(l) {}

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

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

// A data type could either be
//
// 1. A scalar type, including `int`, `bool`.
// 2. A composited type, including the fundamental type and the mdspan type.
// 3. An 'ituple' type.
//
struct DataType : public Node, public TypeIDProvider<DataType> {
  BaseType base_type;
  ptr<Node> mdspan_type = nullptr;

 public:
  DataType(const location& l, BaseType t)
      : Node(l), base_type(t), mdspan_type(nullptr) {}

  DataType(const location& l, BaseType bt, const ptr<Node>& st)
      : Node(l), base_type(bt), mdspan_type(st) {
    assert(bt != BaseType::ITUPLE && "Unexpected type!");
    assert(bt != BaseType::INT && "Unexpected type!");
    assert(bt != BaseType::BOOL && "Unexpected type!");
  }

  BaseType getBaseType() const { return base_type; }
  Node* getPartialType() const { return mdspan_type.get(); }

  bool isScalar() const { return (base_type == BaseType::INT) || (base_type == BaseType::BOOL); }
  bool isITuple() const { return base_type == BaseType::ITUPLE; }
  bool isSpanned() const { return (bool)mdspan_type; }

  ptr<Type> MakeSemaType() {
    switch (base_type) {
      case BaseType::INT:
        return std::make_shared<IntegerType>();
      case BaseType::BOOL:
        return std::make_shared<BooleanType>();
      case BaseType::F32:
      case BaseType::F16:
      case BaseType::BF16:
      case BaseType::U32:
      case BaseType::S32:
      case BaseType::U16:
      case BaseType::S16:
      case BaseType::U8:
      case BaseType::S8:
        assert(mdspan_type != nullptr && "Expecting a valid mdspan.");
        return GetType(); // the type has be deduced already
      case BaseType::ITUPLE:
        return MakeUninitITupleType();  // need type inference to retrieve the
                                        // dim count
      default:
        choreo_unreachable("Unexpected BaseType.");
        break;
    }
    return nullptr;
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << getStringFrom(base_type);
    if (isSpanned()) {
      os << " ";
      mdspan_type->Print(os);
    }
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Identifier : public Node, public TypeIDProvider<Identifier> {
  std::string name;
  Identifier(const location& l,
             const std::string& n = SymbolTable::GetAnonName())
      : Node(l), name(n) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << name;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Parameter : public Node, public TypeIDProvider<Parameter> {
  ptr<DataType> type = nullptr;
  ptr<Identifier> sym = nullptr;

  Parameter(const location& l, const ptr<DataType> t,
            ptr<Identifier> n = nullptr)
      : Node(l), type(t), sym(n) {
    assert(t && "invalid parameter without a type.");
  }

  bool HasSymbol() const { return (bool)sym; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    type->Print(os, prefix + " type: ");
    if (sym) sym->Print(os, ", symbol: ");
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct ParamList : public Node, public TypeIDProvider<ParamList> {
  std::vector<ptr<Parameter>> values;
  explicit ParamList(const location& l) : Node(l) {}
  ParamList(const location& l, std::vector<ptr<Parameter>>& v)
      : Node(l), values(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "Parameters";
    for (auto& item : values) item->Print(os, "\n  " + prefix);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct IfElse : public Node, public TypeIDProvider<IfElse> {
  ptr<Node> cond;
  ptr<MultiNodes> if_stmts;
  ptr<MultiNodes> else_stmts;  // optional requirements

  IfElse(const location& l, const ptr<Node>& c, const ptr<MultiNodes>& if_s)
      : Node(l), cond(c), if_stmts(if_s) {}
  IfElse(const location& l, const ptr<Node>& c, const ptr<MultiNodes>& if_s,
         const ptr<MultiNodes>& else_s)
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

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct ParallelBy : public Node, public TypeIDProvider<ParallelBy> {
  std::string biv;
  int bound;
  ptr<MultiNodes> statms;

  ParallelBy(const location& l, const std::string v, int b)
      : Node(l), biv(v), bound(b) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Parellelization: ";
    os << " index symbol: " << biv << ", bound [0, " << bound << ")";
    if (!statms)
      os << std::endl;
    else
      statms->Print(os, prefix + " ");
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

// `require_bind` parsing "idx_1 <-> idx_2"
struct RequireBind : public Node, public TypeIDProvider<RequireBind> {
  ptr<Node> lhs;
  ptr<Node> rhs;

  RequireBind(const location& l, const ptr<Node>& lhs, const ptr<Node>& rhs)
      : Node(l), lhs(lhs), rhs(rhs) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "`- ";
    lhs->Print(os);
    os << " bind-to ";
    rhs->Print(os);
    os << "\n";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct WithIn : public Node, public TypeIDProvider<WithIn> {
  ptr<Node> with;
  ptr<Node> in;
  ptr<MultiNodes> with_matchers;  // optional requirements

  WithIn(const location& l, const ptr<Node>& w, const ptr<Node>& i)
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

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct WithBlock : public Node, public TypeIDProvider<WithBlock> {
  ptr<MultiNodes> withins;
  ptr<MultiNodes> reqs;    // optional requirements
  ptr<MultiNodes> statms;  // may be empty

  explicit WithBlock(const location& l) : Node(l) {}

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

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct DMA : public Node, public TypeIDProvider<DMA> {
  std::string operation;
  ptr<Identifier> future;
  ptr<Node> from;
  ptr<Node> to;

  DMA(const location& l, const std::string& o, const ptr<Identifier>& r,
      const ptr<Node>& f, const ptr<Node>& t)
      : Node(l, MakeFutureType()), operation(o), future(r), from(f), to(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- DMA" << operation;
    os << "\n" << prefix << "  `- future: ";
    future->Print(os);
    os << "\n" << prefix << "  `- from: ";
    from->Print(os);
    os << "\n" << prefix << "  `- to: ";
    to->Print(os);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct ChunkAt : public Node, public TypeIDProvider<ChunkAt> {
  ptr<Node> data;
  ptr<MultiNodes> positions;

  ChunkAt(const location& l, const ptr<Node>& d, const ptr<MultiNodes>& p)
      : Node(l), data(d), positions(p) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    data->Print(os);
    os << ".ChunkAt(";
    positions->Print(os);
    os << ")";

    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Wait : public Node, public TypeIDProvider<Wait> {
  ptr<Node> target;

  Wait(const location& l, const ptr<Node>& t) : Node(l), target(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- WAIT: ";
    target->Print(os);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Return : public Node, public TypeIDProvider<Return> {
  ptr<Node> value = nullptr;

  Return(const location& l) : Node(l) {}
  Return(const location& l, const ptr<Node>& t) : Node(l), value(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Return: ";
    if (!value)
      os << "void";
    else
      value->Print(os);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Call : public Node, public TypeIDProvider<Call> {
  ptr<Node> function;
  ptr<Node> arguments;

  Call(const location& l, const ptr<Node>& f, const ptr<Node>& a)
      : Node(l), function(f), arguments(a) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Call: ";
    function->Print(os);
    os << "\n" << prefix << "  `- with arguements:";
    arguments->Print(os);
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct ForeachBlock : public Node, public TypeIDProvider<ForeachBlock> {
  ptr<MultiNodes> ivs;
  ptr<MultiNodes> statms;

  explicit ForeachBlock(const location& l, const ptr<MultiNodes>& i,
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

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct FunctionDecl : public Node, public TypeIDProvider<FunctionDecl> {
  std::string name;
  ptr<DataType> ret_type;
  ptr<ParamList> params;

  FunctionDecl(const location& l) : Node(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "Name: " << name;
    os << "\n" << prefix << "Return type: ";
    ret_type->Print(os);
    params->Print(os, prefix);
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct ChoreoFunction : public Node, public TypeIDProvider<ChoreoFunction> {
  std::string name;
  FunctionDecl f_decl;
  ptr<MultiNodes> statms;

  ChoreoFunction(const location& l) : Node(l), f_decl(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "ChoreoFunction";
    f_decl.Print(os, prefix + " `- ");
    if (statms) statms->Print(os, prefix + " ");
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct CppSourceCode : public Node, public TypeIDProvider<CppSourceCode> {
  std::string code;
  CppSourceCode(const location& l, const std::string& c) : Node(l), code(c) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << code;
    (void)prefix;
  }

  std::string GetCode() { return code; }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

// Top-level program structure
struct Program : public Node, public TypeIDProvider<Program> {
  std::vector<ptr<Node>> nodes;

  Program(const location& l) : Node(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    for (auto& node : nodes) node->Print(os, "");
    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

// Utility to generate shared_ptr<Node>
template <typename T, typename... Args>
ptr<T> Make(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

// Utility to check the type
template <typename T>
bool typeof(const Node *n) {
  return isa<T>(n->GetType().get());
}

}  // end of namespace AST

}  // end of namespace Choreo

#endif  // __CHOREO_AST_HPP__
