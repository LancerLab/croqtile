#ifndef __CHOREO_AST_HPP__
#define __CHOREO_AST_HPP__

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "aux.hpp"
#include "dmaconf.hpp"
#include "enums.hpp"
#include "location.hh"
#include "symtab.hpp"

extern Choreo::SymbolTable symtab;

namespace Choreo {
struct Visitor;

namespace AST {

// short hands
template <typename T>
using ptr = Choreo::ptr<T>;

// Utility to generate shared_ptr<Node>
template <typename T, typename... Args>
ptr<T> Make(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

//------------------------- AST Node Fundamentals ----------------------------//

class Identifier;
class DataType;

// interface class for all AST nodes
struct Node {
  location loc;
  ptr<Type> pty = MakeUnknownType();

  Node(const location& l, const ptr<Type>& p = MakeUnknownType())
      : loc(l), pty(p) {}

  virtual bool TypeUnknown() const { return isa<UnknownType>(pty.get()); }

  virtual void SetType(const ptr<Type>& t) { pty = t; }
  virtual const ptr<Type>& GetType() const { return pty; }
  virtual const location& LOC() const { return loc; }

  virtual ~Node() = default;

  virtual std::string getRefName() const { return ""; }

  virtual void Print(std::ostream& os,
                     const std::string& prefix = {}) const = 0;

  virtual void PrintType(std::ostream& os,
                         const std::string& prefix = {}) const {
    os << prefix;
    pty->Print(os);
  };

  virtual std::string EmitTo(
      const std::string& prefix = {},
      Choreo::Target target = Choreo::Target::Factor) const {
    std::ostringstream _os;
    _os << prefix;
    _os << pty->EmitTo(target);
    return _os.str();
  };

  virtual void accept(Visitor&) = 0;

  // for runtime type disambiguition
  virtual const std::string TypeNameString() = 0;
  virtual uint64_t RuntimeID() const { return 0ULL; }
  static uint64_t TypeID() { return 0ULL; }
};

// utility functions
template <typename T>
bool typeof(const Node* n) {
  return isa<T>(n->GetType().get());
}
template <typename T>
bool typeof(const ptr<Node>& n) {
  return isa<T>(n->GetType().get());
}

inline std::string STR(const AST::Node& n) {
  std::ostringstream oss;
  n.Print(oss);
  return oss.str();
}
inline std::string STR(const std::shared_ptr<AST::Node>& n) { return STR(*n); }
inline std::string TYPE_STR(const AST::Node& n) { return STR(*n.GetType()); }
inline std::string TYPE_STR(const std::shared_ptr<AST::Node>& n) {
  return STR(*n->GetType());
}

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

  size_t Count() const { return values.size(); }

  void SetDelimiter(const std::string& d) { delimiter = d; }

  std::vector<ptr<Node>> AllSubs() { return values; }

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

  void accept(Visitor& visitor) override;

  __UDT_TYPE_INFO__
};

struct MultiValues : public Node, public TypeIDProvider<MultiValues> {
  std::vector<ptr<Node>> values;
  std::string delimiter;

  explicit MultiValues(const location& l, std::string d = "")
      : Node(l), delimiter(d){};

  void Append(const ptr<Node>& m) {
    assert(m != nullptr && "Unexpected: null pointer.");
    values.push_back(m);
  }

  size_t Count() const { return values.size(); }

  void SetDelimiter(const std::string& d) { delimiter = d; }

  ptr<Node> ValueAt(const size_t idx) const {
    assert(idx < this->Count() &&
           "Out-of-bound error when querying MultiValues\n");
    return values[idx];
  }

  ptr<Node> operator[](const size_t idx) const { return ValueAt(idx); }

  const std::vector<ptr<Node>>& AllValues() const { return values; }

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
  Boolean(const location& l, std::string v)
      : Node(l, MakeBooleanType()), value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << value;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct IntLiteral : public Node, public TypeIDProvider<IntLiteral> {
  int value;
  IntLiteral(const location& l, int v = GetUnKnownInteger())
      : Node(l, MakeIntegerType()), value(v) {}

  int Val() const { return value; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (IsUnKnownInteger(value))
      os << prefix << "?";
    else
      os << prefix << value;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Expr : public Node, public TypeIDProvider<Expr> {
  // Different expression type
  enum Form { Unary, Binary, Ternary, Reference };

  std::string op;

 private:
  ptr<Expr> value_c = nullptr;
  ptr<Node> value_l = nullptr;
  ptr<Node> value_r = nullptr;
  Form t;

 public:
  const ptr<Node>& GetR() const { return value_r; }
  const ptr<Node>& GetL() const { return value_l; }
  const ptr<Expr>& GetC() const { return value_c; }
  Form GetForm() const { return t; }
  void SetR(const ptr<Node>& r) {
    assert(r);
    value_r = r;
  }
  void SetL(const ptr<Node>& l) {
    assert(l);
    assert((t == Binary) || (t == Ternary));
    value_l = l;
  }
  void SetC(const ptr<Expr>& c) {
    assert(c);
    assert((t == Ternary));
    value_c = c;
  }

 public:
  Shape s;  // to pass information between shape inference & type inference

  explicit Expr(const location& l, const ptr<Node>& v)
      : Node(l), op("ref"), value_r(v), t(Reference) {
    assert(value_r);
  }
  explicit Expr(const location& l, const std::string& o, const ptr<Node>& v2)
      : Node(l), op(o), value_r(v2), t(Unary) {
    assert(value_r);
  }
  explicit Expr(const location& l, const std::string& o, const ptr<Node>& v1,
                const ptr<Node>& v2)
      : Node(l), op(o), value_l(v1), value_r(v2), t(Binary) {
    assert(value_l);
    assert(value_r);
  }
  explicit Expr(const location& l, const std::string& o, const ptr<Expr>& c,
                const ptr<Node>& v1, const ptr<Node>& v2)
      : Node(l), op(o), value_c(c), value_l(v1), value_r(v2), t(Ternary) {
    assert(value_c);
    assert(value_l);
    assert(value_r);
  }

  ptr<Node> GetReference() {
    if (t == Reference) return value_r;
    return nullptr;
  }

  Identifier* GetSymbol() {
    if (t != Reference) return nullptr;
    return dyn_cast<Identifier>(value_r);
  }

  bool IsUnary() const { return t == Unary; }
  bool IsBinary() const { return t == Binary; }
  bool IsTernary() const { return t == Ternary; }
  bool IsReference() const { return t == Reference; }

  bool IsArith() const {
    if (!IsBinary()) return false;
    if ((op == "+") || (op == "-") || (op == "*") || (op == "/") ||
        (op == "%") || (op == "cdiv"))
      return true;
    return false;
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (t == Reference) {
      value_r->Print(os, prefix);
      return;
    }

    assert(op.size() > 0 && "must have an operand.");

    if (op == "dimof") {
      value_l->Print(os, prefix);
      value_r->Print(os);
      return;
    }

    os << " (";
    switch (t) {
      case Unary:
        os << op << " ";
        value_r->Print(os);
        break;
      case Binary:
        value_l->Print(os);
        os << " " << op << " ";
        value_r->Print(os);
        break;
      case Ternary:
        value_c->Print(os);
        os << " ? ";
        value_l->Print(os);
        os << " : ";
        value_r->Print(os);
        break;
      default:
        choreo_unreachable("unhandled expression type.");
        break;
    }
    os << ") ";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

// Represents both dimensions and s like {3, 4, 5} or {1, 2, 1}
struct MultiDimSpans : public Node, public TypeIDProvider<MultiDimSpans> {
  std::string ref_name;            // syntax suger, could be empty
  ptr<Node> list;                  // null if the span is dynamically valued
  size_t rank = GetInvalidRank();  // dynamic value with known dimension count

  // If the mdspan is known
  explicit MultiDimSpans(const location& l, const std::string& n,
                         const ptr<Node>& lst)
      : Node(l, MakeUninitMDSpanType()),
        ref_name(n),
        list(lst),
        rank(GetInvalidRank()) {
    assert(list && "Unexpected: span list is not provided");
  }

  // set both the mdspan and dim count
  explicit MultiDimSpans(const location& l, const std::string& n,
                         const ptr<Node>& lst, size_t dc)
      : Node(l, MakeRankedMDSpanType(dc)), ref_name(n), list(lst), rank(dc) {
    assert(list && "Unexpected: span list is not provided");
    // check the consistent between rank and span list in semantic time
  }

  // mdspan is unknown - for parameter passing
  explicit MultiDimSpans(const location& l, const std::string& n, size_t c)
      : Node(l, MakeRankedMDSpanType(c)), ref_name(n), list(nullptr), rank(c) {
    assert(IsValidRank(rank) && "Invalid dimensions.");
  }

  explicit MultiDimSpans(const location& l, const std::string& n,
                         const ptr<MDSpanType>& pty)
      : Node(l, pty), ref_name(n), list(nullptr), rank(pty->Dims()) {}

  bool HasValidRank() const { return IsValidRank(rank); }
  size_t Rank() const { return rank; }
  void SetRank(size_t n) { rank = n; }

  void SetTypeDetail(const Shape& s) {
    assert(typeof<MDSpanType>(this) && "Incorrect type for mdspan.");
    cast<MDSpanType>(GetType())->SetShape(s);
  }

  const Shape GetTypeDetail() {
    assert(typeof<MDSpanType>(this) && "Incorrect type for mdspan.");
    return cast<MDSpanType>(GetType())->GetShape();
  }

  Shape MakeValueList() {
    if (!list) {
      // dynamically valued
      return {Shape(rank)};
    } else
      return {Shape(0) /*TODO: make Type from the list*/};
  }

  std::string getRefName() const override {
    std::ostringstream oss;
    if (list) {
      list->Print(oss, "");
    }
    return oss.str();
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (!list)
      os << "<" << rank << ">";
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
  const ptr<Node> init_expr;       // associated init_expr
  size_t rank = GetInvalidRank();  // rank annotation only

  explicit NamedTypeDecl(const location& l, const std::string& n,
                         const ptr<Node>& v, const std::string& d = "-")
      : Node(l), name_str(n), init_str(d), init_expr(v) {
    assert(name_str.size() > 0 && "Invalid name string.");
    assert(init_expr && "Invalid value.");
  }

  explicit NamedTypeDecl(const location& l, const std::string& n,
                         const ptr<Node>& v, size_t r,
                         const std::string& d = "-")
      : Node(l), name_str(n), init_str(d), init_expr(v), rank(r) {
    assert(name_str.size() > 0 && "Invalid name string.");
    assert(init_expr && "Invalid value.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Type Decl: ";
    os << name_str << " " << init_str << " " << STR(*init_expr);
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
    os << STR(st);
  }

  Storage Get() const { return st; }
  void Set(Storage s) { st = s; }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

// Represents declarations like: ituple t = {3, 4, 5};
struct IntTuple : public Node, public TypeIDProvider<IntTuple> {
  std::string ref_name;  // could be anonymous
  ptr<MultiValues> vlist;

  explicit IntTuple(const location& l, const std::string& n,
                    ptr<MultiValues> lst)
      : Node(l, MakeUninitITupleType()), ref_name(n), vlist(lst) {
    vlist->SetDelimiter(", ");
  }

  const ptr<MultiValues>& GetValues() const { return vlist; }
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (ref_name.size() > 0) os << ref_name << " ";
    os << "{" << STR(*vlist) << "}";
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

// A data type could either be
//
// 1. A scalar type, including `int`, `bool`.
// 2. A composited type, including the fundamental type and the mdspan type.
// 3. An 'ituple' type.
//
struct DataType : public Node, public TypeIDProvider<DataType> {
  BaseType base_type;
  size_t rank = GetInvalidRank();  // for annotated ituple only
  ptr<Node> mdspan_type = nullptr;

 public:
  explicit DataType(const location& l, BaseType t)
      : Node(l), base_type(t), mdspan_type(nullptr) {
    InitSemaType();
  }

  explicit DataType(const location& l, BaseType bt, const ptr<Node>& st)
      : Node(l), base_type(bt), mdspan_type(st) {
    assert(bt != BaseType::ITUPLE && "Unexpected type!");
    assert(bt != BaseType::INT && "Unexpected type!");
    assert(bt != BaseType::BOOL && "Unexpected type!");
    InitSemaType();
  }

  explicit DataType(const location& l, BaseType bt, int r)
      : Node(l), base_type(bt), rank(r) {
    assert(bt == BaseType::ITUPLE && "Unexpected type!");
    InitSemaType();
  }

  BaseType getBaseType() const { return base_type; }
  FundamentalType getFundamentalType() const {
    return (FundamentalType)base_type;
  }
  Node* getPartialType() const { return mdspan_type.get(); }

  bool IsVoid() const { return base_type == BaseType::VOID; }
  bool IsUnknown() const { return base_type == BaseType::UNKNOWN; }
  bool isScalar() const {
    return (base_type == BaseType::INT) || (base_type == BaseType::BOOL);
  }
  bool isITuple() const { return base_type == BaseType::ITUPLE; }
  bool isSpanned() const { return (bool)mdspan_type; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << STR(base_type);
    if (isSpanned()) {
      os << " ";
      mdspan_type->Print(os);
    }
  }

  void accept(Visitor&) override;

 private:
  ptr<Type> InitSemaType() {
    switch (base_type) {
      case BaseType::INT:
        SetType(MakeIntegerType());
        break;
      case BaseType::BOOL:
        SetType(MakeBooleanType());
        break;
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
        SetType(MakeSpannedType(base_type,
                                GenUninitShape()));  // need type inference
        break;
      case BaseType::ITUPLE:
        if (!IsValidRank(rank))
          SetType(MakeUninitITupleType());  // type inference to deduce the dim
                                            // count
        else
          SetType(MakeITupleType(rank));
        break;
      case BaseType::UNKNOWN:
        SetType(MakeUnknownType());  // need type inference
        break;
      case BaseType::VOID:
        SetType(MakeVoidType());
        break;
      default:
        choreo_unreachable("Unexpected BaseType.");
        break;
    }
    return nullptr;
  }

 public:
  __UDT_TYPE_INFO__
};

struct NamedVariableDecl : public Node,
                           public TypeIDProvider<NamedVariableDecl> {
  const std::string name_str;
  const std::string init_str;
  const ptr<Memory> mem = nullptr;  // storage location
  const ptr<DataType> type = nullptr;
  const ptr<Node> init_expr = nullptr;  // associated initializer

  explicit NamedVariableDecl(const location& l, const std::string& n,
                             const ptr<DataType>& t,
                             const ptr<Memory>& s = nullptr,
                             const ptr<Node>& v = nullptr,
                             const std::string& d = "=")
      : Node(l), name_str(n), init_str(d), mem(s), type(t), init_expr(v) {
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
    if (init_expr) {
      os << " " << init_str << " ";
      init_expr->Print(os);
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
  Attribute attr = ATT_NONE;

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
  ptr<MultiValues> id_list;
  ptr<MultiValues> iv_list;
  ptr<MultiNodes> stmts;

  ParallelBy(const location& l, const std::string v, int b)
      : Node(l), biv(v), bound(b) {}

  ParallelBy(const location &l, const ptr<MultiValues> &id_l,
             const ptr<MultiValues> &iv_l, const ptr<MultiNodes> &ss)
      : Node(l), id_list(id_l), iv_list(iv_l), stmts(ss) {
    auto id = id_l->ValueAt(0);
    auto *identifier = dyn_cast<Identifier>(id);
    assert(identifier != nullptr);
    auto iv = iv_l->ValueAt(0);
    auto *num = dyn_cast<IntLiteral>(iv);
    assert(num != nullptr);
    biv = identifier->name;
    bound = num->value;
    symtab.AddSymbol(biv, MakeBoundedIntegerType(bound));
    if (id_l->Count() > 1) 
      stmts = ConstructParallelByRecursively(stmts, 1, id_list, iv_list);
  }

  ptr<MultiNodes> ConstructParallelByRecursively(const ptr<MultiNodes> &ss,
                                                 size_t idx,
                                                 const ptr<MultiValues> &id_l,
                                                 const ptr<MultiValues> &iv_l) {
    auto id = id_l->ValueAt(idx);
    auto *identifier = dyn_cast<Identifier>(id);
    assert(identifier != nullptr);
    auto iv = iv_l->ValueAt(idx);
    auto *num = dyn_cast<IntLiteral>(iv);
    assert(num != nullptr);
    symtab.AddSymbol(identifier->name, MakeBoundedIntegerType(num->value));
    auto pb = Make<ParallelBy>(id->loc, identifier->name, num->value);
    if (idx == id_l->Count() - 1)
      pb->stmts = ss;
    else
      pb->stmts = ConstructParallelByRecursively(ss, idx + 1, id_l, iv_l);
    auto mn = Make<MultiNodes>(id->loc);
    mn->Append(pb);
    return mn;
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Parallelization: ";
    os << " index symbol: " << biv << ", bound [0, " << bound << ")";
    if (!stmts)
      os << std::endl;
    else
      stmts->Print(os, prefix + " ");
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

// `require_bind` parsing "idx_1 <-> idx_2"
struct WhereBind : public Node, public TypeIDProvider<WhereBind> {
  ptr<Node> lhs;
  ptr<Node> rhs;

  WhereBind(const location& l, const ptr<Node>& lhs, const ptr<Node>& rhs)
      : Node(l), lhs(lhs), rhs(rhs) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "`- " << STR(*lhs) << " bind-to " << STR(*rhs) << "\n";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct WithIn : public Node, public TypeIDProvider<WithIn> {
  ptr<Identifier> with;  // either with or with_matcher
  ptr<Node> in;
  ptr<MultiValues> with_matchers;

  WithIn(const location& l, const ptr<Identifier>& w, const ptr<Node>& i)
      : Node(l), with(w), in(i), with_matchers(nullptr) {}

  WithIn(const location& l, const ptr<Node>& i, const ptr<MultiValues>& m)
      : Node(l), with(), in(i), with_matchers(m) {}

  WithIn(const location& l, const ptr<Identifier>& w, const ptr<Node>& i,
         ptr<MultiValues> m)
      : Node(l), with(w), in(i), with_matchers(m) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "`- ";
    if (with) os << with->name;
    if (with_matchers) {
      os << " = {";
      with_matchers->InlinePrint(os);
      os << "}";
    }
    os << " in " << STR(*in) << "\n";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct WithBlock : public Node, public TypeIDProvider<WithBlock> {
  ptr<MultiNodes> withins;
  ptr<MultiNodes> reqs;   // optional requirements
  ptr<MultiNodes> stmts;  // may be empty

  explicit WithBlock(const location& l) : Node(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- With Block:\n";
    os << prefix << "  (within constraints)\n";
    withins->Print(os, prefix + "  ");
    if (reqs) {
      os << prefix << "  (where clause)\n";
      reqs->Print(os, prefix + "  ");
    }
    if (stmts) {
      if (stmts->values.size() == 0) {
        os << prefix << "  (with empty statements)\n";
        return;
      }
      os << prefix << "  (with statements)";
      stmts->Print(os, prefix + "  ");
    }
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct ChunkAt : public Node, public TypeIDProvider<ChunkAt> {
  ptr<Identifier> data;
  ptr<MultiValues> positions = nullptr;

  ChunkAt(const location& l, const ptr<Identifier>& d,
          const ptr<MultiValues>& p = nullptr)
      : Node(l), data(d), positions(p) {}

  std::string RefSymbol() const {
    assert(data && "ref data is not set.");
    return data->name;
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (!positions) {
      os << STR(data);
      return;
    }

    os << STR(data) << ".ChunkAt(" << STR(positions) << ")";

    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Select : public Node, public TypeIDProvider<Select> {
  std::string future;
  ptr<Expr> select_factor = nullptr;
  int bound;
  ptr<MultiValues> span_expr_list = nullptr;
  bool inDMA = false;

  Select(const location& l, const ptr<Expr>& sf,
         const ptr<MultiValues>& list = nullptr)
      : Node(l), select_factor(sf), span_expr_list(list) {}

  // TODO(wsj)
  // x = select(IntLiteral, a, b, c)

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "select(" << STR(select_factor) << ", " << STR(span_expr_list) << ")";
    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};


struct DMA : public Node, public TypeIDProvider<DMA> {
  std::string operation;
  std::string future;
  bool async;
  // if this DMA is chained with other DMA in pipeline mode
  bool chained;
  // SYMBOL string of its chained DMA B, direction is B->A
  std::string chain_from;
  // SYMBOL string of its chained DMA B, direction is A->B
  std::string chain_to;
  ptr<Node> from;
  ptr<Node> to;
  ptr<DMAConfig> config;

  explicit DMA(const location& l, const std::string& o, const std::string& r,
               const ptr<Node>& f, const ptr<Node>& t, bool a,
               const ptr<DMAConfig>& c = nullptr)
      : Node(l, MakeDummyFutureType(a)),
        operation(o),
        future(r),
        async(a),
        from(f),
        to(t),
        config(c) {
          chained = false;
          chain_to = "";
          chain_from = "";
          if (auto tptr = dyn_cast<AST::Select>(t)) {
            tptr->inDMA = true;
            tptr->future = future + "_buffer";
          }
        }

  explicit DMA(const location& l, const std::string& o, const std::string& r,
               const std::string& chained_from,
               const ptr<Node>& f, const ptr<Node>& t, bool a,
               const ptr<DMAConfig>& c = nullptr)
      : Node(l, MakeDummyFutureType(a)),
        operation(o),
        future(r),
        async(a),
        from(f),
        to(t),
        config(c){
          chained = true;
          chain_from = chained_from;
          if (auto tptr = dyn_cast<AST::Select>(t)) {
            tptr->inDMA = true;
            tptr->future = future + "_buffer";
          }
        }

  std::string FromSymbol() const { return cast<ChunkAt>(from)->RefSymbol(); }

  std::string ToSymbol() const {
    if (auto tochunk = dyn_cast<ChunkAt>(to)) return tochunk->data->name;
    return "";
  }

  void SetConfig(const ptr<DMAConfig>& cfg) { config = cfg; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- DMA" << operation << ((async) ? ".async" : "");
    if (config) os << "\n" << prefix << "  `- config: " << STR(*config);
    if (!future.empty()) os << "\n" << prefix << "  `- future: " << future;
    os << "\n" << prefix << "  `- from: " << STR(from);
    os << "\n" << prefix << "  `- to: " << STR(to);
    if (chained) {
      if (chain_to != "")
        os << "\n" << prefix << "  `- chained to: " << chain_to;
      if (chain_from != "")
        os << "\n" << prefix << "  `- chained from: " << chain_from;
    }
  }

  std::string SourceString() {
    return future + " = dma" + operation + " " + STR(*from) + " => " + STR(*to);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct Wait : public Node, public TypeIDProvider<Wait> {
  ptr<MultiValues> targets;

  Wait(const location& l, const ptr<MultiValues>& t) : Node(l), targets(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- WAIT: " << AST::STR(*targets);
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
  ptr<MultiValues> arguments;

  Call(const location& l, const ptr<Node>& f, const ptr<MultiValues>& a)
      : Node(l), function(f), arguments(a) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Call: " << STR(*function);
    os << "\n" << prefix << "  `- with arguements: " << STR(*arguments);
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct ForeachBlock : public Node, public TypeIDProvider<ForeachBlock> {
  ptr<MultiValues> ivs;
  ptr<MultiNodes> stmts;
  // for now, lb_offset >= 0, ub_offset <= 0
  int lb_offset = 0;
  int ub_offset = 0;

  explicit ForeachBlock(const location& l, const ptr<MultiValues>& i,
                        const ptr<MultiNodes>& s)
      : Node(l), ivs(i), stmts(s) {
    assert(i != nullptr && "missing iteration variables for the statement.");
  }

  explicit ForeachBlock(const location& l, const ptr<Node>& i,
                        const ptr<MultiNodes>& s, const int lb_o, const int ub_o)
      : Node(l), stmts(s), lb_offset(lb_o), ub_offset(-1 * ub_o) {
    assert(i != nullptr && "missing iteration variables for the statement.");
    ivs = Make<MultiValues>(i->LOC());
    ivs->Append(i);
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Foreach Block:";
    os << "\n" << prefix << " `- Iteration variables: ";
    ivs->Print(os);
    if (ivs->Count() == 1) {
      if (lb_offset != 0)
        os << "\n" << prefix << "      (lower bound offset is " << lb_offset << ")";
      if (ub_offset != 0)
        os << "\n" << prefix << "      (upper bound offset is " << ub_offset << ")";
    }
    if (stmts) {
      stmts->Print(os, prefix + " ");
    }
  }

  ptr<MultiValues> getIterationVars() const { return ivs; }

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
  ptr<MultiNodes> stmts;

  ChoreoFunction(const location& l) : Node(l), f_decl(l) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "ChoreoFunction";
    f_decl.Print(os, prefix + " `- ");
    if (stmts) stmts->Print(os, prefix + " ");
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__
};

struct CppSourceCode : public Node, public TypeIDProvider<CppSourceCode> {
  std::string code;
  bool host;  // host or kernel
  CppSourceCode(const location& l, const std::string& c, bool h = true)
      : Node(l), code(c), host(h) {}
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

}  // end of namespace AST

}  // end of namespace Choreo

#endif  // __CHOREO_AST_HPP__
