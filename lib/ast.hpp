#ifndef __CHOREO_AST_HPP__
#define __CHOREO_AST_HPP__

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "aux.hpp"
#include "context.hpp"
#include "dmaconf.hpp"
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
  std::string note;

  Node(const location& l, const ptr<Type>& p = MakeUnknownType())
      : loc(l), pty(p) {}

  virtual bool TypeUnknown() const { return isa<UnknownType>(pty.get()); }

  virtual void SetType(const ptr<Type>& t) { pty = t; }
  virtual const ptr<Type>& GetType() const { return pty; }
  virtual const location& LOC() const { return loc; }

  virtual ~Node() = default;

  virtual std::string getRefName() const { return ""; }

  virtual const std::string& GetNote() const { return note; }
  virtual void SetNote(const std::string& n) { note = n; }

  virtual void Print(std::ostream& os,
                     const std::string& prefix = {}) const = 0;

  virtual void PrintType(std::ostream& os,
                         const std::string& prefix = {}) const {
    os << prefix;
    pty->Print(os);
  };

  virtual void accept(Visitor&) = 0;

  // for runtime type disambiguition
  __UDT_TYPE_INFO_BASE__(node)
};

// utility functions
template <typename T>
bool typeof(const Node* n) {
  return isa<T>(n->GetType());
}
template <typename T>
bool typeof(const ptr<Node>& n) {
  return isa<T>(n->GetType());
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
      : Node(l), delimiter(d) {};

  void Append(const ptr<Node>& m) {
    assert(m != nullptr && "Unexpected: null pointer.");
    values.push_back(m);
  }

  size_t Count() const { return values.size(); }

  void SetDelimiter(const std::string& d) { delimiter = d; }

  std::vector<ptr<Node>> AllSubs() { return values; }

  // retrieve the index if the element is inside the MultiNodes
  int GetIndex(Node* n) const {
    for (size_t i = 0; i < values.size(); ++i) {
      if (values[i].get() == n) return i;
    }
    return -1;
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

  void accept(Visitor& visitor) override;

  __UDT_TYPE_INFO__(Node, MultiNodes)
};

struct MultiValues : public Node, public TypeIDProvider<MultiValues> {
  std::vector<ptr<Node>> values;
  std::string delimiter;

  explicit MultiValues(const location& l, std::string d = "")
      : Node(l), delimiter(d) {}

  template <typename... T>
  explicit MultiValues(const location& l, std::string d, T... args)
      : Node(l), delimiter(d) {
    (Append(args), ...);
  }

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

  __UDT_TYPE_INFO__(Node, MultiValues)
};

struct Boolean : public Node, public TypeIDProvider<Boolean> {
  std::string value;
  Boolean(const location& l, std::string v)
      : Node(l, MakeBooleanType()), value(v) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << value;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Boolean)
};

struct IntLiteral : public Node, public TypeIDProvider<IntLiteral> {
  int value;
  IntLiteral(const location& l, int v = GetUnKnownInteger())
      : Node(l, MakeIntegerType()), value(v) {}

  // allow copy construction
  explicit IntLiteral(const IntLiteral& il) : IntLiteral(il.LOC(), il.value) {}

  int Val() const { return value; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (IsUnKnownInteger(value))
      os << prefix << "?";
    else
      os << prefix << value;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, IntLiteral)
};

struct Expr : public Node, public TypeIDProvider<Expr> {
  // Different expression type
  enum Form { Unary, Binary, Ternary, Reference };

  std::string op;
  OptimizedValues opt_vals;

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
  void SetForm(const Form& form) {
    // the form must be set after operand
    if (form == Reference)
      assert(!isa<Expr>(value_r));
    else if (form == Unary)
      assert(value_r);
    else if (form == Binary)
      assert(value_r && value_l);
    else if (form == Ternary)
      assert(value_r && value_l && value_c);
    t = form;
  }
  void SetR(const ptr<Node>& r) {
    assert(r);
    value_r = r;
  }
  void SetL(const ptr<Node>& l) {
    assert(l);
    value_l = l;
  }
  void SetC(const ptr<Expr>& c) {
    assert(c);
    value_c = c;
  }

public:
  Shape s; // to pass information between shape inference & type inference

  explicit Expr(const location& l, const ptr<Node>& v)
      : Node(l), op("ref"), value_r(v), t(Reference) {
    assert(value_r && "null node is provided.");
    assert(!isa<Expr>(v) && "can not reference an expression.");
  }
  explicit Expr(const location& l, const std::string& o, const ptr<Node>& v2)
      : Node(l), op(o), value_r(v2), t(Unary) {
    assert(value_r && "null node is provided.");
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

  // copy constructor for reconstructing expr in SymReplace pass
  // TODO(wsj): loc?
  explicit Expr(const Expr& e) : Node(e.LOC()) { OverWrite(e); }

  void OverWrite(const Expr& e) {
    if (e.IsReference()) {
      op = "ref";
      SetR(e.GetR());
      SetForm(Reference);
    } else if (e.IsUnary()) {
      op = e.op;
      SetR(e.GetR());
      SetForm(Unary);
    } else if (e.IsBinary()) {
      op = e.op;
      SetL(e.GetL());
      SetR(e.GetR());
      SetForm(Binary);
    } else if (e.IsTernary()) {
      op = e.op;
      SetC(e.GetC());
      SetL(e.GetL());
      SetR(e.GetR());
      SetForm(Ternary);
    }
  }

  ptr<Node> GetReference() {
    if (t == Reference) return value_r;
    return nullptr;
  }

  ptr<Identifier> GetSymbol() {
    if (t != Reference) return nullptr;
    return dyn_cast<Identifier>(value_r);
  }

  ptr<IntLiteral> GetInt() {
    if (t != Reference) return nullptr;
    return dyn_cast<IntLiteral>(value_r);
  }

  bool IsUnary() const { return t == Unary; }
  bool IsBinary() const { return t == Binary; }
  bool IsTernary() const { return t == Ternary; }
  bool IsReference() const { return t == Reference; }

  bool IsArith() const {
    if (!IsBinary()) return false;
    if ((op == "+") || (op == "-") || (op == "*") || (op == "/") ||
        (op == "%") || (op == "cdiv") || (op == "#"))
      return true;
    return false;
  }

  bool IsLogical() const {
    if ((op == "||") || (op == "&&") || (op == "!") || (op == "<") ||
        (op == "<=") || (op == "==") || (op == ">") || (op == ">=") ||
        (op == "!="))
      return true;
    return false;
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (t == Reference) {
      value_r->Print(os, prefix);
      return;
    }

    assert(op.size() > 0 && "must have an operand.");

    if (op == "dimof" || op == "getith") {
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
    default: choreo_unreachable("unhandled expression type."); break;
    }
    os << ") ";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Expr)
};

// Represents both dimensions and s like {3, 4, 5} or {1, 2, 1}
struct MultiDimSpans : public Node, public TypeIDProvider<MultiDimSpans> {
  std::string ref_name;           // syntax suger, could be empty
  ptr<Node> list;                 // null if the span is dynamically valued
  size_t rank = GetInvalidRank(); // dynamic value with known dimension count

  // If the mdspan is known
  explicit MultiDimSpans(const location& l, const std::string& n,
                         const ptr<Node>& lst)
      : Node(l, MakeUninitMDSpanType()), ref_name(n), list(lst),
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
    if (list) { list->Print(oss, ""); }
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

  __UDT_TYPE_INFO__(Node, MultiDimSpans)
};

struct SpanAs : public Node, public TypeIDProvider<SpanAs> {
  ptr<Identifier> id = nullptr;
  ptr<Identifier> nid = nullptr;
  ptr<MultiValues> list = nullptr;

  explicit SpanAs(const location& l, const ptr<Identifier>& n,
                  const ptr<Identifier>& nn, const ptr<MultiValues>& lst)
      : Node(l, MakeUninitMDSpanType()), id(n), nid(nn), list(lst) {
    assert(list && "Unexpected: span list is not provided");
  }

  explicit SpanAs(const location& l, const ptr<Identifier>& n,
                  const ptr<MultiValues>& lst)
      : SpanAs(l, n, Make<Identifier>(l), lst) {
    assert(list && "Unexpected: span list is not provided");
  }

  // allow copy construction
  explicit SpanAs(const SpanAs& sa) : SpanAs(sa.LOC(), sa.id, sa.nid, sa.list) {
    assert(list && "Unexpected: span list is not provided");
  }

  void SetTypeDetail(const Shape& s) {
    assert(typeof<SpannedType>(this) && "Incorrect type for mdspan.");
    cast<MDSpanType>(GetType())->SetShape(s);
  }

  const Shape GetTypeDetail() {
    assert(typeof<SpannedType>(this) && "Incorrect type for mdspan.");
    return cast<MDSpanType>(GetType())->GetShape();
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    assert(id && "no original span is specified.");
    assert(nid && "no new span is specified.");
    assert(list && "no span_as is specified.");

    os << PSTR(id) << ".span_as[";
    list->Print(os, " ");
    os << " ]";

    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, SpanAs)
};

struct NamedTypeDecl : public Node, public TypeIDProvider<NamedTypeDecl> {
  const std::string name_str;
  const std::string init_str;
  const ptr<Node> init_expr;      // associated init_expr
  size_t rank = GetInvalidRank(); // rank annotation only

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

  __UDT_TYPE_INFO__(Node, NamedTypeDecl)
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

  __UDT_TYPE_INFO__(Node, Memory)
};

// Represents declarations like: ituple t = {3, 4, 5};
struct IntTuple : public Node, public TypeIDProvider<IntTuple> {
  std::string ref_name; // could be anonymous
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

  __UDT_TYPE_INFO__(Node, IntTuple)
};

struct Assignment : public Node, public TypeIDProvider<Assignment> {
  std::string name;
  ptr<Node> value;

  explicit Assignment(const location& l, const std::string& n,
                      const ptr<Node>& v)
      : Node(l), name(n), value(v) {
    assert(n.size() > 0 && "invalid assignment to the un-named value.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Assign: " << name << " = " << PSTR(value);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Assignment)
};

struct IntIndex : public Node, public TypeIDProvider<IntIndex> {
  ptr<Node> value;
  explicit IntIndex(const location& l, const ptr<Node>& v)
      : Node(l), value(v) {}

  char lb = '(';
  char rb = ')';

  // TODO(wsj): loc?
  explicit IntIndex(const IntIndex& ii) : Node(ii.LOC()), value(ii.value) {}

  void UseBracket() {
    lb = '[';
    rb = ']';
  }

  bool IsNegative() const {
    if (auto il = dyn_cast<IntLiteral>(value)) return il->Val() < 0;
    return false;
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << lb << STR(value) << rb;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, IntIndex)
};

// A data type could either be
//
// 1. A scalar type, including `int`, `bool`.
// 2. A composited type, including the fundamental type and the mdspan type.
// 3. An 'ituple' type.
//
struct DataType : public Node, public TypeIDProvider<DataType> {
  BaseType base_type;
  size_t rank = GetInvalidRank(); // for annotated ituple only
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
    case BaseType::INT: SetType(MakeIntegerType()); break;
    case BaseType::BOOL: SetType(MakeBooleanType()); break;
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
                              GenUninitShape())); // need type inference
      break;
    case BaseType::ITUPLE:
      if (!IsValidRank(rank))
        SetType(MakeUninitITupleType()); // type inference to deduce the dim
                                         // count
      else
        SetType(MakeITupleType(rank));
      break;
    case BaseType::UNKNOWN:
      SetType(MakeUnknownType()); // need type inference
      break;
    case BaseType::VOID: SetType(MakeVoidType()); break;
    default: choreo_unreachable("Unexpected BaseType."); break;
    }
    return nullptr;
  }

public:
  __UDT_TYPE_INFO__(Node, DataType)
};

struct NamedVariableDecl : public Node,
                           public TypeIDProvider<NamedVariableDecl> {
  const std::string name_str;
  const std::string init_str;
  const ptr<Memory> mem = nullptr;            // storage location
  ptr<DataType> type = nullptr;               // type annotation
  const ptr<Node> init_expr = nullptr;        // associated initializer
  const ptr<IntLiteral> init_value = nullptr; // associated initial value

  explicit NamedVariableDecl(const location& l, const std::string& n,
                             const ptr<DataType>& t = nullptr,
                             const ptr<Memory>& s = nullptr,
                             const ptr<Node>& i = nullptr,
                             const ptr<IntLiteral>& v = nullptr,
                             const std::string& d = "=")
      : Node(l), name_str(n), init_str(d), mem(s), type(t), init_expr(i),
        init_value(v) {

    if (init_expr)
      assert(!init_value && "initial value can not be set when initialization "
                            "expression is specified.");
    else if (init_value)
      assert(!init_expr && "initialization expression can not be set when init "
                           "value is specified.");

    assert(name_str.size() > 0 && "Invalid name string.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Var Decl (";
    if (type) type->Print(os);
    if (mem) {
      os << ", ";
      mem->Print(os);
    }
    os << "): " << name_str;
    if (init_expr)
      os << " " << init_str << " " << PSTR(init_expr);
    else if (init_value)
      os << " " << init_str << " {" << PSTR(init_value) << "}";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, NamedVariableDecl)
};

struct Identifier : public Node, public TypeIDProvider<Identifier> {
  std::string name;
  Identifier(const location& l,
             const std::string& n = SymbolTable::GetAnonName())
      : Node(l), name(n) {}
  Identifier(const Identifier& id) : Node(id.LOC()), name(id.name) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << name;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Identifier)
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

  __UDT_TYPE_INFO__(Node, Parameter)
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

  __UDT_TYPE_INFO__(Node, ParamList)
};

struct IfElse : public Node, public TypeIDProvider<IfElse> {
  ptr<Node> cond;
  ptr<MultiNodes> if_stmts;
  ptr<MultiNodes> else_stmts; // optional requirements

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

  __UDT_TYPE_INFO__(Node, IfElse)
};

struct ParallelBy : public Node, public TypeIDProvider<ParallelBy> {
  std::string biv;
  int bound;
  ptr<MultiValues> id_list;
  ptr<MultiValues> iv_list;
  ptr<MultiNodes> stmts;

  ParallelBy(const location& l, const std::string v, int b)
      : Node(l), biv(v), bound(b) {}

  ParallelBy(const location& l, const ptr<MultiValues>& id_l,
             const ptr<MultiValues>& iv_l, const ptr<MultiNodes>& ss)
      : Node(l), id_list(id_l), iv_list(iv_l), stmts(ss) {
    auto id = id_l->ValueAt(0);
    auto identifier = cast<Identifier>(id);
    auto iv = iv_l->ValueAt(0);
    auto num = cast<IntLiteral>(iv);
    biv = identifier->name;
    bound = num->value;
    if (id_l->Count() > 1)
      stmts = ConstructParallelByRecursively(stmts, 1, id_list, iv_list);
  }

  ptr<MultiNodes> ConstructParallelByRecursively(const ptr<MultiNodes>& ss,
                                                 size_t idx,
                                                 const ptr<MultiValues>& id_l,
                                                 const ptr<MultiValues>& iv_l) {
    auto id = id_l->ValueAt(idx);
    auto identifier = cast<Identifier>(id);
    auto iv = iv_l->ValueAt(idx);
    auto num = cast<IntLiteral>(iv);
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

  __UDT_TYPE_INFO__(Node, ParallelBy)
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

  __UDT_TYPE_INFO__(Node, WhereBind)
};

struct WithIn : public Node, public TypeIDProvider<WithIn> {
  ptr<Identifier> with; // either with or with_matcher
  ptr<Node> in;
  ptr<MultiValues> with_matchers;

  WithIn(const location& l, const ptr<Identifier>& w, const ptr<Node>& i)
      : Node(l), with(w), in(i), with_matchers(nullptr) {}

  WithIn(const location& l, const ptr<Node>& i, const ptr<MultiValues>& m)
      : Node(l), with(), in(i), with_matchers(m) {}

  WithIn(const location& l, const ptr<Identifier>& w, const ptr<Node>& i,
         ptr<MultiValues> m)
      : Node(l), with(w), in(i), with_matchers(m) {}

  const std::vector<ptr<Node>>& GetMatchers() const {
    return with_matchers->AllValues();
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << prefix << "`- ";
    if (with) os << with->name;
    if (with_matchers) {
      if (with) os << " = ";
      os << "{";
      with_matchers->InlinePrint(os);
      os << "}";
    }
    os << " in " << STR(*in) << "\n";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, WithIn)
};

struct WithBlock : public Node, public TypeIDProvider<WithBlock> {
  ptr<MultiNodes> withins;
  ptr<MultiNodes> reqs;  // optional requirements
  ptr<MultiNodes> stmts; // may be empty

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

  __UDT_TYPE_INFO__(Node, WithBlock)
};

struct ChunkAt : public Node, public TypeIDProvider<ChunkAt> {
  ptr<Identifier> data;
  ptr<SpanAs> sa = nullptr; // for span_as expression
  ptr<MultiValues> positions = nullptr;

  ChunkAt(const location& l, const ptr<Identifier>& d,
          const ptr<MultiValues>& p = nullptr)
      : Node(l), data(d), sa(nullptr), positions(p) {}

  ChunkAt(const location& l, const ptr<SpanAs>& s,
          const ptr<MultiValues>& p = nullptr)
      : Node(l), data(s->nid), sa(s), positions(p) {}

  std::string RefSymbol() const {
    assert(data && "ref data is not set.");
    return RemoveSuffix(data->name, ".data");
  }

  bool SymbolicBufferName() { return !positions; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (sa)
      os << PSTR(sa);
    else
      os << PSTR(data);

    if (positions) os << ".ChunkAt(" << STR(positions) << ")";

    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, ChunkAt)
};

struct Select : public Node, public TypeIDProvider<Select> {
  std::string rname;
  ptr<Expr> select_factor = nullptr;
  int bound;
  ptr<MultiValues> expr_list = nullptr;
  bool inDMA = false;

  Select(const location& l, const ptr<Expr>& sf,
         const ptr<MultiValues>& list = nullptr)
      : Node(l), select_factor(sf), expr_list(list) {}

  // TODO(wsj)
  // x = select(IntLiteral, a, b, c)

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "select(" << STR(select_factor) << ", " << STR(expr_list) << ")";
    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Select)
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
  ptr<Node> from = nullptr;
  ptr<Node> to = nullptr;
  ptr<DMAConfig> config = nullptr;

  explicit DMA(const location& l, const std::string& o, const std::string& r,
               const ptr<Node>& f, const ptr<Node>& t, bool a,
               const ptr<DMAConfig>& c = nullptr)
      : Node(l, MakeDummyFutureType(a)), operation(o), future(r), async(a),
        from(f), to(t), config(c) {
    chained = false;
    chain_to = "";
    chain_from = "";
    if (auto tptr = dyn_cast<AST::Select>(t)) tptr->inDMA = true;
  }

  explicit DMA(const location& l, const std::string& o, const std::string& r,
               const std::string& chained_from, const ptr<Node>& f,
               const ptr<Node>& t, bool a, const ptr<DMAConfig>& c = nullptr)
      : Node(l, MakeDummyFutureType(a)), operation(o), future(r), async(a),
        from(f), to(t), config(c) {
    chained = true;
    chain_from = chained_from;
    if (auto tptr = dyn_cast<AST::Select>(t)) tptr->inDMA = true;
  }

  // The dummy dma
  explicit DMA(const location& l, const std::string& f)
      : Node(l, MakePlaceHolderFutureType()), operation(".any"), future(f),
        async(true) {}

  std::string FromSymbol() const { return cast<ChunkAt>(from)->RefSymbol(); }

  std::string ToSymbol() const {
    if (auto tochunk = dyn_cast<ChunkAt>(to)) return tochunk->data->name;
    return "";
  }

  void SetConfig(const ptr<DMAConfig>& cfg) { config = cfg; }
  const ptr<DMAConfig>& GetConfig() const { return config; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    if (operation == ".any") {
      os << "\n" << prefix << "`- DMA" << operation;
      if (!future.empty()) os << "\n" << prefix << "  `- future: " << future;
      return;
    }

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

  __UDT_TYPE_INFO__(Node, DMA)
};

struct Wait : public Node, public TypeIDProvider<Wait> {
  ptr<MultiValues> targets;

  Wait(const location& l, const ptr<MultiValues>& t) : Node(l), targets(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- WAIT: " << AST::STR(*targets);
  }

  const std::vector<ptr<Node>>& GetFutures() const {
    return targets->AllValues();
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Wait)
};

struct Return : public Node, public TypeIDProvider<Return> {
  ptr<Node> value = nullptr;

  Return(const location& l) : Node(l) {}
  Return(const location& l, const ptr<Node>& t) : Node(l), value(t) {}

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Return: ";
    os << ((!value) ? "void" : STR(value));
    if (!note.empty()) os << " (" << note << ")";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Return)
};

struct Call : public Node, public TypeIDProvider<Call> {
  ptr<Identifier> function;
  ptr<MultiValues> arguments;
  ptr<MultiValues> template_args;

  Call(const location& l, const ptr<Identifier>& f, const ptr<MultiValues>& a)
      : Node(l), function(f), arguments(a), template_args(nullptr) {}

  Call(const location& l, const ptr<Identifier>& f, const ptr<MultiValues>& a,
       const ptr<MultiValues>& b)
      : Node(l), function(f), arguments(a), template_args(b) {
    arguments->SetDelimiter(", ");
    template_args->SetDelimiter(", ");
  }

  const std::vector<ptr<Node>>& GetArguments() const {
    return arguments->AllValues();
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Call: " << STR(*function);
    os << "\n" << prefix << "  `- with arguments: " << STR(*arguments);
    if (template_args)
      os << "\n"
         << prefix << "  `- with template parameters: " << STR(*template_args);
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Call)
};

struct Rotate : public Node, public TypeIDProvider<Rotate> {
  ptr<MultiValues> ids;

  Rotate(const location& loc, const ptr<MultiValues>& v) : Node(loc), ids(v) {
    ids->SetDelimiter(", ");
  }

  ptr<Node> ValueAt(int index) { return ids->ValueAt(index); }
  ptr<Identifier> IdAt(int index) {
    return cast<Identifier>(ids->ValueAt(index));
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n"
       << prefix << "`- " << ((ids->Count() == 2) ? "Swap: " : "Rotate: ")
       << PSTR(ids);
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Rotate)
};

struct LoopRange : public Node, public TypeIDProvider<LoopRange> {
  ptr<Identifier> iv; // induction variable
  int lbound = GetInvalidBound();
  int ubound = GetInvalidBound();
  int stride = GetInvalidStride();

  LoopRange(const location& l, const ptr<Identifier> i)
      : Node(l), iv(i) {} // the bounds are yet to be inferenced
  LoopRange(const location& l, const ptr<Identifier> i, int lb, int ub,
            int s = 1)
      : Node(l), iv(i), lbound(lb), ubound(ub), stride(s) {}

  const std::string IVName() const { return iv->name; }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Iteration variables: " << iv->name;

    if (!IsValidBound(lbound) && !IsValidBound(ubound) &&
        !IsValidStride(stride))
      return;

    os << "\n" << prefix << "`- Loop Control: (";
    os << (IsValidBound(lbound) ? std::to_string(lbound) : std::string("?"))
       << ":";
    os << (IsValidBound(ubound) ? std::to_string(ubound) : std::string("?"))
       << ":";
    os << (IsValidStride(stride) ? std::to_string(stride) : std::string("?"))
       << ")";
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, LoopRange)
};

struct ForeachBlock : public Node, public TypeIDProvider<ForeachBlock> {
  ptr<MultiValues> ranges;
  ptr<MultiNodes> stmts;

  explicit ForeachBlock(const location& l, const ptr<MultiValues>& i,
                        const ptr<MultiNodes>& s)
      : Node(l), ranges(i), stmts(s) {
    assert(i != nullptr && "missing iteration variables for the statement.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Foreach Block:";
    ranges->Print(os, prefix + " ");
    if (stmts) { stmts->Print(os, prefix + " "); }
  }

  ptr<MultiValues> GetRangeNodes() const { return ranges; }
  const std::vector<ptr<Node>>& GetRanges() const {
    return ranges->AllValues();
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, ForeachBlock)
};

struct IncrementBlock : public Node, public TypeIDProvider<IncrementBlock> {
  ptr<MultiValues> bvs;
  ptr<Node> pred;
  ptr<MultiNodes> stmts;

  explicit IncrementBlock(const location& l, const ptr<MultiValues>& i,
                          const ptr<Node>& p, const ptr<MultiNodes>& s)
      : Node(l), bvs(i), pred(p), stmts(s) {
    assert(i != nullptr && "missing iteration variables for the statement.");
    assert(p != nullptr && "missing predication for the increment block.");
  }

  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << "\n" << prefix << "`- Increment Block:";
    os << "\n" << prefix << " `- Iteration variables: " << STR(bvs);
    os << "\n" << prefix << " `- Predicate: " << STR(pred);
    if (stmts) { stmts->Print(os, prefix + " "); }
  }

  void accept(Visitor&) override;

  const std::vector<ptr<Node>>& GetIterationVars() const {
    return bvs->AllValues();
  }

  const ptr<Node>& GetPredicate() const { return pred; }

  __UDT_TYPE_INFO__(Node, IncrementBlock)
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

  __UDT_TYPE_INFO__(Node, FunctionDecl)
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

  __UDT_TYPE_INFO__(Node, ChoreoFunction)
};

struct CppSourceCode : public Node, public TypeIDProvider<CppSourceCode> {
  std::string code;
  bool host; // host or kernel
  CppSourceCode(const location& l, const std::string& c, bool h = true)
      : Node(l), code(c), host(h) {}
  void Print(std::ostream& os, const std::string& prefix = {}) const override {
    os << code;
    (void)prefix;
  }

  std::string GetCode() { return code; }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, CppSourceCode)
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

  __UDT_TYPE_INFO__(Node, Program)
};

//---------------------------------------------------------------------------//
// Utility Functions
//---------------------------------------------------------------------------//
inline std::optional<std::string> GetName(const Node& n) {
  if (auto id = dyn_cast<AST::Identifier>(&n))
    return id->name;
  else if (auto exp = dyn_cast<AST::Expr>(&n)) {
    if (auto id = exp->GetSymbol()) return id->name;
  }
  return std::nullopt;
}

inline Identifier* GetIdentifier(const Node& n) {
  if (auto id = dyn_cast<AST::Identifier>(&n))
    return id;
  else if (auto expr = dyn_cast<AST::Expr>(&n))
    return expr->GetSymbol().get();
  else
    return nullptr;
}

inline ptr<Node> Ref(const ptr<Node>& n) {
  if (auto expr = dyn_cast<Expr>(n)) return expr->GetReference();
  return n;
}

inline std::string NodeName(const Node& n) { return n.TypeNameString(); }

// symbol reference specific expr
inline ptr<Expr> MakeIdExpr(const location& l, const std::string& n) {
  return Make<Expr>(l, Make<Identifier>(l, n));
}

// symbol reference specific expr
inline ptr<Expr> MakeIntExpr(const location& l, int val) {
  return Make<Expr>(l, Make<IntLiteral>(l, val));
}

} // end of namespace AST

} // end of namespace Choreo

#endif // __CHOREO_AST_HPP__
