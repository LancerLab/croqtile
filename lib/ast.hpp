#ifndef __CHOREO_AST_HPP__
#define __CHOREO_AST_HPP__

#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "context.hpp"
#include "dmaconf.hpp"
#include "loc.hpp"
#include "loop_utils.hpp"
#include "symtab.hpp"
#include "symvals.hpp"

extern Choreo::SymbolTable symtab;

namespace Choreo {
struct Visitor;
struct VisitorWithScope;

namespace AST {

// short hands
template <typename T>
using ptr = Choreo::ptr<T>;

// Utility to generate shared_ptr<Node>
template <typename T, typename... Args>
inline ptr<T> Make(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

//------------------------- AST Node Fundamentals ----------------------------//

struct Identifier;
struct DataType;

// interface class for all AST nodes
struct Node {
  location loc;
  ptr<Type> pty = MakeUnknownType();
  using NoteMapType = std::unordered_map<std::string, std::string>;

private:
  NoteMapType note;
  DiversityShape dshape;

protected:
  ParallelLevel level = ParallelLevel::NONE; // belongs to a specific level

public:
  Node(const location& l, const ptr<Type>& p = MakeUnknownType())
      : loc(l), pty(p) {}

  virtual bool TypeUnknown() const { return isa<UnknownType>(pty.get()); }

  virtual void SetType(const ptr<Type>& t) { pty = t; }
  virtual const ptr<Type>& GetType() const { return pty; }
  virtual const location& LOC() const { return loc; }
  virtual void SetLOC(const location& l) { loc = l; }
  virtual void SetDiversityShape(const DiversityShape ds) { dshape = ds; }
  virtual DiversityShape GetDiversityShape() const { return dshape; }
  virtual ~Node() = default;

  virtual const NoteMapType& Note() const { return note; }
  virtual bool HasNote(const std::string& k) const { return note.count(k); }
  virtual std::string GetNote(const std::string& k) const { return note.at(k); }
  virtual void AddNote(const std::string& k, const std::string& v = "") {
    note.emplace(k, v);
  }
  virtual void EraseNote(const std::string& k) { note.erase(k); }

  virtual bool IsBlock() const { return false; }
  virtual ParallelLevel GetLevel() const { return level; }
  virtual void SetLevel(ParallelLevel l) { level = l; }
  virtual const ptr<Node> Clone() const {
    auto n = CloneImpl();
    n->SetType(GetType());
    n->SetLevel(GetLevel());
    n->SetDiversityShape(GetDiversityShape());
    if (!note.empty()) n->note = note;
    return n;
  }
  virtual ptr<Node> CloneImpl() const = 0;

  virtual void Print(std::ostream& os, const std::string& prefix = {},
                     bool with_type = false) const = 0;

  virtual void InlinePrint(std::ostream& os, const std::string& prefix = {},
                           bool with_type = false) const {
    if (!IsBlock()) Print(os, prefix, with_type);
  }

  virtual void accept(Visitor&) = 0;

  // for runtime type disambiguation
  __UDT_TYPE_INFO_BASE__(node)
};

using NodeList = std::vector<ptr<Node>>;

// utility functions
template <typename T>
bool istypeof(const Node* n) {
  return isa<T>(n->GetType());
}
template <typename T>
bool istypeof(const ptr<Node>& n) {
  return isa<T>(n->GetType());
}

inline std::string STR(const Node& n) {
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
  NodeList values;
  std::string delimiter;

  MultiNodes() = delete;

  explicit MultiNodes(const location& l, std::string d = "")
      : Node(l), delimiter(d) {}

  template <typename... Tys>
  explicit MultiNodes(const location& l, Tys&&... args)
      : Node(l), delimiter(", ") {
    (Append(std::forward<Tys>(args)), ...);
  }

  void Append(const ptr<Node>& m) {
    assert(m != nullptr && "Unexpected: null pointer.");
    values.push_back(m);
  }
  void Insert(const ptr<Node>& m, size_t index) {
    assert(m != nullptr && "Unexpected: null pointer.");
    assert(index <= values.size());
    values.insert(values.begin() + index, m);
  }

  void PopBack() { values.pop_back(); }

  size_t Count() const { return values.size(); }
  bool None() const { return Count() == 0; }

  void SetDelimiter(const std::string& d) { delimiter = d; }

  NodeList AllSubs() { return values; }

  ptr<Node> SubAt(const size_t idx) const {
    assert(idx < this->Count() &&
           "Out-of-bound error when querying MultiNodes\n");
    return values[idx];
  }

  const ptr<Node> Last() { return values.back(); }
  const ptr<Node> First() { return values.front(); }

  // retrieve the index if the element is inside the MultiNodes
  int GetIndex(Node* n) const {
    for (size_t i = 0; i < values.size(); ++i) {
      if (values[i].get() == n) return i;
    }
    return -1;
  }

  bool IsBlock() const override { return true; }

  ptr<Node> CloneImpl() const override {
    auto mv = Make<MultiNodes>(LOC(), delimiter);
    for (auto v : values) mv->Append(CloneP(v));
    return mv;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    if (delimiter != "" && values.size() > 1) {
      auto i = values.begin();
      auto e = values.end();
      (*i)->Print(os, prefix, with_type);
      ++i;
      for (; i != e; ++i) {
        os << delimiter;
        (*i)->Print(os, prefix, with_type);
      }
    } else {
      for (auto& v : values) v->Print(os, prefix, with_type);
    }
  }

  void accept(Visitor& visitor) override;

  __UDT_TYPE_INFO__(Node, MultiNodes)
};

struct MultiValues : public Node, public TypeIDProvider<MultiValues> {
  NodeList values;
  std::string delimiter;

private:
  OptimizedValues opt_vals;

public:
  explicit MultiValues(const location& l, std::string d = "")
      : Node(l), delimiter(d) {}

  explicit MultiValues(const location& l, size_t len, const ptr<Node>& v,
                       std::string d = "")
      : Node(l), delimiter(d) {
    for (size_t i = 0; i < len; ++i) Append(v->Clone());
  }

  template <typename... T>
  explicit MultiValues(const location& l, std::string d, T... args)
      : Node(l), delimiter(d) {
    (Append(args), ...);
  }

  void Append(const ptr<Node>& m) {
    assert(m != nullptr && "Unexpected: null pointer.");
    values.push_back(m);
  }

  void Insert(const ptr<Node>& m, size_t index) {
    assert(m != nullptr && "Unexpected: null pointer.");
    assert(index <= values.size());
    values.insert(values.begin() + index, m);
  }

  size_t Count() const { return values.size(); }
  bool None() const { return values.empty(); }

  OptimizedValues& Opts() { return opt_vals; }
  const OptimizedValues& Opts() const { return opt_vals; }

  void SetDelimiter(const std::string& d) { delimiter = d; }

  ptr<Node> ValueAt(const size_t idx) const {
    assert(idx < this->Count() &&
           "Out-of-bound error when querying MultiValues\n");
    return values[idx];
  }

  void SetValueAt(const size_t idx, ptr<Node> v) {
    assert(idx < this->Count() &&
           "Out-of-bound error when querying MultiValues\n");
    values[idx] = v;
  }

  ptr<Node> operator[](const size_t idx) const { return ValueAt(idx); }

  const NodeList& AllValues() const { return values; }

  bool IsBlock() const override { return true; }

  ptr<Node> CloneImpl() const override {
    auto mv = Make<MultiValues>(LOC(), delimiter);
    for (auto v : values) mv->Append(CloneP(v));
    mv->opt_vals = opt_vals;
    return mv;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    if (delimiter != "" && values.size() > 1) {
      auto i = values.begin();
      auto e = values.end();
      (*i)->Print(os, prefix, with_type);
      ++i;
      for (; i != e; ++i) {
        os << delimiter;
        (*i)->Print(os, prefix, with_type);
      }
    } else {
      for (auto& v : values) v->Print(os, prefix, with_type);
    }
  }

  // TODO: workaround for "x, y" like print, we may need typeid to merge this
  // print logic into trivial Print()
  void InlinePrint(std::ostream& os, const std::string& prefix = {},
                   bool = false) const override {
    for (auto& v : values) {
      v->Print(os, prefix);
      if (&v != &values.back()) os << ", ";
    }
  }

  void accept(Visitor& visitor) override;

  __UDT_TYPE_INFO__(Node, MultiValues)
};

inline const NodeList MakeNodeList(const ptr<MultiValues>& mv) {
  if (!mv) return {};

  NodeList nl;
  for (auto n : mv->AllValues()) nl.push_back(n);
  return nl;
}

struct BoolLiteral : public Node, public TypeIDProvider<BoolLiteral> {
  bool value;
  explicit BoolLiteral(const location& l, bool v)
      : Node(l, MakeBooleanType()), value(v) {}
  explicit BoolLiteral(const BoolLiteral& b) : Node(b.LOC()) {
    value = b.value;
  }

  bool Val() const { return value; }

  ptr<Node> CloneImpl() const override {
    return Make<BoolLiteral>(LOC(), value);
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    os << prefix << ((value) ? "true" : "false");
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, BoolLiteral)
};

struct NoValue : public Node, public TypeIDProvider<NoValue> {
  explicit NoValue(const location& l) : Node(l, MakeNoValueType()) {}

  ptr<Node> CloneImpl() const override { return Make<NoValue>(LOC()); }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    os << prefix << "__noval__";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, NoValue)
};

struct IntLiteral : public Node, public TypeIDProvider<IntLiteral> {
  std::variant<int, uint32_t, int64_t, uint64_t> value;

  IntLiteral(const location& l)
      : Node(l, MakeIntegerType()), value(GetUnKnownInteger()) {}

  IntLiteral(const location& l, int v)
      : Node(l, MakeScalarIntegerType(BaseType::S32)), value(v) {}
  IntLiteral(const location& l, uint32_t v)
      : Node(l, MakeScalarIntegerType(BaseType::U32)), value(v) {}
  IntLiteral(const location& l, int64_t v)
      : Node(l, MakeScalarIntegerType(BaseType::S64)), value(v) {}
  IntLiteral(const location& l, uint64_t v)
      : Node(l, MakeScalarIntegerType(BaseType::U64)), value(v) {}
  IntLiteral(const location& l,
             const std::variant<int, uint32_t, int64_t, uint64_t>& v)
      : Node(l, MakeScalarIntegerType(BaseType::UNKNOWN)), value(v) {}
  IntLiteral(const location& l, BaseType bt)
      : Node(l, MakeScalarIntegerType(bt)), value(0) {
    assert(IsIntegerType(bt) &&
           "BaseType must be an integer fundamental type.");
    switch (bt) {
    case BaseType::S32: value = static_cast<int>(0); break;
    case BaseType::U32: value = static_cast<uint32_t>(0); break;
    case BaseType::S64: value = static_cast<int64_t>(0); break;
    case BaseType::U64: value = static_cast<uint64_t>(0); break;
    default: value = GetUnKnownInteger();
    }
  }

  // allow copy construction
  explicit IntLiteral(const IntLiteral& il)
      : Node(il.LOC(), MakeScalarIntegerType(BaseType::UNKNOWN)),
        value(il.value) {}

  int64_t Val() const {
    return std::visit([](auto x) -> int64_t { return static_cast<int64_t>(x); },
                      value);
  }

  int ValS32() const { return std::get<int>(value); }
  uint32_t ValU32() const { return std::get<uint32_t>(value); }
  uint64_t ValU64() const { return std::get<uint64_t>(value); }
  int64_t ValS64() const { return std::get<int64_t>(value); }

  bool IsInt() const { return std::holds_alternative<int>(value); }
  bool IsUint32() const { return std::holds_alternative<uint32_t>(value); }
  bool IsInt64() const { return std::holds_alternative<int64_t>(value); }
  bool IsUint64() const { return std::holds_alternative<uint64_t>(value); }

  std::string ValAsString() const {
    std::ostringstream oss;
    std::visit([&oss](const auto& val) { oss << val; }, value);
    return oss.str();
  }

  ptr<Node> CloneImpl() const override {
    return Make<IntLiteral>(LOC(), value);
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    if (IsUnKnownInteger(Val()))
      os << prefix << "?";
    else
      os << prefix << ValAsString();
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, IntLiteral)
};

// Single/Double precision floating-point number
struct FloatLiteral : public Node, public TypeIDProvider<FloatLiteral> {
  std::variant<float, double> value;

  FloatLiteral(const location& l, float v = GetUnKnownFloat())
      : Node(l, MakeF32Type()) {
    value = v;
  }

  FloatLiteral(const location& l, double v) : Node(l, MakeF64Type()) {
    value = v;
  }

  FloatLiteral(const location& l, BaseType bt) : Node(l, MakeF32Type()) {
    assert(IsFloatType(bt) &&
           "BaseType must be a float-point fundamental type.");
    switch (bt) {
    case BaseType::F32: value = static_cast<float>(0.0); break;
    case BaseType::F64: value = static_cast<double>(0.0); break;
    default: value = GetUnKnownFloat();
    }
  }

  // allow copy construction
  explicit FloatLiteral(const FloatLiteral& fl)
      : Node(fl.LOC()), value(fl.value) {}

  float Val_f32() const {
    assert(
        IsFloat32() &&
        "Cannot get f32 value from float-point number whose type is not f32.");
    return std::get<float>(value);
  }
  double Val_f64() const {
    assert(
        IsFloat64() &&
        "Cannot get f64 value from float-point number whose type is not f64.");
    return std::get<double>(value);
  }

  bool IsFloat32() const { return isa<F32Type>(GetType()); }
  bool IsFloat64() const { return isa<F64Type>(GetType()); }

  ptr<Node> CloneImpl() const override {
    auto n = Make<FloatLiteral>(LOC());
    n->value = value;
    return n;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    std::ostringstream oss;
    if (IsFloat32()) {
      auto f32 = std::get<float>(value);
      if (IsUnKnownFloatPoint(f32))
        oss << prefix << "?";
      else
        oss << prefix << std::fixed << f32 << "f";
    } else if (IsFloat64()) {
      auto f64 = std::get<double>(value);
      if (IsUnKnownFloatPoint(f64))
        oss << prefix << "?";
      else
        oss << prefix << std::fixed << f64;
    } else {
      choreo_unreachable("unhandled floating-point type.");
    }
    os << oss.str();
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, FloatLiteral)
};

struct StringLiteral : public Node, public TypeIDProvider<StringLiteral> {
  std::string value;
  StringLiteral(const location& l, std::string v)
      : Node(l, MakeStringType()), value(v) {}

  // allow copy construction
  explicit StringLiteral(const StringLiteral& il)
      : StringLiteral(il.LOC(), il.value) {}

  const std::string Val() const { return value; }

  const std::string EscapedVal() const {
    std::ostringstream oss;
    for (char c : value) {
      switch (c) {
      case '\n': oss << "\\n"; break;
      case '\t': oss << "\\t"; break;
      case '\\': oss << "\\\\"; break;
      case '\"': oss << "\\\""; break;
      default:
        if (isprint(c)) {
          oss << c; // Print printable characters as is
        } else {
          oss << "\\x" << std::hex << std::setw(2) << std::setfill('0')
              << (static_cast<unsigned char>(c));
        }
      }
    }
    return oss.str();
  }

  ptr<Node> CloneImpl() const override {
    return Make<StringLiteral>(LOC(), value);
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    os << prefix << "\"" << EscapedVal() << "\"";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, StringLiteral)
};

struct Call;
struct Expr : public Node, public TypeIDProvider<Expr> {
  // Different expression type
  enum Form { Unary, Binary, Ternary, Reference };

  std::string op;

private:
  ptr<Expr> value_c = nullptr;
  ptr<Node> value_l = nullptr;
  ptr<Node> value_r = nullptr;
  Form t;

private:
  OptimizedValues opt_vals;
  ptr<SCEV> scev = nullptr;

public:
  const ptr<Node>& GetR() const { return value_r; }
  const ptr<Node>& GetL() const { return value_l; }
  const ptr<Expr>& GetC() const { return value_c; }
  const std::string GetOp() const { return op; }
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
  void ResetL() { value_l = nullptr; }

  OptimizedValues& Opts() { return opt_vals; }
  const OptimizedValues& Opts() const { return opt_vals; }
  ptr<SCEV> GetSCEV() const { return scev; }
  void SetSCEV(const ptr<SCEV>& s) { scev = s; }

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
  explicit Expr(const location& l, const std::string& o, const ptr<Expr>& c,
                const ptr<Node>& v1, const ptr<Node>& v2, Form f,
                const OptimizedValues& ov, const Shape& sp)
      : Node(l), op(o), value_c(c), value_l(v1), value_r(v2), t(f),
        opt_vals(ov), s(sp) {}

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

  ptr<FloatLiteral> GetFloat() {
    if (t != Reference) return nullptr;
    return dyn_cast<FloatLiteral>(value_r);
  }

  ptr<StringLiteral> GetString() {
    if (t != Reference) return nullptr;
    return dyn_cast<StringLiteral>(value_r);
  }

  ptr<BoolLiteral> GetBoolean() {
    if (t != Reference) return nullptr;
    return dyn_cast<BoolLiteral>(value_r);
  }

  bool IsUnary() const { return t == Unary; }
  bool IsBinary() const { return t == Binary; }
  bool IsTernary() const { return t == Ternary; }
  bool IsReference() const { return t == Reference; }

  bool IsArith() const {
    if (!IsBinary()) return false;
    if ((op == "+") || (op == "-") || (op == "*") || (op == "/") ||
        (op == "%") || (op == "cdiv") || (op == "#") || (op == "#+") ||
        (op == "#-") || (op == "#*") || (op == "#/") || (op == "#%"))
      return true;
    return false;
  }

  bool IsLogical() const {
    if ((op == "||") || (op == "&&") || (op == "!")) return true;
    return false;
  }

  bool IsCompare() const {
    if ((op == "<") || (op == "<=") || (op == "==") || (op == ">") ||
        (op == ">=") || (op == "!="))
      return true;
    return false;
  }

  bool isBitwise() const {
    if ((op == "&") || (op == "|") || (op == "^") || (op == "~") ||
        (op == "<<") || (op == ">>"))
      return true;
    return false;
  }

  bool IsUBArith() const {
    if ((op == "#") || (op == "#+") || (op == "#-") || (op == "#*") ||
        (op == "#/") || (op == "#%"))
      return true;
    return false;
  }

public:
  ptr<Node> CloneImpl() const override;

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override;

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Expr)
};

struct AttributeExpr final : public Node, public TypeIDProvider<AttributeExpr> {
private:
  std::string attr_name;
  ptr<MultiValues> attr_values = nullptr;

public:
  AttributeExpr(const location& l, const std::string& n,
                const ptr<MultiValues>& v)
      : Node(l, MakeUnknownType()), attr_name(n), attr_values(v) {
    assert(attr_values && "null node is provided.");
  }

  std::string AttrName() const { return attr_name; }
  ptr<Node> AttrValueAt(const size_t idx) { return attr_values->ValueAt(idx); }
  size_t AttrValueCount() const { return attr_values->Count(); }

public:
  ptr<Node> CloneImpl() const override {
    return AST::Make<AST::AttributeExpr>(loc, attr_name, CloneP(attr_values));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << prefix << "@" << attr_name << " ";
    if (attr_values) attr_values->Print(os, "", with_type);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, AttributeExpr)
};

struct CastExpr : public Expr, public TypeIDProvider<CastExpr> {
private:
  BaseType from;
  BaseType to;
  size_t element_count = 1;

public:
  CastExpr(const location& l, const ptr<Node>& val) : Expr(l, "cast", val) {
    assert(isa<Expr>(val));
  }

  BaseType FromType() const { return from; }
  BaseType ToType() const { return to; }
  bool IsVectorType() const { return element_count > 1; }
  size_t ElementCount() const { return element_count; }

  void SetFrom(BaseType bty, size_t ec = 1) {
    from = bty;
    if (ec > 1) element_count = ec;
  }
  void SetTo(BaseType bty, size_t ec = 1) {
    to = bty;
    if (ec > 1) element_count = ec;
  }

public:
  ptr<Node> CloneImpl() const override {
    auto n = Make<CastExpr>(LOC(), CloneP(GetR()));
    n->from = from;
    n->to = to;
    return n;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << prefix << "CAST(" << FromType() << "=>" << ToType() << ": '";
    GetR()->Print(os, {}, with_type);
    os << "') ";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Expr, CastExpr);
};

// Represents both dimensions and s like {3, 4, 5} or {1, 2, 1}
struct MultiDimSpans : public Node, public TypeIDProvider<MultiDimSpans> {
  std::string ref_name;           // syntax sugar, could be empty
  ptr<Node> list = nullptr;       // null if the span is dynamically valued
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

  bool IsSymbolReference() {
    if (ref_name != "") return false;
    if (auto e = dyn_cast<Expr>(list)) {
      if (e->GetSymbol()) return true;
    }
    return false;
  }

  void SetTypeDetail(const Shape& s) {
    assert(istypeof<MDSpanType>(this) && "Incorrect type for mdspan.");
    cast<MDSpanType>(GetType())->SetShape(s);
  }

  const Shape GetTypeDetail() {
    assert(istypeof<MDSpanType>(this) && "Incorrect type for mdspan.");
    return cast<MDSpanType>(GetType())->GetShape();
  }

  ptr<Node> CloneImpl() const override {
    return Make<MultiDimSpans>(LOC(), ref_name, CloneP(list));
  }

  void Print(std::ostream& os, const std::string& = {},
             bool with_type = false) const override {
    if (!list)
      os << "<" << rank << ">";
    else {
      os << "[";
      list->Print(os, " " + ref_name, with_type);
      os << " ]";
    }
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
    assert(istypeof<SpannedType>(this) && "Incorrect type for mdspan.");
    cast<MDSpanType>(GetType())->SetShape(s);
  }

  const Shape GetTypeDetail() {
    assert(istypeof<SpannedType>(this) && "Incorrect type for mdspan.");
    return cast<MDSpanType>(GetType())->GetShape();
  }

  ptr<Node> CloneImpl() const override {
    return Make<SpanAs>(LOC(), id, nid, cast<MultiValues>(CloneP(list)));
  }

  void Print(std::ostream& os, const std::string& = {},
             bool with_type = false) const override {
    assert(id && "no original span is specified.");
    assert(nid && "no new span is specified.");
    assert(list && "no span_as is specified.");

    os << PSTR(id) << ".span_as[";
    list->Print(os, " ", with_type);
    os << " ]";

    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
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
    assert(init_expr && "Invalid expression.");
  }

  explicit NamedTypeDecl(const location& l, const std::string& n,
                         const ptr<Node>& v, size_t r,
                         const std::string& d = "-")
      : Node(l), name_str(n), init_str(d), init_expr(v), rank(r) {
    assert(name_str.size() > 0 && "Invalid name string.");
    assert(init_expr && "Invalid value.");
  }

  ptr<Node> CloneImpl() const override {
    return Make<NamedTypeDecl>(LOC(), name_str, CloneP(init_expr), rank,
                               init_str);
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- Type Decl: ";
    os << name_str << " " << init_str << " " << STR(*init_expr);
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, NamedTypeDecl)
};

struct Memory : public Node, public TypeIDProvider<Memory> {
  Storage st;
  Memory(const location& l, const Storage s = Storage::DEFAULT)
      : Node(l), st(s) {}

  ptr<Node> CloneImpl() const override { return Make<Memory>(LOC(), st); }

  void Print(std::ostream& os, const std::string& = {},
             bool with_type = false) const override {
    os << STR(st);
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
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

  ptr<Node> CloneImpl() const override {
    return Make<IntTuple>(LOC(), ref_name, CloneP(vlist));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    if (ref_name.size() > 0) os << ref_name << " ";
    os << "{" << STR(*vlist) << "}";
    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, IntTuple)
};

struct Identifier : public Node, public TypeIDProvider<Identifier> {
  std::string name;
  Identifier(const location& l,
             const std::string& n = SymbolTable::GetAnonName())
      : Node(l), name(n) {}
  Identifier(const Identifier& id) : Node(id.LOC()), name(id.name) {}

  ptr<Node> CloneImpl() const override { return Make<Identifier>(LOC(), name); }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    os << prefix << name;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Identifier)
};

struct DataAccess : public Node, public TypeIDProvider<DataAccess> {
  ptr<Identifier> data = nullptr;
  ptr<MultiValues> indices = nullptr;
  ptr<SCEV> scev = nullptr;

private:
  bool is_decl = false; // decl or ref

public:
  DataAccess(const location& l, const ptr<Identifier>& i,
             const ptr<MultiValues>& m = nullptr, bool isd = false)
      : Node(l), data(i), indices(m), is_decl(isd) {
    assert(i != nullptr && "no data is specified.");
    if (m) assert((m->Count() > 0) && "requires at least one index.");
  }

  explicit DataAccess(const location& l, const std::string& s)
      : Node(l), data(Make<Identifier>(l, s)), indices(nullptr) {}

  const ptr<Identifier>& GetData() const { return data; };
  const std::string& GetDataName() const { return data->name; };
  ptr<SCEV> GetSCEV() const { return scev; }

  bool AccessElement() const { return indices != nullptr; }
  void SetDecl(bool isd = true) { is_decl = isd; }
  bool IsDecl() const { return is_decl; }

  const NodeList& GetIndices() const {
    if (!indices) choreo_unreachable("unexpected null indices.");
    return indices->AllValues();
  }

  ptr<Node> CloneImpl() const override {
    return Make<DataAccess>(LOC(), CloneP(data), CloneP(indices), is_decl);
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    (void)prefix;
    os << data->name;
    if (indices) {
      os << "[";
      int i = 0;
      for (auto e : GetIndices()) {
        if (i++ != 0) os << ", ";
        os << PSTR(e);
      }
      os << "]";
    }
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, DataAccess)
};

struct Assignment : public Node, public TypeIDProvider<Assignment> {
  ptr<DataAccess> da = nullptr;
  ptr<Node> value = nullptr;

public:
  explicit Assignment(const location& l, const std::string& n,
                      const ptr<Node>& v)
      : Node(l), da(Make<DataAccess>(l, n)), value(v) {}

  explicit Assignment(const location& l, const ptr<DataAccess>& n,
                      const ptr<Node>& v)
      : Node(l), da(n), value(v) {}

  void SetDecl(bool d = true) { da->SetDecl(d); }
  bool IsDecl() const { return da->IsDecl(); }

  ptr<Node> CloneImpl() const override {
    auto c = Make<Assignment>(LOC(), CloneP(da), CloneP(value));
    return c;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- Assign";
    if (with_type) os << (IsDecl() ? "(decl)" : "");
    // if (HasNote("update")) os << "(u)";
    os << ": ";
    da->Print(os, "", with_type);
    os << " = ";
    value->Print(os, "", with_type);
  }

  const std::string& GetName() const {
    if (da->AccessElement())
      choreo_unreachable("element access can not be named.");
    return da->GetDataName();
  }

  const std::string& GetDataArrayName() const {
    if (!da->AccessElement()) choreo_unreachable("not a data array access.");
    return da->GetDataName();
  }

  bool AssignToDataElement() { return da->AccessElement(); }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Assignment)
};

struct IntIndex : public Node, public TypeIDProvider<IntIndex> {
  ptr<Node> value;
  explicit IntIndex(const location& l, const ptr<Node>& v)
      : Node(l), value(v) {}

  char lb = '(';
  char rb = ')';

  explicit IntIndex(const IntIndex& ii) : Node(ii.LOC()), value(ii.value) {}

  void UseBracket() {
    lb = '[';
    rb = ']';
  }

  const ptr<Node> Val() const { return value; }

  bool IsNegative() const {
    if (auto il = dyn_cast<IntLiteral>(value)) return il->Val() < 0;
    return false;
  }

  ptr<Node> CloneImpl() const override {
    return Make<IntIndex>(LOC(), CloneP(value));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << prefix << lb;
    value->Print(os, "", with_type);
    os << rb;
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
  ValueList array_dims;
  bool is_mutable = false;
  bool infer_span = false; // the span must be inferenced

public:
  explicit DataType(const location& l, BaseType t, bool m = false)
      : Node(l), base_type(t), mdspan_type(nullptr), is_mutable(m) {
    InitSemaType();
  }

  explicit DataType(const location& l, BaseType bt, const ptr<Node>& st,
                    bool m = false)
      : Node(l), base_type(bt), mdspan_type(st), is_mutable(m) {
    assert(bt != BaseType::ITUPLE && "unexpected type!");
    assert(bt != BaseType::BOOL && "unexpected type!");
    InitSemaType();
  }

  explicit DataType(const location& l, BaseType bt, int r)
      : Node(l), base_type(bt), rank(r) {
    assert(bt == BaseType::ITUPLE && "unexpected type!");
    InitSemaType();
  }

  explicit DataType(const location& l, BaseType bt, const ptr<MultiValues>& ad)
      : Node(l), base_type(bt), rank(1) {
    // infer the real array dim at shapeinfer.
    if (ad != nullptr) array_dims = GenUninitValueList(ad->Count());
    assert(((bt == BaseType::EVENT) || (bt == BaseType::UNKNOWN)) &&
           "unexpected type!");
    InitSemaType();
  }

  // used for clone
  explicit DataType(const location& l, BaseType bt, size_t r,
                    const ptr<Node> pt, const ValueList& ad, bool im,
                    bool infer)
      : Node(l), base_type(bt), rank(r), mdspan_type(pt), array_dims(ad),
        is_mutable(im), infer_span(infer) {}

  BaseType getBaseType() const { return base_type; }
  ptr<Node> getPartialType() const { return mdspan_type; }

  bool IsVoid() const { return base_type == BaseType::VOID; }
  bool IsUnknown() const { return base_type == BaseType::UNKNOWN; }
  bool IsScalar() const {
    return (base_type != BaseType::ITUPLE) && (base_type != BaseType::EVENT) &&
           (base_type != BaseType::ARRAY) && (base_type != BaseType::ADDR) &&
           (base_type != BaseType::VOID) && (base_type != BaseType::VOID);
  }
  bool IsArrayType() const { return !array_dims.empty(); }
  ValueList ArrayDims() const {
    if (!IsArrayType()) choreo_unreachable("The type is not array.");
    return array_dims;
  }
  bool isITuple() const { return base_type == BaseType::ITUPLE; }
  bool ExplicitSpanned() const { return (bool)mdspan_type; }

  bool IsMutable() const { return is_mutable; }
  void SetMutable(bool m) { is_mutable = m; }

  // force regeneration of sema type
  void ReGenSemaType() { InitSemaType(); }

  ptr<Node> CloneImpl() const override {
    return Make<DataType>(LOC(), base_type, rank, CloneP(mdspan_type),
                          array_dims, is_mutable, infer_span);
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    os << prefix;
    if (is_mutable) os << "mutable ";
    os << STR(base_type);
    if (ExplicitSpanned())
      os << " " << STR(mdspan_type);
    else if (infer_span)
      os << " [?]";
  }

  void accept(Visitor&) override;

private:
  ptr<Type> InitSemaType() {
    if (ExplicitSpanned()) {
      if (IsIntegerType(base_type) || IsFloatType(base_type)) {
        assert(mdspan_type != nullptr && "Expecting a valid mdspan.");
        // need type inference
        if (!IsArrayType())
          SetType(MakeUnRankedSpannedType(base_type));
        else
          SetType(MakeStridedSpannedArrayType(base_type, GenUnknownShape(), {},
                                              ArrayDims()));
      } else {
        choreo_unreachable("Unexpected BaseType: " + STR(base_type) + ".");
      }
    } else {
      if (IsScalarType(base_type)) {
        SetType(MakeScalarType(base_type, is_mutable));
      } else if (base_type == BaseType::EVENT) {
        if (!IsArrayType())
          SetType(MakeEventType(Storage::DEFAULT));
        else
          SetType(MakeEventArrayType(Storage::DEFAULT, ArrayDims()));
      } else if (base_type == BaseType::ITUPLE) {
        if (!IsValidRank(rank))
          // type inference to deduce the dim count
          SetType(MakeUninitITupleType());
        else
          SetType(MakeITupleType(rank));
      } else if (base_type == BaseType::STREAM) {
        SetType(MakeStreamType());
      } else if (base_type == BaseType::UNKNOWN) {
        SetType(MakeUnknownType()); // need type inference
      } else if (base_type == BaseType::VOID) {
        SetType(MakeVoidType());
      } else {
        choreo_unreachable("Unexpected BaseType: " + STR(base_type) + ".");
      }
    }
    return nullptr;
  }

public:
  __UDT_TYPE_INFO__(Node, DataType)
};

struct DeviceFunctionDecl;
struct Call : public Node, public TypeIDProvider<Call> {
  enum CallAttr : uint8_t {
    NONE = 0,
    BIF = 0x1,
    COMPTIME = 0x2,
    ARITH = 0x4,
    EXPR = 0x8,
    ANNO = 0x10
  };
  // Overload bitwise OR
  friend constexpr CallAttr operator|(CallAttr lhs, CallAttr rhs) {
    return static_cast<CallAttr>(
        static_cast<std::underlying_type_t<CallAttr>>(lhs) |
        static_cast<std::underlying_type_t<CallAttr>>(rhs));
  }

public:
  ptr<Identifier> function = nullptr;
  ptr<MultiValues> arguments = nullptr;
  ptr<MultiValues> template_args = nullptr;
  std::vector<ptr<DeviceFunctionDecl>> device_functions;

private:
  CallAttr attr;

public:
  Call(const location& l, const ptr<Identifier>& f, const ptr<MultiValues>& a,
       CallAttr ba = NONE)
      : Node(l), function(f), arguments(a), template_args(nullptr), attr(ba) {}

  Call(const location& l, const ptr<Identifier>& f, const ptr<MultiValues>& a,
       const ptr<MultiValues>& b)
      : Node(l), function(f), arguments(a), template_args(b), attr(NONE) {
    if (arguments) arguments->SetDelimiter(", ");
    if (template_args) template_args->SetDelimiter(", ");
  }

  const NodeList& GetArguments() const { return arguments->AllValues(); }

  bool IsBIF() const { return (bool)(attr & BIF); }
  bool CompileTimeEval() const { return (bool)(attr & COMPTIME); }
  bool IsArith() const { return (bool)(attr & ARITH); }
  bool IsExpr() const { return (bool)(attr & EXPR); }
  bool IsAnno() const { return (bool)(attr & ANNO); }

  void SetBIF() { attr = attr | BIF; }
  void SetCompileTimeEval() { attr = attr | COMPTIME; }
  void SetArith() { attr = attr | ARITH; }
  void SetExpr() { attr = attr | EXPR; }

  ptr<Node> CloneImpl() const override {
    auto n = Make<Call>(LOC(), CloneP(function), CloneP(arguments),
                        CloneP(template_args));
    n->attr = attr;
    n->device_functions = device_functions;
    return n;
  }

  void InlinePrint(std::ostream& os, const std::string& prefix = {},
                   bool with_type = false) const override {
    os << prefix << "call " << STR(*function);
    if (template_args) {
      os << "<";
      template_args->Print(os, {}, with_type);
      os << ">";
    }
    os << "(";
    if (arguments->Count()) arguments->Print(os, {}, with_type);
    os << ")";
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- Call: " << STR(*function);
    if (IsBIF()) {
      if (CompileTimeEval())
        os << " (compile-time built-in)";
      else
        os << " (built-in)";
    }
    if (arguments->Count()) {
      os << "\n" << prefix << "  `- with arguments: ";
      arguments->Print(os, {}, with_type);
    }
    if (template_args) {
      os << "\n" << prefix << "  `- with template parameters: ";
      template_args->Print(os, {}, with_type);
    }
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Call)
};

struct NamedVariableDecl : public Node,
                           public TypeIDProvider<NamedVariableDecl> {
  const std::string name_str;
  const std::string init_str;
  ptr<Memory> mem = nullptr;             // storage location (null for stack)
  ptr<DataType> type = nullptr;          // type annotation
  ptr<Node> init_expr = nullptr;         // associated initializer
  const ptr<Node> init_value = nullptr;  // associated initial value
  ptr<MultiValues> array_dims = nullptr; // has element when it is an array
  bool is_mutable = false;

  explicit NamedVariableDecl(const location& l, const std::string& n,
                             const ptr<DataType>& t = nullptr,
                             const ptr<Memory>& s = nullptr,
                             const ptr<Node>& i = nullptr,
                             const ptr<Node>& v = nullptr,
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

  void SetArrayDims(ptr<MultiValues> ad) {
    if (ad == nullptr) return;
    array_dims = ad;
    array_dims->SetDelimiter(", ");
  }
  bool IsArray() const { return array_dims != nullptr; }
  ptr<Node> ArrayDimension(size_t idx) const {
    return array_dims->ValueAt(idx);
  }
  const ptr<MultiValues>& ArrayDimensions() const { return array_dims; }

  bool IsMutable() const { return is_mutable; }
  void SetMutable(bool m) { is_mutable = m; }

  ptr<Memory> GetMemory() const { return mem; }
  void SetMemory(ptr<Memory> m) { mem = m; }

  ptr<Node> CloneImpl() const override {
    auto n = Make<NamedVariableDecl>(LOC(), name_str, CloneP(type), CloneP(mem),
                                     CloneP(init_expr), CloneP(init_value),
                                     init_str);
    n->SetMutable(IsMutable());
    n->SetArrayDims(ArrayDimensions());
    return n;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- Var Decl (";
    if (type) type->Print(os, "", with_type);
    if (mem) os << ", " << PSTR(mem);
    os << "): " << name_str;
    if (IsArray())
      for (const auto& d : array_dims->AllValues()) os << "[" << STR(d) << "]";
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
    if (init_expr) {
      os << " " << init_str << " ";
      if (auto call = dyn_cast<Call>(init_expr))
        call->InlinePrint(os, "", with_type);
      else
        init_expr->Print(os, " ", with_type);
    } else if (init_value) {
      os << " " << init_str << " {";
      init_value->Print(os, "", with_type);
      os << "}";
    }
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, NamedVariableDecl)
};

struct Parameter : public Node, public TypeIDProvider<Parameter> {
  ptr<DataType> type = nullptr;
  ptr<Identifier> sym = nullptr;
  bool pass_by_ref = false;
  ParamAttr attr = ParamAttr::NONE;

  Parameter(const location& l, const ptr<DataType>& t,
            const ptr<Identifier>& n = nullptr, bool r = false,
            ParamAttr a = ParamAttr::NONE)
      : Node(l), type(t), sym(n), pass_by_ref(r), attr(a) {
    assert(t && "invalid parameter without a type.");
  }

  bool HasSymbol() const { return (bool)sym; }
  ParamAttr GetAttr() const { return attr; }

  ptr<Node> CloneImpl() const override {
    return Make<Parameter>(LOC(), CloneP(type), CloneP(sym), pass_by_ref, attr);
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    type->Print(os, prefix + " type: ");
    if (sym) sym->Print(os, ", symbol: ");
    if (pass_by_ref) os << "(pass by ref)";
    if (attr != ParamAttr::NONE) os << ", attr: " << STR(attr);
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Parameter)
};

struct ParamList : public Node, public TypeIDProvider<ParamList> {
  std::vector<ptr<Parameter>> values;
  explicit ParamList(const location& l) : Node(l) {}
  ParamList(const location& l, std::vector<ptr<Parameter>>& v)
      : Node(l), values(v) {}

  ptr<Node> CloneImpl() const override {
    auto n = Make<ParamList>(LOC());
    for (auto v : values) n->values.push_back(CloneP(v));
    return n;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "Parameters";
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
    os << ":";
    for (auto& item : values) item->Print(os, "\n  " + prefix, with_type);
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, ParamList)
};

struct IfElseBlock : public Node, public TypeIDProvider<IfElseBlock> {
  ptr<Node> pred;
  ptr<MultiNodes> if_stmts;
  ptr<MultiNodes> else_stmts; // optional requirements

  IfElseBlock(const location& l, const ptr<Node>& c,
              const ptr<MultiNodes>& if_s,
              const ptr<MultiNodes>& else_s = nullptr)
      : Node(l), pred(c), if_stmts(if_s), else_stmts(else_s) {
    assert(if_stmts != nullptr && "must contains the if statements.");
  }

  bool IsBlock() const override { return true; }

  const ptr<Node> GetPred() const { return pred; }

  ptr<Node> CloneImpl() const override {
    return Make<IfElseBlock>(LOC(), CloneP(pred), CloneP(if_stmts),
                             CloneP(else_stmts));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- Branch On Condition: ";
    pred->Print(os, " ");
    if (!pred->GetDiversityShape().Unknown()) {
      if (pred->GetDiversityShape().Uniform())
        os << " (uniform predicate)";
      else
        os << " (divergent predicate)";
    }
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
    if (if_stmts->Count()) {
      os << "\n" << prefix << " `- If-Block:";
      if_stmts->Print(os, prefix + "  ", with_type);
    }
    if (else_stmts && else_stmts->Count()) {
      os << "\n" << prefix << " `- Else-Block:";
      else_stmts->Print(os, prefix + "  ", with_type);
    }
  }

  bool HasElse() const { return else_stmts && else_stmts->Count(); }

  bool IsNorm() const {
    if (pred->GetDiversityShape().Uniform()) return true; // uniform predicate
    return false;
  }

  bool IsDivergent() const {
    if (pred->GetDiversityShape().Divergent())
      return true; // divergent predicate
    return false;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, IfElseBlock)
};

struct ParallelBy : public Node, public TypeIDProvider<ParallelBy> {
private:
  ptr<Identifier> bpv = nullptr; // bounded parallel variables
  ptr<Expr> bound_expr = nullptr;

  // components
  ptr<MultiValues> cmpt_bpvs = nullptr;
  ptr<MultiValues> cmpt_bounds = nullptr;

public:
  ptr<MultiNodes> stmts = nullptr;

private:
  bool async = false;
  bool bracketed = false; // for late syntax check

  ParallelLevel max_lvl = ParallelLevel::NONE;
  bool is_outer = false; // if it is the outer-most pb
  bool enforced = false;

public:
  ParallelBy(const location& l, const ptr<Identifier>& pv,
             const ptr<Expr>& pb = nullptr, const ptr<MultiValues>& c = nullptr,
             const ptr<MultiValues>& cbs = nullptr,
             const ptr<MultiNodes>& ss = nullptr, bool a = false,
             ParallelLevel s = ParallelLevel::NONE, bool b = false)
      : Node(l), bpv(pv), bound_expr(pb), cmpt_bpvs(c), cmpt_bounds(cbs),
        stmts(ss), async(a), bracketed(b) {

    assert(bpv != nullptr && "requires a parallel variable.");
    if (cmpt_bpvs == nullptr) {
      assert(cmpt_bounds == nullptr);
      cmpt_bpvs = Make<MultiValues>(pv->LOC());
      cmpt_bounds = Make<MultiValues>(pv->LOC());
    } else {
      assert(cmpt_bounds != nullptr);
      for (auto sv : cmpt_bpvs->AllValues())
        assert(isa<Identifier>(sv) && "expect components to be identifiers.");
    }

    if (stmts == nullptr) stmts = Make<MultiNodes>(l);

    // note: leave component count check to early sema

    SetLevel(s); // override it for the parallel-level annotation

    // fill the upper-bound for `parallel p={px,py,pz} by [2,3,4] {}`
    if (bpv != nullptr && bound_expr == nullptr && cmpt_bounds->Count() > 0) {
      auto e = cast<Expr>(cmpt_bounds->ValueAt(0)->Clone());
      for (size_t i = 1; i < cmpt_bounds->Count(); ++i)
        e = Make<Expr>(e->LOC(), "*", e, CloneP(cmpt_bounds->ValueAt(i)));
      bound_expr = e;
    }
  }

  bool HasSubPVs() const { return !cmpt_bpvs->None(); }

  bool IsBracketed() const { return bracketed; }
  void SetBracketed(bool b) { bracketed = b; }

  void SetEnforced(bool e = true) { enforced = e; }
  bool IsEnforced() const { return enforced; }

  bool IsAsync() const { return async; }
  void SetAsync(bool a) { async = a; }
  //  size_t SubCount() const { return cmpt_bpvs->Count(); }

  const ptr<Identifier> BPV() const { return bpv; }
  void SetPV(const ptr<Identifier>& pv) { bpv = pv; }

  const ptr<MultiValues> SubPVs() const { return cmpt_bpvs; }
  void SetSubPVs(const ptr<MultiValues>& spv) { cmpt_bpvs = spv; }
  size_t SubPVCount() const { return cmpt_bpvs->Count(); }
  const ptr<MultiValues> BoundExprs() const { return cmpt_bounds; }
  size_t SubBoundCount() const { return cmpt_bounds->Count(); }
  void SetBoundExprs(const ptr<MultiValues>& sbs) { cmpt_bounds = sbs; }
  const NodeList AllSubPVs() const { return cmpt_bpvs->AllValues(); }
  const NodeList AllBoundExprs() const { return cmpt_bounds->AllValues(); }

  ValueItem BoundValue() const {
    if (!bound_expr->Opts().HasVal()) return GetInvalidValueItem();
    return bound_expr->Opts().GetVal();
  }
  ptr<Expr> BoundExpr() const { return bound_expr; }
  void SetBoundExpr(const ptr<Expr>& be) { bound_expr = be; }
  ptr<Expr> BoundExprAt(size_t idx) {
    assert(cmpt_bounds != nullptr);
    return cast<Expr>(cmpt_bounds->ValueAt(idx));
  }

  const ptr<Identifier> GetSubPV(size_t idx) const {
    assert(idx < cmpt_bpvs->Count() && "index out of bound!");
    return cast<Identifier>(cmpt_bpvs->ValueAt(idx));
  }

  // Get the parallel variable and its bound
  std::pair<ptr<Identifier>, ptr<Node>> GetPVB(size_t idx) const {
    assert(idx < cmpt_bpvs->Count() && "index out of bound!");
    return std::make_pair(cast<Identifier>(cmpt_bpvs->ValueAt(idx)),
                          cmpt_bounds->ValueAt(idx));
  }

  // Return a ValueList which contains values of bound items.
  // notice: the values are just number or identifier names.
  const ValueList BoundValues() const {
    ValueList vl;
    for (auto b : cmpt_bounds->AllValues()) {
      auto e = cast<Expr>(b);
      if (!e->Opts().HasVal()) return {};
      vl.push_back(e->Opts().GetVal());
    }
    return vl;
  }

  ParallelLevel GetMaxLevel() const { return max_lvl; }
  void SetMaxLevel(ParallelLevel pl) { max_lvl = pl; }
  bool IsOuter() const { return is_outer; }
  void SetOuter(bool o) { is_outer = o; }

public:
  bool IsBlock() const override { return true; }

  ptr<Node> CloneImpl() const override {
    auto pb = Make<ParallelBy>(LOC(), CloneP(bpv), CloneP(bound_expr),
                               CloneP(cmpt_bpvs), CloneP(cmpt_bounds),
                               CloneP(stmts), async);
    pb->SetBracketed(IsBracketed());
    pb->SetMaxLevel(GetMaxLevel());
    pb->SetOuter(IsOuter());
    pb->SetEnforced(IsEnforced());
    return pb;
  }

  void InlinePrint(std::ostream& os, const std::string& prefix = {},
                   bool with_type = false) const override {
    os << prefix << "parallel ";
    if (bpv) bpv->Print(os, "", with_type);
    if (bpv && cmpt_bpvs) os << " = ";
    if (cmpt_bpvs) {
      os << "{";
      cmpt_bpvs->InlinePrint(os, "", with_type);
      os << "}";
    }
    os << " by [";
    PrintBounds(os);
    os << "]";
    if (GetLevel() != ParallelLevel::NONE) os << " : " << STR(GetLevel());
  }

  void PrintBound(std::ostream& os) const {
    auto ob = BoundValue();
    os << ((IsValidValueItem(ob)) ? ob->ToString() : STR(bound_expr));
  }

  void PrintBounds(std::ostream& os) const {
    auto obvs = BoundValues();
    if (obvs.empty())
      cmpt_bounds->InlinePrint(os);
    else
      os << Choreo::STR(obvs);
  }

  void PrintWithoutStmts(std::ostream& os, const std::string& prefix,
                         bool = false) const {
    os << "\n" << prefix << "`- ";
    if (GetLevel() != ParallelLevel::NONE) os << STR(GetLevel()) << " ";
    os << "Parallelization" << (IsOuter() ? "(o)" : "") << ":"
       << " index symbol: " << bpv->name << ", bound [0, ";
    PrintBound(os);
    os << ")";
    if (HasSubPVs()) {
      os << "\n"
         << prefix << "                   "
         << " index component: {";
      cmpt_bpvs->InlinePrint(os);
      os << "}, corresponding ubound: [";
      PrintBounds(os);
      os << "]";
    }
    if (!Note().empty()) {
      for (const auto& [k, v] : Note())
        os << "\n" << prefix << "   (note: [" << k << ", " << v << "])";
    }
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    PrintWithoutStmts(os, prefix, with_type);
    stmts->Print(os, prefix + " ", with_type);
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

  ptr<Node> CloneImpl() const override {
    return Make<WhereBind>(LOC(), CloneP(lhs), CloneP(rhs));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    os << prefix << "`- " << STR(*lhs) << " bind-to " << STR(*rhs) << "\n";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, WhereBind)
};

struct WithIn : public Node, public TypeIDProvider<WithIn> {
  ptr<Identifier> with;
  ptr<Node> in;
  ptr<MultiValues> with_matchers;

  WithIn(const location& l, const ptr<Identifier>& w, const ptr<Node>& i)
      : Node(l), with(w), in(i), with_matchers(nullptr) {}

  WithIn(const location& l, const ptr<Node>& i, const ptr<MultiValues>& m)
      : Node(l), with(nullptr), in(i), with_matchers(m) {}

  WithIn(const location& l, const ptr<Identifier>& w, const ptr<Node>& i,
         ptr<MultiValues> m)
      : Node(l), with(w), in(i), with_matchers(m) {}

  const NodeList& GetMatchers() const { return with_matchers->AllValues(); }

  ptr<Node> CloneImpl() const override {
    return Make<WithIn>(LOC(), CloneP(with), CloneP(in), CloneP(with_matchers));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
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

  explicit WithBlock(const location& l, const ptr<MultiNodes>& w = nullptr,
                     const ptr<MultiNodes>& r = nullptr,
                     const ptr<MultiNodes>&& ss = nullptr)
      : Node(l), withins(w), reqs(r), stmts(ss) {}

  bool IsBlock() const override { return true; }

  ptr<Node> CloneImpl() const override {
    return Make<WithBlock>(LOC(), CloneP(withins), CloneP(reqs), CloneP(stmts));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- With Block:\n";
    os << prefix << "  (within constraints)\n";
    withins->Print(os, prefix + "  ", with_type);
    if (reqs) {
      os << prefix << "  (where clause)\n";
      reqs->Print(os, prefix + "  ", with_type);
    }
    if (stmts) {
      if (stmts->values.size() == 0) {
        os << prefix << "  (with empty statements)\n";
        return;
      }
      os << prefix << "  (with statements)";
      stmts->Print(os, prefix + "  ", with_type);
    }
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, WithBlock)
};

// Information about operation on data, including tile/tiling, subscription, or
// reshape, and etc..
struct SpannedOperation {
protected:
  const location loc;

  Shape block_shape;       // block shape after applying the operation
  ValueList block_strides; // block strides after applying the operation

public:
  SpannedOperation(const location& l) : loc(l) {}
  virtual ~SpannedOperation();

public:
  virtual const location& LOC() const { return loc; }
  virtual void SetBlockShape(const Shape shape) {
    if (!shape.IsValidOrDummy())
      choreo_unreachable("invalid shape is specified.");
    block_shape = shape;
  }

  virtual const Shape& GetBlockShape() const {
    if (!block_shape.IsValidOrDummy())
      choreo_unreachable("retrieving an invalid shape.");
    return block_shape;
  }

  virtual size_t GetRank() const {
    if (!block_shape.IsValidOrDummy())
      choreo_unreachable("retrieving an invalid shape.");
    return block_shape.Rank();
  }
  virtual const ValueList GetBlockStrides() const { return block_strides; }
  virtual void SetBlockStrides(const ValueList& s) { block_strides = s; }

  virtual const NodeList TilingFactorNodes() const { return {}; }
  virtual const NodeList SubSpanNodes() const { return {}; }
  virtual const NodeList StrideNodes() const { return {}; }
  virtual const NodeList StepNodes() const { return {}; }
  virtual const NodeList IndexNodes() const { return {}; }
  virtual const NodeList OffsetNodes() const { return {}; }
  virtual const NodeList ReferredNodes() const = 0;
  virtual const ptr<MultiValues> GetTilingFactors() const { return nullptr; }
  virtual const ptr<MultiValues> GetSubSpan() const { return nullptr; }
  virtual const ptr<MultiValues> GetStrides() const { return nullptr; }
  virtual const ptr<MultiValues> GetSteps() const { return nullptr; }
  virtual const ptr<MultiValues> GetIndices() const { return nullptr; }
  virtual const ptr<MultiValues> GetOffsets() const { return nullptr; }

  virtual const ptr<SpannedOperation> CloneImpl() const = 0;
  virtual const ptr<SpannedOperation> Clone() const {
    auto sop = CloneImpl();
    sop->block_shape = block_shape;
    sop->block_strides = block_strides;
    return sop;
  }

  virtual void accept(Visitor& v) = 0;
  virtual void Print(std::ostream& os) const = 0;

  virtual void Dump(std::ostream& os) const {
    Print(os);
    os << "(shape: " << STR(GetBlockShape()) << ", strides: ";
    PrintValueList(GetBlockStrides(), os);
    os << ")";
  }

  __UDT_TYPE_INFO_BASE__(spanned - operation)
};

inline std::string STR(const SpannedOperation& so) {
  std::ostringstream oss;
  so.Print(oss);
  return oss.str();
}

inline Identifier* GetIdentifier(const Node&);
inline ptr<Expr> MakeIntExpr(const location&, int);

namespace SOP {
struct Tiling : public SpannedOperation, public TypeIDProvider<Tiling> {
  ptr<MultiValues> tfactor = nullptr; // tiling-factor
  Tiling(const location& l, const ptr<MultiValues>& tf)
      : SpannedOperation(l), tfactor(tf) {
    if (tf == nullptr) choreo_unreachable("must provide the tiling factor.");
  }
  const NodeList TilingFactorNodes() const override {
    return MakeNodeList(GetTilingFactors());
  }
  const NodeList IndexNodes() const override { return MakeNodeList(tfactor); }
  const NodeList ReferredNodes() const override { return TilingFactorNodes(); }
  const ptr<MultiValues> GetTilingFactors() const override {
    auto res = AST::Make<AST::MultiValues>(tfactor->LOC());

    for (auto v : tfactor->AllValues()) {
      if (auto id = GetIdentifier(*v); id && (id->name == "_"))
        res->Append(AST::MakeIntExpr(id->LOC(), 1));
      else
        res->Append(AST::Make<AST::Expr>(v->LOC(), "ubound", v->Clone()));
    }
    res->SetDelimiter(", ");
    return res;
  }

  const ptr<MultiValues> GetIndices() const override { return tfactor; }
  const ptr<SpannedOperation> CloneImpl() const override {
    return Make<Tiling>(LOC(), CloneP(tfactor));
  }

  void accept(Visitor& v) override;

  void Print(std::ostream& os) const override {
    os << ".ChunkAt(" << STR(tfactor) << ")";
  }

  __UDT_TYPE_INFO__(SpannedOperation, Tiling)
};

struct TileAt : public SpannedOperation, public TypeIDProvider<TileAt> {
  ptr<MultiValues> tfactor = nullptr; // tiling-factor
  ptr<MultiValues> indices = nullptr; // subscripting indices
  TileAt(const location& l, const ptr<MultiValues>& tf,
         const ptr<MultiValues>& i)
      : SpannedOperation(l), tfactor(tf), indices(i) {}

  const ptr<MultiValues> GetTilingFactors() const override { return tfactor; }
  const ptr<MultiValues> GetIndices() const override { return indices; }
  const NodeList TilingFactorNodes() const override {
    return MakeNodeList(tfactor);
  }
  const NodeList IndexNodes() const override { return MakeNodeList(indices); }
  const NodeList ReferredNodes() const override {
    auto res = TilingFactorNodes();
    auto& idn = IndexNodes();
    res.insert(res.end(), idn.begin(), idn.end());
    return res;
  }
  const ptr<SpannedOperation> CloneImpl() const override {
    return Make<TileAt>(LOC(), CloneP(tfactor), CloneP(indices));
  }

  void accept(Visitor& v) override;

  void Print(std::ostream& os) const override {
    os << ".Chunk(" << STR(tfactor) << ").At(" << STR(indices) << ")";
  }

  __UDT_TYPE_INFO__(SpannedOperation, TileAt)
};

struct SubSpan : public SpannedOperation, public TypeIDProvider<SubSpan> {
  ptr<MultiValues> subspan = nullptr; // subspan shape
  ptr<MultiValues> indices = nullptr; // subscripting indices
  ptr<MultiValues> steps = nullptr;   // optional steps
  ptr<MultiValues> strides = nullptr; // optional strides
  SubSpan(const location& l, const ptr<MultiValues>& s,
          const ptr<MultiValues>& i, const ptr<MultiValues>& stp = nullptr,
          const ptr<MultiValues>& strd = nullptr)
      : SpannedOperation(l), subspan(s), indices(i), steps(stp), strides(strd) {
    if (s == nullptr) choreo_unreachable("must provide the sub-span.");
  }

  const ptr<MultiValues> GetSubSpan() const override { return subspan; }
  const ptr<MultiValues> GetSteps() const override { return steps; }
  const ptr<MultiValues> GetStrides() const override { return strides; }
  const ptr<MultiValues> GetIndices() const override { return indices; }
  const NodeList SubSpanNodes() const override { return MakeNodeList(subspan); }
  const NodeList StepNodes() const override { return MakeNodeList(steps); }
  const NodeList StrideNodes() const override { return MakeNodeList(strides); }
  const NodeList IndexNodes() const override { return MakeNodeList(indices); }
  const NodeList ReferredNodes() const override {
    auto res = SubSpanNodes();
    auto& idn = IndexNodes();
    res.insert(res.end(), idn.begin(), idn.end());
    auto& stn = StepNodes();
    res.insert(res.end(), stn.begin(), stn.end());
    auto& sdn = StrideNodes();
    res.insert(res.end(), sdn.begin(), sdn.end());
    return res;
  }
  void SetIndexNodes(const ptr<AST::MultiValues>& mv) { indices = mv; }
  const ptr<SpannedOperation> CloneImpl() const override {
    return Make<SubSpan>(LOC(), CloneP(subspan), CloneP(indices), CloneP(steps),
                         CloneP(strides));
  }

  void accept(Visitor& v) override;
  void Print(std::ostream& os) const override {
    os << ".SubSpan(" << STR(subspan) << ")";
    if (steps) os << ".Step(" << STR(steps) << ")";
    if (strides) os << ".Stride(" << STR(steps) << ")";
    if (indices) os << ".At(" << STR(indices) << ")";
  }

  __UDT_TYPE_INFO__(SpannedOperation, SubSpan)
};

struct ModSpan : public SpannedOperation, public TypeIDProvider<ModSpan> {
  ptr<MultiValues> subspan = nullptr; // subspan shape
  ptr<MultiValues> indices = nullptr; // subscripting indices
  ptr<MultiValues> steps = nullptr;   // optional steps
  ptr<MultiValues> strides = nullptr; // optional sttrides
  ModSpan(const location& l, const ptr<MultiValues>& s,
          const ptr<MultiValues>& i, const ptr<MultiValues>& stp = nullptr,
          const ptr<MultiValues>& strd = nullptr)
      : SpannedOperation(l), subspan(s), indices(i), steps(stp), strides(strd) {
    if (s == nullptr) choreo_unreachable("must provide the sub-span.");
  }

  const ptr<MultiValues> GetSubSpan() const override { return subspan; }
  const ptr<MultiValues> GetSteps() const override { return steps; }
  const ptr<MultiValues> GetStrides() const override { return strides; }
  const ptr<MultiValues> GetIndices() const override { return indices; }
  const NodeList SubSpanNodes() const override { return MakeNodeList(subspan); }
  const NodeList StepNodes() const override { return MakeNodeList(steps); }
  const NodeList StrideNodes() const override { return MakeNodeList(strides); }
  const NodeList IndexNodes() const override { return MakeNodeList(indices); }
  const NodeList ReferredNodes() const override {
    auto res = SubSpanNodes();
    auto& idn = IndexNodes();
    res.insert(res.end(), idn.begin(), idn.end());
    auto& stn = StepNodes();
    res.insert(res.end(), stn.begin(), stn.end());
    auto& sdn = StrideNodes();
    res.insert(res.end(), sdn.begin(), sdn.end());
    return res;
  }
  void SetIndexNodes(const ptr<AST::MultiValues>& mv) { indices = mv; }
  const ptr<SpannedOperation> CloneImpl() const override {
    return Make<ModSpan>(LOC(), CloneP(subspan), CloneP(indices), CloneP(steps),
                         CloneP(strides));
  }

  void accept(Visitor& v) override;
  void Print(std::ostream& os) const override {
    os << ".ModSpan(" << STR(subspan) << ")";
    if (steps) os << ".Step(" << STR(steps) << ")";
    if (strides) os << ".Stride(" << STR(steps) << ")";
    if (indices) os << ".At(" << STR(indices) << ")";
  }

  __UDT_TYPE_INFO__(SpannedOperation, ModSpan)
};

struct View : public SpannedOperation, public TypeIDProvider<View> {
  ptr<MultiValues> subspan = nullptr; // subspan shape
  ptr<MultiValues> offsets = nullptr; // optional offsets
  ptr<MultiValues> strides = nullptr; // optional strides
  View(const location& l, const ptr<MultiValues>& s,
       const ptr<MultiValues>& off, const ptr<MultiValues>& strd = nullptr)
      : SpannedOperation(l), subspan(s), offsets(off), strides(strd) {
    if (s == nullptr) choreo_unreachable("must provide the sub-span.");
  }
  const ptr<MultiValues> GetSubSpan() const override { return subspan; }
  const ptr<MultiValues> GetStrides() const override { return strides; }
  const ptr<MultiValues> GetOffsets() const override { return offsets; }
  const NodeList SubSpanNodes() const override { return MakeNodeList(subspan); }
  const NodeList StrideNodes() const override { return MakeNodeList(strides); }
  const NodeList OffsetNodes() const override { return MakeNodeList(offsets); }
  const NodeList ReferredNodes() const override {
    auto res = SubSpanNodes();
    auto& ofn = OffsetNodes();
    res.insert(res.end(), ofn.begin(), ofn.end());
    auto& stn = StrideNodes();
    res.insert(res.end(), stn.begin(), stn.end());
    return res;
  }
  const ptr<SpannedOperation> CloneImpl() const override {
    return Make<View>(LOC(), CloneP(subspan), CloneP(offsets), CloneP(strides));
  }

  void accept(Visitor& v) override;
  void Print(std::ostream& os) const override {
    os << ".VIEW(" << STR(subspan) << ")";
    if (strides) os << ".Stride(" << STR(strides) << ")";
    if (offsets) os << ".From(" << STR(offsets) << ")";
  }

  __UDT_TYPE_INFO__(SpannedOperation, View)
};

struct Reshape : public SpannedOperation, public TypeIDProvider<Reshape> {
  ptr<MultiValues> newspan = nullptr; // new shape
  Reshape(const location& l, const ptr<MultiValues>& s)
      : SpannedOperation(l), newspan(s) {
    if (s == nullptr) choreo_unreachable("must provide the sub-span.");
  }
  const ptr<MultiValues> GetNewSpan() const { return newspan; }
  const NodeList ReferredNodes() const override {
    return MakeNodeList(newspan);
  }
  const ptr<SpannedOperation> CloneImpl() const override {
    return Make<Reshape>(LOC(), CloneP(newspan));
  }

  void accept(Visitor& v) override;
  void Print(std::ostream& os) const override {
    os << ".SpanAs(" << STR(newspan) << ")";
  }
  __UDT_TYPE_INFO__(SpannedOperation, Reshape)
};

} // end namespace SOP

// If the block after applying the operation is contiguous in the original
// block. Return true/false: true positive or true negative Return ValueItem if
// the result cannot be determined definitively. Contiguous if
//  the ValueItem is evaluted to true at runtime.
inline std::variant<bool, ValueItem> IsContiguousSOp(const SpannedOperation& k,
                                                     Shape original_shape) {
  if (isa<SOP::Reshape>(&k)) return true;

  Shape new_shape = k.GetBlockShape();
  assert(original_shape.SameRankAs(new_shape));

  size_t rank = original_shape.Rank();

  // TODO: if nil*nil*64, then the stride is ?
  // workaround: modify nil to a dummy value.
  auto ModifyNil = [](Shape& s) {
    auto vl = s.Value();
    for (auto& vi : vl)
      if (VIIsNil(vi)) vi = sbe::nu(2);
    return s = Shape(s.Rank(), vl);
  };
  ModifyNil(original_shape);
  ModifyNil(new_shape);

  auto ComputeStrides = [rank](const Shape s) -> ValueList {
    ValueList strides(rank, sbe::nu(1));
    for (int i = rank - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * s.ValueAt(i + 1);
    return strides;
  };
  auto strides = ComputeStrides(original_shape);

  ValueList last_idx(rank);
  for (size_t k = 0; k < rank; ++k)
    last_idx[k] = new_shape.ValueAt(k) - sbe::nu(1);
  auto first = sbe::nu(0);
  auto last = std::inner_product(last_idx.begin(), last_idx.end(),
                                 strides.begin(), sbe::nu(0));
  auto N = new_shape.ElementCountValue();

  auto res = oc_eq(last - first + sbe::nu(1), N);
  if (sbe::is_true(res)) return true;
  if (sbe::is_false(res)) return false;
  return res->Normalize();
}

struct ChunkAt : public Node, public TypeIDProvider<ChunkAt> {
  ptr<Identifier> data;               // spanned data name
  ptr<MultiValues> indices = nullptr; // indexing of data arrays
  ptr<SpanAs> sa = nullptr;           // for span_as expression. TODO: deprecate

private:
  std::vector<ptr<SpannedOperation>> operations;

private:
  Shape bs; // internally used to pass real shape of the tiled block

public:
  ChunkAt(const location& l, const ptr<Identifier>& d,
          const ptr<MultiValues>& idxes = nullptr,
          const std::vector<ptr<SpannedOperation>>& ops = {})
      : Node(l), data(d), indices(idxes), operations(ops) {}

  ChunkAt(const location& l, const ptr<SpanAs>& s,
          const ptr<MultiValues>& idxes = nullptr,
          const std::vector<ptr<SpannedOperation>>& ops = {})
      : Node(l), data(s->nid), indices(idxes), sa(s), operations(ops) {}

  const std::string RefSymbol() const {
    assert(data && "ref data is not set.");
    return RemoveSuffix(data->name, ".data");
  }

  bool NoOperation() const { return operations.empty(); }
  bool HasOperation() const { return !operations.empty(); }
  const std::vector<ptr<SpannedOperation>>& AllOperations() const {
    return operations;
  }
  const ptr<SpannedOperation>& OpAt(size_t index) const {
    assert(index < operations.size());
    return operations[index];
  }
  const ptr<SpannedOperation>& FirstOp() const {
    assert(!operations.empty());
    return operations.front();
  }
  const ptr<SpannedOperation>& LastOp() const {
    assert(!operations.empty());
    return operations.back();
  }
  size_t OpCount() const { return operations.size(); }
  bool NoTilingOperation() const { return !HasTilingOperation(); }
  bool HasTilingOperation() const { return TilingOperationCount() > 0; }
  size_t TilingOperationCount() const {
    size_t count = 0;
    for (auto& so : AllOperations())
      if (!isa<SOP::Reshape>(so)) count++;
    return count;
  }
  bool HasReshape() const {
    for (auto& so : AllOperations())
      if (isa<SOP::Reshape>(so)) return true;
    return false;
  }
  bool ReshapeOnly() const {
    bool reshape_only = false;
    for (auto& so : AllOperations()) {
      if (!isa<SOP::Reshape>(so)) return false;
      reshape_only = true;
    }
    return reshape_only;
  }

  std::optional<size_t> IndexOfLastSpanAs() const {
    int li = -1;
    int i = 0;
    for (auto sop : AllOperations()) {
      if (isa<SOP::Reshape>(sop)) li = i;
      ++i;
    }
    if (li == -1) return std::nullopt;
    return li;
  }

  const Shape& GetBlockShape() const { return bs; }
  void SetBlockShape(const Shape& shape) { bs = shape; }

  void RemoveOperation(size_t index) {
    operations.erase(operations.begin() + index);
  }

  ptr<Node> CloneImpl() const override {
    std::vector<ptr<SpannedOperation>> ntsis;
    for (auto tsi : operations) ntsis.push_back(tsi->Clone());
    auto n = Make<ChunkAt>(LOC(), CloneP(data), CloneP(indices), ntsis);
    n->sa = (!sa) ? nullptr : CloneP(sa);
    n->bs = bs;
    return n;
  }

  void Print(std::ostream& os, const std::string& = {},
             bool with_type = false) const override {
    if (sa)
      os << PSTR(sa);
    else
      os << PSTR(data);

    if (indices)
      for (auto index : indices->AllValues()) os << "[" << PSTR(index) << "]";

    for (auto tsi : operations) tsi->Print(os);

    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, ChunkAt)
};

struct Select : public Node, public TypeIDProvider<Select> {
  std::string rname;
  ptr<Expr> select_factor = nullptr;
  ptr<MultiValues> expr_list = nullptr;
  bool inDMA = false;

  Select(const location& l, const ptr<Expr>& sf, const ptr<MultiValues>& list)
      : Node(l), select_factor(sf), expr_list(list) {
    assert(sf);
    assert(list);
  }

  ptr<Node> CloneImpl() const override {
    auto n = Make<Select>(LOC(), CloneP(select_factor), CloneP(expr_list));
    n->rname = rname;
    n->inDMA = inDMA;
    return n;
  }

  void Print(std::ostream& os, const std::string& = {},
             bool with_type = false) const override {
    os << "select(";
    select_factor->Print(os, "", with_type);
    os << ", ";
    expr_list->Print(os, "", with_type);
    os << ")";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Select)
};

struct DMAAttribute {
  SwizMode sw_mode = SwizMode::NONE;
  bool zfill = false;
  bool is_sparse = false;
  int sparse_n = 0;
  int sparse_m = 0;
  DMAAttribute(SwizMode swiz = SwizMode::NONE, bool zf = false, bool sp = false,
               int sp_n = 0, int sp_m = 0)
      : sw_mode(swiz), zfill(zf), is_sparse(sp), sparse_n(sp_n),
        sparse_m(sp_m) {}
};

struct DMA : public Node, public TypeIDProvider<DMA> {
  std::string operation;
  std::string future;

private:
  bool async;
  bool enforce_tma;

public:
  // if this DMA is chained with other DMA in pipeline mode
  bool chained;
  // SYMBOL string of its chained DMA B, direction is B->A
  std::string chain_from;
  // SYMBOL string of its chained DMA B, direction is A->B
  std::string chain_to;
  ptr<Node> from = nullptr;
  ptr<Node> to = nullptr;
  DMAAttribute attr;
  ptr<DMAConfig> config = nullptr;

public:
  explicit DMA(const location& l, const std::string& o, const std::string& r,
               const ptr<Node>& f, const ptr<Node>& t, bool a,
               const DMAAttribute& at = {}, bool is_tma = false,
               const ptr<DMAConfig>& c = nullptr)
      : Node(l, MakeDummyFutureType(a)), operation(o), future(r), async(a),
        enforce_tma(is_tma), from(f), to(t), attr(at), config(c) {
    chained = false;
    chain_to = "";
    chain_from = "";
    if (auto tptr = dyn_cast<AST::Select>(t)) tptr->inDMA = true;
  }

  explicit DMA(const location& l, const std::string& o, const std::string& r,
               const std::string& chained_from, const ptr<Node>& f,
               const ptr<Node>& t, bool a, const DMAAttribute& at = {},
               bool is_tma = false, const ptr<DMAConfig>& c = nullptr)
      : Node(l, MakeDummyFutureType(a)), operation(o), future(r), async(a),
        enforce_tma(is_tma), from(f), to(t), attr(at), config(c) {
    chained = true;
    chain_from = chained_from;
    if (auto tptr = dyn_cast<AST::Select>(t)) tptr->inDMA = true;
  }

  // The dummy dma
  explicit DMA(const location& l, const std::string& f, bool is_tma = false)
      : Node(l, MakePlaceHolderFutureType()), operation(".any"), future(f),
        async(true), enforce_tma(is_tma) {}

  bool IsDummy() const { return operation == ".any"; }
  const ptr<Node> GetFrom() const { return from; }
  const ptr<Node> GetTo() const { return to; }
  const ptr<ChunkAt> GetSrc() const { return cast<ChunkAt>(from); }
  const ptr<ChunkAt> GetDst() const { return cast<ChunkAt>(to); }
  bool IsDstInferred() const { return isa<Memory>(to); }

  std::string FromSymbol() const { return cast<ChunkAt>(from)->RefSymbol(); }

  std::string ToSymbol() const {
    if (auto tochunk = dyn_cast<ChunkAt>(to)) return tochunk->data->name;
    return "";
  }

  void SetConfig(const ptr<DMAConfig>& cfg) { config = cfg; }
  bool IsSparse() const { return attr.is_sparse; }
  bool IsOOBZeroFill() const { return attr.zfill; }
  SwizMode GetSwizzleMode() const { return attr.sw_mode; }
  const std::pair<int, int> GetSparsePattern() const {
    return {attr.sparse_n, attr.sparse_m};
  }

  const ptr<DMAConfig>& GetConfig() const { return config; }

  ptr<Node> CloneImpl() const override {
    auto n = Make<DMA>(LOC(), operation, future, CloneP(from), CloneP(to),
                       async, attr, enforce_tma, config);
    n->chained = chained;
    n->chain_from = chain_from;
    n->chain_to = chain_to;
    return n;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    if (operation == ".any") {
      os << "\n" << prefix << "`- DMA" << operation;
      if (with_type) os << "<{" << PSTR(GetType()) << "}>";
      if (!future.empty()) os << "\n" << prefix << "  `- future: " << future;
      return;
    }

    os << "\n" << prefix << "`- DMA" << operation << ((async) ? ".async" : "");
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
    if (config) os << "\n" << prefix << "  `- config: " << STR(*config);
    if (!future.empty()) os << "\n" << prefix << "  `- future: " << future;
    if (attr.is_sparse) {
      os << "\n"
         << prefix << "  `- sparse: " << attr.sparse_n << ":" << attr.sparse_m;
    }
    os << "\n" << prefix << "  `- from: ";
    from->Print(os, "", with_type);
    os << "\n" << prefix << "  `- to: ";
    to->Print(os, "", with_type);

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

  bool IsAsync() const { return async; }
  bool IsTMA() const { return enforce_tma; }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, DMA)
};

struct MMAOperation {
public:
  enum Kind { Fill, Load, Exec, Store, Commit };
  enum ExecMethod { ROW_ROW, ROW_COL, COL_ROW, COL_COL };

  // NOTE: acc, lhs, rhs are not accepted in ast.cpp.

  struct FillInfo {
    // TODO: for now, fill can only be operated on fragment C.
    // If multibuffer is to be enabled, fill should be able to all fragments.
    // And, mma.load should pass frag as parameter rather than return value.
    ptr<Expr> buffer;
    ptr<Expr> fill_expr;
    bool is_decl; // If `mma.fill mc, 0.0f`, false
    BaseType fill_elem_type;
    ptr<MultiValues> array_dims; // nullptr by default.
  };
  struct LoadInfo {
    // for now, load is always decl+load.
    ptr<ChunkAt> ld_expr;
    ptr<Expr> future;
    bool async;
    SwizMode swiz_mode; // 128, 64, or 32; default 128
  };
  struct ExecInfo {
    ExecMethod method;
    ptr<Expr> acc;
    ptr<Expr> lhs;
    ptr<Expr> rhs;
    ptr<Expr> mdata;
    bool is_sparse;
    bool scale;
    ptr<ChunkAt> scale_a;
    ptr<Expr> scale_b;
  };
  struct StoreInfo {
    ptr<Expr> buffer;
    ptr<ChunkAt> st_expr;
  };
  using InfoType = std::variant<FillInfo, LoadInfo, ExecInfo, StoreInfo>;

private:
  Kind tag;
  InfoType info;

public:
  MMAOperation(const ptr<Expr>& n, const ptr<Expr>& e, bool is_decl,
               BaseType t = BaseType::UNKSCALAR)
      : tag(Fill), info(FillInfo{n, e, is_decl, t, nullptr}) {}

  MMAOperation(const ptr<ChunkAt>& e, const ptr<Expr>& fu, bool a = false,
               SwizMode swizzle = SwizMode::B128)
      : tag(Load), info(LoadInfo{e, fu, a, swizzle}) {}

  MMAOperation(ExecMethod m, const ptr<Expr>& o, const ptr<Expr>& l,
               const ptr<Expr>& r, bool sp = false)
      : tag(Exec),
        info(ExecInfo{m, o, l, r, nullptr, sp, false, nullptr, nullptr}) {}
  MMAOperation(ExecMethod m, const ptr<Expr>& o, const ptr<Expr>& l,
               const ptr<Expr>& r, const ptr<Expr>& e, bool sp)
      : tag(Exec), info(ExecInfo{m, o, l, r, e, sp, false, nullptr, nullptr}) {}
  MMAOperation(ExecMethod m, const ptr<Expr>& o, const ptr<Expr>& l,
               const ptr<Expr>& r, const ptr<ChunkAt>& scale_a,
               const ptr<Expr>& scale_b)
      : tag(Exec),
        info(ExecInfo{m, o, l, r, nullptr, false, true, scale_a, scale_b}) {}

  MMAOperation(const ptr<Expr>& n, const ptr<ChunkAt>& c)
      : tag(Store), info(StoreInfo{n, c}) {}

  MMAOperation() : tag(Commit), info() {}

public:
  bool IsKind(Kind k) const { return k == tag; }

  const ptr<Expr> FillingTo() const {
    if (tag != Fill) choreo_unreachable("not a mma fill operation.");
    return std::get<0>(info).buffer;
  }
  ptr<Expr> FillingValue() {
    if (tag != Fill) choreo_unreachable("not a mma fill operation.");
    return std::get<0>(info).fill_expr;
  }
  const ptr<Expr> FillingValue() const {
    if (tag != Fill) choreo_unreachable("not a mma fill operation.");
    return std::get<0>(info).fill_expr;
  }
  BaseType FillingType() const {
    if (tag != Fill) choreo_unreachable("not a mma fill operation.");
    return std::get<0>(info).fill_elem_type;
  }
  bool FillingIsDecl() const {
    if (tag != Fill) choreo_unreachable("not a mma fill operation.");
    return std::get<0>(info).is_decl;
  }
  void SetFillingArrayDims(ptr<MultiValues> d) {
    if (tag != Fill) choreo_unreachable("not a mma fill operation.");
    std::get<0>(info).array_dims = d;
  }
  const ptr<MultiValues> FillingArrayDims() const {
    if (tag != Fill) choreo_unreachable("not a mma fill operation.");
    return std::get<0>(info).array_dims;
  }

  ptr<ChunkAt> LoadFrom() {
    if (tag != Load) choreo_unreachable("not a mma load operation.");
    auto l_info = std::get<1>(info);
    return l_info.ld_expr;
  }
  const ptr<ChunkAt> LoadFrom() const {
    if (tag != Load) choreo_unreachable("not a mma load operation.");
    auto l_info = std::get<1>(info);
    return l_info.ld_expr;
  }
  const ptr<Expr> LoadTo() const {
    if (tag != Load) choreo_unreachable("not a mma load operation.");
    auto l_info = std::get<1>(info);
    return l_info.future;
  }

  ptr<ChunkAt> StoreTo() {
    if (tag != Store) choreo_unreachable("not a mma store operation.");
    return std::get<3>(info).st_expr;
  }
  const ptr<ChunkAt> StoreTo() const {
    if (tag != Store) choreo_unreachable("not a mma store operation.");
    return std::get<3>(info).st_expr;
  }
  const ptr<Expr> StoreFrom() const {
    if (tag != Store) choreo_unreachable("not a mma store operation.");
    return std::get<3>(info).buffer;
  }

  void SetAsync(bool async = true) {
    if (tag != Load) choreo_unreachable("not a mma load operation.");
    auto l_info = std::get<1>(info);
    l_info.async = async;
  }

  bool IsAsync() const {
    if (tag != Load) choreo_unreachable("not a mma load operation.");
    auto l_info = std::get<1>(info);
    return l_info.async;
  }

  const ptr<Expr> ExecOperand(size_t index) const {
    if (tag != Exec) choreo_unreachable("not a mma exec operation.");
    auto e_info = std::get<2>(info);
    if (index == 0)
      return e_info.acc;
    else if (index == 1)
      return e_info.lhs;
    else if (index == 2)
      return e_info.rhs;
    else if (index == 3)
      return e_info.mdata;
    else
      choreo_unreachable("oob for mma exec operands.");
  }

  ExecMethod GetMethod() const {
    if (tag != Exec) choreo_unreachable("not a mma exec operation.");
    auto e_info = std::get<2>(info);
    return e_info.method;
  }

  bool IsSparse() const {
    if (tag != Exec) return false;
    auto e_info = std::get<2>(info);
    return e_info.is_sparse;
  }

  bool HasScale() const {
    if (tag != Exec) return false;
    auto e_info = std::get<2>(info);
    return e_info.scale;
  }

  ptr<ChunkAt> ScaleA() const {
    if (tag != Exec) choreo_unreachable("not a mma exec operation.");
    auto e_info = std::get<2>(info);
    return e_info.scale_a;
  }

  ptr<Expr> ScaleB() const {
    if (tag != Exec) choreo_unreachable("not a mma exec operation.");
    auto e_info = std::get<2>(info);
    return e_info.scale_b;
  }

  void SetFuture(const ptr<AST::Expr>& fut) {
    if (tag != Load) choreo_unreachable("not a mma load operation.");
    auto l_info = std::get<1>(info);
    l_info.future = fut;
  }

  const ptr<Expr> GetFuture() const { return LoadTo(); }

  const ptr<Expr> GetFrag() const {
    if (tag == Fill) return FillingTo();
    if (tag == Load) return LoadTo();
    if (tag == Exec) return ExecOperand(0);
    if (tag == Store) return StoreFrom();
    choreo_unreachable("unexpected mma operation!");
    return nullptr;
  }

  SwizMode GetSwizzleMode() const {
    if (tag != Load) choreo_unreachable("not a mma load operation.");
    auto l_info = std::get<1>(info);
    return l_info.swiz_mode;
  }

  void SetSwizzleMode(SwizMode sm) {
    if (tag != Load) choreo_unreachable("not a mma load operation.");
    auto l_info = std::get<1>(info);
    l_info.swiz_mode = sm;
  }

  Kind Tag() const { return tag; }

public:
  const ptr<MMAOperation> Clone() const {
    switch (tag) {
    case Fill:
      return Make<MMAOperation>(CloneP(FillingTo()), CloneP(FillingValue()),
                                FillingIsDecl(), FillingType());
    case Load: {
      auto l_info = std::get<1>(info);
      return Make<MMAOperation>(CloneP(l_info.ld_expr), CloneP(l_info.future),
                                l_info.async, l_info.swiz_mode);
    }
    case Exec: {
      auto e_info = std::get<2>(info);
      if (e_info.scale)
        return Make<MMAOperation>(
            e_info.method, CloneP(e_info.acc), CloneP(e_info.lhs),
            CloneP(e_info.rhs), CloneP(e_info.scale_a), CloneP(e_info.scale_b));
      return Make<MMAOperation>(e_info.method, CloneP(e_info.acc),
                                CloneP(e_info.lhs), CloneP(e_info.rhs),
                                CloneP(e_info.mdata), e_info.is_sparse);
    }
    case Store:
      return Make<MMAOperation>(CloneP(StoreFrom()), CloneP(StoreTo()));
    case Commit: return Make<MMAOperation>();
    default: choreo_unreachable("unsupported MMA operation kind.");
    }
    return nullptr;
  }

  void Print(std::ostream& os) const {
    switch (tag) {
    case Fill: {
      if (FillingIsDecl())
        os << PSTR(FillingTo()) << " = MMA.FILL " << PSTR(FillingValue());
      else
        os << "MMA.FILL " << PSTR(FillingTo()) << ", " << PSTR(FillingValue());
    } break;
    case Load: {
      auto l_info = std::get<1>(info);
      if (l_info.future) os << PSTR(l_info.future) << " = ";
      os << "MMA.LOAD" << ((l_info.async) ? ".ASYNC" : "") << " "
         << PSTR(l_info.ld_expr);
    } break;
    case Exec: {
      auto e_info = std::get<2>(info);
      os << "MMA.EXEC";
      switch (e_info.method) {
      case ROW_ROW: os << ".ROW.ROW"; break;
      case ROW_COL: os << ".ROW.COL"; break;
      case COL_COL: os << ".COL.COL"; break;
      case COL_ROW: os << ".COL.ROW"; break;
      default: choreo_unreachable("unsupported dma execution mode."); break;
      }
      if (e_info.is_sparse) os << ".SP";
      if (e_info.scale) os << ".SCALE";
      // TODO: missing scale
      os << " " << PSTR(e_info.acc) << ", " << PSTR(e_info.lhs) << ", "
         << PSTR(e_info.rhs);
    } break;
    case Store: {
      os << "MMA.STORE " << PSTR(StoreFrom()) << ", " << PSTR(StoreTo());
    } break;
    default: choreo_unreachable("unsupported MMA operation kind.");
    }
  }

private:
  void Verify() {} // TODO
};

struct MMA : public Node, public TypeIDProvider<MMA> {
  ptr<MMAOperation> operation;

public:
  MMA(const location& l, const ptr<MMAOperation>& op)
      : Node(l), operation(op) {}
  ptr<Node> CloneImpl() const override {
    return Make<MMA>(LOC(), operation->Clone());
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- ";
    operation->Print(os);
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
  }

  const ptr<MMAOperation> GetOperation() const { return operation; }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, MMA)
};

struct Wait : public Node, public TypeIDProvider<Wait> {
  ptr<MultiValues> targets;

  Wait(const location& l, const ptr<MultiValues>& t) : Node(l), targets(t) {}

  ptr<Node> CloneImpl() const override {
    return Make<Wait>(LOC(), CloneP(targets));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- WAIT: ";
    targets->Print(os, "", with_type);
  }

  const NodeList& GetTargets() const { return targets->AllValues(); }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Wait)
};

struct Trigger : public Node, public TypeIDProvider<Trigger> {
  ptr<MultiValues> targets;

  Trigger(const location& l, const ptr<MultiValues>& t) : Node(l), targets(t) {}

  ptr<Node> CloneImpl() const override {
    return Make<Trigger>(LOC(), CloneP(targets));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- TRIGGER: ";
    targets->Print(os, "", with_type);
  }

  const NodeList& GetEvents() const { return targets->AllValues(); }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Trigger)
};

struct Break : public Node, public TypeIDProvider<Break> {
  Break(const location& l) : Node(l) {}

  ptr<Node> CloneImpl() const override { return Make<Break>(LOC()); }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    os << "\n" << prefix << "`- Break";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Break)
};

struct Continue : public Node, public TypeIDProvider<Continue> {
  Continue(const location& l) : Node(l) {}

  ptr<Node> CloneImpl() const override { return Make<Continue>(LOC()); }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    os << "\n" << prefix << "`- Continue";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Continue)
};

struct Return : public Node, public TypeIDProvider<Return> {
  ptr<Node> value = nullptr;

  Return(const location& l) : Node(l) {}
  Return(const location& l, const ptr<Node>& t) : Node(l), value(t) {}

  ptr<Node> CloneImpl() const override {
    return Make<Return>(LOC(), CloneP(value));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- Return";
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
    os << ": ";
    if (!value)
      os << "void";
    else
      value->Print(os, {}, with_type);
    if (!Note().empty())
      for (const auto& [k, v] : Note()) os << " [" << k << ", " << v << "]";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Return)
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
  const NodeList& GetIds() const { return ids->AllValues(); }

  ptr<Node> CloneImpl() const override {
    return Make<Rotate>(LOC(), CloneP(ids));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n"
       << prefix << "`- " << ((ids->Count() == 2) ? "Swap: " : "Rotate: ");
    ids->Print(os, "", with_type);
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Rotate)
};

struct Synchronize : public Node, public TypeIDProvider<Synchronize> {
  Storage buf_ty;

  Synchronize(const location& loc, Storage s) : Node(loc), buf_ty(s) {}

  ptr<Node> CloneImpl() const override {
    return Make<Synchronize>(LOC(), buf_ty);
  }

  Storage Resource() const { return buf_ty; }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    os << "\n" << prefix << "`- " << "Synchronize: " << STR(buf_ty);
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Synchronize)
};

struct LoopRange : public Node, public TypeIDProvider<LoopRange> {
  ptr<Identifier> iv; // induction variable
  // both will be normalized to Expr which ref to anon_x
  ptr<Node> lbound = nullptr;
  ptr<Node> ubound = nullptr;
  int step = GetInvalidStep();

  LoopRange(const location& l, const ptr<Identifier>& i)
      : Node(l), iv(i) {} // the cmpt_bounds are yet to be inferred
  LoopRange(const location& l, const ptr<Identifier>& i, const ptr<Node>& lb,
            const ptr<Node>& ub, int s = 1)
      : Node(l), iv(i), lbound(lb), ubound(ub), step(s) {}

  const std::string IVName() const { return iv->name; }
  const ptr<Identifier> IV() const { return iv; }

  ptr<Node> CloneImpl() const override {
    return Make<LoopRange>(LOC(), (!iv) ? nullptr : CloneP(iv), CloneP(lbound),
                           CloneP(ubound), step);
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    os << "\n" << prefix << "`- Iteration variables: " << iv->name;

    if (!lbound && !ubound && !IsValidStep(step)) return;

    os << "\n" << prefix << "`- Loop Control: (";
    os << (lbound ? PSTR(lbound) : std::string("?")) << ":";
    os << (ubound ? PSTR(ubound) : std::string("?")) << ":";
    os << (IsValidStep(step) ? std::to_string(step) : std::string("?")) << ")";
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, LoopRange)
};

ptr<Call> GetCall(const ptr<Node>& n);
struct ForeachBlock : public Node, public TypeIDProvider<ForeachBlock> {
  ptr<MultiValues> ranges;
  ptr<MultiValues> suffixs;
  ptr<MultiNodes> stmts;
  ptr<Loop> loop;

  explicit ForeachBlock(const location& l, const ptr<MultiValues>& i,
                        const ptr<MultiNodes>& s)
      : Node(l), ranges(i), stmts(s), loop(nullptr) {
    assert(i != nullptr && "missing iteration variables for the statement.");
  }

  explicit ForeachBlock(const location& l, const ptr<MultiValues>& i,
                        const ptr<MultiValues>& se, const ptr<MultiNodes>& s)
      : Node(l), ranges(i), suffixs(se), stmts(s), loop(nullptr) {
    assert(i != nullptr && "missing iteration variables for the statement.");
  }

  bool IsBlock() const override { return true; }

  ptr<Node> CloneImpl() const override {
    auto copied = Make<ForeachBlock>(LOC(), CloneP(ranges), CloneP(suffixs),
                                     CloneP(stmts));
    copied->loop = loop ? loop : nullptr;
    return copied;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- Foreach Block:";
    ranges->Print(os, prefix + " ", with_type);
    if (suffixs) {
      os << "\n" << prefix << " `- Suffixes: ";
      suffixs->Print(os, prefix + " ", with_type);
    }
    if (stmts) { stmts->Print(os, prefix + " ", with_type); }
  }

  ptr<MultiValues> GetRangeNodes() const { return ranges; }
  const NodeList& GetRanges() const { return ranges->AllValues(); }

  void accept(Visitor&) override;

  bool IsNorm() const { return loop != nullptr; }

  ptr<Identifier> GetIV() const {
    if (ranges->Count() == 1) {
      auto range = dyn_cast<AST::LoopRange>(ranges->ValueAt(0));
      return range->IV();
    }
    return nullptr;
  }

  __UDT_TYPE_INFO__(Node, ForeachBlock)
};

struct InThreadsBlock : public Node, public TypeIDProvider<InThreadsBlock> {
  ptr<Expr> pred;
  ptr<MultiNodes> stmts;
  bool async = false;
  bool outer = true;

  bool IsBlock() const override { return true; }

  const ptr<Node> GetPred() const { return pred; }

  explicit InThreadsBlock(const location& l, const ptr<Expr> p,
                          const ptr<MultiNodes>& s, bool a = false,
                          bool o = true)
      : Node(l), pred(p), stmts(s), async(a), outer(o) {
    assert(p != nullptr && "missing predication.");
  }

  ptr<Node> CloneImpl() const override {
    return Make<InThreadsBlock>(LOC(), CloneP(pred), CloneP(stmts), async,
                                outer);
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- InThreads Block:";
    if (async) os << " Async";
    os << "\n" << prefix << " `- Predication: " << PSTR(pred);
    if (stmts) { stmts->Print(os, prefix + " ", with_type); }
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, InThreadsBlock)
};

struct WhileBlock : public Node, public TypeIDProvider<WhileBlock> {
  ptr<Expr> pred;
  ptr<MultiNodes> stmts;

  bool IsBlock() const override { return true; }

  explicit WhileBlock(const location& l, const ptr<Expr> p,
                      const ptr<MultiNodes>& s)
      : Node(l), pred(p), stmts(s) {
    assert(p != nullptr && "predication is requried.");
  }

  ptr<Node> CloneImpl() const override {
    return Make<WhileBlock>(LOC(), CloneP(pred), CloneP(stmts));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- While Block:";
    os << "\n" << prefix << " `- Predication: " << PSTR(pred);
    if (stmts) { stmts->Print(os, prefix + " ", with_type); }
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, WhileBlock)
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

  bool IsBlock() const override { return true; }

  ptr<Node> CloneImpl() const override {
    return Make<IncrementBlock>(LOC(), CloneP(bvs), CloneP(pred),
                                CloneP(stmts));
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "`- Increment Block:";
    os << "\n" << prefix << " `- Iteration variables: " << STR(bvs);
    os << "\n" << prefix << " `- Predicate: " << STR(pred);
    if (stmts) { stmts->Print(os, prefix + " ", with_type); }
  }

  void accept(Visitor&) override;

  const NodeList& GetIterationVars() const { return bvs->AllValues(); }

  const ptr<Node>& GetPredicate() const { return pred; }

  __UDT_TYPE_INFO__(Node, IncrementBlock)
};

struct FunctionDecl : public Node, public TypeIDProvider<FunctionDecl> {
  std::string name;
  ptr<DataType> ret_type;
  ptr<ParamList> params;

  FunctionDecl(const location& l, const ptr<DataType>& rt = nullptr,
               const ptr<ParamList>& pl = nullptr)
      : Node(l), ret_type(rt), params(pl) {}

  ptr<Node> CloneImpl() const override {
    auto n = Make<FunctionDecl>(LOC(), CloneP(ret_type), CloneP(params));
    n->name = name;
    return n;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "Name: " << name;
    os << "\n" << prefix << "Return type: ";
    ret_type->Print(os, "", with_type);
    params->Print(os, prefix, with_type);
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, FunctionDecl)
};

struct ChoreoFunction : public Node, public TypeIDProvider<ChoreoFunction> {
  std::string name;
  FunctionDecl f_decl;
  ptr<MultiNodes> stmts;

  ChoreoFunction(const location& l) : Node(l), f_decl(l) {}

  bool IsBlock() const override { return true; }

  ptr<Node> CloneImpl() const override {
    auto n = Make<ChoreoFunction>(LOC());
    n->name = name;
    n->f_decl = f_decl;
    n->stmts = CloneP(stmts);
    return n;
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    os << "\n" << prefix << "ChoreoFunction";
    if (with_type) os << "<{" << PSTR(GetType()) << "}>";
    f_decl.Print(os, prefix + " `- ", with_type);
    if (stmts) stmts->Print(os, prefix + " ", with_type);
  }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, ChoreoFunction)
};

struct CppSourceCode : public Node, public TypeIDProvider<CppSourceCode> {
  enum Kind { None, Host, Device, Inline };
  std::string code;
  Kind kind;

  CppSourceCode(const location& l, const std::string& c, Kind k = None)
      : Node(l), code(c), kind(k) {}

  bool IsBlock() const override { return true; }

  ptr<Node> CloneImpl() const override {
    return Make<CppSourceCode>(LOC(), code, kind);
  }

  void Print(std::ostream& os, const std::string& prefix = {},
             bool = false) const override {
    if (kind == Inline)
      os << "\n" << prefix << "`- [cpp] " << code;
    else
      os << code;
  }

  std::string GetCode() { return code; }
  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, CppSourceCode)
};

struct DeviceFunctionDecl final : public Node,
                                  public TypeIDProvider<DeviceFunctionDecl> {
  std::string name;
  ptr<DeviceDataType> ret_type;
  std::vector<ptr<DeviceDataType>> param_types;
  std::vector<std::string> attributes;
  std::string templates;

  struct DeviceTemplateParam {
    enum Kind { TYPE, VALUE, UNKNOWN } kind;
    std::string param_name;
    std::string type_name;
    std::string default_value; // for value type, the default value
    DeviceTemplateParam(const std::string& name = "", Kind k = UNKNOWN,
                        const std::string& type = "",
                        const std::string& def = "")
        : kind(k), param_name(name), type_name(type), default_value(def) {};
  };
  std::vector<DeviceTemplateParam> template_params;
  bool IsTemplated() const { return !template_params.empty(); }

  DeviceFunctionDecl(const location& l) : Node(l) {}

  ptr<Node> CloneImpl() const override {
    auto copy = Make<DeviceFunctionDecl>(LOC());
    copy->name = name;
    copy->ret_type = CloneP(ret_type);
    for (const auto& pt : param_types) {
      copy->param_types.push_back(CloneP(pt));
    }
    copy->attributes = attributes;
    for (const auto& tp : template_params) {
      copy->template_params.push_back(tp);
    }
    copy->templates = templates;
    return copy;
  }

  void Print(std::ostream& os, const std::string& = {},
             bool = false) const override {
    os << "DeviceFunction: ";
    os << name;
    os << " -> ";
    if (templates.size() > 0) os << templates << " ";
    os << PSTR(ret_type);
    os << " (";
    for (size_t i = 0; i < param_types.size(); ++i) {
      if (i > 0) os << ", ";
      os << PSTR(param_types[i]);
    }
    os << ")\n";
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, DeviceFunctionDecl)
};

// Top-level program structure
struct Program : public Node, public TypeIDProvider<Program> {
  ptr<MultiNodes> nodes;

  Program(const location& l, const ptr<MultiNodes> ss = nullptr)
      : Node(l), nodes(ss) {
    if (!nodes) nodes = Make<MultiNodes>(l);
  }

  bool IsBlock() const override { return true; }

  ptr<Node> CloneImpl() const override {
    return Make<Program>(LOC(), CloneP(nodes));
  }
  void Print(std::ostream& os, const std::string& prefix = {},
             bool with_type = false) const override {
    nodes->Print(os, "", with_type);
    (void)prefix;
  }

  void accept(Visitor&) override;

  __UDT_TYPE_INFO__(Node, Program)
};

inline ptr<Node> Expr::CloneImpl() const {
  if (op == "cast") {
    auto ce = cast<CastExpr>(this);
    return ce->CloneImpl();
  }

  return Make<Expr>(LOC(), op, CloneP(value_c), CloneP(value_l),
                    CloneP(value_r), t, opt_vals, s);
}
inline void Expr::Print(std::ostream& os, const std::string& prefix,
                        bool with_type) const {
  if (with_type) { os << "<{" << PSTR(GetType()) << "}>"; }
  if (t == Reference) {
    if (isa<Call>(value_r))
      value_r->InlinePrint(os, prefix, with_type);
    else
      value_r->Print(os, prefix, with_type);
    return;
  }

  assert(op.size() > 0 && "must have an operand.");

  if (op == "dimof" || op == "getith") {
    value_l->Print(os, prefix, with_type);
    value_r->Print(os, "", with_type);
    return;
  }

  if (op == "cast") {
    auto ce = cast<CastExpr>(this);
    os << "(" << ce->ToType() << ")(";
    value_r->Print(os, "", with_type);
    os << ")";
    return;
  }

  os << " (";
  switch (t) {
  case Unary:
    os << op << " ";
    value_r->Print(os, "", with_type);
    break;
  case Binary:
    value_l->Print(os, "", with_type);
    os << " " << op << " ";
    value_r->Print(os, "", with_type);
    break;
  case Ternary:
    value_c->Print(os, "", with_type);
    os << " ? ";
    value_l->Print(os, "", with_type);
    os << " : ";
    value_r->Print(os, "", with_type);
    break;
  default: choreo_unreachable("unhandled expression type."); break;
  }
  os << ") ";
}
//---------------------------------------------------------------------------//
// Utility Functions
//---------------------------------------------------------------------------//
inline std::optional<std::string> GetName(const Node& n) {
  if (auto id = dyn_cast<Identifier>(&n))
    return id->name;
  else if (auto exp = dyn_cast<Expr>(&n)) {
    if (auto id = exp->GetSymbol()) return id->name;
  }
  return std::nullopt;
}

inline Identifier* GetIdentifier(const Node& n) {
  if (auto id = dyn_cast<Identifier>(&n))
    return id;
  else if (auto expr = dyn_cast<Expr>(&n))
    return expr->GetSymbol().get();
  else
    return nullptr;
}

inline ptr<Identifier> GetIdentifier(const ptr<Node>& n) {
  if (auto id = dyn_cast<Identifier>(n))
    return id;
  else if (auto expr = dyn_cast<Expr>(n))
    return expr->GetSymbol();
  else
    return nullptr;
}

inline ptr<Call> GetCall(const ptr<Node>& n) {
  if (auto call = dyn_cast<Call>(n))
    return call;
  else if (auto expr = dyn_cast<Expr>(n))
    return GetCall(expr->GetReference());
  else
    return nullptr;
}

inline IntLiteral* GetIntLiteral(const Node& n) {
  if (auto il = dyn_cast<IntLiteral>(&n))
    return il;
  else if (auto expr = dyn_cast<Expr>(&n))
    return expr->GetInt().get();
  else
    return nullptr;
}

inline ptr<IntLiteral> GetIntLiteral(const ptr<Node>& n) {
  if (auto il = dyn_cast<IntLiteral>(n))
    return il;
  else if (auto expr = dyn_cast<Expr>(n))
    return expr->GetInt();
  else
    return nullptr;
}

inline bool AllConstant(const ptr<MultiValues>& mv) {
  for (auto il : mv->AllValues())
    if (!GetIntLiteral(il)) return false;

  return true;
}

inline bool IsSymbolOrArrayRef(const Node& n) {
  auto id = GetName(n);
  if (id.has_value()) return true;
  if (auto e = dyn_cast<Expr>(&n))
    if (e->op == "elemof") return true;
  return false;
}

inline bool HasVectorizationHint(const ForeachBlock& n,
                                 ptr<AST::AttributeExpr>& c) {
  if (!n.suffixs) return false;

  for (auto suffix : n.suffixs->values) {
    if (auto attr = dyn_cast<AST::AttributeExpr>(suffix)) {
      if (attr->AttrName() == "vectorize") {
        c = attr;
        return true;
      }
    }
  }
  c = nullptr;
  return false;
}

inline bool HasVectorizationHint(const ForeachBlock& n) {
  ptr<AST::AttributeExpr> c;
  return HasVectorizationHint(n, c);
}

inline const ptr<Identifier> GetArrayBaseSymbol(const Expr& n) {
  assert(n.op == "elemof");
  if (auto id = dyn_cast<Identifier>(n.GetL())) return id;
  auto expr = cast<Expr>(n.GetL());
  return GetArrayBaseSymbol(*expr);
}

inline size_t GetSubScriptLevel(const Expr& n) {
  assert(n.op == "elemof");
  if (auto id = dyn_cast<Identifier>(n.GetL())) return 1;
  return 1 + GetSubScriptLevel(*cast<Expr>(n.GetL()));
}

inline ptr<Node> Ref(const ptr<Node>& n) {
  if (auto expr = dyn_cast<Expr>(n))
    if (auto r = expr->GetReference()) return r;
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

inline bool IsLiteral(const Node& n) {
  return isa<IntLiteral>(&n) || isa<FloatLiteral>(&n) || isa<BoolLiteral>(&n) ||
         isa<StringLiteral>(&n);
}

inline ptr<ParallelBy>
MakeSimpleParallelBy(const location& l, const ptr<MultiNodes> stmts = nullptr,
                     int bv = 1) {
  auto anon_sym = SymbolTable::GetAnonPBName();
  auto pv = AST::Make<AST::Identifier>(l, anon_sym);
  pv->SetType(MakeBoundedITupleType(Shape(1, bv)));
  // elements
  auto spv = AST::Make<AST::MultiValues>(l, ", ");
  auto epv = AST::Make<AST::Identifier>(l, anon_sym + "__elem__x");
  epv->SetType(MakeBoundedIntegerType(sbe::nu(bv)));
  spv->Append(epv);

  // bound
  auto p_bound = AST::MakeIntExpr(l, bv);
  p_bound->SetType(MakeIntegerType());
  auto spv_bounds = AST::Make<AST::MultiValues>(l, ", ");
  spv_bounds->Append(p_bound->Clone());
  spv_bounds->SetType(MakeITupleType(1));

  auto pb = AST::Make<AST::ParallelBy>(l, pv, p_bound, spv, spv_bounds, stmts);
  pb->SetType(MakeBoundedIntegerType(sbe::nu(bv)));
  return pb;
}

inline const ptr<MultiValues> MakeMultiValues(const location& loc, size_t count,
                                              const ptr<Node>& v) {
  auto mv = AST::Make<AST::MultiValues>(loc);
  for (size_t i = 0; i < count; ++i) mv->Append(v->Clone());
  return mv;
}

inline const ValueList MakeValueList(const ptr<MultiValues>& mv) {
  if (!mv) choreo_unreachable("invalid multi-values.");
  ValueList vl;
  for (auto b : mv->AllValues()) {
    auto e = cast<Expr>(b);
    if (!e->Opts().HasVal()) return {};
    vl.push_back(e->Opts().GetVal());
  }
  return vl;
}

inline bool FragIsArrayElem(const ptr<AST::Expr>& e) {
  if (e->op == "elemof") return true;
  return false;
}

inline const std::string FragName(const ptr<AST::Expr>& e) {
  if (auto id = e->GetSymbol()) return id->name;
  return GetArrayBaseSymbol(*e)->name;
}

} // end of namespace AST

} // end of namespace Choreo

#endif // __CHOREO_AST_HPP__
