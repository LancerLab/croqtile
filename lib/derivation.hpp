#ifndef __CHOREO_ATTRIBUTE_DERIVATION_HPP__
#define __CHOREO_ATTRIBUTE_DERIVATION_HPP__

#include "visitor.hpp"

namespace Choreo {

struct AttributeDeriver {
private:
  std::unordered_set<std::string> symbols;
  std::unordered_set<AST::Node*> nodes;

  VisitorWithScope* vws = nullptr;
  std::string name;
  bool use_sstab = true;
  bool debug = false;

public:
  AttributeDeriver(VisitorWithScope* v, const std::string& n, bool us = true,
                   bool d = false)
      : vws(v), name(n), use_sstab(us), debug(d) {}

  void Add(const std::string& sym) {
    if (!PrefixedWith(sym, "::"))
      choreo_unreachable("expect a scoped symbol: " + sym + ".");
    if (symbols.count(sym))
      choreo_unreachable("already contains the symbol: " + sym + ".");
    symbols.insert(sym);

    if (debug) dbgs() << "[" << name << "] added symbol: " << sym << ".\n";
  }

  void Add(AST::Node& n) {
    nodes.insert(&n);

    if (debug) dbgs() << "[" << name << "] added expr: " << STR(n) << ".\n";
  }

  void Insert(AST::Node& n) {
    Add(n);
    if (auto pname = AST::GetName(n)) Add(vws->InScopeName(*pname));
  }

  bool Contains(const ptr<AST::Node>& n) {
    if (n == nullptr) return false; // handle nullptr for easier processing

    if (auto pname = AST::GetName(*n)) {
      if (!use_sstab || vws->SSTab().IsDeclared(*pname)) {
        return symbols.count(vws->InScopeName(*pname));
      }
    } else if (isa<AST::Expr>(n))
      return nodes.count(n.get());
    else if (auto it = dyn_cast<AST::IntTuple>(n))
      return nodes.count(it->vlist.get());
    else if (auto mds = dyn_cast<AST::MultiDimSpans>(n))
      return nodes.count(mds->list.get());
    else if (auto mds = dyn_cast<AST::DataAccess>(n))
      return nodes.count(mds->indices.get());
    else if (auto sl = dyn_cast<AST::Select>(n))
      return nodes.count(sl.get());
    else if (AST::IsLiteral(*n) || isa<AST::IntIndex>(n) ||
             isa<AST::SpanAs>(n) || isa<AST::ChunkAt>(n) ||
             isa<AST::NoValue>(n) || isa<AST::DataType>(n))
      return false;
    else if (auto c = dyn_cast<AST::Call>(n))
      return nodes.count(c.get());
    else if (auto mv = dyn_cast<AST::MultiValues>(n)) {
      bool res = false;
      for (const auto& v : mv->AllValues()) res |= Contains(v);
      return res;
    } else
      choreo_unreachable("unsupported node: " + n->TypeNameString() + ": " +
                         PSTR(n) + ".");
    return false;
  }
};

} // end namespace Choreo

#endif // __CHOREO_ATTRIBUTE_DERIVATION_HPP__
