#ifndef __CHOREO_TYPE_CONSTRAINT_HPP__
#define __CHOREO_TYPE_CONSTRAINT_HPP__

#include <iostream>

#include "types.hpp"
#include "visitor.hpp"

namespace Choreo {

// simple type constraints collector and resolver
class TypeConstraints {
  std::map<std::string, std::set<std::string>> equals;
  VisitorWithScope* visitor = nullptr;

  bool debug = false;
  bool report_type = false;
  bool ignore_rank = false;

  FutureBufferInfo& FBInfo() {
    return FCtx(visitor->CurrentFunctionName()).GetFutureBufferInfo();
  }

public:
  TypeConstraints(VisitorWithScope* v, bool ir = false)
      : visitor(v), ignore_rank(ir) {}

  void SetDebug(bool d) { debug = d; }
  void SetTypeReport(bool r) { report_type = r; }

  void AddEqual(const std::string& a, const std::string& b) {
    assert(PrefixedWith(a, "::") && "must provide a scoped name");
    assert(PrefixedWith(b, "::") && "must provide a scoped name");

    if (a == b) return;

    if (!equals.count(a)) equals.emplace(a, std::set<std::string>{});
    if (!equals.count(b)) equals.emplace(b, std::set<std::string>{});

    equals[a].insert(b);
    equals[b].insert(a);

    if (debug)
      dbgs() << "[RType] Add equality between '" << a << "' and '" << b
             << "'\n";
  }

  std::optional<const std::vector<std::string>>
  GetEquals(const std::string& name) {
    if (!equals.count(name)) return std::nullopt;

    std::vector<std::string> result;
    std::set<std::string> visited;
    std::deque<std::string> work_list;

    work_list.push_back(name);
    while (!work_list.empty()) {
      auto c = work_list.back();
      work_list.pop_back();

      if (visited.count(c)) continue;
      visited.insert(c);

      if (c != name) result.push_back(c);
      for (auto& e : equals[c]) work_list.push_back(e);
    }
    return result;
  }

  void Reset() { equals.clear(); }

  bool ResolveType(const std::string& n, const ptr<Type>& ty, const location& l,
                   bool report_error = true) {
    assert(PrefixedWith(n, "::"));
    assert(!isa<PlaceHolderType>(ty) &&
           "can not resolve type to be place holder");
    auto equals = GetEquals(n);
    if (!equals) return true;
    for (auto& e : equals.value()) {
      if (visitor->SymTab()->Exists(e)) {
        auto sym = visitor->SymTab()->GetSymbol(e);
        auto sty = sym->GetType();
        // Currently it only allows resolving placeholder types.
        if (isa<PlaceHolderType>(sty)) {
          assert(sty->GetBaseType() == ty->GetBaseType() &&
                 "expect type to have same BaseType.");
          visitor->SSTab().ModifyScopedSymbolType(e, ty);
          if (debug)
            dbgs() << "[RType] Set the type of '" << e << "' to be " << PSTR(ty)
                   << "\n";
          if (auto pty = GetSpannedType(ty)) {
            visitor->SSTab().ModifyScopedSymbolType(e + ".span",
                                                    pty->GetMDSpanType());
            if (debug)
              dbgs() << "[RType] Set the type of '" << e + ".span" << "' to be "
                     << PSTR(pty->GetMDSpanType()) << "\n";
            if (auto fty = dyn_cast<FutureType>(ty)) {
              visitor->SSTab().ModifyScopedSymbolType(e + ".data", pty);
              if (debug)
                dbgs() << "[RType] Set the type of '" << e + ".data"
                       << "' to be " << PSTR(pty) << "\n";
              FBInfo()[n].from_kind = FBInfo()[e].from_kind;
              FBInfo()[n].to_kind = FBInfo()[e].to_kind;
            }
          }
          if (report_type) ReportSymbolType(e, ty);
        } else if (ty->HasSufficientInfo() && !sty->HasSufficientInfo()) {
          assert(ty->ApprxEqual(*sty));
          visitor->SSTab().ModifyScopedSymbolType(e, ty);
          if (debug)
            dbgs() << "[RType] Set the type of '" << e << "' to be " << PSTR(ty)
                   << "\n";
          if (auto pty = GetSpannedType(ty)) {
            visitor->SSTab().ModifyScopedSymbolType(e + ".span",
                                                    pty->GetMDSpanType());
            if (debug)
              dbgs() << "[RType] Set the type of '" << e + ".span" << "' to be "
                     << PSTR(pty->GetMDSpanType()) << "\n";
            if (auto fty = dyn_cast<FutureType>(ty)) {
              visitor->SSTab().ModifyScopedSymbolType(e + ".data", pty);
              if (debug)
                dbgs() << "[RType] Set the type of '" << e + ".data"
                       << "' to be " << PSTR(pty) << "\n";
              FBInfo()[n].from_kind = FBInfo()[e].from_kind;
              FBInfo()[n].to_kind = FBInfo()[e].to_kind;
            }
          }
          if (report_type) ReportSymbolType(e, ty);
        } else {
          bool equal;
          if (ignore_rank)
            equal = ty->ApprxEqual(*sty);
          else
            equal = (*ty == *sty);
          if (!equal && report_error) {
            visitor->Error(l, n + " is typed as " + PSTR(sty) +
                                  " but is resolved as " + PSTR(ty) + ".");
            return false;
          }
        }
      } else
        choreo_unreachable("expect symbol '" + e + "' exists.");
    }
    return true;
  }

  void AddEqualValues(const AST::MultiValues& mv) {
    assert(mv.Count() > 1 && "expect >1 values");
    for (size_t index = 1; index < mv.Count(); ++index) {
      auto lhs = AST::GetIdentifier(*mv.ValueAt(index - 1));
      auto rhs = AST::GetIdentifier(*mv.ValueAt(index));
      assert(lhs && rhs && "Expect identifiers.");
      AddEqual(visitor->SSTab().InScopeName(lhs->name),
               visitor->SSTab().InScopeName(rhs->name));
    }
  }

  // utility function to handle rotate/swap/select where operands of same type
  // is required that type equivalence is established
  ptr<Type> GetOnlyFutureTypeFromMultiValues(const AST::MultiValues& mv,
                                             bool report_error = true) {
    assert(mv.Count() > 1 && "expect two or more values");
    ptr<Type> fty = nullptr;
    auto val0 = mv.ValueAt(0);
    if (GeneralFutureType(visitor->NodeType(*val0))) {
      size_t fidx = 0;
      for (size_t index = 0; index < mv.Count(); ++index) {
        auto futy = visitor->NodeType(*mv.ValueAt(index));
        if (!fty) {
          fty = futy;
          fidx = index;
          continue;
        }
        if (fty->HasSufficientInfo() && futy->HasSufficientInfo()) {
          if (report_error && (*futy != *fty))
            visitor->Error(mv.LOC(), "types are inconsistent: " + PSTR(futy) +
                                         "(" + std::to_string(index) +
                                         ") vs. " + PSTR(fty) + "(" +
                                         std::to_string(fidx) + ").");
        } else if (!fty->HasSufficientInfo() && futy->HasSufficientInfo()) {
          if (report_error && !fty->ApprxEqual(*futy))
            visitor->Error(mv.LOC(), "types are inconsistent: " + PSTR(futy) +
                                         "(" + std::to_string(index) +
                                         ") vs. " + PSTR(fty) + "(" +
                                         std::to_string(fidx) + ").");
          fty = futy;
          fidx = index;
        } else if (fty->HasSufficientInfo() && !futy->HasSufficientInfo()) {
          if (report_error && !fty->ApprxEqual(*futy))
            visitor->Error(mv.LOC(), "types are inconsistent: " + PSTR(futy) +
                                         "(" + std::to_string(index) +
                                         ") vs. " + PSTR(fty) + "(" +
                                         std::to_string(fidx) + ").");
        } else if (!fty->HasSufficientInfo() && !futy->HasSufficientInfo()) {
          if (report_error && !fty->ApprxEqual(*futy))
            visitor->Error(mv.LOC(), "types are inconsistent: " + PSTR(futy) +
                                         "(" + std::to_string(index) +
                                         ") vs. " + PSTR(fty) + "(" +
                                         std::to_string(fidx) + ").");
          if (isa<FutureType>(futy) && isa<PlaceHolderType>(fty)) {
            assert(fty->ApprxEqual(*futy));
            fty = futy;
          }
        }

        if (index < 1) continue;

        auto lid = AST::GetIdentifier(*mv.ValueAt(index - 1));
        auto cid = AST::GetIdentifier(*mv.ValueAt(index));

        assert(lid && "failed to get an identifier.");
        assert(cid && "failed to get an identifier.");

        AddEqual(visitor->SSTab().InScopeName(lid->name),
                 visitor->SSTab().InScopeName(cid->name));
      }
    }
    if (debug) dbgs() << "[RType] Get only future type: " << PSTR(fty) << ".\n";
    return fty;
  }

  void ApplyFutureTypeForMultiValues(const AST::MultiValues& mv,
                                     const ptr<Type>& fty,
                                     bool report_error = true) {
    assert(!isa<PlaceHolderType>(fty) &&
           "can not resolve type to be place holder");
    for (auto& fut : mv.AllValues()) {
      auto sym = AST::GetIdentifier(*fut);
      ResolveType(visitor->SSTab().InScopeName(sym->name), fty, fut->LOC(),
                  report_error);
    }
  }

  const ptr<Type> ResolveEqualFutures(const AST::MultiValues& mv,
                                      bool report_error = true) {
    if (auto fty = GetOnlyFutureTypeFromMultiValues(mv, report_error)) {
      if (isa<PlaceHolderType>(fty)) // do not apply placeholders
        return fty;
      ApplyFutureTypeForMultiValues(mv, fty, report_error);
      return fty;
    }

    return nullptr;
  }

  void ReportSymbolType(const std::string& name, const ptr<Type>& ty) {
    if (isa<SpannedType>(ty))
      dbgs() << "Symbol:    ";
    else if (isa<FutureType>(ty))
      dbgs() << "Future:    ";
    dbgs() << name << ", Type: " << PSTR(ty) << "\n";
  }
};

} // end namespace Choreo

#endif //__CHOREO_TYPE_CONSTRAINT_HPP__
