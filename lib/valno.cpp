#include "shapeinfer.hpp"

using namespace Choreo;
using namespace Choreo::valno;

namespace {

const std::vector<NumTy> CollectValueNumbers(const SignTy& input) {
  std::vector<NumTy> res;
  std::regex valuePattern("#(-?\\d+)");
  auto begin = std::sregex_iterator(input.begin(), input.end(), valuePattern);
  auto end = std::sregex_iterator();

  for (auto i = begin; i != end; ++i) {
    std::smatch match = *i;
    std::string matchStr = match.str(1); // Capture the number part of the match
    res.push_back(std::stoi(matchStr));
  }
  return res;
}

const std::vector<NumTy> GetOperandsValNo(const SignTy& input) {
  std::vector<NumTy> operands;
  size_t startPos = 0;
  size_t colonPos;

  // Loop through all colon-separated parts of the string
  while ((colonPos = input.find(':', startPos)) != std::string::npos) {
    std::string part = input.substr(startPos, colonPos - startPos);
    std::istringstream iss(part.substr(1));
    int number;
    // Try to extract a number from the current part
    if (iss >> number) operands.push_back(number);
    startPos = colonPos + 1;
  }

  // Handle the last part after the last colon
  if (startPos + 1 < input.length()) {
    std::istringstream iss(input.substr(startPos + 1));
    int number;
    if (iss >> number) operands.push_back(number);
  }

  return operands;
}

} // namespace

using namespace valno;

void ValueNumbering::EnterScope() {
  std::string indent = ScopeIndent();
  assert(!indent.empty() && "unexpected empty indent.");
  indent.pop_back();

  vntbl.EnterScope();

  if (trace)
    if (visitor->SSTab().ScopeDepth() > 1)
      dbgs() << indent << "scope-" << visitor->SSTab().ScopeDepth() - 1
             << " {\n";
}

void ValueNumbering::LeaveScope() {
  if (visitor->SSTab().ScopeDepth() <= 1) return;

  std::string sname = std::to_string(visitor->SSTab().ScopeDepth() - 1);

  vntbl.LeaveScope();

  std::string indent = ScopeIndent();
  assert(!indent.empty() && "unexpected empty indent.");
  indent.pop_back();

  bind_info.Clear();

  if (trace) dbgs() << indent << "} // end scope-" << sname << "\n";
}

NumTy ValueNumbering::VNReal(const std::string& input) const {
  assert(!IsUnknownSign(input));
  assert(!IsNoneSign(input));
  if (auto v = RemovePrefixOrNull("#", input)) { // handle #0 string as well
    int vn;
    auto [ptr, ec] = std::from_chars(v->data(), v->data() + v->size(), vn);
    if (ec != std::errc()) choreo_unreachable("input is unexpected: " + input);
    return vn;
  } else
    return GetValueNumberOfSignature(input);
}

const SignTy ValueNumbering::RealSign(const std::string& input) const {
  return GetSignatureFromValueNumber(VNReal(input));
}

ValueItem ValueNumbering::GenValueItemFromValueNumber(NumTy vn) {
  return GenValueItemFromSignature(GetSignatureFromValueNumber(vn));
}

ValueItem ValueNumbering::GenValueItemFromSignature(const SignTy& input) {
  if (auto v = RemovePrefixOrNull("#", input)) { // handle #0 string as well
    int value;
    auto [ptr, ec] = std::from_chars(v->data(), v->data() + v->size(), value);
    if (ec != std::errc()) return nullptr;
    NumTy vn{value};
    if (vn.IsNone()) return sbe::nil();
    if (vntbl.Exists(vn))
      return GenValueItemFromSignature(GetSignatureFromValueNumber(vn));
    return nullptr;
  } else if (IsNoneSign(input)) {
    return sbe::nil();
  } else if (auto iv = RemovePrefixOrNull("const_", input)) {
    if (input.find(".") != std::string::npos) // do not handle floating numbers
      return nullptr;
    uint64_t vn;
    auto [ptr, ec] = std::from_chars(iv->data(), iv->data() + iv->size(), vn);
    if (ec != std::errc()) return nullptr; // can not handle
    return sbe::nu(vn);
  } else if (PrefixedWith(input, "::")) {
    // must be a scoped symbol
    return sbe::sym(input);
  } else if (input == "true") {
    return sbe::bl(true);
  } else if (input == "false") {
    return sbe::bl(false);
  }

  // it is an operation
  auto parts = SplitStringByDelimiter(input, ":");

  if (parts.size() == 4) {
    // ternary operation
    assert(PrefixedWith(input, "?:") && "unexpected ternary operation.");
    auto pvi = GenValueItemFromSignature(RealSign(parts[1]));
    auto lvi = GenValueItemFromSignature(RealSign(parts[2]));
    auto rvi = GenValueItemFromSignature(RealSign(parts[3]));
    if (pvi && lvi && rvi) return sbe::sel(pvi, lvi, rvi)->Normalize();
    return nullptr;
  }

  if (parts.size() == 2) {
    if (PrefixedWith(input, "!:") || PrefixedWith(input, "~:")) {
      if (auto ovi = GenValueItemFromSignature(RealSign(parts[1])))
        return sbe::uop(ToOpCode(input.substr(0, 1)), ovi)->Normalize();
    }
  }

  if (parts.size() != 3) return nullptr;
  if (PrefixedWith(input, "+:") || PrefixedWith(input, "-:") ||
      PrefixedWith(input, "*:") || PrefixedWith(input, "/:") ||
      PrefixedWith(input, "%:") || PrefixedWith(input, ">:") ||
      PrefixedWith(input, "<:") || PrefixedWith(input, "|:") ||
      PrefixedWith(input, "&:") || PrefixedWith(input, "^:")) {
    auto lvi = GenValueItemFromSignature(RealSign(parts[1]));
    auto rvi = GenValueItemFromSignature(RealSign(parts[2]));
    if (lvi && rvi)
      return sbe::bop(ToOpCode(input.substr(0, 1)), lvi, rvi)->Normalize();
  } else if (PrefixedWith(input, "cdiv:")) {
    auto lvi = GenValueItemFromSignature(RealSign(parts[1]));
    auto rvi = GenValueItemFromSignature(RealSign(parts[2]));
    if (lvi && rvi)
      return sbe::bop(OpCode::DIVIDE, lvi + (rvi - sbe::nu(1)), rvi)
          ->Normalize();
  } else if (PrefixedWith(input, ">=:") || PrefixedWith(input, "<=:") ||
             PrefixedWith(input, "==:") || PrefixedWith(input, "!=:") ||
             PrefixedWith(input, ">>:") || PrefixedWith(input, "<<:")) {
    auto lvi = GenValueItemFromSignature(RealSign(parts[1]));
    auto rvi = GenValueItemFromSignature(RealSign(parts[2]));
    if (lvi && rvi)
      return sbe::bop(ToOpCode(input.substr(0, 2)), lvi, rvi)->Normalize();
  }
  return nullptr;
}

const ValueList ValueNumbering::GenValueListFromValueNumber(NumTy vn) {
  return GenValueListFromSignature(GetSignatureFromValueNumber(vn));
}

const ValueList ValueNumbering::GenValueListFromSignature(const SignTy& input) {
  std::vector<ValueItem> res;
  if (CountElementsInSignature(input) > 1) {
    ForeachValueNumber(input, [this, &res](NumTy valno, size_t) {
      auto sign = GetSignatureFromValueNumber(valno);
      res.push_back(GenValueItemFromSignature(sign));
    });
  } else {
    assert(CountElementsInSignature(input) == 1);
    res.push_back(GenValueItemFromSignature(input));
  }
  return res;
}

// Note: it adds value numbers as necessary
const SignTy ValueNumbering::ValueItemToSignature(const ValueItem& vi,
                                                  bool gen) {
  if (VIIsNil(vi)) {
    return NoneSign();
  } else if (auto iv = VIInt(vi)) {
    auto sign = "const_" + STR(vi);
    auto vn = GetOrGenValueNumberFromSignature(sign); // always generate
    return GetSignatureFromValueNumber(vn);
  } else if (auto sym = VISym(vi)) {
    assert(PrefixedWith(sym.value(), "::") && "expected a scoped symbol.");
    assert(HasValueNumberOfSignature(sym.value()) &&
           "the symbol does have a value number.");
    return sym.value();
  } else if (auto bop = VIUop(vi)) {
    auto osign = ValueItemToSignature(bop->GetOperand(), true);
    auto ovn = GetValueNumberOfSignature(osign);
    auto sign = STR(bop->GetOpCode()) + ":" + STR(ovn);
    if (gen) {
      auto vn = GetOrGenValueNumberFromSignature(sign);
      return GetSignatureFromValueNumber(vn);
    } else
      return sign;
  } else if (auto bop = VIBop(vi)) {
    auto lsign = ValueItemToSignature(bop->GetLeft(), true);
    auto rsign = ValueItemToSignature(bop->GetRight(), true);
    auto lvn = GetValueNumberOfSignature(lsign);
    auto rvn = GetValueNumberOfSignature(rsign);
    auto sign = STR(bop->GetOpCode()) + ":" + STR(lvn) + ":" + STR(rvn);
    if (gen) {
      auto vn = GetOrGenValueNumberFromSignature(sign);
      return GetSignatureFromValueNumber(vn);
    } else
      return sign;
  } else if (auto top = VITop(vi)) {
    auto psign = ValueItemToSignature(top->GetPred(), true);
    auto lsign = ValueItemToSignature(top->GetLeft(), true);
    auto rsign = ValueItemToSignature(top->GetRight(), true);
    auto pvn = GetValueNumberOfSignature(psign);
    auto lvn = GetValueNumberOfSignature(lsign);
    auto rvn = GetValueNumberOfSignature(rsign);
    auto sign = STR(top->GetOpCode()) + ":" + STR(pvn) + ":" + STR(lvn) + ":" +
                STR(rvn);
    if (gen) {
      auto vn = GetOrGenValueNumberFromSignature(sign);
      return GetSignatureFromValueNumber(vn);
    } else
      return sign;
  } else
    choreo_unreachable("unsupported value.");
  return "";
}

const SignTy ValueNumbering::ValueListToSignature(const ValueList& vl,
                                                  bool gen) {
  assert(!vl.empty());
  std::string sign;
  int i = 0;
  for (auto vi : vl) {
    if (i++ > 0) sign += ",";
    auto vis = ValueItemToSignature(vi, gen);
    NumTy vn = GetInvalidValueNumber();
    if (gen)
      vn = GetOrGenValueNumberFromSignature(vis);
    else
      vn = GetValueNumberOfSignature(vis);
    sign += STR(vn);
  }
  return sign;
}

// It binds a expression signature with an existing value number.  use it
// carefully.
void ValueNumbering::AssociateSignatureWithValueNumber(const SignTy& sig,
                                                       NumTy valno) {
  assert(vntbl.Exists(valno) && "invalid value number is provided.");
  if (vntbl.Exists(sig))
    assert((vntbl.GetValueNum(sig) == valno) &&
           "must associate signature with different value number.");

  vntbl.Alias(valno, sig);

  if (trace)
    dbgs() << ScopeIndent() << "Alias \"" << sig << "\" -> " << valno << "\n";
}

void ValueNumbering::AssociateSignatureWithInvalidValueNumber(
    const SignTy& sig) {
  assert(!vntbl.Exists(sig) && "signature does exists.");

  vntbl.DummyGen(sig);

  if (trace)
    dbgs() << ScopeIndent() << "Alias \"" << sig << "\" -> #<invalid>\n";
}

void ValueNumbering::RebindSignatureWithValueNumber(const SignTy& s, NumTy v) {
  if (!vntbl.Exists(s))
    return; // could be invalid symbol for scope change
            // TODO: is this code safe?
  vntbl.BindDummy(s, v);

  if (trace)
    dbgs() << ScopeIndent() << "Alias(Rebind) \"" << STR(s) << "\" -> "
           << STR(v) << "\n";
}

const SignTy ValueNumbering::SignBinaryCompositeValues(const location& loc,
                                                       const std::string& op,
                                                       const std::string& l_sig,
                                                       const std::string& r_sig,
                                                       bool verbose) {
  // handle concatenation
  if (op == "concat") {
    auto GetSignature = [this](const std::string& sig) {
      if (CountElementsInSignature(sig) > 1)
        return sig;
      else
        return GetValueNumberOfSignature(sig).ToString();
    };
    return GetSignature(l_sig) + "," + GetSignature(r_sig);
  }

  std::string lhs, rhs;
  // specially handle ituple/mdspan + integer: broadcast integer

  auto l_elem_cnt = CountElementsInSignature(l_sig);
  auto r_elem_cnt = CountElementsInSignature(r_sig);

  if (l_elem_cnt == 1 && r_elem_cnt == 1) {
    NumTy l_valno = GetValueNumberOfSignature(l_sig);
    lhs = STR(l_valno);
    NumTy r_valno = GetValueNumberOfSignature(r_sig);
    rhs = STR(r_valno);
  } else if (l_elem_cnt == 1 && r_elem_cnt > 1) {
    NumTy valno = GetValueNumberOfSignature(l_sig);
    lhs = STR(valno);
    for (int i = 1; i < r_elem_cnt; ++i) lhs += "," + STR(valno);
    rhs = r_sig;
  } else if (l_elem_cnt > 1 && r_elem_cnt == 1) {
    NumTy valno = GetValueNumberOfSignature(r_sig);
    rhs = STR(valno);
    for (int i = 1; i < l_elem_cnt; ++i) rhs += "," + STR(valno);
    lhs = l_sig;
  } else {
    lhs = l_sig;
    rhs = r_sig;
  }

  assert(CountElementsInSignature(lhs) == CountElementsInSignature(rhs));

  auto l_vns = CollectValueNumbers(lhs);
  auto r_vns = CollectValueNumbers(rhs);
  assert(l_vns.size() == r_vns.size());

  std::vector<NumTy> signatures;
  for (size_t i = 0; i < l_vns.size(); ++i) {
    auto l_sig = GetSignatureFromValueNumber(l_vns[i]);
    auto r_sig = GetSignatureFromValueNumber(r_vns[i]);
    auto opt_sig = TryToSimplifyBinary(loc, op, l_sig, r_sig, verbose);
    if (opt_sig)
      signatures.push_back(GetOrGenValueNumberFromSignature(*opt_sig));
    else {
      std::ostringstream oss;
      oss << op << ":" << GetValueNumberOfSignature(l_sig) << ":"
          << GetValueNumberOfSignature(r_sig);
      signatures.push_back(GetOrGenValueNumberFromSignature(oss.str()));
    }
  }
  assert(signatures.size() > 0);

  std::ostringstream oss;
  oss << STR(signatures[0]);
  for (size_t i = 1; i < signatures.size(); ++i)
    oss << "," << STR(signatures[i]);
  return oss.str();
}

const std::string ValueNumbering::SimplifySignature(const location& loc,
                                                    const std::string& sign) {
  assert(CountElementsInSignature(sign) == 1);

  // Applies the algebraic simplification
  std::set<std::string> optimizable = {
      "+",  "-", "*", "/",  "%",  "cdiv", "@",  "@+",
      "@-", "<", ">", "<=", ">=", "==",   "!=",
  };

  auto HandleMultiSigns = [&loc, this](const std::string& op,
                                       const SignTy& lsign,
                                       const SignTy& rsign) {
    // For multiple signatures, it is possible to generate new intermediate
    // value numbers
    auto lvns = SplitStringByDelimiter(lsign, ",");
    auto rvns = SplitStringByDelimiter(rsign, ",");
    assert(lvns.size() == rvns.size());
    std::string o_sign;
    for (size_t i = 0; i < lvns.size(); ++i) {
      auto e_sign = SimplifySignature(loc, op + ":" + lvns[i] + ":" + rvns[i]);
      o_sign += STR(GetOrGenValueNumberFromSignature(e_sign));
      if (i < lvns.size() - 1) o_sign += ",";
    }
    return o_sign;
  };

  auto sparts = SplitStringByDelimiter(sign, ":");
  if ((sparts.size() == 3)) {
    auto op = sparts[0];
    NumTy lvn = VNReal(sparts[1]);
    NumTy rvn = VNReal(sparts[2]);
    auto lsign = GetSignatureFromValueNumber(lvn);
    auto rsign = GetSignatureFromValueNumber(rvn);

    if (op == "concat") {
      auto NumSign = [this](const std::string& s) {
        if (CountElementsInSignature(s) > 1) return s;
        return STR(GetValueNumberOfSignature(s));
      };
      return NumSign(lsign) + "," + NumSign(rsign);
    } else if (optimizable.count(op)) {
      auto lcnt = CountElementsInSignature(lsign);
      auto rcnt = CountElementsInSignature(rsign);
      if (lcnt == rcnt) {
        if (lcnt == 1) {
          if (auto res = TryToSimplifyBinary(loc, op, lsign, rsign))
            return res.value();
          else if (op == "#")
            return "*:" + sparts[1] + sparts[2]; // operation '#' is special
        } else
          return HandleMultiSigns(op, lsign, rsign);
      } else {
        // multivalues operates on a single value
        assert(((lcnt > 1) && (rcnt == 1)) || ((rcnt > 1) && (lcnt == 1)));
        if (lcnt == 1) {
          lsign = STR(lvn);
          for (int i = 1; i < rcnt; ++i) lsign += "," + STR(lvn);
        } else if (rcnt == 1) {
          rsign = STR(rvn);
          for (int i = 1; i < lcnt; ++i) rsign += "," + STR(rvn);
        }
        return HandleMultiSigns(op, lsign, rsign);
      }
    }
  }

  return sign;
}

std::optional<std::string>
ValueNumbering::TryToSimplifyBinary(const location& loc, const std::string& op,
                                    const std::string& lhs,
                                    const std::string& rhs, bool verbose) {
  if (op == "concat") return std::nullopt;

  // try simplify using symbexpr first
  auto lvi = GenValueItemFromSignature(lhs);
  auto rvi = GenValueItemFromSignature(rhs);
  if (lvi && rvi &&
      (op == "+" || op == "-" || op == "*" || op == "/" || op == "%")) {
    auto res_vi = sbe::bop(ToOpCode(op), lvi, rvi);
    auto opt_vi = res_vi->Normalize();
    if (*res_vi != *opt_vi) {
      auto res = ValueItemToSignature(opt_vi);
      if (trace && verbose)
        dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
               << rhs << " to '" << res << "'\n";
      return res;
    }
  }

  auto l_cv = RemovePrefixOrNull("const_", lhs);
  auto r_cv = RemovePrefixOrNull("const_", rhs);

  // constant folding
  if (l_cv && r_cv) {
    std::string res = "const_";
    if (CCtx().SimplifyFpValno()) {
      using Var = std::variant<float, double, long long>;
      auto parseNumeric = [](const std::string& s) -> Var {
        if (auto d = Str2Double(s)) return *d;
        if (auto f = Str2Float(s)) return *f;
        return std::stoll(s);
      };
      auto applyOp = [&](auto&& L, auto&& R, auto f) -> Var {
        return std::visit(
            [&](auto a, auto b) -> Var {
              using CT = std::common_type_t<decltype(a), decltype(b)>;
              return static_cast<CT>(f(a, b));
            },
            L, R);
      };

      Var L = parseNumeric(*l_cv);
      Var R = parseNumeric(*r_cv);

      auto isZero = [](const Var& v) {
        if (std::holds_alternative<double>(v))
          return std::get<double>(v) == 0.0;
        if (std::holds_alternative<float>(v)) return std::get<float>(v) == 0.0f;
        if (std::holds_alternative<long long>(v))
          return std::get<long long>(v) == 0LL;
        else
          choreo_unreachable("unexpected type for zero check.");
      };

      std::unordered_map<std::string,
                         std::function<Var(const Var&, const Var&)>>
          op_table;
      op_table["+"] = [&](auto&& A, auto&& B) {
        return applyOp(A, B, [](auto a, auto b) { return a + b; });
      };
      op_table["-"] = [&](auto&& A, auto&& B) {
        return applyOp(A, B, [](auto a, auto b) { return a - b; });
      };
      op_table["*"] = [&](auto&& A, auto&& B) {
        return applyOp(A, B, [](auto a, auto b) { return a * b; });
      };
      op_table["/"] = [&](auto&& A, auto&& B) {
        if (isZero(B)) {
          dbgs() << ScopeIndent() << "<ERROR> divide by zero: " << lhs << " / "
                 << rhs << "\n";
          choreo_unreachable("divide by zero is found in shape evaluation.");
        }
        return applyOp(A, B, [](auto a, auto b) { return a / b; });
      };
      op_table["%"] = [&](auto&& A, auto&& B) {
        if (std::holds_alternative<long long>(A) &&
            std::holds_alternative<long long>(B)) {
          if (isZero(B)) {
            dbgs() << ScopeIndent() << "<ERROR> mod by zero: " << lhs << " % "
                   << rhs << "\n";
            choreo_unreachable("mod by zero is found in shape evaluation.");
          }
          return std::get<long long>(A) % std::get<long long>(B);
        } else {
          dbgs() << ScopeIndent()
                 << "<ERROR> mod with float-point is not support: " << lhs
                 << " % " << rhs << "\n";
          choreo_unreachable(
              "mod with float-point is found in shape evaluation.");
        }
      };
      op_table["<"] = [&](auto&& A, auto&& B) {
        return Var{static_cast<long long>(
            std::visit([&](auto a, auto b) { return a < b; }, A, B))};
      };
      op_table[">"] = [&](auto&& A, auto&& B) {
        return Var{static_cast<long long>(
            std::visit([&](auto a, auto b) { return a > b; }, A, B))};
      };
      op_table["=="] = [&](auto&& A, auto&& B) {
        // TODO: do we need EPSILON for float-point?
        return Var{static_cast<long long>(
            std::visit([&](auto a, auto b) { return a == b; }, A, B))};
      };
      op_table["!="] = [&](auto&& A, auto&& B) {
        // TODO: do we need EPSILON for float-point?
        return Var{static_cast<long long>(
            std::visit([&](auto a, auto b) { return a != b; }, A, B))};
      };
      op_table["<="] = [&](auto&& A, auto&& B) {
        // TODO: do we need EPSILON for float-point?
        return Var{static_cast<long long>(
            std::visit([&](auto a, auto b) { return a <= b; }, A, B))};
      };
      op_table[">="] = [&](auto&& A, auto&& B) {
        // TODO: do we need EPSILON for float-point?
        return Var{static_cast<long long>(
            std::visit([&](auto a, auto b) { return a >= b; }, A, B))};
      };
      op_table["cdiv"] = [&](auto&& A, auto&& B) {
        if (std::holds_alternative<long long>(A) &&
            std::holds_alternative<long long>(B)) {
          if (isZero(B)) {
            dbgs() << ScopeIndent() << "<ERROR> cdiv by zero: " << lhs << " % "
                   << rhs << "\n";
            choreo_unreachable("cdiv by zero is found in shape evaluation.");
          }
          return (std::get<long long>(A) + std::get<long long>(B) - 1ll) /
                 std::get<long long>(B);
        } else {
          dbgs() << ScopeIndent()
                 << "<ERROR> cdiv with float-point is not support: " << lhs
                 << " % " << rhs << "\n";
          choreo_unreachable(
              "cdiv with float-point is found in shape evaluation.");
        }
      };
      op_table["#"] = [&](auto&& A, auto&& B) {
        if (std::holds_alternative<long long>(A) &&
            std::holds_alternative<long long>(B)) {
          return std::get<long long>(A) * std::get<long long>(B);
        } else {
          dbgs() << ScopeIndent()
                 << "<ERROR> # with float-point is not support: " << lhs
                 << " % " << rhs << "\n";
          choreo_unreachable(
              "# with float-point is found in shape evaluation.");
        }
      };

      op_table["#+"] = [&](auto&& A, auto&& B) {
        if (std::holds_alternative<long long>(A) &&
            std::holds_alternative<long long>(B)) {
          return std::get<long long>(A) + std::get<long long>(B);
        } else {
          dbgs() << ScopeIndent()
                 << "<ERROR> # with float-point is not support: " << lhs
                 << " % " << rhs << "\n";
          choreo_unreachable(
              "# with float-point is found in shape evaluation.");
        }
      };

      op_table["#-"] = [&](auto&& A, auto&& B) {
        if (std::holds_alternative<long long>(A) &&
            std::holds_alternative<long long>(B)) {
          return std::get<long long>(A) - std::get<long long>(B);
        } else {
          dbgs() << ScopeIndent()
                 << "<ERROR> # with float-point is not support: " << lhs
                 << " % " << rhs << "\n";
          choreo_unreachable(
              "# with float-point is found in shape evaluation.");
        }
      };

      if (op_table.count(op)) {
        Var raw = op_table.at(op)(L, R);
        std::visit(
            [&](auto x) -> std::string {
              using T = decltype(x);
              if constexpr (std::is_same_v<T, long long>) {
                if (op == "<" || op == ">" || op == "==" || op == "!=" ||
                    op == "<=" || op == ">=") {
                  return res = x ? "true" : "false";
                }
                return res += std::to_string(x);
              } else if constexpr (std::is_same_v<T, float>) {
                return res += std::to_string(x) + "f";
              } else if constexpr (std::is_same_v<T, double>) {
                return res += std::to_string(x);
              } else {
                choreo_unreachable("unexpected type for simplification.");
              }
            },
            raw);
      } else {
        Error(loc,
              "simplification of operation `" + op + "' is not yet supported.");
        return std::nullopt;
      }
    } else {
      // if the const value is not integer, stop simplification
      if (!(IsInteger(*l_cv) && IsInteger(*r_cv))) return std::nullopt;
      auto l = std::stoll(*l_cv);
      auto r = std::stoll(*r_cv);
      if (op == "+")
        res += std::to_string(l + r);
      else if (op == "-")
        res += std::to_string(l - r);
      else if (op == "*")
        res += std::to_string(l * r);
      else if (op == "/") {
        auto div_end = r;
        if (div_end == 0) {
          dbgs() << ScopeIndent() << "<ERROR> divide by zero: " << lhs << " / "
                 << rhs << "\n";
          choreo_unreachable("divide by zero is found in shape evaluation.");
        }
        res += std::to_string(l / div_end);
      } else if (op == "%") {
        auto div_end = r;
        if (div_end == 0) {
          dbgs() << ScopeIndent() << "<ERROR> divide by zero: " << lhs << " / "
                 << rhs << "\n";
          choreo_unreachable("divide by zero is found in shape evaluation.");
        }
        res += std::to_string(l % r);
      } else if (op == "<") {
        res = l < r ? "true" : "false";
      } else if (op == ">") {
        res = l > r ? "true" : "false";
      } else if (op == "==") {
        res = l == r ? "true" : "false";
      } else if (op == "!=") {
        res = l != r ? "true" : "false";
      } else if (op == "<=") {
        res = l <= r ? "true" : "false";
      } else if (op == ">=") {
        res = l >= r ? "true" : "false";
      } else if (op == "cdiv") {
        auto div_end = r;
        if (div_end == 0) {
          dbgs() << ScopeIndent() << "<ERROR> divide by zero: " << lhs << " / "
                 << rhs << "\n";
          choreo_unreachable("divide by zero is found in shape evaluation.");
        }
        res += std::to_string((l + r - 1) / r);
      } else if (op == "#") {
        // calculate the upper bound result
        res += std::to_string(l * r);
      } else if (op == "#+") {
        res += std::to_string(l + r);
      } else if (op == "#-") {
        res += std::to_string(l - r);
      } else {
        Error(loc,
              "simplification of operation `" + op + "' is not yet supported.");
      }
    }
    if (trace && verbose)
      dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
             << rhs << " to '" << res << "'\n";
    return res;
  }

  // for symbolic values
  if (op == "/") {
    // a/a == 1
    if (GetValueNumberOfSignature(lhs) == GetValueNumberOfSignature(rhs)) {
      std::string res = "const_1";
      if (trace && verbose)
        dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
               << rhs << " to '" << res << "'\n";
      return res;
    }
    // a/1 = a
    if (rhs == "const_1") return lhs;
    // useful simplification: a/(a/b) = b
    if (!PrefixedWith(lhs, "#") /*not multiple values*/) {
      NumTy rvn = GetValueNumberOfSignature(rhs);
      auto bind_set = GetBindSet(rvn);
      bind_set.insert(rvn); // always add self
      for (auto div_vn : bind_set) {
        auto sig = GetSignatureFromValueNumber(div_vn);
        if (!PrefixedWith(rhs, "/:")) continue;
        auto div = GetOperandsValNo(sig);
        assert(div.size() == 2);
        if (GetValueNumberOfSignature(lhs) == div[0]) {
          auto res = GetSignatureFromValueNumber(div[1]);

          if (trace && verbose)
            dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
                   << rhs << " to '" << res << "'\n";

          return res;
        }
      }
    }
  } else if (op == "*") {
#if 0
    // useful simplification: a*(b/a) = b
    // TODO: 6 * ( N/6 ), if N = 15, get error res 15
    // TODO: is there other simplify error?
    if (!PrefixedWith(lhs, "#") /*not multiple values*/) {
      int rvn = GetValueNumberOfSignature(lhs);
      auto bind_set = GetBindSet(rvn);
      bind_set.insert(rvn); // always add self
      for (auto div_vn : bind_set) {
        auto sig = GetSignatureFromValueNumber(div_vn);
        if (!PrefixedWith(lhs, "/:")) continue;
        auto div = GetOperandsValNo(sig);
        assert(div.size() == 2);
        if (GetValueNumberOfSignature(rhs) == div[1]) {
          auto res = GetSignatureFromValueNumber(div[0]);

          if (trace && verbose)
            dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op <<
            " "
                   << rhs << " to '" << res << "'\n";

          return res;
        }
      }
    }
#endif
  } else if (op == "-") {
    // a-a == 1
    if (GetValueNumberOfSignature(lhs) == GetValueNumberOfSignature(rhs)) {
      std::string res = "const_0";
      if (trace && verbose)
        dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
               << rhs << " to '" << res << "'\n";
      return res;
    }
  } else if (op == "+") {
    // useful simplification: a-b+b = a
    if (!PrefixedWith(rhs, "#") /*not multiple values*/) {
      NumTy lvn = GetValueNumberOfSignature(lhs);
      auto bind_set = GetBindSet(lvn);
      bind_set.insert(lvn); // always add self
      for (auto minus_vn : bind_set) {
        auto sig = GetSignatureFromValueNumber(minus_vn);
        if (!PrefixedWith(lhs, "-:")) continue;
        auto minus = GetOperandsValNo(sig);
        assert(minus.size() == 2);
        if (GetValueNumberOfSignature(rhs) == minus[1]) {
          auto res = GetSignatureFromValueNumber(minus[0]);

          if (trace && verbose)
            dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
                   << rhs << " to '" << res << "'\n";

          return res;
        }
      }
    }
  } else if (op == "#") {
    // suppose `a` and `b` are bounded vars
    // `#a` is 4, `#b` is `N/#a` where `N` is dynamic dim
    // if `xx.chunkat(a#b)`, then the result shape should be 1
    // that is, N / (#a * #b) = N / N = 1
    // so, `a#b` should be simplified to a bounded var whose ubound is `N`
    if (!PrefixedWith(lhs, "#") /*not multiple values*/) {
      NumTy rvn = GetValueNumberOfSignature(rhs);
      auto bind_set = GetBindSet(rvn);
      bind_set.insert(rvn); // always add self
      for (auto div_vn : bind_set) {
        auto sig = GetSignatureFromValueNumber(div_vn);
        if (!PrefixedWith(rhs, "/:")) continue;
        auto div = GetOperandsValNo(sig);
        assert(div.size() == 2);
        if (GetValueNumberOfSignature(lhs) == div[1]) {
          auto res = GetSignatureFromValueNumber(div[0]);

          if (trace && verbose)
            dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
                   << rhs << " to '" << res << "'\n";

          return res;
        }
      }
    }
    if (!PrefixedWith(rhs, "#")) {
      // TODO: # is different with *
      // a # (b/a) will alway result in a?
      // if so, we need to emphasize this optimization to our users.
      NumTy lvn = GetValueNumberOfSignature(lhs);
      auto bind_set = GetBindSet(lvn);
      bind_set.insert(lvn); // always add self
      for (auto div_vn : bind_set) {
        auto sig = GetSignatureFromValueNumber(div_vn);
        if (!PrefixedWith(lhs, "/:")) continue;
        auto div = GetOperandsValNo(sig);
        assert(div.size() == 2);
        if (GetValueNumberOfSignature(rhs) == div[1]) {
          auto res = GetSignatureFromValueNumber(div[0]);

          if (trace && verbose)
            dbgs() << ScopeIndent() << "<Simplify> '" << lhs << " " << op << " "
                   << rhs << " to '" << res << "'\n";

          return res;
        }
      }
    }
  }
  return std::nullopt;
}

NumTy ValueNumbering::GetValueNumberOfSignature(const SignTy& signature) const {
  if (signature == "") choreo_unreachable("invalid signature provided.");

  if (IsUnknownSign(signature)) return NumTy::Unknown();
  if (IsNoneSign(signature)) return NumTy::None();

  // Check if this expression has been encountered before
  if (vntbl.Exists(signature))
    return vntbl.GetValueNum(signature); // Return existing value number

  choreo_unreachable("failed to get value number of signature \"" + signature +
                     "\".");

  return GetInvalidValueNumber();
}

bool ValueNumbering::HasValueNumberOfSignature(const SignTy& signature) const {
  if (IsUnknownSign(signature) || IsNoneSign(signature)) return true;
  return vntbl.Exists(signature);
}

bool ValueNumbering::HasValidValueNumberOfSignature(const SignTy& signature) {
  if (IsNoneSign(signature)) return true;
  if (vntbl.Exists(signature)) return vntbl.GetValueNum(signature).IsValid();

  return false;
}

NumTy ValueNumbering::GetOrGenValueNumberFromSignature(
    const SignTy& signature) {
  if (PrefixedWith(signature, "#") &&
      (CountElementsInSignature(signature) == 1)) {
    // works for input like "#1"
    return VNReal(signature);
  }
  if (HasValueNumberOfSignature(signature))
    return GetValueNumberOfSignature(signature);
  return GenerateValueNumberFromSignature(signature);
}

NumTy ValueNumbering::GenerateValueNumberFromSignature(
    const SignTy& signature) {
  if (IsUnknownSign(signature)) return NumTy::Unknown();
  if (IsNoneSign(signature)) return NumTy::None();

  if (HasValueNumberOfSignature(signature))
    choreo_unreachable("signature \"" + signature + "\" has already existed.");

  NumTy valNo = vntbl.Generate(signature);

  if (trace)
    dbgs() << ScopeIndent() << "New VN " << STR(valNo) << ": '" << signature
           << "'\n";

  return valNo;
}

const std::vector<NumTy> ValueNumbering::Flatten(NumTy valno) const {
  if (!valno.IsValid()) choreo_unreachable("expect a valid valno.");
  std::vector<NumTy> mvn;
  std::deque<NumTy> work_list;
  work_list.push_back(valno);

  while (!work_list.empty()) {
    auto val_no = work_list.front();
    work_list.pop_front();
    assert(val_no.IsValid());

    auto valsign = GetSignatureFromValueNumber(val_no);
    auto vn_count = CountElementsInSignature(valsign);

    assert(vn_count >= 1);

    if (vn_count == 1) {
      mvn.push_back(val_no);
      continue;
    }

    for (int i = vn_count - 1; i >= 0; --i)
      work_list.push_front(GetNthValNo(valsign, i));
  }
  assert(mvn.size() > 0);
  return mvn;
}

NumTy ValueNumbering::GetNthValNo(const SignTy& input, NumTy n) const {
  auto ith_str = GetNthElement(input, n.Value());
  if (!ith_str)
    choreo_unreachable("no value number is found for (" + STR(n) +
                       "th): " + input + ".");

  assert(ith_str.value()[0] == '#' ||
         (ith_str.value().substr(0, 6) == "const_"));

  return ith_str.value()[0] == '#' ? VNReal(ith_str.value())
                                   : GetValueNumberOfSignature(ith_str.value());
}

const std::string ValueNumbering::ScopeIndent() {
  std::string indent;
  for (size_t i = 0; i <= visitor->SSTab().ScopeDepth(); ++i) indent += " ";
  return indent;
}

void ValueNumbering::Error(const location& loc, const std::string& message) {
  visitor->Error(loc, message);
}

void ValueNumbering::Warning(const location& loc, const std::string& message) {
  visitor->Warning(loc, message);
}
