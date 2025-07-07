#include "shapeinfer.hpp"

using namespace Choreo;
using namespace Choreo::valno;

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

ValueItem ValueNumbering::GenValueItemFromValueNumber(const NumTy& valno) {
  if (!valno.IsValid()) choreo_unreachable("signature is invalid.");
  return GenValueItemFromSignature(GetSignatureFromValueNumber(valno));
}

ValueItem ValueNumbering::GenValueItemFromSignature(const SignTy& input) {
  auto sign = Simplify(input);
  if (!IsValid(sign)) choreo_unreachable("signature is invalid.");
  if (isa<MultiSigns>(sign)) {
    choreo_unreachable(
        "unable to generate single value item from a multi-signatures:" +
        STR(input));
  } else if (IsUnknown(sign)) {
    choreo_unreachable(
        "unable to generate single value item from an unknown-signatures.");
  } else if (IsNone(sign)) {
    return sbe::nil();
  } else if (auto osn = CSign(sign)) {
    if (osn->Holds<bool>())
      return sbe::bl(osn->Get<bool>());
    else if (osn->Holds<int64_t>())
      return sbe::nu(osn->Get<int64_t>());
  } else if (auto osn = SSign(sign)) {
    // must be a scoped symbol
    return sbe::sym(osn->Value());
  } else if (auto osn = OpSign(sign)) {
    auto op = osn->Operation();
    auto& oprds = osn->OperandValueNums();
    if (oprds.size() == 3) {
      // ternary operation
      assert(op == "?" && "unexpected ternary operation.");
      auto pvi = GenValueItemFromSignature(SignNum(oprds[0]));
      auto lvi = GenValueItemFromSignature(SignNum(oprds[1]));
      auto rvi = GenValueItemFromSignature(SignNum(oprds[2]));
      if (pvi && lvi && rvi) return sbe::sel(pvi, lvi, rvi)->Normalize();
    } else if (oprds.size() == 2) {
      if ((op == "+") || (op == "-") || (op == "*") || (op == "/") ||
          (op == "%") || (op == ">") || (op == "<") || (op == "|") ||
          (op == "&") || (op == "^") || (op == ">=") || (op == "<=") ||
          (op == "==") || (op == "!=") || (op == ">>") || (op == "<<")) {
        auto lvi = GenValueItemFromSignature(SignNum(oprds[0]));
        auto rvi = GenValueItemFromSignature(SignNum(oprds[1]));
        if (lvi && rvi) return sbe::bop(ToOpCode(op), lvi, rvi)->Normalize();
      } else if ((op == "cdiv")) {
        auto lvi = GenValueItemFromSignature(SignNum(oprds[0]));
        auto rvi = GenValueItemFromSignature(SignNum(oprds[1]));
        if (lvi && rvi)
          return sbe::bop(OpCode::DIVIDE, lvi + (rvi - sbe::nu(1)), rvi)
              ->Normalize();
      }
    } else if (oprds.size() == 1) {
      if ((op == "!") || (op == "~")) {
        if (auto ovi = GenValueItemFromSignature(SignNum(oprds[0])))
          return sbe::uop(ToOpCode(op), ovi)->Normalize();
      }
    }
  }

  return nullptr;
}

const ValueList
ValueNumbering::GenValueListFromValueNumber(const NumTy& valno) {
  if (!valno.IsValid()) choreo_unreachable("signature is invalid.");
  return GenValueListFromSignature(SignNum(valno));
}

const ValueList ValueNumbering::GenValueListFromSignature(const SignTy& input) {
  if (!IsValid(input)) choreo_unreachable("signature is invalid.");
  std::vector<ValueItem> res;
  if (auto osn = MSign(input)) {
    for (auto n : osn->AllValueNums())
      res.push_back(GenValueItemFromValueNumber(n));
  } else
    res.push_back(GenValueItemFromSignature(input));

  return res;
}

// Note: it adds value numbers as necessary
const SignTy ValueNumbering::ValueItemToSignature(const ValueItem& vi,
                                                  bool gen) {
  if (VIIsNil(vi)) {
    return non_sn();
  } else if (auto iv = VIInt(vi)) {
    auto sign = c_sn(iv.value());
    auto vn = GetOrGenValueNumberFromSignature(sign); // always generate
    return GetSignatureFromValueNumber(vn);
  } else if (auto sym = VISym(vi)) {
    assert(PrefixedWith(sym.value(), "::") && "expected a scoped symbol.");
    auto sign = s_sn(sym.value());
    assert(HasValueNumberOfSignature(sign) &&
           "the symbol does have a value number.");
    return sign;
  } else if (auto bop = VIUop(vi)) {
    auto osign = ValueItemToSignature(bop->GetOperand(), true);
    auto ovn = GetValueNumberOfSignature(osign);
    auto sign = o_sn(STR(bop->GetOpCode()), ovn);
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
    auto sign = o_sn(STR(bop->GetOpCode()), lvn, rvn);
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
    auto sign = o_sn(STR(top->GetOpCode()), pvn, lvn, rvn);
    if (gen) {
      auto vn = GetOrGenValueNumberFromSignature(sign);
      return GetSignatureFromValueNumber(vn);
    } else
      return sign;
  } else
    choreo_unreachable("unsupported value.");
  return unk_sn();
}

const SignTy ValueNumbering::ValueListToSignature(const ValueList& vl,
                                                  bool gen) {
  assert(!vl.empty());
  if (vl.size() == 1) return ValueItemToSignature(vl[0], gen);

  auto signs = m_sn();
  for (auto vi : vl) {
    auto vis = ValueItemToSignature(vi, gen);
    NumTy vn = GetInvalidValueNumber();
    if (gen)
      vn = GetOrGenValueNumberFromSignature(vis);
    else
      vn = GetValueNumberOfSignature(vis);
    signs->Append(vn);
  }
  return signs;
}

// It binds a expression signature with an existing value number.  use it
// carefully.
void ValueNumbering::AssociateSignatureWithValueNumber(const SignTy& sign,
                                                       const NumTy& valno) {
  if (!IsValid(sign)) choreo_unreachable("signature is invalid.");
  if (!valno.IsValid()) choreo_unreachable("signature is invalid.");
  assert(vntbl.Exists(valno) && "invalid value number is provided.");
  if (vntbl.Exists(sign))
    assert((vntbl.GetValueNum(sign) == valno) &&
           "must associate signature with different value number.");

  vntbl.Alias(valno, sign);

  if (trace)
    dbgs() << ScopeIndent() << "Alias \"" << sign << "\" -> " << valno << "\n";
}

void ValueNumbering::AssociateSignatureWithInvalidValueNumber(
    const SignTy& sign) {
  if (!IsValid(sign)) choreo_unreachable("signature is invalid.");
  assert(!vntbl.Exists(sign) && "signature does exists.");

  vntbl.DummyGen(sign);

  if (trace)
    dbgs() << ScopeIndent() << "Alias \"" << sign << "\" -> #<invalid>\n";
}

void ValueNumbering::RebindSignatureWithValueNumber(const SignTy& s,
                                                    const NumTy& v) {
  if (!IsValid(s)) choreo_unreachable("signature is invalid.");
  if (!v.IsValid()) choreo_unreachable("value number is invalid.");
  if (!vntbl.Exists(s))
    return; // could be invalid symbol for scope change
            // TODO: is this code safe?
  vntbl.BindDummy(s, v);

  if (trace)
    dbgs() << ScopeIndent() << "Alias(Rebind) \"" << STR(s) << "\" -> "
           << STR(v) << "\n";
}

const SignTy ValueNumbering::Simplify(const SignTy& sign) {
  if (!IsValid(sign)) choreo_unreachable("signature is invalid.");
  if (auto mss = MSign(sign); mss && mss->Count() == 1)
    return SignNum(mss->NumAt(0));

  // Applies the algebraic simplification
  std::set<OpTy> optimizable = {
      "+",  "-", "*", "/",  "%",  "cdiv", "@",  "@+",
      "@-", "<", ">", "<=", ">=", "==",   "!=",
  };

  auto HandleMultiSigns = [this](const std::string& op, const SignTy& lhs,
                                 const SignTy& rhs) {
    // For multiple signatures, it is possible to generate new intermediate
    // value numbers
    auto& lvns = MSign(lhs)->AllValueNums();
    auto& rvns = MSign(rhs)->AllValueNums();
    assert(lvns.size() == rvns.size());
    auto r_sign = m_sn();
    for (size_t i = 0; i < lvns.size(); ++i) {
      auto e_sign = Simplify(o_sn(op, lvns[i], rvns[i]));
      r_sign->Append(GetOrGenValueNumberFromSignature(e_sign));
    }
    return r_sign;
  };

  auto osn = OpSign(sign);
  if (!osn) return sign;

  auto op = osn->Operation();
  auto& operands = osn->OperandValueNums();

  if ((operands.size() == 2)) {
    NumTy lvn = operands[0];
    NumTy rvn = operands[1];
    auto lsign = SignNum(lvn);
    auto rsign = SignNum(rvn);

    if (op == "concat") {
      auto lns = NumVector(lvn);
      auto rns = NumVector(rvn);
      auto r_msn = m_sn();
      for (const NumTy& n : lns) r_msn->Append(n);
      for (const NumTy& n : rns) r_msn->Append(n);
      return r_msn;
    } else if (optimizable.count(op)) {
      if (lsign->Count() == rsign->Count()) {
        if (lsign->Count() == 1) {
          if (op == "#") { // operation '#' is special
            auto simple_sign = TryToSimplifyBinary("*", lsign, rsign);
            if (!IsUnknown(simple_sign)) return simple_sign;
            return o_sn("*", lvn, rvn);
          } else {
            auto simple_sign = TryToSimplifyBinary(op, lsign, rsign);
            if (!IsUnknown(simple_sign)) return simple_sign;
            return o_sn(op, lvn, rvn);
          }
        } else
          return HandleMultiSigns(op, MSign(lsign), MSign(rsign));
      } else {
        // else, multivalues operates on a single value
        if (lsign->Count() > rsign->Count())
          return HandleMultiSigns(op, lsign, m_sn(rvn, lsign->Count()));
        else
          return HandleMultiSigns(op, m_sn(lvn, rsign->Count()), rsign);
      }
    }
  }

  return sign;
}

// when failed to optimize, return unknown
const SignTy ValueNumbering::TryToSimplifyBinary(const OpTy& op,
                                                 const SignTy& lhs,
                                                 const SignTy& rhs,
                                                 bool verbose) {
  if (!IsValid(lhs)) choreo_unreachable("signature is invalid.");
  if (!IsValid(rhs)) choreo_unreachable("signature is invalid.");
  if (op == "concat") return unk_sn();

  // try to simplify using symbexpr first
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

  auto l_csn = CSign(lhs);
  auto r_csn = CSign(rhs);

  auto Report = [this, &op, &lhs, &rhs,
                 &verbose](const SignTy& s) -> const SignTy {
    if (trace && verbose)
      dbgs() << ScopeIndent() << "<Simplify> '" << lhs->ToString() << " "
             << STR(op) << " " << rhs->ToString() << " to '" << s->ToString()
             << "'\n";
    return s;
  };

  // try the more general constant folding, including floating-point
  if (l_csn && r_csn) {
    if ((!l_csn->IsFloat() && !r_csn->IsFloat()) || CCtx().SimplifyFpValno()) {
      auto res = std::visit(ConstSign::ArithmeticVisitor{op}, l_csn->Value(),
                            r_csn->Value());
      if (!IsUnknown(res)) return Report(res);
    }
  }
  // or else, apply optimization for the symbolic expression
  else if (op == "/") {
    // a/a == 1
    if (NumSign(lhs) == NumSign(rhs)) {
      return Report(c_sn(1));
    }
    // a/1 = a
    else if (rhs == c_sn(1))
      return Report(lhs);
    // useful simplification: a/(a/b) = b
    else if (lhs->Count() == 1 /*not multiple values*/) {
      NumTy rvn = GetValueNumberOfSignature(rhs);
      auto bind_set = GetBindSet(rvn);
      bind_set.insert(rvn); // always add self
      for (auto div_vn : bind_set) {
        auto osn = OpSign(SignNum(div_vn));
        if (!osn || !osn->IsOp("/")) continue;
        auto& div = osn->OperandValueNums();
        assert(div.size() == 2);
        if (GetValueNumberOfSignature(lhs) == div[0])
          return Report(GetSignatureFromValueNumber(div[1]));
      }
    }
  } else if (op == "-") {
    // a-a == 1
    if (NumSign(lhs) == NumSign(rhs)) return Report(c_sn(0));
  } else if (op == "+") {
    // useful simplification: a-b+b = a
    if (rhs->Count() == 1 /*not multiple values*/) {
      NumTy lvn = NumSign(lhs);
      auto bind_set = GetBindSet(lvn);
      bind_set.insert(lvn); // always add self
      for (auto minus_vn : bind_set) {
        auto osn = OpSign(SignNum(minus_vn));
        if (!osn || !osn->IsOp("-")) continue;
        auto& minus = osn->OperandValueNums();
        assert(minus.size() == 2);
        if (NumSign(rhs) == minus[1]) return Report(SignNum(minus[0]));
      }
    }
  } else if (op == "#") {
    // suppose `a` and `b` are bounded vars
    // `#a` is 4, `#b` is `N/#a` where `N` is dynamic dim
    // if `xx.chunkat(a#b)`, then the result shape should be 1
    // that is, N / (#a * #b) = N / N = 1
    // so, `a#b` should be simplified to a bounded var whose ubound is `N`
    if (lhs->Count() == 1 /*not multiple values*/) {
      NumTy rvn = NumSign(rhs);
      auto bind_set = GetBindSet(rvn);
      bind_set.insert(rvn); // always add self
      for (auto div_vn : bind_set) {
        auto osn = OpSign(SignNum(div_vn));
        if (!osn || !osn->IsOp("/")) continue;
        auto& div = osn->OperandValueNums();
        assert(div.size() == 2);
        if (NumSign(lhs) == div[1]) return Report(SignNum(div[0]));
      }
    } else if (rhs->Count() == 1) {
      // TODO: # is different with *
      // a # (b/a) will alway result in a?
      // if so, we need to emphasize this optimization to our users.
      NumTy lvn = NumSign(lhs);
      auto bind_set = GetBindSet(lvn);
      bind_set.insert(lvn); // always add self
      for (auto div_vn : bind_set) {
        auto osn = OpSign(SignNum(div_vn));
        if (!osn || !osn->IsOp("/")) continue;
        auto& div = osn->OperandValueNums();
        assert(div.size() == 2);
        if (NumSign(rhs) == div[1]) return Report(SignNum(div[0]));
      }
    }
  }

  return unk_sn(); // no optimization
}

const NumTy
ValueNumbering::GetValueNumberOfSignature(const SignTy& signature) const {
  if (!IsValid(signature)) choreo_unreachable("signature is invalid.");
  if (IsUnknown(signature)) return NumTy::Unknown();
  if (IsNone(signature)) return NumTy::None();

  // Check if this expression has been encountered before
  if (vntbl.Exists(signature))
    return vntbl.GetValueNum(signature); // Return existing value number

  choreo_unreachable("failed to get value number of signature \"" +
                     STR(signature) + "\".");

  return GetInvalidValueNumber();
}

bool ValueNumbering::HasValueNumberOfSignature(const SignTy& signature) const {
  if (!IsValid(signature)) choreo_unreachable("signature is invalid.");
  if (IsUnknown(signature) || IsNone(signature)) return true;
  return vntbl.Exists(signature);
}

bool ValueNumbering::HasValidValueNumberOfSignature(const SignTy& signature) {
  if (!IsValid(signature)) choreo_unreachable("signature is invalid.");
  if (IsNone(signature)) return true;
  if (vntbl.Exists(signature)) return vntbl.GetValueNum(signature).IsValid();

  return false;
}

const NumTy
ValueNumbering::GetOrGenValueNumberFromSignature(const SignTy& signature) {
  if (!IsValid(signature)) choreo_unreachable("signature is invalid.");
  if (HasValueNumberOfSignature(signature))
    return GetValueNumberOfSignature(signature);
  return GenerateValueNumberFromSignature(signature);
}

const NumTy
ValueNumbering::GenerateValueNumberFromSignature(const SignTy& signature) {
  if (!IsValid(signature)) choreo_unreachable("signature is invalid.");
  if (IsUnknown(signature)) return NumTy::Unknown();
  if (IsNone(signature)) return NumTy::None();

  if (HasValueNumberOfSignature(signature))
    choreo_unreachable("signature \"" + STR(signature) +
                       "\" has already existed.");

  NumTy valNo = vntbl.Generate(signature);

  if (trace)
    dbgs() << ScopeIndent() << "New VN " << STR(valNo) << ": '" << signature
           << "'\n";

  return valNo;
}

const std::vector<NumTy> ValueNumbering::Flatten(const NumTy& valno) const {
  if (!valno.IsValid()) choreo_unreachable("expect a valid valno.");
  std::vector<NumTy> mvn;
  std::deque<NumTy> work_list;
  work_list.push_back(valno);

  while (!work_list.empty()) {
    auto val_no = work_list.front();
    work_list.pop_front();
    assert(val_no.IsValid());

    auto valsign = SignNum(val_no);
    if (auto osn = MSign(valsign)) {
      auto vns = osn->AllValueNums();
      for (auto v = vns.rbegin(); v != vns.rend(); v++)
        work_list.push_front(*v);
    } else
      mvn.push_back(val_no);
  }
  assert(mvn.size() > 0);
  return mvn;
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
