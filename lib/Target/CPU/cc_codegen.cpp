#include "cc_codegen.hpp"
#include "codegen_utils.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "assert_site.hpp"
#include "ast.hpp"
#include "choreo_cc_header.inc"
#include "choreo_header.inc"
#include "choreo_types_header.inc"
#include "codegen.hpp"
#include "context.hpp"
#include "types.hpp"

using namespace Choreo;
using namespace Choreo::CC;

const std::string CCCodeGen::TypeSTR(const Type& ty) const {
  if (isa<VoidType>(&ty)) return "void";
  if (isa<S8Type>(&ty)) return "char";
  if (isa<U8Type>(&ty)) return "unsigned char";
  if (isa<S16Type>(&ty)) return "short";
  if (isa<U16Type>(&ty)) return "unsigned short";
  if (isa<S32Type>(&ty)) return "int";
  if (isa<U32Type>(&ty)) return "unsigned int";
  if (isa<S64Type>(&ty)) return "long long";
  if (isa<U64Type>(&ty)) return "unsigned long long";
  if (isa<BooleanType>(&ty)) return "bool";
  if (isa<F16Type>(&ty)) return "choreo::f16";
  if (isa<BF16Type>(&ty)) return "choreo::bf16";
  if (isa<F32Type>(&ty) || isa<TF32Type>(&ty)) return "float";
  if (isa<F64Type>(&ty)) return "double";
  if (isa<FloatE4M3Type>(&ty) || isa<FloatE5M2Type>(&ty)) return "uint8_t";
  if (auto sty = dyn_cast<SpannedType>(&ty))
    return "choreo::spanned_view<choreo::" + STR(sty->e_type) + ", " +
           std::to_string(sty->Dims()) + ">";
  if (auto bitt = dyn_cast<BoundedITupleType>(&ty)) {
    (void)bitt;
    return "int";
  }
  choreo_unreachable("unsupported CC type: " + STR(ty) + ".");
  return "";
}

static const char* CCBaseTypeName(BaseType bt) {
  switch (bt) {
  case BaseType::F64: return "double";
  case BaseType::F32: return "float";
  case BaseType::TF32: return "float";
  case BaseType::F16: return "choreo::f16";
  case BaseType::BF16: return "choreo::bf16";
  case BaseType::F8_E4M3: return "uint8_t";
  case BaseType::F8_E5M2: return "uint8_t";
  case BaseType::F8_UE8M0: return "uint8_t";
  case BaseType::F8_UE4M3: return "uint8_t";
  case BaseType::F6_E2M3: return "uint8_t";
  case BaseType::F6_E3M2: return "uint8_t";
  case BaseType::F4_E2M1: return "uint8_t";
  case BaseType::U64: return "unsigned long long";
  case BaseType::U32: return "unsigned int";
  case BaseType::U16: return "unsigned short";
  case BaseType::U8: return "unsigned char";
  case BaseType::S64: return "long long";
  case BaseType::S32: return "int";
  case BaseType::S16: return "short";
  case BaseType::S8: return "signed char";
  case BaseType::BOOL: return "bool";
  default: choreo_unreachable("unsupported CC base-type: " + STR(bt) + ".");
  }
  return "";
}

const std::string CCCodeGen::OpValueSTR(const ValueItem& vi,
                                        const std::string& parent_op,
                                        bool is_left_child,
                                        bool ll_suffix) const {
  auto WrapParen = [&](const std::string& s, const std::string& cur_op) {
    if (Operator::NeedParen(cur_op, parent_op, is_left_child))
      return "(" + s + ")";
    if (parent_op.empty()) return "(" + s + ")";
    return s;
  };

  if (auto iv = VIInt(vi)) {
    auto s = std::to_string(*iv);
    return ll_suffix ? s + "LL" : s;
  }
  if (auto bv = VIBool(vi)) return PSTR(vi);
  if (auto sv = VISym(vi)) {
    auto res = UnScopedExpr(ssm.HostName(sv.value()));
    if (ll_suffix) return "static_cast<long long>(" + res + ")";
    return res;
  }
  if (auto bo = VIBop(vi)) {
    if (bo->GetOpCode() == OpCode::ADD) {
      if (auto rv = VIInt(bo->GetRight()); rv && rv.value() < 0) {
        std::string res = OpValueSTR(bo->GetLeft(), "-", true, ll_suffix) +
                          " - " + std::to_string(-rv.value());
        return WrapParen(res, "-");
      }
    }
    std::string op = STR(bo->GetOpCode());
    std::string res = OpValueSTR(bo->GetLeft(), op, true, ll_suffix) + " " +
                      op + " " +
                      OpValueSTR(bo->GetRight(), op, false, ll_suffix);
    return WrapParen(res, op);
  }
  if (auto uo = VIUop(vi)) {
    std::string op = STR(uo->GetOpCode());
    std::string res = op + OpValueSTR(uo->GetOperand(), op, false, ll_suffix);
    return WrapParen(res, op);
  }
  if (auto to = VITop(vi)) {
    std::string op = "?";
    std::string res = OpValueSTR(to->GetPred(), op, true) + " ? " +
                      OpValueSTR(to->GetLeft(), op, true, ll_suffix) + " : " +
                      OpValueSTR(to->GetRight(), op, false, ll_suffix);
    return WrapParen(res, op);
  }
  choreo_unreachable("unsupported ValueItem in CC codegen.");
  return "";
}

const std::string CCCodeGen::ValueSTR(const ValueItem& vi,
                                      bool ll_suffix) const {
  return OpValueSTR(vi, "", true, ll_suffix);
}

const std::string CCCodeGen::ValueSTR(const ValueList& vl,
                                      const std::string& sep) const {
  std::string res;
  for (size_t i = 0; i < vl.size(); ++i) {
    if (i > 0) res += sep;
    res += ValueSTR(vl[i]);
  }
  return res;
}

const std::string CCCodeGen::ExprSTR(AST::ptr<AST::Node> e) const {
  std::ostringstream oss;

  if (auto id = dyn_cast<AST::Identifier>(e)) {
    if (within_map.count(InScopeNameForRef(id->name))) {
      size_t i = 0;
      for (auto iv_name : within_map.at(InScopeNameForRef(id->name)))
        oss << ((i++ == 0) ? "" : ", ") << UnScopedName(ssm.HostName(iv_name));
    } else {
      oss << UnScopedName(ssm.HostName(InScopeNameForRef(id->name)));
    }
  } else if (auto np = dyn_cast<AST::Nullptr>(e)) {
    (void)np;
    oss << "nullptr";
  } else if (auto il = dyn_cast<AST::IntLiteral>(e)) {
    oss << il->ValAsString();
  } else if (auto fl = dyn_cast<AST::FloatLiteral>(e)) {
    std::ostringstream fp_val;
    if (fl->IsFloat32())
      fp_val << std::fixed << fl->Val_f32() << "f";
    else if (fl->IsFloat64())
      fp_val << std::fixed << fl->Val_f64();
    else
      fp_val << "0.0f";
    oss << fp_val.str();
  } else if (auto sl = dyn_cast<AST::StringLiteral>(e)) {
    oss << sl->EscapedVal();
  } else if (auto b = dyn_cast<AST::BoolLiteral>(e)) {
    oss << b->value;
  } else if (auto ii = dyn_cast<AST::IntIndex>(e)) {
    return ExprSTR(ii->value);
  } else if (auto it = dyn_cast<AST::IntTuple>(e)) {
    int i = 0;
    oss << "{";
    for (auto& v : it->GetValues()->AllValues()) {
      if (i++ > 0) oss << ", ";
      oss << ExprSTR(v);
    }
    oss << "}";
  } else if (auto da = dyn_cast<AST::DataAccess>(e)) {
    if (da->AccessElement()) {
      auto sty = GetSpannedType(GetSymbolType(da->data->name));
      assert(sty && "can only access the element of a spanned type.");
      oss << "*((" << CCBaseTypeName(sty->ElementType()) << "*)"
          << ExprSTR(da->data);
      size_t idx = 0;
      auto shape = sty->GetShape();
      for (auto item : da->GetIndices()) {
        auto offset_vi = sbe::nu(0);
        if (auto id_node = AST::GetIdentifier(item)) {
          if (within_map.count(InScopeNameForRef(id_node->name))) {
            auto ivs = within_map.at(InScopeNameForRef(id_node->name));
            for (auto iv_itr = ivs.begin(); iv_itr != ivs.end(); ++iv_itr) {
              offset_vi = sbe::sym(*iv_itr);
              if (shape.Rank() > idx + 1)
                offset_vi =
                    offset_vi * shape.TrimDims(idx + 1).ElementCountValue();
              SimplifyExpression(offset_vi);
              if (!sbe::ceq(offset_vi, sbe::nu(0)))
                oss << " + " << ValueSTR(offset_vi);
              ++idx;
            }
          } else {
            offset_vi = sbe::sym(InScopeNameForRef(id_node->name));
            if (shape.Rank() > idx + 1)
              offset_vi =
                  offset_vi * shape.TrimDims(idx + 1).ElementCountValue();
            SimplifyExpression(offset_vi);
            if (!sbe::ceq(offset_vi, sbe::nu(0)))
              oss << " + " << ValueSTR(offset_vi);
            ++idx;
          }
        } else if (auto int_lit = AST::GetIntLiteral(*item)) {
          offset_vi = sbe::nu(int_lit->Val());
          if (shape.Rank() > idx + 1)
            offset_vi = offset_vi * shape.TrimDims(idx + 1).ElementCountValue();
          SimplifyExpression(offset_vi);
          if (!sbe::ceq(offset_vi, sbe::nu(0)))
            oss << " + " << ValueSTR(offset_vi);
          ++idx;
        } else {
          oss << " + " << ExprSTR(item);
          ++idx;
        }
      }
      oss << ")";
    } else {
      oss << ExprSTR(da->data) << ".data()";
    }
  } else if (auto ca = dyn_cast<AST::ChunkAt>(e)) {
    auto base = ExprSTR(ca->data);
    auto off_vi = GenOffset(ca);
    if (sbe::ceq(off_vi, sbe::nu(0)))
      oss << base;
    else
      oss << "(" << base << " + " << ValueSTR(off_vi) << ")";
  } else if (auto ce = dyn_cast<AST::CastExpr>(e)) {
    if (ce->IsForeignCast()) {
      oss << "static_cast<" << ce->ForeignType() << ">(" << ExprSTR(ce->GetR())
          << ")";
    } else {
      oss << "static_cast<" << CCBaseTypeName(ce->ToType()) << ">("
          << ExprSTR(ce->GetR()) << ")";
    }
  } else if (auto expr = dyn_cast<AST::Expr>(e)) {
    if (expr->IsReference()) {
      if (PSTR(expr) == "_") return "0";
      return ExprSTR(expr->GetReference());
    } else if (expr->IsUnary()) {
      auto& op = expr->GetOp();
      if (op == Op::LogicNot) {
        oss << "(!" << ExprSTR(expr->GetR()) << ")";
      } else if (op == Op::BitNot) {
        oss << "(~" << ExprSTR(expr->GetR()) << ")";
      } else if (op == Op::SizeOf) {
        auto se = expr->Opts().GetSize();
        if (IsValidValueItem(se))
          oss << ValueSTR(se);
        else {
          auto var = RemoveSuffix(*AST::GetName(*expr->GetR()), ".span");
          auto shape = GetShape(GetSymbolType(var));
          oss << ValueSTR(shape.ElementCountValue());
        }
      } else if (op == Op::DataOf || op == Op::MDataOf) {
        if (auto id = cast<AST::Expr>(expr->GetR())->GetSymbol())
          oss << id->name << ".data()";
        else
          oss << ExprSTR(expr->GetR()) << ".data()";
      } else if (op == Op::GetUBound) {
        auto rty = cast<BoundedType>(NodeType(*expr->GetR()));
        if (rty->Dims() == 1) oss << ValueSTR(rty->GetUpperBound());
      } else if (op == Op::PreInc) {
        oss << "++" << ExprSTR(expr->GetR());
      } else if (op == Op::PreDec) {
        oss << "--" << ExprSTR(expr->GetR());
      } else if (op == Op::AddrOf) {
        if (auto id = AST::GetIdentifier(expr->GetR()))
          oss << ExprSTR(id);
        else
          oss << "&(" << ExprSTR(expr->GetR()) << ")";
      } else {
        choreo_unreachable("unsupported CC unary op: " + STR(op));
      }
    } else if (expr->IsBinary()) {
      auto& op = expr->GetOp();
      if (op == Op::ElemOf) {
        oss << ExprSTR(expr->GetL()) << "[" << ExprSTR(expr->GetR()) << "]";
      } else if (op == Op::GetIth) {
        oss << ExprSTR(expr->GetR());
      } else if (op == Op::Select) {
        if (expr->IsTernary())
          oss << "(" << ExprSTR(expr->GetC()) << " ? " << ExprSTR(expr->GetL())
              << " : " << ExprSTR(expr->GetR()) << ")";
        else
          oss << "(" << ExprSTR(expr->GetL()) << " ? " << ExprSTR(expr->GetR())
              << " : 0)";
      } else if (op == Op::CeilDiv) {
        oss << "((" << ExprSTR(expr->GetL()) << " + " << ExprSTR(expr->GetR())
            << " - 1) / " << ExprSTR(expr->GetR()) << ")";
      } else if (op == Op::UBound) {
        auto lty = NodeType(*expr->GetL());
        auto rty_n = NodeType(*expr->GetR());
        if (IsActualBoundedIntegerType(lty) &&
            IsActualBoundedIntegerType(rty_n)) {
          auto rty = cast<BoundedType>(rty_n);
          oss << "(" << ExprSTR(expr->GetL()) << " * "
              << ValueSTR(rty->GetUpperBound()) << " + "
              << ExprSTR(expr->GetR()) << ")";
        } else {
          oss << "(" << ExprSTR(expr->GetL()) << " * " << ExprSTR(expr->GetR())
              << ")";
        }
      } else if (op == Op::UBoundAdd || op == Op::UBoundSub) {
        oss << ExprSTR(expr->GetL());
      } else {
        auto op_str = STR(op);
        oss << "(" << ExprSTR(expr->GetL()) << " " << op_str << " "
            << ExprSTR(expr->GetR()) << ")";
      }
    } else if (expr->GetOp() == Op::None && expr->GetR()) {
      oss << ExprSTR(expr->GetR());
    } else {
      choreo_unreachable("unsupported CC expr: " + PSTR(expr));
    }
  } else if (auto sa = dyn_cast<AST::SpanAs>(e)) {
    auto base_name = ssm.HostName(InScopeNameForRef(sa->id->name));
    oss << "static_cast<void*>(" << base_name << ")";
  } else if (auto call = dyn_cast<AST::Call>(e)) {
    oss << CallSTR(*call);
  } else {
    choreo_unreachable("unsupported CC ExprSTR node: " +
                       std::string(e->TypeNameString()));
  }

  return oss.str();
}

const std::string CCCodeGen::CallSTR(AST::Call& n) const {
  if (n.IsAtomic()) {
    static const std::unordered_map<std::string, std::string> atomic_map = {
        {"__atomic_add", "__atomic_fetch_add"},
        {"__atomic_sub", "__atomic_fetch_sub"},
        {"__atomic_exch", "__atomic_exchange_n"},
        {"__atomic_min", "__atomic_fetch_min"},
        {"__atomic_max", "__atomic_fetch_max"},
        {"__atomic_and", "__atomic_fetch_and"},
        {"__atomic_or", "__atomic_fetch_or"},
        {"__atomic_xor", "__atomic_fetch_xor"},
    };
    auto it = atomic_map.find(n.function->name);
    if (it != atomic_map.end()) {
      std::string result = it->second + "(";
      size_t i = 0;
      for (auto& a : n.GetArguments()) {
        if (i > 0) result += ", ";
        if (i == 0)
          result += "&(" + ExprSTR(a) + ")";
        else
          result += ExprSTR(a);
        ++i;
      }
      if (n.function->name == "__atomic_exch")
        result += ", __ATOMIC_SEQ_CST)";
      else
        result += ", __ATOMIC_RELAXED)";
      return result;
    }
    if (n.function->name == "__atomic_cas") {
      auto args = n.GetArguments();
      std::string result = "({ auto __cmp = " + ExprSTR(args[1]) +
                           "; __atomic_compare_exchange_n(&(" +
                           ExprSTR(args[0]) + "), &__cmp, " + ExprSTR(args[2]) +
                           ", false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)"
                           " ? __cmp : __cmp; })";
      return result;
    }
  }

  if (n.IsArith()) {
    static const std::unordered_map<std::string, std::string> arith_map = {
        {"__sqrt", "std::sqrt"},   {"__rsqrt", ""},
        {"__exp", "std::exp"},     {"__expm1", "std::expm1"},
        {"__log", "std::log"},     {"__log1p", "std::log1p"},
        {"__pow", "std::pow"},     {"__sin", "std::sin"},
        {"__cos", "std::cos"},     {"__tan", "std::tan"},
        {"__asin", "std::asin"},   {"__acos", "std::acos"},
        {"__atan", "std::atan"},   {"__atan2", "std::atan2"},
        {"__sinh", "std::sinh"},   {"__cosh", "std::cosh"},
        {"__tanh", "std::tanh"},   {"__ceil", "std::ceil"},
        {"__floor", "std::floor"}, {"__round", "std::round"},
        {"__abs", "std::abs"},     {"__fabs", "std::fabs"},
        {"__fmod", "std::fmod"},   {"__fmax", "std::fmax"},
        {"__fmin", "std::fmin"},   {"__isfinite", "std::isfinite"},
    };
    auto it = arith_map.find(n.function->name);
    std::string func;
    if (it != arith_map.end() && !it->second.empty()) {
      func = it->second;
    } else if (n.function->name == "__rsqrt") {
      std::string arg = ExprSTR(n.arguments->ValueAt(0));
      return "(1.0 / std::sqrt(" + arg + "))";
    } else if (n.function->name == "__sign") {
      std::string arg = ExprSTR(n.arguments->ValueAt(0));
      return "std::signbit(" + arg + ")";
    } else if (n.function->name == "__gelu") {
      std::string x = ExprSTR(n.arguments->ValueAt(0));
      return "(" + x + " * 0.5 * (1.0 + std::erf(" + x +
             " * 0.7071067811865476)))";
    } else if (n.function->name == "__sigmoid") {
      std::string x = ExprSTR(n.arguments->ValueAt(0));
      return "(1.0 / (1.0 + std::exp(-(" + x + "))))";
    } else if (n.function->name == "__softplus") {
      std::string x = ExprSTR(n.arguments->ValueAt(0));
      return "std::log(1.0 + std::exp(" + x + "))";
    } else {
      func = n.function->name;
    }
    std::string result = func + "(";
    if (n.arguments) {
      for (size_t i = 0; i < n.arguments->Count(); i++) {
        if (i > 0) result += ", ";
        result += ExprSTR(n.arguments->ValueAt(i));
      }
    }
    result += ")";
    return result;
  }

  if (n.IsLibCall()) {
    auto& fname = n.function->name;
    auto& args = *n.arguments;
    size_t argc = args.Count();
    auto arg = [&](size_t i) { return ExprSTR(args.ValueAt(i)); };

    if (fname == "__lib_gemm") {
      // __lib_gemm(out, A, B, K, N) or (out, A, B, bias, K, N)
      bool has_bias = (argc == 6);
      std::string out = arg(0), A = arg(1), B = arg(2);
      std::string K = has_bias ? arg(4) : arg(3);
      std::string N = has_bias ? arg(5) : arg(4);
      std::string M =
          "(" + UnScopedExpr(SizeExprOf(*NodeType(*args.ValueAt(0)), true)) +
          " / ((" + N + ") * sizeof(*" + out + ")))";
      std::string result = "choreo::cc::lib_gemm(" + out + ", " + A + ", " + B +
                           ", " + M + ", " + N + ", " + K + ")";
      return result;
    }

    static const std::unordered_map<std::string, std::string> unary_map = {
        {"__lib_abs", "std::abs"},
        {"__lib_neg", "-"},
        {"__lib_sqrt", "std::sqrt"},
        {"__lib_rsqrt", ""},
        {"__lib_exp", "std::exp"},
        {"__lib_log", "std::log"},
        {"__lib_ceil", "std::ceil"},
        {"__lib_floor", "std::floor"},
        {"__lib_round", "std::round"},
        {"__lib_sin", "std::sin"},
        {"__lib_cos", "std::cos"},
        {"__lib_tan", "std::tan"},
        {"__lib_asin", "std::asin"},
        {"__lib_acos", "std::acos"},
        {"__lib_atan", "std::atan"},
        {"__lib_sinh", "std::sinh"},
        {"__lib_cosh", "std::cosh"},
        {"__lib_tanh", "std::tanh"},
        {"__lib_erf", "std::erf"},
        {"__lib_erfc", "std::erfc"},
        {"__lib_reciprocal", ""},
        {"__lib_sign", ""},
        {"__lib_cbrt", "std::cbrt"},
        {"__lib_relu", ""},
        {"__lib_gelu", ""},
        {"__lib_sigmoid", ""},
        {"__lib_silu", ""},
    };

    auto uit = unary_map.find(fname);
    if (uit != unary_map.end() && argc >= 3) {
      // Unary: (dst, src, num)
      std::string dst = arg(0), src = arg(1), num = arg(2);
      std::string body;
      if (fname == "__lib_neg")
        body = dst + "[__i] = -" + src + "[__i]";
      else if (fname == "__lib_rsqrt")
        body = dst + "[__i] = 1.0 / std::sqrt(" + src + "[__i])";
      else if (fname == "__lib_reciprocal")
        body = dst + "[__i] = 1.0 / " + src + "[__i]";
      else if (fname == "__lib_sign")
        body = dst + "[__i] = (" + src + "[__i] > 0) - (" + src + "[__i] < 0)";
      else if (fname == "__lib_relu")
        body = dst + "[__i] = " + src + "[__i] > 0 ? " + src + "[__i] : 0";
      else if (fname == "__lib_gelu")
        body = dst + "[__i] = " + src + "[__i] * 0.5 * (1.0 + std::erf(" + src +
               "[__i] * 0.7071067811865476))";
      else if (fname == "__lib_sigmoid")
        body = dst + "[__i] = 1.0 / (1.0 + std::exp(-" + src + "[__i]))";
      else if (fname == "__lib_silu")
        body = dst + "[__i] = " + src + "[__i] / (1.0 + std::exp(-" + src +
               "[__i]))";
      else
        body = dst + "[__i] = " + uit->second + "(" + src + "[__i])";
      return "for (int __i = 0; __i < " + num + "; ++__i) " + body;
    }

    static const std::unordered_map<std::string, std::string> binary_map = {
        {"__lib_add", "+"}, {"__lib_sub", "-"}, {"__lib_mul", "*"},
        {"__lib_div", "/"}, {"__lib_max", ""},  {"__lib_min", ""},
        {"__lib_pow", ""},  {"__lib_fmod", ""}, {"__lib_atan2", ""},
        {"__lib_gt", ">"},  {"__lib_ge", ">="}, {"__lib_lt", "<"},
        {"__lib_le", "<="}, {"__lib_eq", "=="}, {"__lib_ne", "!="},
    };
    auto bit = binary_map.find(fname);
    if (bit != binary_map.end() && argc >= 4) {
      // Binary: (dst, lhs, rhs, num)
      std::string dst = arg(0), lhs = arg(1), rhs = arg(2), num = arg(3);
      std::string body;
      if (fname == "__lib_max")
        body = dst + "[__i] = std::max(" + lhs + "[__i], " + rhs + "[__i])";
      else if (fname == "__lib_min")
        body = dst + "[__i] = std::min(" + lhs + "[__i], " + rhs + "[__i])";
      else if (fname == "__lib_pow")
        body = dst + "[__i] = std::pow(" + lhs + "[__i], " + rhs + "[__i])";
      else if (fname == "__lib_fmod")
        body = dst + "[__i] = std::fmod(" + lhs + "[__i], " + rhs + "[__i])";
      else if (fname == "__lib_atan2")
        body = dst + "[__i] = std::atan2(" + lhs + "[__i], " + rhs + "[__i])";
      else
        body = dst + "[__i] = " + lhs + "[__i] " + bit->second + " " + rhs +
               "[__i]";
      return "for (int __i = 0; __i < " + num + "; ++__i) " + body;
    }

    if (fname == "__lib_reduce_sum" || fname == "__lib_reduce_max" ||
        fname == "__lib_reduce_min" || fname == "__lib_reduce_mean") {
      // (dst, src, num, reduce_dim, num_reduce)
      if (argc >= 5) {
        std::string dst = arg(0), src = arg(1), num = arg(2);
        std::string rdim = arg(3), nreduce = arg(4);
        std::string init;
        std::string accum;
        if (fname == "__lib_reduce_sum" || fname == "__lib_reduce_mean")
          init = "0",
          accum = "__acc += " + src + "[__outer * " + nreduce + " + __inner]";
        else if (fname == "__lib_reduce_max")
          init = src + "[__outer * " + nreduce + "]",
          accum = "__acc = std::max(__acc, " + src + "[__outer * " + nreduce +
                  " + __inner])";
        else
          init = src + "[__outer * " + nreduce + "]",
          accum = "__acc = std::min(__acc, " + src + "[__outer * " + nreduce +
                  " + __inner])";
        std::string result = "{ int __nouter = " + num + " / " + nreduce +
                             "; for (int __outer = 0; __outer < __nouter; "
                             "++__outer) { auto __acc = decltype(" +
                             src + "[0])(" + init +
                             "); for (int __inner = 0; __inner < " + nreduce +
                             "; ++__inner) { " + accum + "; } " + dst +
                             "[__outer] = __acc";
        if (fname == "__lib_reduce_mean") result += " / " + nreduce;
        result += "; } }";
        return result;
      }
    }

    if (fname == "__lib_convert" && argc >= 3) {
      std::string dst = arg(0), src = arg(1), num = arg(2);
      return "for (int __i = 0; __i < " + num + "; ++__i) " + dst +
             "[__i] = static_cast<std::remove_pointer_t<decltype(" + dst +
             ")>>(" + src + "[__i])";
    }

    if (fname == "__lib_where" && argc >= 5) {
      std::string dst = arg(0), cond = arg(1), x = arg(2), y = arg(3),
                  num = arg(4);
      return "for (int __i = 0; __i < " + num + "; ++__i) " + dst +
             "[__i] = " + cond + "[__i] ? " + x + "[__i] : " + y + "[__i]";
    }
  }

  std::string result = n.function->name + "(";
  if (n.arguments) {
    for (size_t i = 0; i < n.arguments->Count(); i++) {
      if (i > 0) result += ", ";
      result += ExprSTR(n.arguments->ValueAt(i));
    }
  }
  result += ")";
  return result;
}

const ValueItem CCCodeGen::GenOffset(const AST::ptr<AST::ChunkAt>& ca) const {
  if (ca->NoOperation()) return sbe::nu(0);

  sbe::ExprSum offset;

  for (size_t i = 0; i < ca->OpCount(); ++i) {
    const auto& sop = ca->OpAt(i);
    if (isa<AST::SOP::Reshape>(sop)) {
      continue;
    } else if (isa<AST::SOP::Tiling>(sop) || isa<AST::SOP::TileAt>(sop) ||
               isa<AST::SOP::SubSpan>(sop)) {
      auto idx = sop->GetIndices()->Opts();
      auto strd = sop->GetBlockStrides();
      auto blk = sop->GetBlockShape();
      assert(idx.HasVals());
      assert(idx.GetVals().size() == strd.size());
      assert(blk.Rank() == strd.size());
      bool has_step = isa<AST::SOP::SubSpan>(sop) && sop->GetSteps();
      for (size_t ith = 0; ith < idx.GetVals().size(); ++ith) {
        if (has_step)
          offset += idx.GetVals()[ith] * strd[ith];
        else
          offset += idx.GetVals()[ith] * strd[ith] * blk.ValueAt(ith);
      }
    } else if (isa<AST::SOP::View>(sop)) {
      auto off = sop->GetOffsets()->Opts();
      auto strd = sop->GetBlockStrides();
      for (size_t ith = 0; ith < off.GetVals().size(); ++ith)
        offset += off.GetVals()[ith] * strd[ith];
    }
  }

  return offset.Get();
}

// ---- Preamble Emission ----

void CCCodeGen::EmitPreamble() {
  os << "#define __CHOREO_TARGET_CC__\n";
  os << "#include \"choreo.h\"\n";
  os << "#include \"choreo_cc.h\"\n";
  os << "#include <algorithm>\n";
  os << "#include <cmath>\n";
  os << "#include <cstring>\n";
  os << "#include <cstdlib>\n";
  os << "#include <future>\n";
  os << "#include <utility>\n\n";
}

// ---- BeforeVisitImpl / InMidVisitImpl / AfterVisitImpl ----

bool CCCodeGen::BeforeVisitImpl(AST::Node& n) {
  EmitPreSiteAssertions(n);
  EmitLineDirective(n);

  if (isa<AST::Program>(&n)) {
    EmitPreamble();
    code_segments.push_back(os.str());
    os.str("");
    ssm.EnterScope();
    levels.push(ParallelLevel::NONE);
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ResetFunctionStates();
    BuildSiteAssertionMap();
    fty = cast<FunctionType>(GetSymbolType(fname));
    ssm.EnterScope();
    levels.push(ParallelLevel::SEQ);
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.push(pb->GetLevel());
  } else if (isa<AST::WithBlock>(&n)) {
    IndStream() << "{\n";
    IncrIndent();
  } else if (isa<AST::ForeachBlock>(&n)) {
  }
  return true;
}

bool CCCodeGen::InMidVisitImpl(AST::Node& n) {
  if (auto ie = dyn_cast<AST::IfElseBlock>(&n)) {
    if (!ie->HasElse()) return true;
    DecrIndent();
    IndStream() << "} else {\n";
    IncrIndent();
  }
  return true;
}

bool CCCodeGen::AfterVisitImpl(AST::Node& n) {
  EmitPostSiteAssertions(n);

  if (isa<AST::Program>(&n)) {
    ssm.LeaveScope();

    switch (CCtx().GetOutputKind()) {
    case OutputKind::TargetSourceCode: EmitSource(); break;
    case OutputKind::TargetModule:
    case OutputKind::TargetExecutable: {
      if (!CompileWithScript("--compile-link")) {
        error_count++;
        return false;
      }
      break;
    }
    case OutputKind::ShellScript: {
      EmitScript(outs());
      break;
    }
    default:
      choreo_unreachable("outputkind: " + STR(CCtx().GetOutputKind()) +
                         " is not supported by the CC target.");
    }
  } else if (isa<AST::ChoreoFunction>(&n)) {
    ssm.LeaveScope();
    if (code_segments.empty()) code_segments.push_back("");
    code_segments.back() += os.str();
    os.str("");
  } else if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
    levels.pop();
    auto pvs = pb->AllSubPVs();
    for (int i = pvs.size() - 1; i >= 0; --i) {
      DecrIndent();
      IndStream() << "}\n";
    }
    if (pb->HasDeviceTarget() && pb->GetLevel() == ParallelLevel::DEVICE) {
      DecrIndent();
      IndStream() << "});\n";
    }
  } else if (isa<AST::WithBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  } else if (auto fb = dyn_cast<AST::ForeachBlock>(&n)) {
    const auto& ranges = fb->GetRangeNodes();
    for (int j = ranges->Count() - 1; j >= 0; --j) {
      auto rng = cast<AST::LoopRange>(ranges->ValueAt(j));
      auto cname = rng->IVName();
      auto ivs = within_map.at(InScopeName(cname));
      for (auto iv_itr = ivs.rbegin(); iv_itr != ivs.rend(); ++iv_itr) {
        DecrIndent();
        IndStream() << "}\n";
        os << indent << ssm.HostName(*iv_itr) << " = 0;\n";
      }
    }
  } else if (auto it = dyn_cast<AST::InThreadsBlock>(&n)) {
    if (!it->stmts->None()) {
      DecrIndent();
      IndStream() << "}\n";
    }
  } else if (isa<AST::IfElseBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  } else if (isa<AST::WhileBlock>(&n)) {
    DecrIndent();
    IndStream() << "}\n";
  }
  return true;
}

// ---- Visit methods ----

bool CCCodeGen::Visit(AST::ParamList& n) {
  for (auto param : n.values) {
    auto ty = GetSymbolType(param->sym->name);
    SSTab().DefineSymbol(param->sym->name, ty);
  }
  return true;
}

void CCCodeGen::EmitFuncDecl() {
  if (!void_return) {
    if (cgi.HasReturnSymbol(fname)) {
      auto& item = cgi.GetReturnDetail(fname);
      if (item.rty_str != "$")
        os << item.rty_str;
      else
        os << HostTypeStringify(*fty->out_ty, true);
    } else {
      os << HostTypeStringify(*fty->out_ty, true);
    }
  } else
    os << "void";
  os << " " << fname << "(";

  size_t pindex = 0;
  for (auto& item : cgi.GetParameters(fname)) {
    if (item.IsParameter()) assert((int)pindex == item.p_index);
    os << ((pindex == 0) ? "" : ", ")
       << HostTypeStringify(*item.type, false, item.IsReference()) << " "
       << item.host_name;
    ++pindex;
  }
  os << ")";
}

void CCCodeGen::EmitEntryAssertions() {
  if (CCtx().DisableRuntimeCheck()) return;

  for (const auto& rc : FCtx(fname).GetRtChecks()) {
    IndStream() << "choreo::runtime_check(" << ValueSTR(sbe::sym(rc.lhs)) << " "
                << rc.op << " " << ValueSTR(sbe::sym(rc.rhs)) << ", \""
                << rc.message << ", " << rc.loc << "\");\n";
  }

  for (const auto& ar : FCtx(fname).GetAssertions(AssessType::ENTRY)) {
    if (!ar.enabled) continue;
    IndStream() << "choreo::runtime_check(" << ValueSTR(ar.expr, true) << ", \""
                << ar.message << ", " << ar.loc << "\");\n";
  }
}

void CCCodeGen::BuildSiteAssertionMap() {
  if (CCtx().DisableRuntimeCheck()) return;
  if (fname.empty()) return;

  for (const auto& ar : FCtx(fname).GetAssertions(AssessType::USE_SITE)) {
    if (!ar.enabled || !ar.EmitTarget()) continue;
    if (ar.emit_position == AssertionEmitPosition::BEFORE_NODE)
      pre_site_assertions[ar.EmitTarget()].push_back(ar);
    else
      post_site_assertions[ar.EmitTarget()].push_back(ar);
  }
  for (const auto& ar : FCtx(fname).GetAssertions(AssessType::HOIST_SITE)) {
    if (!ar.enabled || !ar.EmitTarget()) continue;
    if (ar.emit_position == AssertionEmitPosition::BEFORE_NODE)
      pre_site_assertions[ar.EmitTarget()].push_back(ar);
    else
      post_site_assertions[ar.EmitTarget()].push_back(ar);
  }
}

void CCCodeGen::EmitPreSiteAssertions(AST::Node& n) {
  auto it = pre_site_assertions.find(&n);
  if (it == pre_site_assertions.end()) return;

  for (const auto& ar : it->second) {
    IndStream() << "choreo::choreo_assert(" << ValueSTR(ar.expr, true) << ", \""
                << ar.message << ", " << ar.loc << "\");\n";
  }
}

void CCCodeGen::EmitPostSiteAssertions(AST::Node& n) {
  auto it = post_site_assertions.find(&n);
  if (it == post_site_assertions.end()) return;

  for (const auto& ar : it->second) {
    IndStream() << "choreo::choreo_assert(" << ValueSTR(ar.expr, true) << ", \""
                << ar.message << ", " << ar.loc << "\");\n";
  }
}

void CCCodeGen::EmitLineDirective(AST::Node& n) {
  if (!CCtx().GenDebugInfo()) return;
  if (!(isa<AST::WithBlock>(&n) || isa<AST::ForeachBlock>(&n) ||
        isa<AST::InThreadsBlock>(&n) || isa<AST::IfElseBlock>(&n) ||
        isa<AST::WhileBlock>(&n) || isa<AST::Assignment>(&n) ||
        isa<AST::ParallelBy>(&n) || isa<AST::DMA>(&n) || isa<AST::MMA>(&n) ||
        isa<AST::Wait>(&n) || isa<AST::Trigger>(&n) || isa<AST::Break>(&n) ||
        isa<AST::Continue>(&n) || isa<AST::Call>(&n) ||
        isa<AST::NamedVariableDecl>(&n) || isa<AST::Return>(&n)))
    return;
  auto loc = n.LOC();
  if (loc.begin.line <= 0) return;
  auto file = loc.begin.filename;
  os << "#line " << loc.begin.line << " \"" << file << "\"\n";
}

bool CCCodeGen::Visit(AST::NamedTypeDecl& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  auto sym = n.name_str;
  SSTab().DefineSymbol(sym, nty);

  if (auto mty = dyn_cast<MDSpanType>(nty)) {
    if (mty->Dims() <= 1 && mty->HasSufficientInfo()) {
      IndStream() << "int " << sym << " = "
                  << ValueSTR(mty->GetShape().ValueAt(0)) << ";\n";
    }
  }
  return true;
}

bool CCCodeGen::Visit(AST::FunctionDecl& n) {
  TraceEachVisit(n);

  assert(n.name == fname && "inconsistent in function names.");

  size_t pindex = 0;
  for (auto& item : cgi.GetParameters(fname)) {
    if (item.IsParameter()) {
      assert((int)pindex == item.p_index);
      item.host_name = UnScopedName(item.name);
      if (auto sty = dyn_cast<SpannedType>(item.type))
        ssm.MapHostSymbol(item.name, item.host_name + ".data()");
      else
        ssm.MapHostSymbol(item.name, item.host_name);
    } else
      item.host_name = UnScopedName(item.name);
    pindex++;
  }

  if (isa<VoidType>(fty->out_ty)) void_return = true;

  EmitFuncDecl();
  os << " {\n";
  IncrIndent();

  EmitEntryAssertions();

  return true;
}

bool CCCodeGen::Visit(AST::ChoreoFunction& n) {
  TraceEachVisit(n);
  for (auto& name : global_allocs)
    IndStream() << "std::free(" << name << ");\n";
  DecrIndent();
  os << "}\n\n";
  return true;
}

bool CCCodeGen::Visit(AST::WithIn& n) {
  TraceEachVisit(n);

  if (n.with)
    ssm.MapHostSymbol(InScopeName(n.with->name), "__iv_" + n.with->name);

  assert(n.with_matchers && "expected matchers exist.");

  for (auto& v : n.GetMatchers()) {
    auto id = cast<AST::Identifier>(v);
    ssm.RemapHostSymbol(InScopeName(id->name), "__iv_" + id->name);
    os << indent << "int __iv_" << id->name << " = 0;\n";
  }

  if (n.with && (n.GetMatchers().size() == 1)) {
    auto m1 = cast<AST::Identifier>(n.GetMatchers()[0]);
    ssm.RemapHostSymbol(InScopeName(n.with->name), "__iv_" + m1->name);
  }

  return true;
}

bool CCCodeGen::Visit(AST::WhereBind& n) {
  TraceEachVisit(n);
  return true;
}

bool CCCodeGen::Visit(AST::WithBlock& n) {
  TraceEachVisit(n);
  return true;
}

bool CCCodeGen::Visit(AST::ForeachBlock& n) {
  TraceEachVisit(n);

  bool vectorizable = n.loop && n.loop->CanVectorize();

  for (auto& rn : n.GetRanges()) {
    auto rng = cast<AST::LoopRange>(rn);
    auto cname = rng->IVName();
    for (auto iv_name : within_map.at(InScopeName(cname))) {
      auto iv_ty = GetSymbolType(UnScopedName(iv_name));
      auto iv_bty = dyn_cast<BoundedType>(iv_ty);
      assert(iv_bty && "foreach IV should have bounded type.");
      auto mapped = ssm.HostName(iv_name);
      if (vectorizable) IndStream() << "#pragma omp simd\n";
      IndStream() << "for (" << mapped << " = "
                  << (rng->lbound ? ("(" + ExprSTR(rng->lbound) + ")")
                                  : std::string("0"))
                  << "; " << mapped << " < "
                  << UnScopedExpr(ValueSTR(iv_bty->GetUpperBound()))
                  << (rng->ubound ? (" + " + ExprSTR(rng->ubound))
                                  : std::string(""))
                  << "; ++" << mapped << ") {\n";
      IncrIndent();
    }
  }

  return true;
}

bool CCCodeGen::Visit(AST::InThreadsBlock& n) {
  TraceEachVisit(n);
  if (!n.stmts->None()) {
    IndStream() << "if (" << ExprSTR(n.pred) << ") {\n";
    IncrIndent();
  }
  return true;
}

bool CCCodeGen::Visit(AST::IfElseBlock& n) {
  TraceEachVisit(n);
  IndStream() << "if (" << ExprSTR(n.pred) << ") {\n";
  IncrIndent();
  return true;
}

bool CCCodeGen::Visit(AST::WhileBlock& n) {
  TraceEachVisit(n);
  IndStream() << "while (" << ExprSTR(n.pred) << ") {\n";
  IncrIndent();
  return true;
}

bool CCCodeGen::Visit(AST::ParallelBy& n) {
  TraceEachVisit(n);

  if (n.HasDeviceTarget() && n.GetLevel() == ParallelLevel::DEVICE) {
    auto future_name =
        "__dev_future_" + std::to_string(device_future_counter++);
    pending_device_futures.push_back(future_name);
    IndStream() << "auto " << future_name
                << " = std::async(std::launch::async, [&]() {\n";
    IncrIndent();
    auto pvs = n.AllSubPVs();
    auto bvs = n.BoundValues();
    auto bpv_name = n.BPV()->name;
    for (size_t i = 0; i < pvs.size(); ++i) {
      auto pv_id = cast<AST::Identifier>(pvs[i]);
      auto pv_name = pv_id->name;
      auto bound = (i < bvs.size()) ? ValueSTR(bvs[i]) : std::string("1");
      ssm.MapHostSymbol(InScopeName(pv_name), pv_name);
      IndStream() << "for (int " << pv_name << " = 0; " << pv_name << " < "
                  << bound << "; ++" << pv_name << ") {\n";
      IncrIndent();
    }
    if (pvs.size() == 1) {
      auto pv_name = cast<AST::Identifier>(pvs[0])->name;
      ssm.MapHostSymbol(InScopeName(bpv_name), pv_name);
    }
    return true;
  }

  auto pvs = n.AllSubPVs();
  auto bvs = n.BoundValues();

  auto bpv_name = n.BPV()->name;

  for (size_t i = 0; i < pvs.size(); ++i) {
    auto pv_id = cast<AST::Identifier>(pvs[i]);
    auto pv_name = pv_id->name;
    auto bound = (i < bvs.size()) ? ValueSTR(bvs[i]) : std::string("1");
    ssm.MapHostSymbol(InScopeName(pv_name), pv_name);
    IndStream() << "for (int " << pv_name << " = 0; " << pv_name << " < "
                << bound << "; ++" << pv_name << ") {\n";
    IncrIndent();
  }

  if (pvs.size() == 1) {
    auto pv_name = cast<AST::Identifier>(pvs[0])->name;
    ssm.MapHostSymbol(InScopeName(bpv_name), pv_name);
  }

  return true;
}

bool CCCodeGen::Visit(AST::DMA& n) {
  TraceEachVisit(n);

  if (n.IsDummy() || isa<PlaceHolderType>(NodeType(n))) {
    if (!n.future.empty()) {
      auto& fbi = FCtx(fname).GetFutureBufferInfo();
      auto it = fbi.find(InScopeName(n.future));
      if (it != fbi.end() && !it->second.buffer.empty()) {
        auto host_buf = ssm.HostName(it->second.buffer);
        ssm.MapHostSymbol(InScopeNameForRef(n.future), host_buf);
        ssm.MapHostSymbol(InScopeName(n.future + ".data"), host_buf);
      }
    }
    return true;
  }

  assert(isa<AST::ChunkAt>(n.from) && "DMA source must be ChunkAt.");
  assert(isa<AST::ChunkAt>(n.to) && "DMA destination must be ChunkAt.");

  auto from_ca = cast<AST::ChunkAt>(n.from);
  auto to_ca = cast<AST::ChunkAt>(n.to);
  auto from_sty = cast<SpannedType>(from_ca->GetType());

  auto size_expr = UnScopedExpr(SizeExprOf(*from_sty, true));

  std::string from_expr = ExprSTR(n.from);
  std::string to_expr = ExprSTR(n.to);

  if (!n.future.empty()) {
    auto buf_name = to_ca->RefSymbol();
    auto host_buf = ssm.HostName(InScopeNameForRef(buf_name));
    ssm.MapHostSymbol(InScopeNameForRef(n.future), host_buf);
    ssm.MapHostSymbol(InScopeName(n.future + ".data"), host_buf);
  }

  bool is_async = n.IsAsync();

  if (is_async && !n.future.empty()) {
    IndStream() << "auto " << n.future
                << " = std::async(std::launch::async, [&]() {\n";
    IncrIndent();
    IndStream() << "std::memcpy((void*)(" << to_expr << "), (const void*)("
                << from_expr << "), " << size_expr << ");\n";
    DecrIndent();
    IndStream() << "});\n";
  } else {
    IndStream() << "std::memcpy((void*)(" << to_expr << "), (const void*)("
                << from_expr << "), " << size_expr << ");\n";
  }

  return true;
}

bool CCCodeGen::Visit(AST::MMA& n) {
  TraceEachVisit(n);
  auto& op = *n.GetOperation();

  if (op.Tag() == AST::MMAOperation::Commit ||
      op.Tag() == AST::MMAOperation::Wait)
    return true;

  const ptr<AST::Expr>& frag = op.GetFrag();
  std::string sym = AST::FragName(frag);
  std::string scoped = InScopeName(sym);

  switch (op.Tag()) {
  case AST::MMAOperation::Fill: {
    auto fill_val = ExprSTR(op.FillingValue());
    auto fill_ty = op.FillingType();
    if (fill_ty == BaseType::UNKSCALAR) {
      auto nty = NodeType(*op.FillingValue());
      fill_ty = GetBaseType(*nty);
    }
    auto ty_str = CCBaseTypeName(fill_ty);
    if (op.FillingIsDecl()) {
      if (FCtx(fname).FragHasMMAType(scoped)) {
        auto& ssmi = cgi.GetSymbolMMA(scoped);
        auto shape = ssmi.GetShape();
        int total = 1;
        for (auto& v : shape) {
          auto iv = VIInt(v);
          total *= iv ? *iv : 1;
        }
        os << indent << ty_str << " " << sym << "[" << total << "];\n";
        IndStream() << "choreo::cc::mma_fill<" << ty_str << ">(" << sym << ", "
                    << "static_cast<" << ty_str << ">(" << fill_val << "), "
                    << total << ");\n";
      } else {
        os << indent << ty_str << " " << sym << " = " << fill_val << ";\n";
      }
      ssm.MapHostSymbol(scoped, sym);
    } else {
      if (FCtx(fname).FragHasMMAType(scoped)) {
        auto& ssmi = cgi.GetSymbolMMA(scoped);
        auto shape = ssmi.GetShape();
        int total = 1;
        for (auto& v : shape) {
          auto iv = VIInt(v);
          total *= iv ? *iv : 1;
        }
        IndStream() << "choreo::cc::mma_fill<" << ty_str << ">(" << sym << ", "
                    << "static_cast<" << ty_str << ">(" << fill_val << "), "
                    << total << ");\n";
      }
    }
    break;
  }
  case AST::MMAOperation::Load: {
    auto ca = op.LoadFrom();
    auto from_sty = cast<SpannedType>(ca->GetType());
    auto size_expr = UnScopedExpr(SizeExprOf(*from_sty, true));
    auto from_expr = ExprSTR(ca);
    IndStream() << "std::memcpy(" << sym << ", " << from_expr << ", "
                << size_expr << ");\n";
    ssm.MapHostSymbol(scoped, sym);
    break;
  }
  case AST::MMAOperation::Exec: {
    auto& ssmi = cgi.GetSymbolMMA(scoped);
    auto shape = ssmi.GetShape();
    std::string acc_sym = AST::FragName(op.ExecOperand(0));
    std::string a_sym = AST::FragName(op.ExecOperand(1));
    std::string b_sym = AST::FragName(op.ExecOperand(2));
    auto acc_ty_str = CCBaseTypeName(ssmi.ty);
    auto a_scoped = InScopeName(a_sym);
    auto b_scoped = InScopeName(b_sym);
    auto a_sty = GetSpannedType(GetSymbolType(a_sym));
    auto b_sty = GetSpannedType(GetSymbolType(b_sym));
    auto a_ty_str = CCBaseTypeName(a_sty->ElementType());
    auto b_ty_str = CCBaseTypeName(b_sty->ElementType());
    auto a_shape = a_sty->GetShape();
    auto b_shape = b_sty->GetShape();
    std::string M_str, N_str, K_str;
    switch (op.GetMethod()) {
    case AST::MMAOperation::ROW_COL:
      M_str = ValueSTR(a_shape.ValueAt(0));
      K_str = ValueSTR(a_shape.ValueAt(1));
      N_str = ValueSTR(b_shape.ValueAt(1));
      IndStream() << "choreo::cc::mma_exec_row_col<" << acc_ty_str << ", "
                  << a_ty_str << ", " << b_ty_str << ", " << acc_ty_str << ">("
                  << acc_sym << ", " << a_sym << ", " << b_sym << ", "
                  << acc_sym << ", " << M_str << ", " << N_str << ", " << K_str
                  << ");\n";
      break;
    case AST::MMAOperation::ROW_ROW:
      M_str = ValueSTR(a_shape.ValueAt(0));
      K_str = ValueSTR(a_shape.ValueAt(1));
      N_str = ValueSTR(b_shape.ValueAt(0));
      IndStream() << "choreo::cc::mma_exec_row_row<" << acc_ty_str << ", "
                  << a_ty_str << ", " << b_ty_str << ", " << acc_ty_str << ">("
                  << acc_sym << ", " << a_sym << ", " << b_sym << ", "
                  << acc_sym << ", " << M_str << ", " << N_str << ", " << K_str
                  << ");\n";
      break;
    case AST::MMAOperation::COL_ROW:
      M_str = ValueSTR(a_shape.ValueAt(1));
      K_str = ValueSTR(a_shape.ValueAt(0));
      N_str = ValueSTR(b_shape.ValueAt(0));
      IndStream() << "choreo::cc::mma_exec_col_row<" << acc_ty_str << ", "
                  << a_ty_str << ", " << b_ty_str << ", " << acc_ty_str << ">("
                  << acc_sym << ", " << a_sym << ", " << b_sym << ", "
                  << acc_sym << ", " << M_str << ", " << N_str << ", " << K_str
                  << ");\n";
      break;
    case AST::MMAOperation::COL_COL:
      M_str = ValueSTR(a_shape.ValueAt(1));
      K_str = ValueSTR(a_shape.ValueAt(0));
      N_str = ValueSTR(b_shape.ValueAt(1));
      IndStream() << "choreo::cc::mma_exec_col_col<" << acc_ty_str << ", "
                  << a_ty_str << ", " << b_ty_str << ", " << acc_ty_str << ">("
                  << acc_sym << ", " << a_sym << ", " << b_sym << ", "
                  << acc_sym << ", " << M_str << ", " << N_str << ", " << K_str
                  << ");\n";
      break;
    }
    break;
  }
  case AST::MMAOperation::Store: {
    auto ca = op.StoreTo();
    auto to_sty = cast<SpannedType>(ca->GetType());
    auto to_expr = ExprSTR(ca);
    if (op.StoreIsTranspose()) {
      auto shape = to_sty->GetShape();
      auto M_str = ValueSTR(shape.ValueAt(0));
      auto N_str = ValueSTR(shape.ValueAt(1));
      auto ty_str = CCBaseTypeName(to_sty->ElementType());
      IndStream() << "choreo::cc::mma_store_transpose<" << ty_str << ">("
                  << to_expr << ", " << sym << ", " << M_str << ", " << N_str
                  << ");\n";
    } else {
      auto size_expr = UnScopedExpr(SizeExprOf(*to_sty, true));
      IndStream() << "std::memcpy(" << to_expr << ", " << sym << ", "
                  << size_expr << ");\n";
    }
    break;
  }
  case AST::MMAOperation::Scale: {
    auto acc_sym = ssm.HostName(InScopeNameForRef(PSTR(op.ScaleAccumulator())));
    auto scale_a_expr = ExprSTR(op.ScaleA());
    if (FCtx(fname).FragHasMMAType(scoped)) {
      auto& ssmi = cgi.GetSymbolMMA(scoped);
      auto shape = ssmi.GetShape();
      int total = 1;
      for (auto& v : shape) {
        auto iv = VIInt(v);
        total *= iv ? *iv : 1;
      }
      auto ty_str = CCBaseTypeName(ssmi.ty);
      IndStream() << "choreo::cc::mma_scale<" << ty_str << ">(" << acc_sym
                  << ", " << scale_a_expr << ", " << total << ");\n";
    }
    break;
  }
  default: break;
  }
  return true;
}

bool CCCodeGen::Visit(AST::Wait& n) {
  TraceEachVisit(n);

  for (auto& t : n.GetTargets()) {
    if (auto id = AST::GetIdentifier(t)) {
      auto wty = NodeType(*t);
      if (isa<FutureType>(wty))
        IndStream() << id->name << ".get();\n";
      else if (isa<EventType>(wty))
        IndStream() << id->name << ".wait();\n";
    }
  }
  return true;
}

bool CCCodeGen::Visit(AST::Trigger& n) {
  TraceEachVisit(n);
  for (auto& t : n.targets->AllValues()) {
    if (auto id = AST::GetIdentifier(t))
      IndStream() << id->name << ".trigger();\n";
  }
  return true;
}

bool CCCodeGen::Visit(AST::Break& n) {
  TraceEachVisit(n);
  IndStream() << "break;\n";
  return true;
}

bool CCCodeGen::Visit(AST::Continue& n) {
  TraceEachVisit(n);
  IndStream() << "continue;\n";
  return true;
}

bool CCCodeGen::Visit(AST::Yield& n) {
  TraceEachVisit(n);
  IndStream() << "return;\n";
  return true;
}

bool CCCodeGen::Visit(AST::Rotate& n) {
  TraceEachVisit(n);
  auto& ids = n.GetIds();
  if (ids.size() == 2) {
    auto a = cast<AST::Identifier>(ids[0])->name;
    auto b = cast<AST::Identifier>(ids[1])->name;
    IndStream() << "std::swap(" << a << ", " << b << ");\n";
  } else {
    auto first = cast<AST::Identifier>(ids[0])->name;
    IndStream() << "{ auto __tmp = std::move(" << first << ");\n";
    IncrIndent();
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
      auto cur = cast<AST::Identifier>(ids[i])->name;
      auto nxt = cast<AST::Identifier>(ids[i + 1])->name;
      IndStream() << cur << " = std::move(" << nxt << ");\n";
    }
    auto last = cast<AST::Identifier>(ids.back())->name;
    IndStream() << last << " = std::move(__tmp);\n";
    DecrIndent();
    IndStream() << "}\n";
  }
  return true;
}

bool CCCodeGen::Visit(AST::Synchronize& n) {
  TraceEachVisit(n);
  if (n.Resource() == Storage::GLOBAL && !pending_device_futures.empty()) {
    for (auto& f : pending_device_futures) IndStream() << f << ".get();\n";
    pending_device_futures.clear();
  }
  // sync.shared is a no-op: CC target executes sequentially, so shared
  // memory is always coherent within the same address space.
  return true;
}

bool CCCodeGen::Visit(AST::Assignment& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);

  if (n.AssignToDataElement()) {
    os << indent << ExprSTR(n.da) << " = " << ExprSTR(n.value) << ";\n";
    return true;
  }

  if (auto sa = dyn_cast<AST::SpanAs>(n.value)) {
    auto name = n.GetName();
    auto sname = InScopeName(name);
    auto base_name = ssm.HostName(InScopeNameForRef(sa->id->name));
    auto nty_s = dyn_cast<SpannedType>(nty);
    std::string cast_expr = "static_cast<" +
                            std::string(CCBaseTypeName(nty_s->ElementType())) +
                            "*>(" + base_name + ")";
    ssm.MapHostSymbol(sname, cast_expr);
    os << indent << "auto* " << name << " = " << cast_expr << ";\n";
    return true;
  }

  if (auto s = dyn_cast<AST::Select>(n.value)) {
    auto name = n.GetName();
    size_t val_count = s->expr_list->Count();
    std::string arr = name + "_select_array__";
    if (isa<FutureType>(nty)) {
      os << indent << "std::future<void>* " << arr << "[] = {";
      for (size_t i = 0; i < val_count; i++) {
        if (i > 0) os << ", ";
        os << "&" << ExprSTR(s->expr_list->ValueAt(i));
      }
      os << "};\n";
      os << indent << "std::future<void>& " << name << " = *" << arr << "["
         << ExprSTR(s->select_factor) << "];\n";
    } else if (auto sty = dyn_cast<SpannedType>(nty)) {
      auto bts = CCBaseTypeName(sty->ElementType());
      os << indent << bts << "* " << arr << "[] = {";
      for (size_t i = 0; i < val_count; i++) {
        if (i > 0) os << ", ";
        os << ExprSTR(s->expr_list->ValueAt(i));
      }
      os << "};\n";
      os << indent << bts << "* " << name << " = " << arr << "["
         << ExprSTR(s->select_factor) << "];\n";
      ssm.MapHostSymbol(InScopeName(name), name);
    }
    return true;
  }

  if (isa<ScalarType>(nty) || isa<BoundedType>(nty)) {
    auto name = n.GetName();
    os << indent << ((!n.IsDecl()) ? "" : "auto ") << name << " = "
       << ExprSTR(n.value) << ";\n";
    return true;
  }

  if (isa<SpannedType>(nty)) {
    auto name = n.GetName();
    if (!ssm.HasHostName(InScopeName(name)))
      ssm.MapHostSymbol(InScopeName(name), name);
    os << indent;
    if (!n.IsDecl())
      os << name << " = ";
    else
      os << "auto " << name << " = ";
    os << ExprSTR(n.value) << ";\n";
    return true;
  }

  if (isa<FutureType>(nty)) { return true; }

  errs() << "CC codegen: unprocessed assignment type " << PSTR(nty) << "\n";
  return false;
}

bool CCCodeGen::Visit(AST::Call& n) {
  TraceEachVisit(n);

  if (n.IsBIF()) {
    const auto func_name = n.function->name;
    if (func_name == "assert") {
      if (n.arguments && n.arguments->Count() > 0)
        os << indent << "assert(" << ExprSTR(n.arguments->ValueAt(0)) << ");\n";
      return true;
    }
    if (func_name == "print" || func_name == "println") {
      os << indent << "std::cout";
      if (n.arguments) {
        for (auto& arg : n.arguments->AllValues()) os << " << " << ExprSTR(arg);
      }
      if (func_name == "println") os << " << \"\\n\"";
      os << ";\n";
      return true;
    }
    if (func_name == "launch_bounds" || func_name == "setreg") return true;
    if (n.IsArith() || n.IsAtomic()) {
      if (!n.IsExpr()) os << indent << CallSTR(n) << ";\n";
      return true;
    }
  }

  if (!n.IsExpr()) { os << indent << CallSTR(n) << ";\n"; }

  return true;
}

bool CCCodeGen::Visit(AST::NamedVariableDecl& n) {
  TraceEachVisit(n);

  auto nty = NodeType(n);
  auto sym = n.name_str;
  SSTab().DefineSymbol(sym, nty);
  auto sname = InScopeName(sym);

  if (isa<FutureType>(nty)) {
    if (auto s = dyn_cast<AST::Select>(n.init_expr)) {
      size_t val_count = s->expr_list->Count();
      std::string arr = sym + "_select_array__";
      os << indent << "std::future<void>* " << arr << "[] = {";
      for (size_t i = 0; i < val_count; i++) {
        if (i > 0) os << ", ";
        os << "&" << ExprSTR(s->expr_list->ValueAt(i));
      }
      os << "};\n";
      os << indent << "std::future<void>& " << sym << " = *" << arr << "["
         << ExprSTR(s->select_factor) << "];\n";
    } else {
      os << indent << "std::future<void> " << sym << ";\n";
    }
    ssm.MapHostSymbol(sname, sym);
    return true;
  }

  if (isa<EventType>(nty)) {
    os << indent << "choreo::cc::event_t " << sym << ";\n";
    ssm.MapHostSymbol(sname, sym);
    return true;
  }

  if (auto eat = dyn_cast<EventArrayType>(nty)) {
    auto dims = eat->Dimensions();
    os << indent << "choreo::cc::event_t " << sym;
    for (auto& d : dims) os << "[" << ValueSTR(d) << "]";
    os << ";\n";
    ssm.MapHostSymbol(sname, sym);
    return true;
  }

  if (auto sty = dyn_cast<SpannedType>(nty)) {
    ssm.MapHostSymbol(sname, sym);
    auto elem_ty = CCBaseTypeName(sty->ElementType());
    auto size = UnScopedExpr(SizeExprOf(*sty, true));
    if (sty->GetStorage() == Storage::LOCAL ||
        sty->GetStorage() == Storage::SHARED) {
      os << indent << "alignas(64) " << elem_ty << " " << sym << "["
         << UnScopedExpr(ValueSTR(sty->GetShape().ElementCountValue()))
         << "];\n";
    } else {
      os << indent << elem_ty << "* " << sym << " = static_cast<" << elem_ty
         << "*>(std::aligned_alloc(64, " << size << "));\n";
      global_allocs.push_back(sym);
    }
    return true;
  }

  if (isa<ScalarType>(nty)) {
    if (n.init_expr) {
      os << indent << TypeSTR(*nty) << " " << sym << " = "
         << ExprSTR(n.init_expr) << ";\n";
    } else {
      os << indent << TypeSTR(*nty) << " " << sym << " = {};\n";
    }
    ssm.MapHostSymbol(sname, sym);
    return true;
  }

  if (isa<BoundedType>(nty)) {
    if (n.init_expr)
      os << indent << TypeSTR(*nty) << " " << sym << " = "
         << ExprSTR(n.init_expr) << ";\n";
    else
      os << indent << "int " << sym << " = 0;\n";
    ssm.MapHostSymbol(sname, sym);
    return true;
  }

  if (auto bitt = dyn_cast<BoundedITupleType>(nty)) {
    (void)bitt;
    os << indent << "int " << sym << " = 0;\n";
    ssm.MapHostSymbol(sname, sym);
    return true;
  }

  errs() << "CC codegen: unprocessed variable decl type " << PSTR(nty)
         << " for " << sym << "\n";
  return true;
}

bool CCCodeGen::Visit(AST::CppSourceCode& n) {
  TraceEachVisit(n);

  if (n.kind == AST::CppSourceCode::Inline) {
    os << n.GetCode();
  } else {
    if (code_segments.empty() || n.kind != AST::CppSourceCode::Host)
      code_segments.push_back("");
    code_segments.back() += n.GetCode();
  }

  return true;
}

bool CCCodeGen::Visit(AST::Return& n) {
  TraceEachVisit(n);

  auto vty = NodeType(*n.value);

  if (isa<ScalarType>(vty)) {
    os << indent << "return " << ExprSTR(n.value) << ";\n";
  } else if (auto sty = dyn_cast<SpannedType>(vty)) {
    if (auto id = AST::GetIdentifier(*n.value)) {
      auto sym_ty = GetSymbolType(id->name);
      auto sym_sty = dyn_cast<SpannedType>(sym_ty);
      if (sym_sty && sym_sty->GetStorage() == Storage::GLOBAL) {
        auto shape = sym_sty->GetShape();
        os << indent << "return choreo::copy_as_spanned<" << shape.Rank()
           << ">(" << id->name << ", {";
        for (size_t d = 0; d < shape.Rank(); ++d) {
          if (d > 0) os << ", ";
          os << "(size_t)" << ValueSTR(shape.ValueAt(d));
        }
        os << "});\n";
      } else {
        os << indent << "return choreo::copy_as_spanned(" << id->name << ", "
           << id->name << ".shape());\n";
      }
    } else {
      os << indent << "return " << ExprSTR(n.value) << ";\n";
    }
  } else {
    os << indent << "return " << ExprSTR(n.value) << ";\n";
  }

  return true;
}

// ---- Output Emission ----

void CCCodeGen::EmitSource() {
  for (auto& code : code_segments) outs() << code << "\n";
}

void CCCodeGen::EmitScript(std::ostream& out, const std::string& exe_fn) {
  auto filename = RemoveDirectoryPrefix(
      RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));
  out << "#!/usr/bin/env bash\n\n";
  out << "# Choreo CC target compile script\n\n";

  std::string build_path = CreateUniquePath();
  auto cc_file = build_path + "/__choreo_cc_" + filename + ".cpp";
  auto exe_file = exe_fn;
  if (exe_file.empty())
    exe_file = build_path + "/__choreo_cc_" + filename + ".exe";

  out << "rm -fr " << build_path << "\n";
  out << "mkdir -p " << build_path << "\n\n";

  out << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  out << __choreo_header_as_string << "\nEOF\n\n";
  out << "cat <<'EOF' > " << build_path << "/choreo_types.h\n";
  out << __choreo_types_header_as_string << "\nEOF\n\n";
  out << "cat <<'EOF' > " << build_path << "/choreo_cc.h\n";
  out << __choreo_cc_header_as_string << "\nEOF\n\n";

  out << "cat <<'EOF' > " << cc_file << "\n";
  for (auto& code : code_segments) out << code << "\n";
  out << "\nEOF\n\n";

  out << R"script(
CXX="${CXX:-g++}"
CXX_LIBDIR="$(${CXX} -print-file-name=libstdc++.so | xargs dirname | xargs realpath 2>/dev/null)"
RPATH_FLAG=""
if [ -n "$CXX_LIBDIR" ] && [ -d "$CXX_LIBDIR" ]; then
  RPATH_FLAG="-Wl,-rpath,$CXX_LIBDIR"
fi
CFLAGS="-std=c++17 -O2 -pthread -fopenmp-simd -I)script"
      << build_path << R"script("

show_usage() {
  echo "  Usage: $0 <actions>"
  echo ""
  echo "  Options:"
  echo "   --compile-link,     Compile and link to an executable"
  echo "   --compile-module,   Compile to an object file"
  echo "   --execute,          Compile and execute"
}

do_compile() {
  ${CXX} ${CFLAGS} ${RPATH_FLAG} -o )script"
      << exe_file << " " << cc_file << R"script(
  if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
  fi
}

if [ "$1" = "--compile-link" ] || [ "$1" = "--compile-module" ]; then
  do_compile
elif [ "$1" = "--execute" ]; then
  do_compile && )script"
      << exe_file << R"script(
else
  show_usage
fi
)script";
}

bool CCCodeGen::CompileWithScript(const std::string& action) {
  assert(!action.empty() && "no action is specified.");

  char tempFileName[] = "/tmp/choreo_cc_script_XXXXXX";
  int fd = mkstemp(tempFileName);
  if (fd == -1) {
    errs() << "Failed to create temporary file.\n";
    return false;
  }
  close(fd);

  std::ofstream tempFile(tempFileName);
  if (!tempFile) {
    errs() << "Failed to open temporary file for writing.\n";
    return false;
  }

  auto outfile = OptionRegistry::GetInstance().GetOutputFileName();
  EmitScript(tempFile, outfile);
  tempFile.close();

  std::string command = "bash " + std::string(tempFileName) + " " + action;
  int result = system(command.c_str());
  if (result == -1) {
    errs() << "Failed to execute the compile script.\n";
    return false;
  }

  if (remove(tempFileName) != 0)
    errs() << "Warning: failed to remove temporary script file.\n";

  return result == 0;
}
