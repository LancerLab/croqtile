#include "interpreter.hpp"
#include "opcode.hpp"
#include "symvals.hpp"
#include <cmath>

using namespace Choreo;
using Op = Choreo::Opcode::Kind;

namespace CoWeb {

std::string Interpreter::Run(AST::Program& root) {
  output_.str("");
  output_.clear();
  vars_.clear();
  arrays_.clear();

  ExecBlock(root);
  return output_.str();
}

void Interpreter::ExecBlock(AST::Node& node) {
  if (!node.HasBody()) return;
  auto body = node.GetBody();
  if (!body) return;
  for (size_t i = 0; i < body->Count(); ++i) {
    ExecNode(*body->SubAt(i));
  }
}

void Interpreter::ExecNode(AST::Node& node) {
  if (auto* fn = dyn_cast<AST::ChoreoFunction>(&node)) {
    ExecFunction(*fn);
  } else if (auto* pb = dyn_cast<AST::ParallelBy>(&node)) {
    ExecParallel(*pb);
  } else if (auto* fb = dyn_cast<AST::ForeachBlock>(&node)) {
    ExecForeach(*fb);
  } else if (auto* ife = dyn_cast<AST::IfElseBlock>(&node)) {
    ExecIfElse(*ife);
  } else if (auto* vd = dyn_cast<AST::NamedVariableDecl>(&node)) {
    ExecVarDecl(*vd);
  } else if (auto* asgn = dyn_cast<AST::Assignment>(&node)) {
    ExecAssign(*asgn);
  } else if (auto* call = dyn_cast<AST::Call>(&node)) {
    ExecCall(*call);
  } else if (auto* blk = dyn_cast<AST::Block>(&node)) {
    ExecBlock(*blk);
  }
}

void Interpreter::ExecFunction(AST::ChoreoFunction& fn) {
  ExecBlock(fn);
}

void Interpreter::ExecParallel(AST::ParallelBy& pb) {
  size_t n = pb.SubPVCount();
  if (n == 0) {
    ExecBlock(pb);
    return;
  }

  if (n == 1) {
    std::string idx_name = pb.GetSubPV(0)->name;
    int64_t bound = ToInt(Eval(*pb.BoundExprAt(0)));
    for (int64_t i = 0; i < bound; ++i) {
      vars_[idx_name] = i;
      ExecBlock(pb);
    }
    vars_.erase(idx_name);
  } else if (n == 2) {
    std::string idx0 = pb.GetSubPV(0)->name;
    std::string idx1 = pb.GetSubPV(1)->name;
    int64_t b0 = ToInt(Eval(*pb.BoundExprAt(0)));
    int64_t b1 = ToInt(Eval(*pb.BoundExprAt(1)));
    for (int64_t i = 0; i < b0; ++i) {
      for (int64_t j = 0; j < b1; ++j) {
        vars_[idx0] = i;
        vars_[idx1] = j;
        ExecBlock(pb);
      }
    }
    vars_.erase(idx0);
    vars_.erase(idx1);
  } else if (n == 3) {
    std::string idx0 = pb.GetSubPV(0)->name;
    std::string idx1 = pb.GetSubPV(1)->name;
    std::string idx2 = pb.GetSubPV(2)->name;
    int64_t b0 = ToInt(Eval(*pb.BoundExprAt(0)));
    int64_t b1 = ToInt(Eval(*pb.BoundExprAt(1)));
    int64_t b2 = ToInt(Eval(*pb.BoundExprAt(2)));
    for (int64_t i = 0; i < b0; ++i)
      for (int64_t j = 0; j < b1; ++j)
        for (int64_t k = 0; k < b2; ++k) {
          vars_[idx0] = i;
          vars_[idx1] = j;
          vars_[idx2] = k;
          ExecBlock(pb);
        }
    vars_.erase(idx0);
    vars_.erase(idx1);
    vars_.erase(idx2);
  }
}

void Interpreter::ExecForeach(AST::ForeachBlock& fb) {
  auto& ranges_list = fb.GetRanges();
  if (ranges_list.empty()) {
    ExecBlock(fb);
    return;
  }

  auto* range = dyn_cast<AST::LoopRange>(ranges_list[0].get());
  if (!range) {
    ExecBlock(fb);
    return;
  }

  int64_t start = 0, end_val = 0;
  int step = range->step;
  if (!IsValidStep(step)) step = 1;
  if (range->lbound) start = ToInt(Eval(*range->lbound));
  if (range->ubound) end_val = ToInt(Eval(*range->ubound));
  if (step == 0) step = 1;

  std::string idx_name = range->GetIVName();
  for (int64_t i = start; i < end_val; i += step) {
    vars_[idx_name] = i;
    ExecBlock(fb);
  }
  vars_.erase(idx_name);
}

void Interpreter::ExecIfElse(AST::IfElseBlock& ife) {
  auto pred = ife.GetPredicate();
  if (!pred) return;
  Value cond = Eval(*pred);
  if (ToBool(cond)) {
    ExecBlock(ife);
  } else if (ife.HasElse()) {
    auto body = ife.else_stmts;
    if (body) {
      for (size_t i = 0; i < body->Count(); ++i)
        ExecNode(*body->SubAt(i));
    }
  }
}

void Interpreter::ExecVarDecl(AST::NamedVariableDecl& vd) {
  std::string name = vd.GetName();

  if (vd.IsArray() && vd.mem) {
    int64_t total = 1;
    auto dims = vd.ArrayDimensions();
    std::vector<int64_t> dim_sizes;
    if (dims) {
      for (size_t i = 0; i < dims->Count(); ++i) {
        int64_t d = ToInt(Eval(*dims->ValueAt(i)));
        dim_sizes.push_back(d);
        total *= d;
      }
    }
    if (total <= 0) total = 1;
    arrays_[name] = {std::vector<double>(total, 0.0), dim_sizes};
    return;
  }

  if (vd.init_expr) {
    vars_[name] = Eval(*vd.init_expr);
  } else if (vd.init_value) {
    vars_[name] = Eval(*vd.init_value);
  } else {
    vars_[name] = int64_t(0);
  }
}

void Interpreter::ExecAssign(AST::Assignment& asgn) {
  Value rhs_val = Eval(*asgn.value);

  auto& da = asgn.da;
  std::string arr_name = da->GetDataName();

  if (!da->AccessElement()) {
    vars_[arr_name] = rhs_val;
    return;
  }

  auto ait = arrays_.find(arr_name);
  if (ait == arrays_.end()) {
    vars_[arr_name] = rhs_val;
    return;
  }

  auto& indices = da->GetIndices();
  auto& arr = ait->second;

  if (indices.size() == 1) {
    int64_t idx = ToInt(Eval(*indices[0]));
    if (idx >= 0 && idx < (int64_t)arr.data.size())
      arr.data[idx] = ToDouble(rhs_val);
  } else if (indices.size() >= 2) {
    int64_t r = ToInt(Eval(*indices[0]));
    int64_t c = ToInt(Eval(*indices[1]));
    int64_t cols = arr.dims.size() >= 2 ? arr.dims[1] : 1;
    int64_t flat = r * cols + c;
    if (flat >= 0 && flat < (int64_t)arr.data.size())
      arr.data[flat] = ToDouble(rhs_val);
  }
}

void Interpreter::ExecCall(AST::Call& call) {
  std::string name = call.function->name;
  if (name == "println") {
    auto& args = call.arguments;
    bool first = true;
    if (args) {
      for (size_t i = 0; i < args->Count(); ++i) {
        if (!first) output_ << " ";
        first = false;
        Value v = Eval(*args->ValueAt(i));
        output_ << ToString(v);
      }
    }
    output_ << "\n";
  }
}

void Interpreter::ExecMemoryDecl(AST::Memory&) {
  // Memory nodes are handled via NamedVariableDecl
}

Value Interpreter::Eval(AST::Node& node) {
  if (auto* il = dyn_cast<AST::IntLiteral>(&node)) {
    return il->Val();
  }
  if (auto* fl = dyn_cast<AST::FloatLiteral>(&node)) {
    if (fl->IsFloat64()) return fl->Val_f64();
    return (double)fl->Val_f32();
  }
  if (auto* sl = dyn_cast<AST::StringLiteral>(&node)) {
    return sl->Val();
  }
  if (auto* bl = dyn_cast<AST::BoolLiteral>(&node)) {
    return bl->Val();
  }
  if (auto* id = dyn_cast<AST::Identifier>(&node)) {
    auto it = vars_.find(id->name);
    if (it != vars_.end()) return it->second;
    return int64_t(0);
  }
  if (auto* expr = dyn_cast<AST::Expr>(&node)) {
    return EvalExpr(*expr);
  }
  if (auto* da = dyn_cast<AST::DataAccess>(&node)) {
    return EvalDataAccess(*da);
  }
  if (auto* ce = dyn_cast<AST::CastExpr>(&node)) {
    auto& val_node = ce->value_r ? ce->value_r : ce->value_l;
    if (!val_node) return int64_t(0);
    Value inner = Eval(*val_node);
    auto bt = ce->ToType();
    if (bt == BaseType::F32 || bt == BaseType::F64)
      return ToDouble(inner);
    return ToInt(inner);
  }
  return int64_t(0);
}

Value Interpreter::EvalExpr(AST::Expr& expr) {
  auto op = expr.op.GetKind();

  if (op == Op::Ref) {
    if (expr.value_l) return Eval(*expr.value_l);
    if (expr.value_r) return Eval(*expr.value_r);
    return int64_t(0);
  }

  if (op == Op::Cast) {
    if (expr.value_l) return Eval(*expr.value_l);
    return int64_t(0);
  }

  if (op == Op::ElemOf) {
    if (auto name = AST::GetName(expr)) {
      auto ait = arrays_.find(*name);
      if (ait != arrays_.end() && expr.value_r) {
        auto* da = dyn_cast<AST::DataAccess>(&*expr.value_r);
        if (da) return EvalDataAccess(*da);
      }
    }
    if (expr.value_r) return Eval(*expr.value_r);
    if (expr.value_l) return Eval(*expr.value_l);
    return int64_t(0);
  }

  if (expr.value_l && expr.value_r) {
    Value lv = Eval(*expr.value_l);
    Value rv = Eval(*expr.value_r);

    bool use_float = std::holds_alternative<double>(lv) ||
                     std::holds_alternative<double>(rv);

    switch (op) {
    case Op::Add:
      return use_float ? Value(ToDouble(lv) + ToDouble(rv))
                       : Value(ToInt(lv) + ToInt(rv));
    case Op::Sub:
      return use_float ? Value(ToDouble(lv) - ToDouble(rv))
                       : Value(ToInt(lv) - ToInt(rv));
    case Op::Mul:
      return use_float ? Value(ToDouble(lv) * ToDouble(rv))
                       : Value(ToInt(lv) * ToInt(rv));
    case Op::Div:
      if (use_float) {
        double d = ToDouble(rv);
        return d != 0.0 ? Value(ToDouble(lv) / d) : Value(0.0);
      } else {
        int64_t d = ToInt(rv);
        return d != 0 ? Value(ToInt(lv) / d) : Value(int64_t(0));
      }
    case Op::Mod: {
      int64_t d = ToInt(rv);
      return d != 0 ? Value(ToInt(lv) % d) : Value(int64_t(0));
    }
    case Op::Lt: return Value(use_float ? ToDouble(lv) < ToDouble(rv)
                                        : ToInt(lv) < ToInt(rv));
    case Op::Gt: return Value(use_float ? ToDouble(lv) > ToDouble(rv)
                                        : ToInt(lv) > ToInt(rv));
    case Op::Le: return Value(use_float ? ToDouble(lv) <= ToDouble(rv)
                                        : ToInt(lv) <= ToInt(rv));
    case Op::Ge: return Value(use_float ? ToDouble(lv) >= ToDouble(rv)
                                        : ToInt(lv) >= ToInt(rv));
    case Op::Eq: return Value(use_float ? ToDouble(lv) == ToDouble(rv)
                                        : ToInt(lv) == ToInt(rv));
    case Op::Ne: return Value(use_float ? ToDouble(lv) != ToDouble(rv)
                                        : ToInt(lv) != ToInt(rv));
    case Op::LogicAnd: return Value(ToBool(lv) && ToBool(rv));
    case Op::LogicOr:  return Value(ToBool(lv) || ToBool(rv));
    case Op::BitAnd: return Value(ToInt(lv) & ToInt(rv));
    case Op::BitOr:  return Value(ToInt(lv) | ToInt(rv));
    case Op::BitXor: return Value(ToInt(lv) ^ ToInt(rv));
    case Op::Shl:    return Value(ToInt(lv) << ToInt(rv));
    case Op::Shr:    return Value(ToInt(lv) >> ToInt(rv));
    default: break;
    }
  }

  if (expr.value_l || expr.value_r) {
    auto& operand = expr.value_l ? expr.value_l : expr.value_r;
    Value v = Eval(*operand);
    switch (op) {
    case Op::LogicNot: return Value(!ToBool(v));
    case Op::BitNot:   return Value(~ToInt(v));
    default: return v;
    }
  }

  return int64_t(0);
}

Value Interpreter::EvalDataAccess(AST::DataAccess& da) {
  std::string name = da.GetDataName();

  if (!da.AccessElement()) {
    auto vit = vars_.find(name);
    if (vit != vars_.end()) return vit->second;
    return int64_t(0);
  }

  auto ait = arrays_.find(name);
  if (ait == arrays_.end()) {
    auto vit = vars_.find(name);
    if (vit != vars_.end()) return vit->second;
    return int64_t(0);
  }

  auto& indices = da.GetIndices();
  auto& arr = ait->second;

  if (indices.size() == 1) {
    int64_t idx = ToInt(Eval(*indices[0]));
    if (idx >= 0 && idx < (int64_t)arr.data.size())
      return arr.data[idx];
  } else if (indices.size() >= 2) {
    int64_t r = ToInt(Eval(*indices[0]));
    int64_t c = ToInt(Eval(*indices[1]));
    int64_t cols = arr.dims.size() >= 2 ? arr.dims[1] : 1;
    int64_t flat = r * cols + c;
    if (flat >= 0 && flat < (int64_t)arr.data.size())
      return arr.data[flat];
  }
  return 0.0;
}

double Interpreter::ToDouble(const Value& v) {
  if (auto* d = std::get_if<double>(&v)) return *d;
  if (auto* i = std::get_if<int64_t>(&v)) return (double)*i;
  if (auto* b = std::get_if<bool>(&v)) return *b ? 1.0 : 0.0;
  return 0.0;
}

int64_t Interpreter::ToInt(const Value& v) {
  if (auto* i = std::get_if<int64_t>(&v)) return *i;
  if (auto* d = std::get_if<double>(&v)) return (int64_t)*d;
  if (auto* b = std::get_if<bool>(&v)) return *b ? 1 : 0;
  return 0;
}

std::string Interpreter::ToString(const Value& v) {
  if (auto* s = std::get_if<std::string>(&v)) return *s;
  if (auto* i = std::get_if<int64_t>(&v)) return std::to_string(*i);
  if (auto* d = std::get_if<double>(&v)) {
    double val = *d;
    if (val == (int64_t)val && std::abs(val) < 1e15)
      return std::to_string((int64_t)val);
    std::ostringstream oss;
    oss << val;
    return oss.str();
  }
  if (auto* b = std::get_if<bool>(&v)) return *b ? "true" : "false";
  return "";
}

bool Interpreter::ToBool(const Value& v) {
  if (auto* b = std::get_if<bool>(&v)) return *b;
  if (auto* i = std::get_if<int64_t>(&v)) return *i != 0;
  if (auto* d = std::get_if<double>(&v)) return *d != 0.0;
  if (auto* s = std::get_if<std::string>(&v)) return !s->empty();
  return false;
}

} // namespace CoWeb
