#include "mock_interp.hpp"
#include "ast.hpp"
#include "context.hpp"
#include "debugger.hpp"
#include "io.hpp"
#include "types.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <set>
#include <thread>

namespace Choreo {
namespace Mock {

MockInterpreter::MockInterpreter() : VisitorWithSymTab("mock-interp") {}

// Override RunOnProgramImpl to walk the AST manually instead of using the
// visitor traversal. This avoids double-execution from the accept/visit
// pattern which would visit child nodes unconditionally.
bool MockInterpreter::RunOnProgramImpl(AST::Node& root) {
  if (!isa<AST::Program>(&root)) {
    Error(root.LOC(), "Not running a choreo program.");
    return false;
  }

  auto& prog = static_cast<AST::Program&>(root);

  // Collect all choreo function definitions
  for (auto& stmt : prog.stmts->AllSubs()) {
    if (auto fn = dyn_cast<AST::ChoreoFunction>(stmt))
      functions[fn->name] = fn.get();
  }

  // Execute each top-level function
  for (auto& stmt : prog.stmts->AllSubs()) {
    if (auto fn = dyn_cast<AST::ChoreoFunction>(stmt)) {
      mem.EnterScope();
      ExecBlock(fn->stmts);
      if (cf.kind == ControlFlow::Return) cf.kind = ControlFlow::None;
      mem.LeaveScope();
    }
  }

  return !HasError();
}

// -----------------------------------------------------------------------
// ExecBlock -- sequentially execute each statement in a block
// -----------------------------------------------------------------------
void MockInterpreter::ExecBlock(const ptr<AST::MultiNodes>& stmts) {
  if (!stmts) return;
  if (debugger_ && debugger_->IsActive()) debugger_->EnterBlock();
  auto subs = stmts->AllSubs();
  for (size_t i = 0; i < subs.size(); ++i) {
    if (cf.kind != ControlFlow::None || quit_requested_) return;
    if (auto wb = dyn_cast<AST::WhileBlock>(subs[i])) {
      std::vector<AST::Assignment*> refreshes;
      if (i > 0)
        if (auto a = dyn_cast<AST::Assignment>(subs[i - 1]))
          refreshes.push_back(a.get());
      ExecWhile(*wb, refreshes);
    } else {
      ExecStatement(subs[i]);
    }
  }
  if (debugger_ && debugger_->IsActive()) debugger_->LeaveBlock();
}

void MockInterpreter::ExecStatement(const ptr<AST::Node>& stmt) {
  if (cf.kind != ControlFlow::None) return;
  if (quit_requested_) return;

  if (debugger_ && debugger_->IsActive()) {
    if (!debugger_->OnStatement(*stmt)) {
      quit_requested_ = true;
      return;
    }
  }

  if (auto assign = dyn_cast<AST::Assignment>(stmt))
    ExecAssignment(*assign);
  else if (auto decl = dyn_cast<AST::NamedVariableDecl>(stmt))
    ExecNamedVariableDecl(*decl);
  else if (auto pb = dyn_cast<AST::ParallelBy>(stmt))
    ExecParallelBy(*pb);
  else if (auto ifb = dyn_cast<AST::IfElseBlock>(stmt))
    ExecIfElse(*ifb);
  else if (auto wb = dyn_cast<AST::WhileBlock>(stmt))
    ExecWhile(*wb);
  else if (auto fb = dyn_cast<AST::ForeachBlock>(stmt))
    ExecForeach(*fb);
  else if (auto call = dyn_cast<AST::Call>(stmt))
    ExecCall(*call);
  else if (auto ret = dyn_cast<AST::Return>(stmt))
    ExecReturn(*ret);
  else if (auto dma = dyn_cast<AST::DMA>(stmt))
    ExecDMA(*dma);
  else if (auto brk = dyn_cast<AST::Break>(stmt))
    cf.kind = ControlFlow::Break;
  else if (auto cont = dyn_cast<AST::Continue>(stmt))
    cf.kind = ControlFlow::Continue;
  else if (auto wb = dyn_cast<AST::WithBlock>(stmt))
    ExecBlock(wb->stmts);
  else if (auto mn = dyn_cast<AST::MultiNodes>(stmt))
    ExecBlock(mn);
  else if (isa<AST::Synchronize>(stmt) || isa<AST::Yield>(stmt) ||
           isa<AST::Trigger>(stmt)) {
  } // no-ops for mock interpreter
  else if (auto cpp = dyn_cast<AST::CppSourceCode>(stmt)) {
    if (cpp->kind == AST::CppSourceCode::Host ||
        cpp->kind == AST::CppSourceCode::Device)
      Warning(cpp->LOC(), "mock: host C++ source code is not interpretable.");
  } else if (auto mma = dyn_cast<AST::MMA>(stmt))
    Error1(mma->LOC(), "mock: MMA operations are not supported.");
  else if (auto rot = dyn_cast<AST::Rotate>(stmt))
    ExecRotate(*rot);
  else if (auto wait = dyn_cast<AST::Wait>(stmt))
    ExecWait(*wait);
  else if (auto itb = dyn_cast<AST::InThreadsBlock>(stmt))
    ExecInThreads(*itb);
  else if (isa<AST::NamedTypeDecl>(stmt)) {
  } // type alias, skip
  else {
    Warning(stmt->LOC(), "mock: unhandled statement type '" +
                             std::string(stmt->TypeNameString()) + "'");
  }
}

// -----------------------------------------------------------------------
// ParallelBy -- threaded execution over bound range
// -----------------------------------------------------------------------
void MockInterpreter::ExecParallelByBody(
    AST::ParallelBy& n, int64_t i, int64_t /*bound*/,
    const std::string& pv_name, bool has_sub,
    const std::vector<std::string>& sub_names,
    const std::vector<int64_t>& sub_bounds) {
  mem.EnterScope();
  mem.PushThread(pv_name, i);
  mem.Define(pv_name, Value::MakeInt(i));

  if (has_sub) {
    int64_t remaining = i;
    for (int j = (int)sub_names.size() - 1; j >= 0; --j) {
      mem.Define(sub_names[j], Value::MakeInt(remaining % sub_bounds[j]));
      remaining /= sub_bounds[j];
    }
  }

  ExecBlock(n.stmts);
  mem.PopThread();
  mem.LeaveScope();
}

void MockInterpreter::ExecParallelBy(AST::ParallelBy& n) {
  auto bound_val = ExprEval(n.BoundExpr());
  int64_t bound = bound_val.AsInt();

  std::string pv_name = n.BPV()->name;

  bool has_sub = n.HasSubPVs();
  std::vector<std::string> sub_names;
  std::vector<int64_t> sub_bounds;

  if (has_sub) {
    for (auto& sv : n.AllSubPVs())
      sub_names.push_back(cast<AST::Identifier>(sv)->name);
    for (auto& sb : n.AllBoundExprs()) {
      auto bv = ExprEval(sb);
      sub_bounds.push_back(bv.AsInt());
    }
  }

#ifdef __EMSCRIPTEN__
  for (int64_t i = 0; i < bound; ++i) {
    ExecParallelByBody(n, i, bound, pv_name, has_sub, sub_names, sub_bounds);
  }
#else
  // Shared mutex for serializing print output across threads
  std::mutex pm;
  bool own_mutex = (print_mutex_ == nullptr);
  if (own_mutex) print_mutex_ = &pm;

  std::vector<std::thread> threads;
  threads.reserve(bound);

  for (int64_t i = 0; i < bound; ++i) {
    threads.emplace_back([this, &n, i, bound, &pv_name, has_sub, &sub_names,
                          &sub_bounds, &pm]() {
      MockInterpreter child;
      child.mem = mem.Fork();
      child.functions = functions;
      child.print_mutex_ = &pm;

      child.ExecParallelByBody(n, i, bound, pv_name, has_sub, sub_names,
                               sub_bounds);
    });
  }

  for (auto& t : threads) t.join();

  if (own_mutex) print_mutex_ = nullptr;
#endif
}

// -----------------------------------------------------------------------
// Assignment
// -----------------------------------------------------------------------
void MockInterpreter::ExecAssignment(AST::Assignment& n) {
  Value rhs = ExprEval(n.value);

  if (!n.da->AccessElement()) {
    std::string name = n.da->GetDataName();
    if (mem.Exists(name)) {
      auto& existing = mem.Lookup(name);
      if (existing.kind == Value::Pointer && rhs.kind == Value::Pointer) {
        size_t bytes =
            std::min(existing.alloc->TotalBytes(), rhs.alloc->TotalBytes());
        std::memcpy(existing.alloc->RawPtr(), rhs.alloc->RawPtr(), bytes);
      } else {
        existing = rhs;
      }
    } else {
      mem.Define(name, rhs);
    }
  } else {
    std::string arr_name = n.da->GetDataName();
    auto& arr_val = mem.Lookup(arr_name);
    assert(arr_val.kind == Value::Pointer && arr_val.alloc);

    std::vector<size_t> indices;
    for (auto& idx : n.da->GetIndices())
      indices.push_back((size_t)ExprEval(idx).AsInt());

    size_t linear = ComputeLinearIndex(indices, arr_val.alloc->shape);
    size_t byte_offset = linear * arr_val.alloc->ElemSize() + arr_val.offset;
    arr_val.WriteToAlloc(byte_offset, rhs);
  }
}

// -----------------------------------------------------------------------
// NamedVariableDecl
// -----------------------------------------------------------------------
void MockInterpreter::ExecNamedVariableDecl(AST::NamedVariableDecl& n) {
  std::string name = n.GetName();
  auto ty = n.GetType();

  if (auto sty = dyn_cast<SpannedType>(ty.get())) {
    auto shape = ResolveShape(ty);
    auto storage = sty->GetStorage();
    auto elem_bt = sty->ElementType();
    auto alloc = mem.Allocate(elem_bt, shape, storage);

    if (n.init_expr) {
      auto init = ExprEval(n.init_expr);
      if (init.kind == Value::Pointer && init.alloc) {
        size_t bytes = std::min(alloc->TotalBytes(), init.alloc->TotalBytes());
        std::memcpy(alloc->RawPtr(), init.alloc->RawPtr(), bytes);
      }
    }

    mem.Define(name, Value::MakePointer(alloc, elem_bt));
  } else if (n.IsArray()) {
    auto bt = ResolveBaseType(ty);
    std::vector<size_t> shape;
    for (size_t i = 0; i < n.ArrayDimensions()->Count(); ++i) {
      auto dim = ExprEval(n.ArrayDimension(i));
      shape.push_back((size_t)dim.AsInt());
    }
    auto alloc = mem.Allocate(bt, shape, Storage::LOCAL);

    if (n.init_value) {
      auto init = ExprEval(n.init_value);
      for (size_t i = 0; i < alloc->TotalElements(); ++i) {
        size_t byte_off = i * alloc->ElemSize();
        Value tmp = Value::MakePointer(alloc, bt);
        tmp.WriteToAlloc(byte_off, init);
      }
    }

    mem.Define(name, Value::MakePointer(alloc, bt));
  } else {
    Value val;
    if (n.init_expr)
      val = ExprEval(n.init_expr);
    else if (n.init_value)
      val = ExprEval(n.init_value);
    else {
      auto bt = ResolveBaseType(ty);
      val = Value::MakeInt(0);
      val.base_type = bt;
    }
    mem.Define(name, val);
  }
}

// -----------------------------------------------------------------------
// IfElseBlock
// -----------------------------------------------------------------------
void MockInterpreter::ExecIfElse(AST::IfElseBlock& n) {
  auto cond = ExprEval(n.GetPred());
  if (cond.AsBool()) {
    mem.EnterScope();
    ExecBlock(n.GetThenBody());
    mem.LeaveScope();
  } else if (n.HasElse()) {
    mem.EnterScope();
    ExecBlock(n.GetElseBody());
    mem.LeaveScope();
  }
}

// -----------------------------------------------------------------------
// WhileBlock
// -----------------------------------------------------------------------
void MockInterpreter::ExecWhile(
    AST::WhileBlock& n, const std::vector<AST::Assignment*>& pred_refreshes) {
  while (true) {
    for (auto* a : pred_refreshes) ExecAssignment(*a);
    auto cond = ExprEval(n.GetPred());
    if (!cond.AsBool()) break;

    mem.EnterScope();
    ExecBlock(n.stmts);
    mem.LeaveScope();

    if (cf.kind == ControlFlow::Return) return;
    if (cf.kind == ControlFlow::Break) {
      cf.kind = ControlFlow::None;
      break;
    }
    if (cf.kind == ControlFlow::Continue) cf.kind = ControlFlow::None;
  }
}

// -----------------------------------------------------------------------
// ForeachBlock
// -----------------------------------------------------------------------
void MockInterpreter::ExecForeach(AST::ForeachBlock& n) {
  if (!n.ranges || n.ranges->Count() == 0) return;

  struct RangeInfo {
    std::string iv_name;
    int64_t lb, ub, step;
  };

  std::vector<RangeInfo> ranges;
  for (auto& r : n.ranges->AllValues()) {
    auto lr = cast<AST::LoopRange>(r);
    RangeInfo ri;
    ri.iv_name = lr->GetIVName();

    if (lr->lbound)
      ri.lb = ExprEval(lr->lbound).AsInt();
    else
      ri.lb = 0;

    if (lr->ubound) {
      ri.ub = ExprEval(lr->ubound).AsInt();
    } else {
      ri.ub = 0;
      auto iv_ty = lr->GetIV()->GetType();
      if (auto bit = dyn_cast<BoundedITupleType>(iv_ty.get())) {
        auto ub_val = bit->GetUpperBound();
        if (auto iv = VIInt(ub_val)) ri.ub = *iv;
      } else if (auto bint = dyn_cast<BoundedIntegerType>(iv_ty.get())) {
        auto ub_val = bint->GetUpperBound();
        if (auto iv = VIInt(ub_val)) ri.ub = *iv;
      }
    }

    ri.step = IsValidStep(lr->step) ? lr->step : 1;
    ranges.push_back(ri);
  }

  std::function<void(size_t)> run_loop = [&](size_t depth) {
    if (cf.kind != ControlFlow::None) return;
    if (depth == ranges.size()) {
      ExecBlock(n.stmts);
      return;
    }
    auto& ri = ranges[depth];
    for (int64_t i = ri.lb; i < ri.ub; i += ri.step) {
      if (cf.kind != ControlFlow::None) break;
      mem.Define(ri.iv_name, Value::MakeInt(i));
      run_loop(depth + 1);
      if (cf.kind == ControlFlow::Break) {
        if (depth == 0) cf.kind = ControlFlow::None;
        return;
      }
      if (cf.kind == ControlFlow::Continue) {
        if (depth == ranges.size() - 1) cf.kind = ControlFlow::None;
      }
    }
  };

  mem.EnterScope();
  run_loop(0);
  mem.LeaveScope();
}

// -----------------------------------------------------------------------
// Call
// -----------------------------------------------------------------------
void MockInterpreter::ExecCall(AST::Call& n) {
  std::string func_name = n.function->name;

  std::vector<Value> args;
  if (n.arguments)
    for (auto& arg : n.GetArguments()) args.push_back(ExprEval(arg));

  if (n.IsBIF() || IsKnownBIF(func_name)) {
    CallBIF(func_name, args, n);
    return;
  }

  auto it = functions.find(func_name);
  if (it == functions.end()) {
    Warning(n.LOC(), "mock: unknown function '" + func_name + "', skipping.");
    return;
  }

  auto* fn = it->second;
  mem.EnterScope();

  auto& params = fn->f_decl.params->values;
  for (size_t i = 0; i < params.size() && i < args.size(); ++i) {
    auto param = cast<AST::Parameter>(params[i]);
    if (param->HasSymbol()) mem.Define(param->sym->name, args[i]);
  }

  ExecBlock(fn->stmts);

  if (cf.kind == ControlFlow::Return) cf.kind = ControlFlow::None;

  mem.LeaveScope();
}

// -----------------------------------------------------------------------
// Return
// -----------------------------------------------------------------------
void MockInterpreter::ExecReturn(AST::Return& n) {
  cf.kind = ControlFlow::Return;
  if (n.value) cf.return_value = ExprEval(n.value);
}

// -----------------------------------------------------------------------
// DMA -- synchronous memcpy for Phase 1
// -----------------------------------------------------------------------
void MockInterpreter::ExecDMA(AST::DMA& n) {
  if (!n.from || !n.to) return;

  auto src = ExprEval(n.from);
  auto dst = ExprEval(n.to);

  if (src.kind != Value::Pointer || dst.kind != Value::Pointer || !src.alloc ||
      !dst.alloc)
    return;

  size_t bytes = std::min(src.alloc->TotalBytes() - src.offset,
                          dst.alloc->TotalBytes() - dst.offset);

  if (n.IsAsync() && !n.future.empty()) {
    auto src_ptr = std::const_pointer_cast<Allocation>(src.alloc);
    auto dst_ptr = dst.alloc;
    size_t s_off = src.offset, d_off = dst.offset;
    auto fut = std::async(
        std::launch::async, [src_ptr, dst_ptr, s_off, d_off, bytes]() {
          std::memcpy((uint8_t*)dst_ptr->RawPtr() + d_off,
                      (const uint8_t*)src_ptr->RawPtr() + s_off, bytes);
        });
    std::string src_sym, dst_sym;
    if (auto ca = dyn_cast<AST::ChunkAt>(n.from)) src_sym = ca->RefSymbol();
    if (auto ca = dyn_cast<AST::ChunkAt>(n.to)) dst_sym = ca->RefSymbol();
    mem.Define(n.future,
               Value::MakeFuture(fut.share(), src_sym, dst_sym, bytes));
  } else {
    std::memcpy((uint8_t*)dst.alloc->RawPtr() + dst.offset,
                (const uint8_t*)src.alloc->RawPtr() + src.offset, bytes);
    if (!n.future.empty()) mem.Define(n.future, Value::MakeBool(true));
  }
}

// -----------------------------------------------------------------------
// Wait -- block until futures complete
// -----------------------------------------------------------------------
void MockInterpreter::ExecWait(AST::Wait& n) {
  for (auto& t : n.GetTargets()) {
    std::string name;
    if (auto id = dyn_cast<AST::Identifier>(t))
      name = id->name;
    else if (auto da = dyn_cast<AST::DataAccess>(t))
      name = da->GetDataName();
    else if (auto expr = dyn_cast<AST::Expr>(t)) {
      if (expr->GetForm() == AST::Expr::Reference) {
        if (auto id = dyn_cast<AST::Identifier>(expr->GetR()))
          name = id->name;
        else if (auto da = dyn_cast<AST::DataAccess>(expr->GetR()))
          name = da->GetDataName();
      }
    }
    if (name.empty() || !mem.Exists(name)) continue;
    auto& val = mem.Lookup(name);
    if (val.kind == Value::Future && val.future_info &&
        val.future_info->handle.valid())
      val.future_info->handle.get();
  }
}

// -----------------------------------------------------------------------
// Rotate -- circular rotation of variable values
// -----------------------------------------------------------------------
void MockInterpreter::ExecRotate(AST::Rotate& n) {
  auto& ids = n.GetIds();
  if (ids.size() < 2) return;

  std::vector<std::string> names;
  for (auto& id_node : ids) {
    if (auto id = dyn_cast<AST::Identifier>(id_node)) names.push_back(id->name);
  }
  if (names.size() < 2) return;

  Value last = mem.Lookup(names.back());
  for (int i = (int)names.size() - 1; i > 0; --i)
    mem.Define(names[i], mem.Lookup(names[i - 1]));
  mem.Define(names[0], last);
}

// -----------------------------------------------------------------------
// InThreadsBlock -- execute body under predicate
// -----------------------------------------------------------------------
void MockInterpreter::ExecInThreads(AST::InThreadsBlock& n) {
  auto pred = ExprEval(n.GetPredicate());
  if (pred.AsBool()) {
    mem.EnterScope();
    ExecBlock(n.stmts);
    mem.LeaveScope();
  }
}

// -----------------------------------------------------------------------
// ExprEval -- evaluate an expression AST node to a Value
// -----------------------------------------------------------------------
Value MockInterpreter::ExprEval(const ptr<AST::Node>& e) {
  if (!e) return Value::MakeInt(0);

  if (auto lit = dyn_cast<AST::IntLiteral>(e)) {
    return std::visit(
        [](auto v) -> Value {
          using T = decltype(v);
          if constexpr (std::is_same_v<T, int>)
            return Value::MakeInt(v);
          else if constexpr (std::is_same_v<T, uint32_t>)
            return Value::MakeUInt(v);
          else if constexpr (std::is_same_v<T, int64_t>)
            return Value::MakeInt(v);
          else if constexpr (std::is_same_v<T, uint64_t>)
            return Value::MakeUInt(v);
          else
            return Value::MakeInt(0);
        },
        lit->value);
  }

  if (auto lit = dyn_cast<AST::FloatLiteral>(e)) {
    return std::visit(
        [](auto v) -> Value {
          using T = decltype(v);
          if constexpr (std::is_same_v<T, float>)
            return Value::MakeFloat(v);
          else if constexpr (std::is_same_v<T, double>)
            return Value::MakeDouble(v);
          else
            return Value::MakeFloat(0.0f);
        },
        lit->value);
  }

  if (auto lit = dyn_cast<AST::BoolLiteral>(e))
    return Value::MakeBool(lit->value);

  if (auto lit = dyn_cast<AST::StringLiteral>(e))
    return Value::MakeString(lit->Val());

  if (auto id = dyn_cast<AST::Identifier>(e)) {
    if (mem.Exists(id->name)) return mem.Lookup(id->name);
    Warning(id->LOC(),
            "mock: undefined variable '" + id->name + "', defaulting to 0.");
    return Value::MakeInt(0);
  }

  if (auto da = dyn_cast<AST::DataAccess>(e)) {
    std::string name = da->GetDataName();
    if (!mem.Exists(name)) {
      Warning(da->LOC(),
              "mock: undefined variable '" + name + "', defaulting to 0.");
      return Value::MakeInt(0);
    }

    auto& val = mem.Lookup(name);
    if (!da->AccessElement()) return val;

    assert(val.kind == Value::Pointer && val.alloc);
    std::vector<size_t> indices;
    for (auto& idx : da->GetIndices())
      indices.push_back((size_t)ExprEval(idx).AsInt());

    size_t linear = ComputeLinearIndex(indices, val.alloc->shape);
    size_t byte_offset = linear * val.alloc->ElemSize() + val.offset;
    return val.ReadFromAlloc(byte_offset, val.base_type);
  }

  if (auto cast_expr = dyn_cast<AST::CastExpr>(e)) {
    auto inner = ExprEval(cast_expr->GetR());
    auto to_type = cast_expr->ToType();
    return CastValue(inner, to_type);
  }

  if (auto expr = dyn_cast<AST::Expr>(e)) {
    if (expr->GetForm() == AST::Expr::Reference) return ExprEval(expr->GetR());

    if (expr->GetForm() == AST::Expr::Unary)
      return EvalUnaryOp(expr->op, ExprEval(expr->GetR()));

    if (expr->GetForm() == AST::Expr::Binary) {
      if (expr->op == Op::ElemOf) {
        auto base = ExprEval(expr->GetL());
        if (base.kind == Value::Pointer && base.alloc) {
          auto idx_val = ExprEval(expr->GetR());
          size_t idx = (size_t)idx_val.AsInt();
          size_t byte_offset = idx * base.alloc->ElemSize() + base.offset;
          return base.ReadFromAlloc(byte_offset, base.base_type);
        }
        return base;
      }

      if (expr->op == Op::DataOf || expr->op == Op::MDataOf)
        return ExprEval(expr->GetR());

      if (expr->op == Op::SizeOf) {
        auto base = ExprEval(expr->GetR());
        if (base.kind == Value::Pointer && base.alloc)
          return Value::MakeInt((int64_t)base.alloc->TotalElements());
        return Value::MakeInt(0);
      }

      if (expr->op == Op::DimOf) {
        auto base = ExprEval(expr->GetL());
        auto dim_idx = ExprEval(expr->GetR());
        if (base.kind == Value::Pointer && base.alloc) {
          size_t d = (size_t)dim_idx.AsInt();
          if (d < base.alloc->shape.size())
            return Value::MakeInt((int64_t)base.alloc->shape[d]);
        }
        return Value::MakeInt(0);
      }

      auto lhs = ExprEval(expr->GetL());
      auto rhs = ExprEval(expr->GetR());
      return EvalBinaryOp(expr->op, lhs, rhs);
    }

    if (expr->GetForm() == AST::Expr::Ternary) {
      auto cond = ExprEval(expr->GetC());
      if (cond.AsBool())
        return ExprEval(expr->GetL());
      else
        return ExprEval(expr->GetR());
    }
  }

  if (auto call = dyn_cast<AST::Call>(e)) {
    std::string func_name = call->function->name;
    std::vector<Value> args;
    if (call->arguments)
      for (auto& arg : call->GetArguments()) args.push_back(ExprEval(arg));

    if (call->IsBIF() || IsKnownBIF(func_name))
      return CallBIF(func_name, args, *call);

    auto it = functions.find(func_name);
    if (it != functions.end()) {
      auto* fn = it->second;
      mem.EnterScope();
      auto& params = fn->f_decl.params->values;
      for (size_t i = 0; i < params.size() && i < args.size(); ++i) {
        auto param = cast<AST::Parameter>(params[i]);
        if (param->HasSymbol()) mem.Define(param->sym->name, args[i]);
      }
      ExecBlock(fn->stmts);
      Value ret;
      if (cf.kind == ControlFlow::Return) {
        ret = cf.return_value;
        cf.kind = ControlFlow::None;
      }
      mem.LeaveScope();
      return ret;
    }

    return Value::MakeInt(0);
  }

  if (auto mv = dyn_cast<AST::MultiValues>(e)) {
    if (mv->Count() == 1) return ExprEval(mv->ValueAt(0));
  }

  if (auto chunk = dyn_cast<AST::ChunkAt>(e)) {
    std::string base_name = chunk->data->name;
    auto dot = base_name.rfind(".data");
    if (dot != std::string::npos) base_name = base_name.substr(0, dot);
    if (!mem.Exists(base_name)) return Value::MakeInt(0);

    auto& base_val = mem.Lookup(base_name);

    if (chunk->HasOperation() && base_val.kind == Value::Pointer &&
        base_val.alloc) {
      for (auto& op : chunk->AllOperations()) {
        auto indices_mv = op->GetIndices();
        if (!indices_mv || indices_mv->Count() == 0) continue;

        std::vector<size_t> idx_vals;
        for (auto& idx_node : indices_mv->AllValues())
          idx_vals.push_back((size_t)ExprEval(idx_node).AsInt());

        size_t linear = ComputeLinearIndex(idx_vals, base_val.alloc->shape);
        size_t byte_offset =
            linear * base_val.alloc->ElemSize() + base_val.offset;
        return base_val.ReadFromAlloc(byte_offset, base_val.base_type);
      }
    }

    return base_val;
  }

  if (auto sel = dyn_cast<AST::Select>(e)) {
    auto factor = ExprEval(sel->select_factor);
    size_t idx = (size_t)factor.AsInt();
    if (sel->expr_list && idx < sel->expr_list->Count())
      return ExprEval(sel->expr_list->ValueAt(idx));
    return Value::MakeInt(0);
  }

  if (e->LOC().begin.line > 0) {
    Warning(e->LOC(), "mock: unhandled expression type '" +
                          std::string(e->TypeNameString()) + "'");
  }
  return Value::MakeInt(0);
}

// -----------------------------------------------------------------------
// Binary operator evaluation
// -----------------------------------------------------------------------
Value MockInterpreter::EvalBinaryOp(Opcode op, const Value& lhs,
                                    const Value& rhs) {
  bool use_float =
      (lhs.base_type == BaseType::F32 || lhs.base_type == BaseType::F64 ||
       rhs.base_type == BaseType::F32 || rhs.base_type == BaseType::F64);

  auto k = op.GetKind();

  if (use_float) {
    double l = lhs.AsDouble();
    double r = rhs.AsDouble();
    switch (k) {
    case Op::Add: return Value::MakeDouble(l + r);
    case Op::Sub: return Value::MakeDouble(l - r);
    case Op::Mul: return Value::MakeDouble(l * r);
    case Op::Div: return Value::MakeDouble(r != 0 ? l / r : 0);
    case Op::Mod: return Value::MakeDouble(std::fmod(l, r));
    case Op::Eq: return Value::MakeBool(l == r);
    case Op::Ne: return Value::MakeBool(l != r);
    case Op::Gt: return Value::MakeBool(l > r);
    case Op::Ge: return Value::MakeBool(l >= r);
    case Op::Lt: return Value::MakeBool(l < r);
    case Op::Le: return Value::MakeBool(l <= r);
    default: break;
    }
  }

  int64_t l = lhs.AsInt();
  int64_t r = rhs.AsInt();
  switch (k) {
  case Op::Add: return Value::MakeInt(l + r);
  case Op::Sub: return Value::MakeInt(l - r);
  case Op::Mul: return Value::MakeInt(l * r);
  case Op::Div: return Value::MakeInt(r != 0 ? l / r : 0);
  case Op::Mod: return Value::MakeInt(r != 0 ? l % r : 0);
  case Op::CeilDiv: return Value::MakeInt(r != 0 ? (l + r - 1) / r : 0);
  case Op::Eq: return Value::MakeBool(l == r);
  case Op::Ne: return Value::MakeBool(l != r);
  case Op::Gt: return Value::MakeBool(l > r);
  case Op::Ge: return Value::MakeBool(l >= r);
  case Op::Lt: return Value::MakeBool(l < r);
  case Op::Le: return Value::MakeBool(l <= r);
  case Op::LogicAnd: return Value::MakeBool(lhs.AsBool() && rhs.AsBool());
  case Op::LogicOr: return Value::MakeBool(lhs.AsBool() || rhs.AsBool());
  case Op::BitAnd: return Value::MakeInt(l & r);
  case Op::BitOr: return Value::MakeInt(l | r);
  case Op::BitXor: return Value::MakeInt(l ^ r);
  case Op::Shl: return Value::MakeInt(l << r);
  case Op::Shr: return Value::MakeInt(l >> r);
  default: return Value::MakeInt(0);
  }
}

// -----------------------------------------------------------------------
// Unary operator evaluation
// -----------------------------------------------------------------------
Value MockInterpreter::EvalUnaryOp(Opcode op, const Value& operand) {
  auto k = op.GetKind();
  bool is_float = (operand.base_type == BaseType::F32 ||
                   operand.base_type == BaseType::F64);
  switch (k) {
  case Op::Sub:
    if (is_float) return Value::MakeDouble(-operand.AsDouble());
    return Value::MakeInt(-operand.AsInt());
  case Op::LogicNot: return Value::MakeBool(!operand.AsBool());
  case Op::BitNot: return Value::MakeInt(~operand.AsInt());
  case Op::PreInc:
    if (is_float) return Value::MakeDouble(operand.AsDouble() + 1.0);
    return Value::MakeInt(operand.AsInt() + 1);
  case Op::PreDec:
    if (is_float) return Value::MakeDouble(operand.AsDouble() - 1.0);
    return Value::MakeInt(operand.AsInt() - 1);
  case Op::GetUBound: return operand;
  default: return operand;
  }
}

bool MockInterpreter::IsKnownBIF(const std::string& name) const {
  static const std::set<std::string> bifs = {
      "print",   "println",   "assert",   "sizeof", "min",   "max",
      "abs",     "sqrt",      "rsqrt",    "sin",    "cos",   "tan",
      "exp",     "log",       "floor",    "ceil",   "round", "pow",
      "alignup", "aligndown", "isfinite",
  };
  return bifs.count(name) > 0;
}

// -----------------------------------------------------------------------
// Built-in function calls
// -----------------------------------------------------------------------
Value MockInterpreter::CallBIF(const std::string& name,
                               const std::vector<Value>& args,
                               const AST::Call& node) {
  if (name == "print" || name == "println") {
    std::unique_lock<std::mutex> lock;
    if (print_mutex_) lock = std::unique_lock<std::mutex>(*print_mutex_);
    for (size_t i = 0; i < args.size(); ++i) {
      if (i > 0) std::cout << " ";
      std::cout << args[i].ToString();
    }
    if (name == "println") std::cout << "\n";
    return Value::MakeInt(0);
  }

  if (name == "assert") {
    if (args.empty() || !args[0].AsBool())
      Error1(node.LOC(), "mock: assertion failed.");
    return Value::MakeInt(0);
  }

  if (name == "sizeof") {
    if (!args.empty() && args[0].kind == Value::Pointer && args[0].alloc)
      return Value::MakeInt((int64_t)args[0].alloc->TotalBytes());
    return Value::MakeInt(0);
  }

  if (name == "min") {
    if (args.size() >= 2) {
      int64_t a = args[0].AsInt(), b = args[1].AsInt();
      return Value::MakeInt(a < b ? a : b);
    }
    return args.empty() ? Value::MakeInt(0) : args[0];
  }

  if (name == "max") {
    if (args.size() >= 2) {
      int64_t a = args[0].AsInt(), b = args[1].AsInt();
      return Value::MakeInt(a > b ? a : b);
    }
    return args.empty() ? Value::MakeInt(0) : args[0];
  }

  if (name == "abs") {
    if (!args.empty()) {
      if (args[0].base_type == BaseType::F32 ||
          args[0].base_type == BaseType::F64)
        return Value::MakeDouble(std::abs(args[0].AsDouble()));
      return Value::MakeInt(std::abs(args[0].AsInt()));
    }
    return Value::MakeInt(0);
  }

  {
    using MathFn = double (*)(double);
    static const std::map<std::string, MathFn> math_fns = {
        {"sqrt", std::sqrt},   {"sin", std::sin},   {"cos", std::cos},
        {"tan", std::tan},     {"exp", std::exp},   {"log", std::log},
        {"floor", std::floor}, {"ceil", std::ceil}, {"round", std::round},
    };
    auto it = math_fns.find(name);
    if (it != math_fns.end()) {
      if (!args.empty())
        return Value::MakeDouble(it->second(args[0].AsDouble()));
      return Value::MakeDouble(0);
    }
    if (name == "rsqrt") {
      if (!args.empty())
        return Value::MakeDouble(1.0 / std::sqrt(args[0].AsDouble()));
      return Value::MakeDouble(0);
    }
  }

  if (name == "pow") {
    if (args.size() >= 2)
      return Value::MakeDouble(
          std::pow(args[0].AsDouble(), args[1].AsDouble()));
    return Value::MakeDouble(0);
  }

  if (name == "alignup") {
    if (args.size() >= 2) {
      int64_t v = args[0].AsInt(), a = args[1].AsInt();
      return Value::MakeInt(a > 0 ? ((v + a - 1) / a) * a : v);
    }
    return Value::MakeInt(0);
  }

  if (name == "aligndown") {
    if (args.size() >= 2) {
      int64_t v = args[0].AsInt(), a = args[1].AsInt();
      return Value::MakeInt(a > 0 ? (v / a) * a : v);
    }
    return Value::MakeInt(0);
  }

  if (name == "isfinite") {
    if (!args.empty())
      return Value::MakeBool(std::isfinite(args[0].AsDouble()));
    return Value::MakeBool(false);
  }

  Warning(node.LOC(), "mock: unknown BIF '" + name + "', returning 0.");
  return Value::MakeInt(0);
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

BaseType MockInterpreter::ResolveBaseType(const ptr<Type>& ty) const {
  if (!ty) return BaseType::S32;
  if (auto sty = dyn_cast<SpannedType>(ty.get())) return sty->ElementType();
  if (isa<S32Type>(ty.get())) return BaseType::S32;
  if (isa<U32Type>(ty.get())) return BaseType::U32;
  if (isa<S64Type>(ty.get())) return BaseType::S64;
  if (isa<U64Type>(ty.get())) return BaseType::U64;
  if (isa<F32Type>(ty.get())) return BaseType::F32;
  if (isa<F64Type>(ty.get())) return BaseType::F64;
  if (isa<S16Type>(ty.get())) return BaseType::S16;
  if (isa<U16Type>(ty.get())) return BaseType::U16;
  if (isa<S8Type>(ty.get())) return BaseType::S8;
  if (isa<U8Type>(ty.get())) return BaseType::U8;
  if (isa<BooleanType>(ty.get())) return BaseType::BOOL;
  if (isa<F16Type>(ty.get())) return BaseType::F16;
  if (isa<BF16Type>(ty.get())) return BaseType::BF16;
  if (dyn_cast<BoundedIntegerType>(ty.get())) return BaseType::S32;
  if (dyn_cast<BoundedITupleType>(ty.get())) return BaseType::S32;
  return BaseType::S32;
}

std::vector<size_t> MockInterpreter::ResolveShape(const ptr<Type>& ty) const {
  std::vector<size_t> shape;
  if (auto sty = dyn_cast<SpannedType>(ty.get())) {
    auto s = sty->GetShape();
    for (size_t d = 0; d < s.Rank(); ++d) {
      auto dim_val = s.ValueAt(d);
      if (auto iv = VIInt(dim_val))
        shape.push_back((size_t)*iv);
      else
        shape.push_back(1);
    }
  }
  return shape;
}

Storage MockInterpreter::ResolveStorage(const ptr<Type>& ty) const {
  if (auto sty = dyn_cast<SpannedType>(ty.get())) return sty->GetStorage();
  return Storage::LOCAL;
}

size_t
MockInterpreter::ComputeLinearIndex(const std::vector<size_t>& indices,
                                    const std::vector<size_t>& shape) const {
  size_t linear = 0;
  size_t stride = 1;
  for (int i = (int)shape.size() - 1; i >= 0; --i) {
    size_t idx = (i < (int)indices.size()) ? indices[i] : 0;
    linear += idx * stride;
    stride *= shape[i];
  }
  return linear;
}

Value MockInterpreter::CastValue(const Value& v, BaseType target_type) const {
  Value result;
  result.kind = Value::Scalar;
  result.base_type = target_type;

  double dval;
  switch (v.base_type) {
  case BaseType::F64: dval = v.scalar.f64; break;
  case BaseType::F32: dval = (double)v.scalar.f32; break;
  case BaseType::S32:
  case BaseType::S64:
  case BaseType::S16:
  case BaseType::S8: dval = (double)v.scalar.i64; break;
  case BaseType::U32:
  case BaseType::U64:
  case BaseType::U16:
  case BaseType::U8: dval = (double)v.scalar.u64; break;
  case BaseType::BOOL: dval = v.scalar.b ? 1.0 : 0.0; break;
  default: dval = (double)v.scalar.i64; break;
  }

  switch (target_type) {
  case BaseType::F64: result.scalar.f64 = dval; break;
  case BaseType::F32: result.scalar.f32 = (float)dval; break;
  case BaseType::S8:
  case BaseType::S16:
  case BaseType::S32:
  case BaseType::S64: result.scalar.i64 = (int64_t)dval; break;
  case BaseType::U8:
  case BaseType::U16:
  case BaseType::U32:
  case BaseType::U64: result.scalar.u64 = (uint64_t)dval; break;
  case BaseType::BOOL: result.scalar.b = dval != 0; break;
  default: result.scalar.i64 = (int64_t)dval; break;
  }

  return result;
}

} // namespace Mock
} // namespace Choreo
