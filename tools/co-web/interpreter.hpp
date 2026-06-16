#ifndef __COWEB_INTERPRETER_HPP__
#define __COWEB_INTERPRETER_HPP__

#include "ast.hpp"
#include <map>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace CoWeb {

using Value = std::variant<int64_t, double, std::string, bool>;

struct ArrayData {
  std::vector<double> data;
  std::vector<int64_t> dims;
};

class Interpreter {
public:
  std::string Run(Choreo::AST::Program& root);

private:
  std::ostringstream output_;
  std::map<std::string, Value> vars_;
  std::map<std::string, ArrayData> arrays_;

  void ExecBlock(Choreo::AST::Node& node);
  void ExecNode(Choreo::AST::Node& node);
  void ExecFunction(Choreo::AST::ChoreoFunction& fn);
  void ExecParallel(Choreo::AST::ParallelBy& pb);
  void ExecForeach(Choreo::AST::ForeachBlock& fb);
  void ExecIfElse(Choreo::AST::IfElseBlock& ife);
  void ExecVarDecl(Choreo::AST::NamedVariableDecl& vd);
  void ExecAssign(Choreo::AST::Assignment& asgn);
  void ExecCall(Choreo::AST::Call& call);
  void ExecMemoryDecl(Choreo::AST::Memory& mem);

  Value Eval(Choreo::AST::Node& node);
  Value EvalExpr(Choreo::AST::Expr& expr);
  Value EvalDataAccess(Choreo::AST::DataAccess& da);

  double ToDouble(const Value& v);
  int64_t ToInt(const Value& v);
  std::string ToString(const Value& v);
  bool ToBool(const Value& v);
};

} // namespace CoWeb

#endif // __COWEB_INTERPRETER_HPP__
