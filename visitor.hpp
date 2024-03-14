#ifndef __CHOREO_VISITOR_HPP__
#define __CHOREO_VISITOR_HPP__

#include <unordered_set>
#include "ast.hpp"

namespace Choreo {

struct Visitor {
  //virtual bool Visit(AST::Node&) = 0;
  virtual bool BeforeVisit(AST::Node &) { return 0; }
  virtual bool AfterVisit(AST::Node &) { return 0; }

  // For any visitor, it should implement all the necessary steps
	virtual bool Visit(AST::MultiNodes&) = 0;
	virtual bool Visit(AST::IntLiteral&) = 0;
	virtual bool Visit(AST::IntList&) = 0;
	virtual bool Visit(AST::SValList&) = 0;
	virtual bool Visit(AST::Expr&) = 0;
	virtual bool Visit(AST::MultiDimSpans&) = 0;
	virtual bool Visit(AST::NamedTypeDecl&) = 0;
	virtual bool Visit(AST::NamedVariableDecl&) = 0;
	virtual bool Visit(AST::IntTuple&) = 0;
	virtual bool Visit(AST::Assignment&) = 0;
	virtual bool Visit(AST::IntIndex&) = 0;
	virtual bool Visit(AST::NthBound&) = 0;
	virtual bool Visit(AST::IntIndexList&) = 0;
	virtual bool Visit(AST::DataType&) = 0;
	virtual bool Visit(AST::Identifier&) = 0;
	virtual bool Visit(AST::ParamList&) = 0;
	virtual bool Visit(AST::ParallelBy&) = 0;
	virtual bool Visit(AST::RequireBind&) = 0;
	virtual bool Visit(AST::WithIn&) = 0;
	virtual bool Visit(AST::WithBlock&) = 0;
	virtual bool Visit(AST::Memory&) = 0;
	virtual bool Visit(AST::DMA&) = 0;
	virtual bool Visit(AST::ChunkAt&) = 0;
	virtual bool Visit(AST::Wait&) = 0;
	virtual bool Visit(AST::Call&) = 0;
	virtual bool Visit(AST::ForeachBlock&) = 0;
	virtual bool Visit(AST::FunctionDecl&) = 0;
	virtual bool Visit(AST::ChoreoFunction&) = 0;
	virtual bool Visit(AST::CppSourceCode&) = 0;
	virtual bool Visit(AST::Program&) = 0;

  // general scoped variable handling
	std::vector<std::unordered_set<std::string>> scopeStack;

	virtual void enterScope() {
		scopeStack.emplace_back(); // Push a new scope
	}

	virtual void leaveScope() {
		if (!scopeStack.empty()) {
			scopeStack.pop_back(); // Pop the last scope
		}
	}

	virtual bool isDeclared(const std::string& varName) {
		// Iterate in reverse order to simulate stack behavior
		for (auto it = scopeStack.rbegin(); it != scopeStack.rend(); ++it) {
			if (it->find(varName) != it->end()) {
				return true; // Found varName in the current or an enclosing scope
			}
		}
		return false; // varName not found in any scope
	}

	virtual void declareVariable(const std::string& varName) {
		if (!scopeStack.empty()) {
			scopeStack.back().insert(varName); // Insert into the current (top) scope
		}
	}
};

} // end namespace Choreo

#endif // __CHOREO_VISITOR_HPP__

