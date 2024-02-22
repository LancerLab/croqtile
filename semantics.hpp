#ifndef __CHOREO_SEMANTIC_CHECK_HPP__
#define __CHOREO_SEMANTIC_CHECK_HPP__

#include "visitor.hpp"

namespace Choreo {

struct SemanticChecker : public Visitor {
  //bool Visit(AST::Node&) override { return true; };

	bool Visit(AST::NodeRef&) override { return true; };
	bool Visit(AST::MultiNodes&) override { return true; };
	bool Visit(AST::IntLiteral&) override { return true; };
	bool Visit(AST::IntList&) override { return true; };
	bool Visit(AST::SValList&) override { return true; };
	bool Visit(AST::Expr&) override { return true; };
	bool Visit(AST::MultiSpans&) override { return true; };
	bool Visit(AST::NamedDecl&) override { return true; };
	bool Visit(AST::IntTuple&) override { return true; };
	bool Visit(AST::Assignment&) override { return true; };
	bool Visit(AST::IntIndex&) override { return true; };
	bool Visit(AST::NthBound&) override { return true; };
	bool Visit(AST::IntIndexList&) override { return true; };
	bool Visit(AST::DataType&) override { return true; };
	bool Visit(AST::Identifier&) override { return true; };
	bool Visit(AST::ParamList&) override { return true; };
	bool Visit(AST::ParallelBy&) override { return true; };
	bool Visit(AST::RequireBind&) override { return true; };
	bool Visit(AST::WithIn&) override { return true; };
	bool Visit(AST::WithBlock&) override { return true; };
	bool Visit(AST::Memory&) override { return true; };
	bool Visit(AST::DMA&) override { return true; };
	bool Visit(AST::ChunkAt&) override { return true; };
	bool Visit(AST::Wait&) override { return true; };
	bool Visit(AST::Call&) override { return true; };
	bool Visit(AST::ForeachBlock&) override { return true; };
	bool Visit(AST::FunctionDecl&) override { return true; };
	bool Visit(AST::ChoreoFunction&) override { return true; };
	bool Visit(AST::CppSourceCode&) override { return true; };
	bool Visit(AST::Program&) override { return true; };
};

} // end namespace Choreo
  //
#endif // __CHOREO_SEMANTIC_CHECK_HPP__
