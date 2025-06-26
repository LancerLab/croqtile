#ifndef __CHOREO_NODE_VERIFIER__
#define __CHOREO_NODE_VERIFIER__

#include "visitor.hpp"

namespace Choreo {

struct ASTVerify : public VisitorWithScope {
  ASTVerify() : VisitorWithScope("astverify") {}

  bool BeforeVisitImpl(AST::Node& n) override {
    if (n.IsBlock()) return true;
    if (isa<AST::Program>(&n) || isa<AST::ChoreoFunction>(&n) ||
        isa<AST::Wait>(&n) || isa<AST::Call>(&n) || isa<AST::Return>(&n) ||
        isa<AST::Rotate>(&n) || isa<AST::Identifier>(&n))
      return true;

    if (auto c = dyn_cast<AST::Call>(&n)) {
      if (!c->IsBIF()) {
        if (c->IsArith())
          dbgs() << "can not annotate non-bif as arithmetic.\n";
        else if (c->CompileTimeEval())
          dbgs() << "can not evaluate a non-bif at compile-time.\n";
      }
    }

    if (n.GetType() == nullptr) {
      choreo_unreachable("[" + n.TypeNameString() +
                         "] is not typed: " + STR(n));
    } else if (n.GetType()->GetBaseType() == BaseType::UNKNOWN)
      dbgs() << "[" << n.TypeNameString() << "] is unknown: " << STR(n) << "\n";

    if (debug_visit)
      errs() << "[" << n.TypeNameString() << "] type: " << PSTR(NodeType(n))
             << ".\n";

    return true;
  }

  bool AfterVisitImpl(AST::Node&) override { return true; }
};

} // end namespace Choreo

#endif // __CHOREO_NODE_VERIFIER__
