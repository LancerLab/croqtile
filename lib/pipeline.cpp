#include "pipeline.hpp"

using namespace Choreo;

void ASTPipeline::Dump() const {
  dbgs() << "++ Pipeline Stages\n";
  for (auto& ps : pl)
    if ((ps.pred && ps.pred()) || !ps.pred)
      if (ps.v) dbgs() << " |-" << ps.v->GetName() << "\n";

  dbgs() << "++ END Pipeline\n";
}

bool ASTPipeline::RunOnProgram(AST::Node& root) {
  if (debug) Dump();
  // verify the input
  if (verify) vf.RunOnProgram(root);

  for (auto& ps : pl) {
    if ((ps.pred && ps.pred()) || !ps.pred) {
      if (ps.v) {
        if (!ps.v->RunOnProgram(root)) {
          state = ps.v->Status();
          return false; // abend immediately
        }
        symtab = ps.v->SymTab();
        // TODO: force abend when failing on verifiers
        if (verify) vf.RunOnProgram(root);
      }
      if (ps.cond_action) ps.cond_action(*this);
    }

    if (abend) return false;
    if (ps.action) ps.action(*this);
  }
  return !abend;
}
