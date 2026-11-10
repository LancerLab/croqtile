#include "choreo_api.hpp"
#include "scanner.hpp"

namespace Choreo {

bool CompilerAPI::Parse(std::stringstream& preprocessed) {
  Scanner s;
  PContext pctx;
  Parser p(pctx, s);
  Scanner::SetRemoveComments();
  Scanner::SetLocationUpdate(true);
  s.yyrestart(preprocessed);
  if (p.parse() != 0 || pctx.HasError()) {
    errs() << "error: parsing failed\n";
    return false;
  }
  return true;
}

} // namespace Choreo
