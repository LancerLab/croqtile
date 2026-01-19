#ifndef __CHOREO_TOPSCC_PREPROCESS_HPP__
#define __CHOREO_TOPSCC_PREPROCESS_HPP__

#include "aux.hpp"
#include "io.hpp"
#include "preprocess.hpp"
#include "unistd.h"
#include <istream>
#include <regex>
#include <sstream>
#include <thread>

namespace Choreo {

class TopsccPreprocess : public Preprocess {
public:
  TopsccPreprocess(std::ostream& o) : Preprocess(o) {};

public:
  bool ExtractDeviceKernel(std::ostream&) override;

private:
  void EmitScript(std::ostream& os);
  std::vector<Range> extract_cok_sections(const std::string& code,
                                          size_t base_offset = 0);
  std::vector<Bundle> find_bundles(const std::string& code);
}; // TopsccPreprocess

} // end namespace Choreo

#endif // __CHOREO_TOPSCC_PREPROCESS_HPP__
