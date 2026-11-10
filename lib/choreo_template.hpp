#ifndef __CHOREO_TEMPLATE_HPP__
#define __CHOREO_TEMPLATE_HPP__

#include <cstddef>
#include <iosfwd>
#include <string>

namespace Choreo {

struct ChoreoTemplateInstanceInfo {
  std::string display_name;
  std::string filename;
  size_t request_line = 0;
  size_t request_column = 0;
  size_t definition_begin_line = 0;
  size_t definition_end_line = 0;
};

// Expand explicitly instantiated Choreo function templates into ordinary
// __co__ functions. The ordinary parser and semantic pipeline only see the
// resulting concrete functions.
bool ExpandChoreoTemplates(const std::string& input, std::ostream& output,
                           const std::string& filename = {});

// The line-oriented preprocessor must not mistake an explicit Choreo template
// instantiation for the beginning of a function body.
bool IsChoreoTemplateInstantiation(const std::string& line);

// Semantic passes use this metadata to add the explicit-instantiation site to
// diagnostics emitted while visiting a generated concrete function.
const ChoreoTemplateInstanceInfo*
FindChoreoTemplateInstance(const std::string& internal_name);

void SetActiveChoreoTemplateInstance(const std::string& internal_name);
const ChoreoTemplateInstanceInfo*
FindActiveChoreoTemplateInstance(const std::string& filename, size_t line);

} // namespace Choreo

#endif // __CHOREO_TEMPLATE_HPP__
