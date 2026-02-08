#include "types.hpp"
#include "context.hpp"

namespace Choreo {

ValueListRepo Shape::values;

void SpannedType::Print(std::ostream& os) const {
  if (m_type != Storage::NONE && m_type != Storage::DEFAULT)
    os << STR(m_type) << " ";
  os << STR(e_type) << " ";
  s_type->Print(os);
  if (CCtx().ShowStrides() && !strides.empty())
    os << " {" << STR(strides) << "}";
}

} // end namespace Choreo
