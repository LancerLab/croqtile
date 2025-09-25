#ifndef CATZILLA_CATZ_COORD_H_
#define CATZILLA_CATZ_COORD_H_

#include <string>
#include <type_traits>

#include "index.h"
#include "trait.h"

namespace catz {

////////////////////////////////////////////////////
#define DEFINE_BINARY_OPERATOR_FOR_COORD(op)                                   \
  template <typename OtherRowType, typename OtherColType>                      \
  constexpr auto operator op(const Coord<OtherRowType, OtherColType> &other)   \
      const {                                                                  \
    auto new_rows = rows op other.rows;                                        \
    auto new_cols = cols op other.cols;                                        \
    return Coord<decltype(new_rows), decltype(new_cols)>(new_rows, new_cols);  \
  }

template <typename RowType, typename ColType>
struct Coord {
  RowType rows;
  ColType cols;

  constexpr Coord(RowType r, ColType c) : rows(r), cols(c) {}

  constexpr bool isStatic() { return (rows.isStatic() && cols.isStatic()); }

  constexpr auto volume() const { return rows() * cols(); }

  DEFINE_BINARY_OPERATOR_FOR_COORD(+)
  DEFINE_BINARY_OPERATOR_FOR_COORD(-)
  DEFINE_BINARY_OPERATOR_FOR_COORD(*)
  DEFINE_BINARY_OPERATOR_FOR_COORD(/)
  DEFINE_BINARY_OPERATOR_FOR_COORD(%)

  template <typename OtherRowType, typename OtherColType>
  constexpr auto
  ceil_div(const Coord<OtherRowType, OtherColType> &other) const {
    auto new_rows = (rows + other.rows - I1()) / other.rows;
    auto new_cols = (cols + other.cols - I1()) / other.cols;
    return Coord<decltype(new_rows), decltype(new_cols)>(new_rows, new_cols);
  }
};

#define make_coord(R, C) Coord(Index<R>(), Index<C>())

#define make_coord_dyn(R, C) Coord(IndexDyn(R), IndexDyn(C))

} // namespace catz

#endif // CATZILLA_CATZ_COORD_H_
