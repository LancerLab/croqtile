#ifndef __CHOREO_LOCATION_HH__
#define __CHOREO_LOCATION_HH__

#include <string>

namespace Choreo {

class position {
public:
  using filename_type = std::string; // Use std::string directly
  /// Type for line and column numbers.
  using counter_type = int;

  explicit position(const filename_type& file = "", counter_type line = 1,
                    counter_type column = 1)
      : filename(file), line(line), column(column) {}

  void initialize(const filename_type& file, counter_type line,
                  counter_type column) {
    filename = file; // Copy the filename
    this->line = line;
    this->column = column;
  }

  /** \name Line and Column related manipulators
   ** \{ */
  /// (line related) Advance to the COUNT next lines.
  void lines(counter_type count = 1) {
    if (count) {
      column = 1;
      line = add_(line, count, 1);
    }
  }

  /// (column related) Advance to the COUNT next columns.
  void columns(counter_type count = 1) { column = add_(column, count, 1); }
  const filename_type& get_filename() const { return filename; }
  counter_type get_line() const { return line; }
  counter_type get_column() const { return column; }

public:
  filename_type filename;
  counter_type line;
  counter_type column;
  /// Compute max (min, lhs+rhs).
  static counter_type add_(counter_type lhs, counter_type rhs,
                           counter_type min) {
    return lhs + rhs < min ? min : lhs + rhs;
  }
};

/// Add \a width columns, in place.
inline position& operator+=(position& res, position::counter_type width) {
  res.columns(width);
  return res;
}

/// Add \a width columns.
inline position operator+(position res, position::counter_type width) {
  return res += width;
}

/// Subtract \a width columns, in place.
inline position& operator-=(position& res, position::counter_type width) {
  return res += -width;
}

/// Subtract \a width columns.
inline position operator-(position res, position::counter_type width) {
  return res -= width;
}

/** \brief Intercept output stream redirection.
 ** \param ostr the destination output stream
 ** \param pos a reference to the position to redirect
 */
template <typename YYChar>
inline std::basic_ostream<YYChar>& operator<<(std::basic_ostream<YYChar>& ostr,
                                              const position& pos) {
  if (!pos.filename.empty()) ostr << pos.filename << ':';
  return ostr << pos.line << '.' << pos.column;
}

class location {
public:
  /// Type for file name.
  typedef position::filename_type filename_type;
  /// Type for line and column numbers.
  typedef position::counter_type counter_type;

public:
  position begin;
  position end;

public:
  location(const position& b, const position& e) : begin(b), end(e) {}

  /// Construct a 0-width location in \a p.
  explicit location(const position& p = position()) : begin(p), end(p) {}

  /// Construct a 0-width location in \a f, \a l, \a c.
  explicit location(filename_type f, counter_type l = 1, counter_type c = 1)
      : begin(f, l, c), end(f, l, c) {}
  void initialize(const position::filename_type& file, counter_type line,
                  counter_type column) {
    begin.initialize(file, line, column);
    end = begin;
  }
  void step() { begin = end; }

  /// Extend the current location to the COUNT next columns.
  void columns(counter_type count = 1) { end += count; }

  /// Extend the current location to the COUNT next lines.
  void lines(counter_type count = 1) { end.lines(count); }
};

template <typename YYChar>
inline std::basic_ostream<YYChar>& operator<<(std::basic_ostream<YYChar>& ostr,
                                              const location& loc) {
  return ostr << loc.begin; // Print only the beginning position for simplicity
}

} // end namespace Choreo

#endif // __CHOREO_LOCATION_HH__
