#ifndef __CHOREO_COLORS_HPP__
#define __CHOREO_COLORS_HPP__

#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

namespace Choreo {
namespace color {

constexpr const char* kReset = "\033[0m";
constexpr const char* kBold = "\033[1m";
constexpr const char* kDim = "\033[2m";
constexpr const char* kRed = "\033[31m";
constexpr const char* kGreen = "\033[32m";
constexpr const char* kYellow = "\033[33m";
constexpr const char* kBlue = "\033[34m";
constexpr const char* kMagenta = "\033[35m";
constexpr const char* kCyan = "\033[36m";
constexpr const char* kBoldGreen = "\033[1;32m";
constexpr const char* kBoldCyan = "\033[1;36m";
constexpr const char* kBoldMagenta = "\033[1;35m";
constexpr const char* kBoldYellow = "\033[1;33m";
constexpr const char* kBoldBlue = "\033[1;34m";
constexpr const char* kBoldRed = "\033[1;31m";

namespace detail {

inline bool termSupportsColor() {
  const char* term = getenv("TERM");
  if (!term) return false;
  return strcmp(term, "dumb") != 0;
}

inline bool computeColorEnabled(int fd) {
  if (!isatty(fd)) return false;
  return termSupportsColor();
}

// Match a keyword at position i that sits at a word boundary.
// Preceding char must be start-of-string, '>', ' ', or '('.
// Following char must be ' ', end-of-string, or '<' (for "mdspan<").
inline bool matchKeyword(const std::string& s, size_t i, const char* kw,
                         size_t kwlen) {
  if (s.compare(i, kwlen, kw) != 0) return false;
  if (i > 0) {
    char prev = s[i - 1];
    if (prev != '>' && prev != ' ' && prev != '(' && prev != ',') return false;
  }
  size_t end = i + kwlen;
  if (end < s.size()) {
    char next = s[end];
    if (next != ' ' && next != '<' && next != ')' && next != ',' &&
        next != '\0')
      return false;
  }
  return true;
}

} // namespace detail

// GCC 10 may duplicate static locals in inline functions under aggressive
// inlining.  Prevent that by keeping the detection out of line.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline))
#endif
inline bool
stdoutHasColor() {
  static bool enabled = detail::computeColorEnabled(STDOUT_FILENO);
  return enabled;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((noinline))
#endif
inline bool
stderrHasColor() {
  static bool enabled = detail::computeColorEnabled(STDERR_FILENO);
  return enabled;
}

inline const char* out(const char* code) {
  return stdoutHasColor() ? code : "";
}

inline const char* err(const char* code) {
  return stderrHasColor() ? code : "";
}

// Syntax-highlight a type string for terminal display.
// Highlights: storage qualifiers (yellow), element types (green),
//             bracket extents (dim).  Returns unmodified string
//             when use_color is false.
inline std::string colorizeType(const std::string& s, bool use_color) {
  if (!use_color) return s;

  std::string r;
  r.reserve(s.size() + 128);
  size_t i = 0, n = s.size();

  auto emit = [&](const char* clr, size_t kwlen) {
    r += clr;
    r.append(s, i, kwlen);
    r += kReset;
    i += kwlen;
  };

  while (i < n) {
    // async=> / sync=> prefixes
    if (s.compare(i, 7, "async=>") == 0) {
      emit(kYellow, 7);
      continue;
    }
    if (s.compare(i, 6, "sync=>") == 0) {
      emit(kYellow, 6);
      continue;
    }

    // Storage qualifiers -> yellow
    using detail::matchKeyword;
    if (matchKeyword(s, i, "global", 6)) {
      emit(kYellow, 6);
      continue;
    }
    if (matchKeyword(s, i, "shared", 6)) {
      emit(kYellow, 6);
      continue;
    }
    if (matchKeyword(s, i, "local", 5)) {
      emit(kYellow, 5);
      continue;
    }
    if (matchKeyword(s, i, "register", 8)) {
      emit(kYellow, 8);
      continue;
    }
    if (matchKeyword(s, i, "node", 4)) {
      emit(kYellow, 4);
      continue;
    }

    // Element types -> green
    if (matchKeyword(s, i, "f8_ue8m0", 8)) {
      emit(kGreen, 8);
      continue;
    }
    if (matchKeyword(s, i, "f8_ue4m3", 8)) {
      emit(kGreen, 8);
      continue;
    }
    if (matchKeyword(s, i, "f8_e4m3", 7)) {
      emit(kGreen, 7);
      continue;
    }
    if (matchKeyword(s, i, "f8_e5m2", 7)) {
      emit(kGreen, 7);
      continue;
    }
    if (matchKeyword(s, i, "f6_e2m3", 7)) {
      emit(kGreen, 7);
      continue;
    }
    if (matchKeyword(s, i, "f6_e3m2", 7)) {
      emit(kGreen, 7);
      continue;
    }
    if (matchKeyword(s, i, "f4_e2m1", 7)) {
      emit(kGreen, 7);
      continue;
    }
    if (matchKeyword(s, i, "bf16", 4)) {
      emit(kGreen, 4);
      continue;
    }
    if (matchKeyword(s, i, "tf32", 4)) {
      emit(kGreen, 4);
      continue;
    }
    if (matchKeyword(s, i, "bin1", 4)) {
      emit(kGreen, 4);
      continue;
    }
    if (matchKeyword(s, i, "bool", 4)) {
      emit(kGreen, 4);
      continue;
    }
    if (matchKeyword(s, i, "void", 4)) {
      emit(kGreen, 4);
      continue;
    }
    if (matchKeyword(s, i, "f64", 3)) {
      emit(kGreen, 3);
      continue;
    }
    if (matchKeyword(s, i, "f32", 3)) {
      emit(kGreen, 3);
      continue;
    }
    if (matchKeyword(s, i, "f16", 3)) {
      emit(kGreen, 3);
      continue;
    }
    if (matchKeyword(s, i, "s64", 3)) {
      emit(kGreen, 3);
      continue;
    }
    if (matchKeyword(s, i, "s32", 3)) {
      emit(kGreen, 3);
      continue;
    }
    if (matchKeyword(s, i, "s16", 3)) {
      emit(kGreen, 3);
      continue;
    }
    if (matchKeyword(s, i, "u64", 3)) {
      emit(kGreen, 3);
      continue;
    }
    if (matchKeyword(s, i, "u32", 3)) {
      emit(kGreen, 3);
      continue;
    }
    if (matchKeyword(s, i, "u16", 3)) {
      emit(kGreen, 3);
      continue;
    }
    if (matchKeyword(s, i, "s8", 2)) {
      emit(kGreen, 2);
      continue;
    }
    if (matchKeyword(s, i, "s6", 2)) {
      emit(kGreen, 2);
      continue;
    }
    if (matchKeyword(s, i, "s4", 2)) {
      emit(kGreen, 2);
      continue;
    }
    if (matchKeyword(s, i, "s2", 2)) {
      emit(kGreen, 2);
      continue;
    }
    if (matchKeyword(s, i, "u8", 2)) {
      emit(kGreen, 2);
      continue;
    }
    if (matchKeyword(s, i, "u6", 2)) {
      emit(kGreen, 2);
      continue;
    }
    if (matchKeyword(s, i, "u4", 2)) {
      emit(kGreen, 2);
      continue;
    }
    if (matchKeyword(s, i, "u2", 2)) {
      emit(kGreen, 2);
      continue;
    }
    if (matchKeyword(s, i, "u1", 2)) {
      emit(kGreen, 2);
      continue;
    }

    // Bracket extents [...] -> dim (handles nesting)
    if (s[i] == '[') {
      int depth = 1;
      r += kDim;
      r += '[';
      i++;
      while (i < n && depth > 0) {
        if (s[i] == '[') depth++;
        if (s[i] == ']') depth--;
        r += s[i++];
      }
      r += kReset;
      continue;
    }

    r += s[i++];
  }
  return r;
}

} // namespace color
} // namespace Choreo

#endif // __CHOREO_COLORS_HPP__
