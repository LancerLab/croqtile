#ifndef __CHOREO_AUX_HPP__
#define __CHOREO_AUX_HPP__

#define STRINGIZE_HELPER(x) #x
#define STRINGIZE(x) STRINGIZE_HELPER(x)

#include <cassert>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <vector>

#if 0
[[noreturn]] inline void choreo_unreachable_impl(
    const char* file, int line, const char* msg = "Unreachable code reached") {
  std::cerr << file << ":" << line << ": Assertion failed: ";
  std::cerr << msg << std::endl;
  std::abort();
}
#endif

[[noreturn]] inline void
choreo_unreachable_impl(const char* file, int line,
                        const std::string& msg = "Unreachable code reached") {
  std::cerr << file << ":" << line << ": Assertion failed: ";
  std::cerr << msg << std::endl;
  std::abort();
}

// Note: __VA_OPT__ requires C++20. But it is available starting from gcc-8.1
// and clang-6. This pre-requisition should be satisfied.

#if defined(__GNUC__) && !defined(__clang__)
  #if (__GNUC__ < 8) || (__GNUC__ == 8 && __GNUC_MINOR__ < 1)
    #error "GCC version must be at least 8.1"
  #endif
#endif

#if defined(__clang__)
  #if (__clang_major__ < 6)
    #error "Clang version must be at least 6"
  #endif
#endif

// Macro that captures the file and line
#define choreo_unreachable(...)                                                \
  choreo_unreachable_impl(__FILE__, __LINE__ __VA_OPT__(, ) __VA_ARGS__)

template <typename Container>
inline static std::string DelimitedString(const Container& container,
                                          std::string delimiter = ", ") {
  std::ostringstream oss;
  auto it = container.begin();
  if (it != container.end()) {
    if constexpr (std::is_same_v<typename Container::value_type, std::string>)
      oss << *it;
    else
      oss << std::to_string(*it);
    ++it;
  }
  for (; it != container.end(); ++it) {
    if constexpr (std::is_same_v<typename Container::value_type, std::string>)
      oss << delimiter << *it;
    else
      oss << delimiter << std::to_string(*it);
  }
  return oss.str();
}

template <typename Container>
inline static std::string DelimitedSTR(const Container& container,
                                       std::string delimiter = ", ") {
  std::ostringstream oss;
  auto it = container.begin();
  if (it != container.end()) {
    oss << STR(*it);
    ++it;
  }
  for (; it != container.end(); ++it) oss << delimiter << STR(*it);
  return oss.str();
}

// split `input` to a vector
inline static std::vector<std::string>
SplitStringByDelimiter(std::string input, std::string delimiter = ",",
                       bool trim = true) {
  std::vector<std::string> tokens;
  size_t pos = 0;
  std::string token;
  while ((pos = input.find(delimiter)) != std::string::npos) {
    token = input.substr(0, pos);
    if (!token.empty()) tokens.push_back(token);
    input.erase(0, pos + delimiter.length());
  }
  if (!input.empty()) tokens.push_back(input);
  if (!trim) return tokens;
  for (auto& token : tokens)
    token = std::regex_replace(token, std::regex("^\\s+|\\s+$"), "");
  return tokens;
}

// Function to check if 'str' starts with 'prefix'
inline static bool PrefixedWith(const std::string& str,
                                const std::string& prefix) {
  if (prefix.size() > str.size()) return false;
  return str.compare(0, prefix.size(), prefix) == 0;
}

// Function to check if 'str' ends with 'suffix'
inline static bool SuffixedWith(const std::string& str,
                                const std::string& suffix) {
  if (suffix.size() > str.size()) return false;
  return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline static std::optional<std::string>
RemovePrefixOrNull(const std::string& prefix, const std::string& str) {
  if (str.find(prefix) == 0)            // Check if 'prefix' is at the beginning
    return str.substr(prefix.length()); // Return the substring after 'prefix'
  else
    return std::nullopt; // Return an empty string if 'prefix' is not at the
                         // beginning
}

// remove suffix
inline static std::string RemoveSuffix(const std::string& str,
                                       const std::string& suffix) {
  if (suffix.size() > str.size()) return str;
  if (str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0)
    return str.substr(0, str.size() - suffix.size());
  else
    return str;
}

inline static std::string Ordinal(int n) {
  assert(n > 0);
  static const char* suffixes[] = {"th", "st", "nd", "rd", "th"};
  int v = n % 100;
  int index = (v >= 11 && v <= 13) ? 0 : std::min(v % 10, 4);
  return std::to_string(n) + suffixes[index];
}

inline static bool ContainsExact(const std::string& str,
                                 const std::string& pattern) {
  std::regex wordBoundaryPattern("\\b" + pattern + "\\b");
  return std::regex_search(str, wordBoundaryPattern);
}

inline static std::string RegexReplaceAll(const std::string& input,
                                          const std::string& pattern,
                                          const std::string& replacement) {
  std::regex regex_pattern(pattern);
  return std::regex_replace(input, regex_pattern, replacement);
}

inline static std::string SearchPattern(const std::string& input,
                                        const std::string& pattern) {
  std::regex re(pattern);
  std::smatch match;
  if (std::regex_search(input, match, re)) return match.str();
  return ""; // or throw an exception if no match is found
}

// Convert string(1.5f) to float
inline static std::optional<float> Str2Float(std::string str) {
  auto res = SearchPattern(str, R"(^[-]?[0-9]+\.[0-9]+f$)");
  if (res == "") return std::nullopt;
  return std::stof(str);
}

// Convert string(1.5) to double
inline static std::optional<double> Str2Double(const std::string& str) {
  auto res = SearchPattern(str, R"(^[-]?[0-9]+\.[0-9]+$)");
  if (res == "") return std::nullopt;
  return std::stod(str);
}

inline static bool IsInteger(const std::string& str) {
  std::istringstream iss(str);
  int num;
  return iss >> num && iss.eof();
}

inline static const std::string ToUpper(const std::string& s) {
  std::string r(s.size(), '\0');
  transform(s.begin(), s.end(), r.begin(), ::toupper);
  return r;
}

inline static const std::string ToLower(const std::string& s) {
  std::string r(s.size(), '\0');
  transform(s.begin(), s.end(), r.begin(), ::tolower);
  return r;
}

inline const std::string RemoveDirectoryPrefix(const std::string& path) {
  size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) { return path; }
  return path.substr(pos + 1);
}

// A range class since we lack c++20 range
template <typename T>
class FilterRange {
private:
  std::vector<T>& vec;
  std::function<bool(const T&)> predicate;

public:
  FilterRange(std::vector<T>& vec, std::function<bool(const T&)> pred)
      : vec(vec), predicate(pred) {}

  class Iterator {
  private:
    using viter = typename std::vector<T>::iterator;
    viter current;
    viter begin;
    viter end;
    std::function<bool(const T&)> predicate;

    void skip_to_next_valid() {
      while (current != end && !predicate(*current)) { ++current; }
    }

    void skip_to_prev_valid() {
      while (current != begin && !predicate(*current)) { --current; }
      // If we reached begin and it's not valid, move to end
      if (current == begin && !predicate(*current)) { current = end; }
    }

  public:
    Iterator(viter current, viter begin, viter end,
             std::function<bool(const T&)> pred)
        : current(current), begin(begin), end(end), predicate(pred) {
      if (current != end)
        if (!predicate(*current)) skip_to_next_valid();
    }

    Iterator& operator++() {
      if (current != end) {
        ++current;
        skip_to_next_valid();
      }
      return *this;
    }

    Iterator& operator--() {
      if (current != begin) {
        --current;
        skip_to_prev_valid();
      }
      return *this;
    }

    T& operator*() {
      if (current == end) choreo_unreachable("Dereferencing empty iterator");
      return *current;
    }

    T* operator->() {
      if (current == end) choreo_unreachable("Dereferencing empty iterator");
      return &(*current);
    }

    bool operator==(const Iterator& other) const {
      return current == other.current;
    }
    bool operator!=(const Iterator& other) const { return !(*this == other); }
  };

  Iterator begin() {
    if (vec.empty()) return end();
    return Iterator(vec.begin(), vec.begin(), vec.end(), predicate);
  }

  Iterator end() {
    return Iterator(vec.end(), vec.begin(), vec.end(), predicate);
  }

  Iterator back() {
    if (vec.empty()) return end();
    return --end();
  }

  bool empty() const {
    if (vec.empty()) return true;
    return std::none_of(vec.begin(), vec.end(), predicate);
  }
};

template <typename T>
inline bool Contains(const std::vector<T>& vec, const T& value) {
  return std::find(vec.begin(), vec.end(), value) != vec.end();
}

#endif // __CHOREO_AUX_HPP__
