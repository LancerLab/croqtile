#include "topscc_preprocess.hpp"
#include "choreo_header.inc"
#include "choreo_types_header.inc"
#include "context.hpp"
#include <filesystem>

#ifndef __CHOREO_TOPSCC_DIR__
  #error "missing macro definition of __CHOREO_TOPSCC_DIR__"
#endif

using namespace Choreo;

static inline bool isIdent(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static inline const std::string UniquePath() {
  // Get a high-resolution timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();

  // Convert timestamp to a more granular unit, like nanoseconds
  auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

  // Get the thread or process ID
  std::stringstream ss;
  ss << std::this_thread::get_id();
  std::string thread_id = ss.str();

  // Construct the path
  std::string path = "/tmp/" + std::to_string(nanoseconds) + "_" + thread_id;

  return path;
}

// Advance i while predicate holds; returns new index
template <class Pred>
static inline size_t advance_while(const std::string& s, size_t i, Pred p) {
  while (i < s.size() && p(s[i])) ++i;
  return i;
}

// Skip whitespace and comments starting at i. Returns index of first
// non-space/comment char.
static size_t skip_ws_and_comments(const std::string& s, size_t i) {
  for (;;) {
    // skip whitespace
    i = advance_while(s, i, [](char c) {
      return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' ||
             c == '\v';
    });
    if (i >= s.size()) return i;
    // line comment
    if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '/') {
      i += 2;
      while (i < s.size() && s[i] != '\n') ++i;
      continue;
    }
    // block comment
    if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '*') {
      i += 2;
      while (i + 1 < s.size() && !(s[i] == '*' && s[i + 1] == '/')) ++i;
      if (i + 1 < s.size()) i += 2;
      continue;
    }
    break;
  }
  return i;
}

static void skip_raw_string(const std::string& s, size_t& i) {
  // assumes s[i] == 'R' and s[i+1] == '"'
  i += 2; // position at start of delimiter
  std::string delim;
  while (i < s.size() && s[i] != '(') {
    delim.push_back(s[i]);
    ++i;
  }
  if (i < s.size() && s[i] == '(') ++i; // consume '('
  // find ")delim"
  for (;;) {
    if (i >= s.size()) return; // unmatched, give up
    if (s[i] == ')') {
      size_t j = i + 1;
      // check for closing delimiter delim then '"'
      if (j + delim.size() < s.size() &&
          s.compare(j, delim.size(), delim) == 0 &&
          j + delim.size() < s.size() && s[j + delim.size()] == '"') {
        i = j + delim.size() + 1; // past closing '"'
        return;
      }
    }
    ++i;
  }
}

static void skip_char_literal(const std::string& s, size_t& i) {
  // assumes s[i] == '\''
  ++i;
  while (i < s.size()) {
    if (s[i] == '\\') {
      i += 2;
      continue;
    }
    if (s[i] == '\'') {
      ++i;
      break;
    }
    ++i;
  }
}

static void skip_string_literal(const std::string& s, size_t& i) {
  // assumes s[i] == '"'
  ++i; // past opening quote
  while (i < s.size()) {
    if (s[i] == '\\') {
      i += 2;
      continue;
    }
    if (s[i] == '"') {
      ++i;
      break;
    }
    ++i;
  }
}

bool TopsccPreprocess::ExtractDeviceKernel(std::ostream& cok_ss) {
#ifdef __EMSCRIPTEN__
  return Preprocess::ExtractDeviceKernel(cok_ss);
#else
  char temp_script_file_name[] = "/tmp/choreo_topscc_script_XXXXXX";

  int script_fd = mkstemp(temp_script_file_name);
  if (script_fd == -1) {
    errs() << "Cannot create temporary file.\n";
    return false;
  }
  close(script_fd);

  std::ofstream temp_script_file(temp_script_file_name);
  if (!temp_script_file.is_open()) {
    errs() << "Cannot open temporary file.\n";
    return false;
  }

  auto filename = RemoveDirectoryPrefix(
      RemoveSuffix(OptionRegistry::GetInstance().GetInputFileName(), ".co"));
  build_path = UniquePath();

  cc_file = build_path + "/__choreo_topscc_" + filename + ".cpp";
  pp_file = build_path + "/__choreo_topscc_" + filename + ".i";

  std::string line;
  bool has_include = false;
  for (auto code_line : include_lines) {
    if (isDirective(code_line, "#include", false)) {
      has_include = true;
      break;
    }
  }

  if (!has_include && cok_codes.empty()) {
    if (debug) dbgs() << "No #include found and no __cok__ region found\n";
    return true;
  }

  EmitScript(temp_script_file);
  temp_script_file.close();

  std::string cmd =
      "bash " + std::string(temp_script_file_name) + " 2>/dev/null";
  int ret = system(cmd.c_str());
  if (ret != 0) {
    if (debug) dbgs() << "Command failed: " << cmd << "\n";
    remove(temp_script_file_name);
    return Preprocess::ExtractDeviceKernel(cok_ss);
  }

  if (remove(temp_script_file_name) != 0) {
    errs() << "Cannot remove temporary file.\n";
    return false;
  }

  // scan the pp_file to extract code inside __cok__ { ... }
  std::ifstream pp_ifs(pp_file);
  if (!pp_ifs.is_open()) {
    errs() << "Cannot open preprocessed file: " << pp_file << "\n";
    return false;
  }
  std::string pp_code((std::istreambuf_iterator<char>(pp_ifs)),
                      std::istreambuf_iterator<char>());

  // find bundles between "__CLANG_OFFLOAD_BUNDLE____START__
  // tops-dtu-enflame-topsXXX" and
  // "__CLANG_OFFLOAD_BUNDLE____END__ tops-dtu-enflame-tops"
  auto bundles = find_bundles(pp_code);
  if (bundles.empty()) return true;
  std::vector<std::string> device_codes;
  for (auto& b : bundles) {
    std::string sub = pp_code.substr(b.start, b.end - b.start);
    auto ranges = extract_cok_sections(sub);
    for (auto& r : ranges)
      device_codes.push_back(sub.substr(r.start, r.end - r.start + 1));
  }

  cok_ss << "__real_cok__ {\n";
  for (auto& c : device_codes) { cok_ss << c << "\n"; }
  cok_ss << "}\n\n";
  return true;
#endif
}

std::vector<Bundle> TopsccPreprocess::find_bundles(const std::string& code) {
  std::vector<Bundle> bundles;
  std::istringstream iss(code);
  std::string line;
  size_t offset = 0;
  size_t start_offset = std::string::npos;
  while (getline(iss, line)) {
    // include newline length
    size_t line_len = line.size() + 1;
    if (line.rfind("// __CLANG_OFFLOAD_BUNDLE____START__ tops-dtu-enflame-tops",
                   0) == 0) {
      start_offset = offset + line_len; // content starts after this line
    } else if (line.rfind(
                   "// __CLANG_OFFLOAD_BUNDLE____END__ tops-dtu-enflame-tops",
                   0) == 0) {
      if (start_offset != std::string::npos) {
        bundles.push_back(
            {start_offset, offset}); // content ends before this line
        start_offset = std::string::npos;
      }
    }
    offset += line_len;
  }
  return bundles;
}

void TopsccPreprocess::EmitScript(std::ostream& os) {
  os << R"script(#!/usr/bin/env bash

# This is the choreo generated bash script to compile topscc code
if [[ -z ${TOPSCC_INSTALL} ]]; then
  if [[ \"$1\" == \"-st\" ]]; then
    TOPSCC_INSTALL=/opt/tops;
    shift 1;
  fi
)script";

  os << "  TOPSCC_INSTALL=" << STRINGIZE(__CHOREO_TOPSCC_DIR__) << "\n";
  os << R"script(
fi
if [[ -z "${TOPSCC_INSTALL}" ]]; then
echo "failed to find the topscc installation."
echo "install topscc or set TOPSCC_INSTALL to topscc installation directory."
exit 1
fi

TOPSCC=${TOPSCC_INSTALL}/bin/topscc
TOPSCC_LIB=${TOPSCC_INSTALL}/lib
)script";
  os << "rm -fr " << build_path << "\n";
  os << "mkdir -p " << build_path << "\n\n";
  os << "cat <<'EOF' > " << build_path << "/choreo.h\n";
  os << __choreo_header_as_string << "\nEOF\n\n";
  os << "cat <<'EOF' > " << build_path << "/choreo_types.h\n";
  os << __choreo_types_header_as_string << "\nEOF\n\n";
  os << "cat <<'EOF' > " << cc_file << "\n";
  for (auto& line : include_lines) { os << line << "\n"; }
  if (!cok_codes.empty()) {
    os << "\n__cok__ {\n";
    for (auto& line : cok_codes) { os << line << "\n"; }
    os << "}\n";
  }
  os << "\nEOF\n\n";
  os << R"(export CFLAGS=")";
  std::filesystem::path cwd = std::filesystem::current_path();
  const std::string input_file =
      OptionRegistry::GetInstance().GetInputFileName();
  std::filesystem::path input_rel_path(input_file);
  auto input_abs_path = std::filesystem::weakly_canonical(cwd / input_rel_path)
                            .parent_path()
                            .string();
  os << " -I" << input_abs_path;
  for (auto inc_path : CCtx().GetIncPaths()) os << " -I" << inc_path;
  for (auto lib_path : CCtx().GetLibPaths()) os << " -L" << lib_path;
  os << "\"";

  os << "\n ${TOPSCC} -E ${CFLAGS}"
     << " " << cc_file << " -o " << pp_file << "\n";
}

std::vector<Range>
TopsccPreprocess::extract_cok_sections(const std::string& code,
                                       size_t base_offset) {
  std::vector<Range> ranges;
  const std::string token = "__cok__";
  size_t i = 0, N = code.size();

  while (i < N) {
    char c = code[i];

    // skip comments/strings
    if (c == '/' && i + 1 < N) {
      if (code[i + 1] == '/') {
        i += 2;
        while (i < N && code[i] != '\n') ++i;
        continue;
      }
      if (code[i + 1] == '*') {
        i += 2;
        while (i + 1 < N && !(code[i] == '*' && code[i + 1] == '/')) ++i;
        if (i + 1 < N) i += 2;
        continue;
      }
    }
    if (c == '"') {
      skip_string_literal(code, i);
      continue;
    }
    if (c == '\'') {
      skip_char_literal(code, i);
      continue;
    }
    if (c == 'R' && i + 1 < N && code[i + 1] == '"') {
      skip_raw_string(code, i);
      continue;
    }

    // found __cok__
    if (c == '_' && i + token.size() <= N &&
        code.compare(i, token.size(), token) == 0) {
      bool left_ok = (i == 0) || !isIdent(code[i - 1]);
      bool right_ok =
          (i + token.size() >= N) || !isIdent(code[i + token.size()]);
      if (left_ok && right_ok) {
        size_t j = i + token.size();
        j = skip_ws_and_comments(code, j);
        if (j < N && code[j] == '{') {
          size_t block_start = j + 1;
          int depth = 1;
          ++j;

          size_t content_start = block_start;
          while (j < N && depth > 0) {
            char d = code[j];

            // handle nested comments/strings
            if (d == '/' && j + 1 < N) {
              if (code[j + 1] == '/') {
                j += 2;
                while (j < N && code[j] != '\n') ++j;
                continue;
              }
              if (code[j + 1] == '*') {
                j += 2;
                while (j + 1 < N && !(code[j] == '*' && code[j + 1] == '/'))
                  ++j;
                if (j + 1 < N) j += 2;
                continue;
              }
            }
            if (d == '"') {
              skip_string_literal(code, j);
              continue;
            }
            if (d == '\'') {
              skip_char_literal(code, j);
              continue;
            }
            if (d == 'R' && j + 1 < N && code[j + 1] == '"') {
              skip_raw_string(code, j);
              continue;
            }

            // detect nested __cok__
            if (d == '_' && j + token.size() <= N &&
                code.compare(j, token.size(), token) == 0) {
              // flush parent content before nested
              if (j > content_start) {
                ranges.push_back(
                    {base_offset + content_start, base_offset + j - 1});
              }
              // recursively extract inner __cok__ blocks
              auto inner =
                  extract_cok_sections(code.substr(j), base_offset + j);
              ranges.insert(ranges.end(), inner.begin(), inner.end());

              // skip whole nested block
              return ranges; // stop here, recursion will handle rest
            }

            if (d == '{') {
              ++depth;
              ++j;
              continue;
            }
            if (d == '}') {
              --depth;
              ++j;
              continue;
            }
            ++j;
          }
          if (depth == 0 && j > block_start) {
            ranges.push_back({base_offset + block_start, base_offset + j - 2});
          }
          i = j;
          continue;
        }
      }
    }
    ++i;
  }
  return ranges;
}
