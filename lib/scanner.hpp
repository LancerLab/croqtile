/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2014 Krzysztof Narkiewicz <krzysztof.narkiewicz@ezaquarii.com>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef __CHOREO_SCANNER_H__
#define __CHOREO_SCANNER_H__

/**
 * Generated Flex class name is yyFlexLexer by default. If we want to use more
 * flex-generated classes we should name them differently. See scanner.l prefix
 * option.
 *
 * Unfortunately the implementation relies on this trick with redefining class
 * name with a preprocessor macro. See GNU Flex manual, "Generating C++
 * Scanners" section
 */
#if !defined(yyFlexLexerOnce)
#undef yyFlexLexer
#define yyFlexLexer                                                            \
  Choreo_FlexLexer // the trick with prefix; no namespace here :(
#include <FlexLexer.h>
#endif

// Scanner method signature is defined by this macro. Original yylex() returns
// int. Since Bison 3 uses symbol_type, we must change returned type. We also
// rename it to something sane, since you cannot overload return type.
#undef YY_DECL
#define YY_DECL Choreo::Parser::symbol_type Choreo::Scanner::get_next_token()

#include <cstring>
#include <unistd.h>

#include "parser.tab.hh" // this is needed for symbol_type

namespace Choreo {

// Forward declare interpreter to avoid include. Header is added
// inimplementation file.
// class Interpreter;

class Scanner : public yyFlexLexer {
public:
  //      Scanner(Interpreter &driver) : m_driver(driver) {}
  Scanner() {}
  virtual ~Scanner() {}
  virtual Parser::symbol_type get_next_token();

public:
  static void Debug(std::string s);
  static void SetDebug(bool d = true) { debug = d; }
  static void SetRemoveComments() { keep_comments = false; };
  static bool KeepComments() { return keep_comments; };
  static void SetLocationUpdate(bool u = true) { loc_update = u; };
  static bool LocationUpdate() { return loc_update; };

  void Error(const location& loc, const std::string& error_message) {
    errs() << loc << ": ";
    errs() << ((should_use_colors()) ? color_red : "")
           << "error: " << ((should_use_colors()) ? color_reset : "");
    errs() << error_message << std::endl;
    std::exit(EXIT_FAILURE); // Terminate the program immediately
  }

private:
  static bool debug;
  static bool keep_comments;
  static bool loc_update;

  const char* color_red = "\033[31m";
  const char* color_reset = "\033[0m";

  static bool shell_supports_colors() {
    const char* term = getenv("TERM");
    return term &&
           (strcmp(term, "xterm-256color") == 0 || strcmp(term, "xterm") == 0);
  }

  static bool should_use_colors() {
    return isatty(fileno(stdout)) && shell_supports_colors();
  }

  // private:
  //    Interpreter &m_driver;
};

} // namespace Choreo

#endif // __CHOREO_SCANNER_H__
