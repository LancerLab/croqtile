%option c++
%option noyywrap nounput noinput batch debug
%option yyclass="Scanner"
%option prefix="Choreo_"

%{ /* -*- C++ -*- */

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring> // strerror
#include <iostream>
#include <cstdlib>
#include <string>
#include <memory>
#include "ast.hpp"
#include "symtab.hpp"
#include "scanner.hpp"
#include "parser.tab.hh"
#include "location.hh"


static int scope = 0;

extern Choreo::location loc;

void update_location(const char *text) {
  const char* it = text;
  while (*it) {
    if (*it == '\n') {
      loc.lines();
      loc.step(); // Reset column to 0 or 1 based on your implementation
    } else {
      loc.columns();
    }
    ++it;
  }
}

// Original yyterminate() macro returns int. Since we're using Bison 3 variants
// as tokens, we must redefine it to change type from `int` to `Parser::semantic_type`
#define yyterminate() { Choreo::Parser::make_END(loc); break; }


/* update location on each matching */
#define YY_USER_ACTION update_location(yytext);

%}

%x CO_FUNC

id    [a-zA-Z_][a-zA-Z0-9_]*
num   [0-9]+
blank [ \t\r]

%%
<INITIAL>"//".*   {
  return Choreo::Parser::make_CPP_CODE(yytext, loc);
}
<INITIAL>"__co__" {
  BEGIN(CO_FUNC);
  Choreo::Scanner::Debug("Entering Choreo Function");
  return Choreo::Parser::make_ATTR_CO(yytext, loc);
}
<INITIAL>.  {
  Choreo::Scanner::Debug("token (char): " + std::string(yytext));
  return Choreo::Parser::make_CPP_CODE(yytext, loc);
} // This matches everything outside the __co__ functions.
<INITIAL>\n+ {
  Choreo::Scanner::Debug("token (newline)");
  return Choreo::Parser::make_CPP_CODE(yytext, loc);
}
<INITIAL><<EOF>> { return Choreo::Parser::make_END(loc); }


<CO_FUNC>{
  {blank}+   {}
  \n+        {}
  {num}      return Choreo::Parser::make_NUM(std::stoi(yytext), loc);
  "//".*     {}  // Ignore everything from // to the end of the line
  "mdspan"   return Choreo::Parser::make_MDSPAN(yytext, loc);
  "ituple"   return Choreo::Parser::make_ITUPLE(yytext, loc);
  "local"    return Choreo::Parser::make_LOCAL(AST::Storage::LOCAL, loc);
  "shared"   return Choreo::Parser::make_SHARED(AST::Storage::SHARED, loc);
  "global"   return Choreo::Parser::make_GLOBAL(AST::Storage::GLOBAL, loc);
  "f32"      return Choreo::Parser::make_F32(AST::BaseType::F32, loc);
  "f16"      return Choreo::Parser::make_F16(AST::BaseType::F16, loc);
  "bf16"     return Choreo::Parser::make_BF16(AST::BaseType::BF16, loc);
  "u32"      return Choreo::Parser::make_U32(AST::BaseType::U32, loc);
  "s32"      return Choreo::Parser::make_S32(AST::BaseType::S32, loc);
  "u16"      return Choreo::Parser::make_U16(AST::BaseType::U16, loc);
  "s16"      return Choreo::Parser::make_S16(AST::BaseType::S16, loc);
  "u8"       return Choreo::Parser::make_U8(AST::BaseType::U8, loc);
  "s8"       return Choreo::Parser::make_S8(AST::BaseType::S8, loc);
  "int"      return Choreo::Parser::make_INT(AST::BaseType::INT, loc);
  "dte"      return Choreo::Parser::make_DMA(yytext, loc);
  ".linear"  return Choreo::Parser::make_DLIN(yytext, loc);
  ".slice"   return Choreo::Parser::make_DSLICE(yytext, loc);
  ".pad"     return Choreo::Parser::make_DPAD(yytext, loc);
  "copy"     return Choreo::Parser::make_COPY(yytext, loc);
  ".span"    return Choreo::Parser::make_FNSPAN(yytext, loc);
  ".data"    return Choreo::Parser::make_FNDATA(yytext, loc);
  ".chunkat" return Choreo::Parser::make_CHUNKAT(yytext, loc);
  "=>"       return Choreo::Parser::make_TRANS  (loc);
  "if"       return Choreo::Parser::make_IF(yytext, loc);
  "else"     return Choreo::Parser::make_ELSE(yytext, loc);
  "parallel" return Choreo::Parser::make_PARA(yytext, loc);
  "by"       return Choreo::Parser::make_BY(yytext, loc);
  "with"     return Choreo::Parser::make_WITH(yytext, loc);
  "in"       return Choreo::Parser::make_IN(yytext, loc);
  "foreach"  return Choreo::Parser::make_ITER(yytext, loc);
  "wait"     return Choreo::Parser::make_WAIT(yytext, loc);
  "call"     return Choreo::Parser::make_CALL(yytext, loc);
  "return"   return Choreo::Parser::make_RET(yytext, loc);
  "requires" return Choreo::Parser::make_REQUIRE(yytext, loc);
  "<->"      return Choreo::Parser::make_BIND   (loc);
  "-"        return Choreo::Parser::make_MINUS  (loc);
  "+"        return Choreo::Parser::make_PLUS   (loc);
  "*"        return Choreo::Parser::make_STAR   (loc);
  "/"        return Choreo::Parser::make_SLASH  (loc);
  "%"        return Choreo::Parser::make_PECET  (loc);
  "="        return Choreo::Parser::make_ASSIGN (loc);
  "<"        return Choreo::Parser::make_LT     (loc);
  ">"        return Choreo::Parser::make_GT     (loc);
  "=="       return Choreo::Parser::make_EQ     (loc);
  "!="       return Choreo::Parser::make_NE     (loc);
  "<="       return Choreo::Parser::make_LE     (loc);
  ">="       return Choreo::Parser::make_GE     (loc);
  ","        return Choreo::Parser::make_COMMA  (loc);
  ";"        return Choreo::Parser::make_SEMCOL (loc);
  ":"        return Choreo::Parser::make_COL    (loc);
  "("        return Choreo::Parser::make_LPAREN (loc);
  ")"        return Choreo::Parser::make_RPAREN (loc);
  "["        return Choreo::Parser::make_LBRAKT (loc);
  "]"        return Choreo::Parser::make_RBRAKT (loc);
  "{"        { scope++; return Choreo::Parser::make_LBRACE(loc); }
  "}"        { if (--scope == 0) {
                 BEGIN(INITIAL);
               }
               return Choreo::Parser::make_RBRACE(loc);
             }
  {id}       { return Choreo::Parser::make_IDENTIFIER(yytext, loc); }
  .          { return Choreo::Parser::make_CHAR(yytext[0], loc); } // Any other single character within the function

} // state <CO_FUNC>

%%

// Any helper functions or C code goes here.
int yywrap() {
    return 1;
}

bool Choreo::Scanner::debug = false;
void Choreo::Scanner::Debug(std::string s) { if (debug) std::cout << "[scanner] " << s << std::endl; }

