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


static int scope = 0;

Choreo::location loc;

// Original yyterminate() macro returns int. Since we're using Bison 3 variants
// as tokens, we must redefine it to change type from `int` to `Parser::semantic_type`
#define yyterminate() Choreo::Parser::make_END(loc);

/* update location on matching */
#define YY_USER_ACTION loc.step(); loc.columns(yyleng);

%}

%x CO_FUNCTION

id    [a-zA-Z][a-zA-Z_0-9]*
num   [0-9]+
blank [ \t\r]

%%
<INITIAL>"__co__" {
  BEGIN(CO_FUNCTION);
  return Choreo::Parser::make_ATTR_CO(yytext, loc);
}
<INITIAL>. {
  // This matches everything outside the __co__ functions.
  return Choreo::Parser::make_CPP_CODE(yytext, loc);
}

<CO_FUNCTION>{blank}+   {}
<CO_FUNCTION>\n+        {}
<CO_FUNCTION>{num}      return Choreo::Parser::make_NUM(std::stoi(yytext), loc);
<CO_FUNCTION>"mdspans"  return Choreo::Parser::make_MDSPANS(yytext, loc);
<CO_FUNCTION>"minds"    return Choreo::Parser::make_MINDS(yytext, loc);
<CO_FUNCTION>"local"    return Choreo::Parser::make_LOCAL(yytext, loc);
<CO_FUNCTION>"shared"   return Choreo::Parser::make_SHARED(yytext, loc);
<CO_FUNCTION>"global"   return Choreo::Parser::make_GLOBAL(yytext, loc);
<CO_FUNCTION>"f32"      return Choreo::Parser::make_F32(AST::BaseType::F32, loc);
<CO_FUNCTION>"f16"      return Choreo::Parser::make_F16(AST::BaseType::F16, loc);
<CO_FUNCTION>"bf16"     return Choreo::Parser::make_BF16(AST::BaseType::BF16, loc);
<CO_FUNCTION>"u32"      return Choreo::Parser::make_U32(AST::BaseType::U32, loc);
<CO_FUNCTION>"s32"      return Choreo::Parser::make_S32(AST::BaseType::S32, loc);
<CO_FUNCTION>"u16"      return Choreo::Parser::make_U16(AST::BaseType::U16, loc);
<CO_FUNCTION>"s16"      return Choreo::Parser::make_S16(AST::BaseType::S16, loc);
<CO_FUNCTION>"u8"       return Choreo::Parser::make_U8(AST::BaseType::U8, loc);
<CO_FUNCTION>"s8"       return Choreo::Parser::make_S8(AST::BaseType::S8, loc);
<CO_FUNCTION>"int"      return Choreo::Parser::make_INT(AST::BaseType::INT, loc);
<CO_FUNCTION>"dma"      return Choreo::Parser::make_DMA(yytext, loc);
<CO_FUNCTION>"copy"     return Choreo::Parser::make_COPY(yytext, loc);
<CO_FUNCTION>"=>"       return Choreo::Parser::make_TRANS  (loc);
<CO_FUNCTION>"if"       return Choreo::Parser::make_IF(yytext, loc);
<CO_FUNCTION>"else"     return Choreo::Parser::make_ELSE(yytext, loc);
<CO_FUNCTION>"with"     return Choreo::Parser::make_WITH(yytext, loc);
<CO_FUNCTION>"foreach"  return Choreo::Parser::make_ITER(yytext, loc);
<CO_FUNCTION>"return"   return Choreo::Parser::make_RET(yytext, loc);
<CO_FUNCTION>"-"        return Choreo::Parser::make_MINUS  (loc);
<CO_FUNCTION>"+"        return Choreo::Parser::make_PLUS   (loc);
<CO_FUNCTION>"*"        return Choreo::Parser::make_STAR   (loc);
<CO_FUNCTION>"/"        return Choreo::Parser::make_SLASH  (loc);
<CO_FUNCTION>"%"        return Choreo::Parser::make_PECET  (loc);
<CO_FUNCTION>"="        return Choreo::Parser::make_ASSIGN (loc);
<CO_FUNCTION>"<"        return Choreo::Parser::make_LT     (loc);
<CO_FUNCTION>">"        return Choreo::Parser::make_GT     (loc);
<CO_FUNCTION>"=="       return Choreo::Parser::make_EQ     (loc);
<CO_FUNCTION>"!="       return Choreo::Parser::make_NE     (loc);
<CO_FUNCTION>"<="       return Choreo::Parser::make_LE     (loc);
<CO_FUNCTION>">="       return Choreo::Parser::make_GE     (loc);
<CO_FUNCTION>","        return Choreo::Parser::make_COMMA  (loc);
<CO_FUNCTION>";"        return Choreo::Parser::make_SEMCOL (loc);
<CO_FUNCTION>":"        return Choreo::Parser::make_COL    (loc);
<CO_FUNCTION>"("        return Choreo::Parser::make_LPAREN (loc);
<CO_FUNCTION>")"        return Choreo::Parser::make_RPAREN (loc);
<CO_FUNCTION>"["        return Choreo::Parser::make_LBRAC  (loc);
<CO_FUNCTION>"]"        return Choreo::Parser::make_RBRAC  (loc);
<CO_FUNCTION>"{" 				{ scope++; return Choreo::Parser::make_LBRACE(loc); }
<CO_FUNCTION>"}"  { 
  if (--scope == 0) {
		BEGIN(INITIAL); 
  }
  return Choreo::Parser::make_RBRACE(loc);
}
<CO_FUNCTION>[a-zA-Z_][a-zA-Z0-9_]* {
  return Choreo::Parser::make_IDENTIFIER(yytext, loc);
}
<CO_FUNCTION>.  {
  return Choreo::Parser::make_CHAR(yytext[0], loc);
}  // Any other single character within the function

%%

// Any helper functions or C code goes here.
int yywrap() {
    return 1;
}
