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
#include "loc.hpp"


static int scope = 0;

extern Choreo::location loc;

std::string kernel_buffer; // buffering the kernel code

void update_location(const char *text);
void handle_line_directive(const char* text);

// Original yyterminate() macro returns int. Since we're using Bison 3 variants
// as tokens, we must redefine it to change type from `int` to `Parser::semantic_type`
#define yyterminate() { Choreo::Parser::make_END(loc); break; }

/* update location on each matching */
#define YY_USER_ACTION update_location(yytext);

%}

%x CO_FUNC KERNEL

id          [a-zA-Z_][a-zA-Z0-9_]*
num         [0-9]+
fpnum       -?[0-9]+\.[0-9]+
fpnum_f     -?[0-9]+\.[0-9]+[fF]
blank       [ \t\r]
cstr        \"([^"\\]|\\.)*\"

%%
<INITIAL,KERNEL,CO_FUNC>{
    #line[ \t]+([0-9]+)[ \t]+\"([^\"]*)\" {
      handle_line_directive(yytext);
    }
    #line[ \t]+([0-9]+) {
      handle_line_directive(yytext);
    }
}

<INITIAL>{
  "//".*   {
    if (Choreo::Scanner::KeepComments())
      return Choreo::Parser::make_HOST_CODE(yytext, loc);
  }
  "__co__" {
    BEGIN(CO_FUNC);
    Choreo::Scanner::Debug("Entering Choreo Function");
    return Choreo::Parser::make_ATTR_CO(yytext, loc);
  }
  "__cok__" {
    BEGIN(KERNEL);
    if (scope != 0)
      Error(loc, "unexpected kernel code (mustn't be inside of existing scope).");
    Choreo::Scanner::Debug("Begin Kernel code");
  }
  . {
    Choreo::Scanner::Debug("token (char): " + std::string(yytext));
    return Choreo::Parser::make_HOST_CODE(yytext, loc);
  } // This matches everything outside the __co__ functions.
  \n+ {
    Choreo::Scanner::Debug("token (newline)");
    return Choreo::Parser::make_HOST_CODE(yytext, loc);
  }
  <<EOF>> { return Choreo::Parser::make_END(loc); }
}

<KERNEL>{
  "{"      { scope++; if (scope > 1) kernel_buffer += yytext; }
  "}"      { if (--scope == 0) {
               BEGIN(INITIAL);
               auto res = Choreo::Parser::make_KERNEL_CODE(kernel_buffer, loc);
               kernel_buffer = "";  // reset the buffer
               return res;
             } else
               kernel_buffer += yytext;
           }
  .        { if (scope > 0) kernel_buffer += yytext; }
  \n+      { if (scope > 0) kernel_buffer += yytext; }
  <<EOF>>  { if (scope > 0)
                Choreo::Scanner::Error(loc, "missing '}' for the kernel code.");
             Choreo::Scanner::Debug("token (char): " + std::string(yytext));
             auto res = Choreo::Parser::make_KERNEL_CODE(kernel_buffer, loc);
             kernel_buffer = "";  // reset the buffer
             return res;
           }
} // state <KERNEL>

<CO_FUNC>{
  {blank}+   {}
  \n+        {}
  {num}      return Choreo::Parser::make_NUM(std::stoi(yytext), loc);
  {fpnum}    return Choreo::Parser::make_DFPVAL(std::stod(yytext), loc);
  {fpnum_f}  return Choreo::Parser::make_FPVAL(std::stof(yytext), loc);
  {cstr}     { // Remove the surrounding quotes and unescape
               std::string str = yytext;
               str = str.substr(1, str.length() - 2); // Remove the quotes

               // Replace escape sequences
               size_t pos = 0;
               while ((pos = str.find("\\", pos)) != std::string::npos) {
                 if (pos + 1 < str.length()) {
                   switch (str[pos + 1]) {
                     case 'n': str.replace(pos, 2, "\n"); break;
                     case 't': str.replace(pos, 2, "\t"); break;
                     case '"': str.replace(pos, 2, "\""); break;
                     case '\\': str.replace(pos, 2, "\\"); break;
                     default: ++pos; break; // Leave unsupported sequences
                   }
                 } else {
                   ++pos;
                 }
               }
               return Choreo::Parser::make_STRING(str, loc);
             }
  "//".*     {}  // Ignore everything from // to the end of the line
  "auto"     return Choreo::Parser::make_AUTO(yytext, loc);
  "mdspan"   return Choreo::Parser::make_MDSPAN(yytext, loc);
  "ituple"   return Choreo::Parser::make_ITUPLE(yytext, loc);
  "print"    return Choreo::Parser::make_PRINT(yytext, loc);
  "local"    return Choreo::Parser::make_LOCAL(Storage::LOCAL, loc);
  "shared"   return Choreo::Parser::make_SHARED(Storage::SHARED, loc);
  "global"   return Choreo::Parser::make_GLOBAL(Storage::GLOBAL, loc);
  "f32"      return Choreo::Parser::make_F32(BaseType::F32, loc);
  "f16"      return Choreo::Parser::make_F16(BaseType::F16, loc);
  "bf16"     return Choreo::Parser::make_BF16(BaseType::BF16, loc);
  "u32"      return Choreo::Parser::make_U32(BaseType::U32, loc);
  "s32"      return Choreo::Parser::make_S32(BaseType::S32, loc);
  "u16"      return Choreo::Parser::make_U16(BaseType::U16, loc);
  "s16"      return Choreo::Parser::make_S16(BaseType::S16, loc);
  "u8"       return Choreo::Parser::make_U8(BaseType::U8, loc);
  "s8"       return Choreo::Parser::make_S8(BaseType::S8, loc);
  "int"      return Choreo::Parser::make_INT(BaseType::INT, loc);
  "bool"     return Choreo::Parser::make_BOOL(BaseType::BOOL, loc);
  "half8"    return Choreo::Parser::make_HALF8(BaseType::HALF8, loc);
  "half"     return Choreo::Parser::make_HALF(BaseType::HALF, loc);
  "bfp16"    return Choreo::Parser::make_BFP16(BaseType::BFP16, loc);
  "float"    return Choreo::Parser::make_FLOAT(BaseType::FLOAT, loc);
  "double"   return Choreo::Parser::make_DOUBLE(BaseType::DOUBLE, loc);
  "void"     return Choreo::Parser::make_VOID(BaseType::VOID, loc);
  "true"     return Choreo::Parser::make_TRUE(yytext, loc);
  "false"    return Choreo::Parser::make_FALSE(yytext, loc);
  "dma"      return Choreo::Parser::make_DMA(yytext, loc);
  ".copy"    return Choreo::Parser::make_COPY(yytext, loc);
  ".transp"  return Choreo::Parser::make_TRANSPOSE(yytext, loc);
  ".pad"     return Choreo::Parser::make_PAD(yytext, loc);
  ".any"     return Choreo::Parser::make_NONE(yytext, loc);
  ".async"   return Choreo::Parser::make_ASYNC(yytext, loc);
  ".span"    return Choreo::Parser::make_FNSPAN(yytext, loc);
  ".span_as" return Choreo::Parser::make_FNSPANAS(yytext, loc);
  ".data"    return Choreo::Parser::make_FNDATA(yytext, loc);
  ".chunkat" return Choreo::Parser::make_CHUNKAT(yytext, loc);
  ".chunk"   return Choreo::Parser::make_CHUNK(yytext, loc);
  ".at"      return Choreo::Parser::make_AT(yytext, loc);
  "=>"       return Choreo::Parser::make_TRANS(loc);
  "swap"     return Choreo::Parser::make_SWAP(yytext, loc);
  "rotate"   return Choreo::Parser::make_ROTATE(yytext, loc);
  "assert"   return Choreo::Parser::make_ASSERT (yytext,loc);
  "if"       return Choreo::Parser::make_IF(yytext, loc);
  "else"     return Choreo::Parser::make_ELSE(yytext, loc);
  "inthreads" return Choreo::Parser::make_INTHDS(yytext, loc);
  "parallel" return Choreo::Parser::make_PARA(yytext, loc);
  "by"       return Choreo::Parser::make_BY(yytext, loc);
  "with"     return Choreo::Parser::make_WITH(yytext, loc);
  "in"       return Choreo::Parser::make_IN(yytext, loc);
  "foreach"  return Choreo::Parser::make_FOREACH(yytext, loc);
  "incr"     return Choreo::Parser::make_INCR(yytext, loc);
  "wait"     return Choreo::Parser::make_WAIT(yytext, loc);
  "call"     return Choreo::Parser::make_CALL(yytext, loc);
  "select"   return Choreo::Parser::make_SELECT(yytext, loc);
  "return"   return Choreo::Parser::make_RET(yytext, loc);
  "where"    return Choreo::Parser::make_WHERE(yytext, loc);
  "while"    return Choreo::Parser::make_WHILE(yytext, loc);
  "<->"      return Choreo::Parser::make_BIND   (loc);
  "-"        return Choreo::Parser::make_MINUS  (loc);
  "+"        return Choreo::Parser::make_PLUS   (loc);
  "*"        return Choreo::Parser::make_STAR   (loc);
  "/"        return Choreo::Parser::make_SLASH  (loc);
  "%"        return Choreo::Parser::make_PECET  (loc);
  "cdiv"     return Choreo::Parser::make_CDIV   (loc);
  "after"    return Choreo::Parser::make_CHAIN  (loc);
  "="        return Choreo::Parser::make_ASSIGN (loc);
  "<"        return Choreo::Parser::make_LT     (loc);
  ">"        return Choreo::Parser::make_GT     (loc);
  "=="       return Choreo::Parser::make_EQ     (loc);
  "!="       return Choreo::Parser::make_NE     (loc);
  "<="       return Choreo::Parser::make_LE     (loc);
  ">="       return Choreo::Parser::make_GE     (loc);
  "&&"       return Choreo::Parser::make_AND    (loc);
  "||"       return Choreo::Parser::make_OR     (loc);
  "!"        return Choreo::Parser::make_NOT    (loc);
  "?"        return Choreo::Parser::make_QES    (loc);
  ","        return Choreo::Parser::make_COMMA  (loc);
  ";"        return Choreo::Parser::make_SEMCOL (loc);
  ":"        return Choreo::Parser::make_COL    (loc);
  "#"        return Choreo::Parser::make_UBOUND (loc);
  "("        return Choreo::Parser::make_LPAREN (loc);
  ")"        return Choreo::Parser::make_RPAREN (loc);
  "["        return Choreo::Parser::make_LBRAKT (loc);
  "]"        return Choreo::Parser::make_RBRAKT (loc);
  "{"        { scope++; return Choreo::Parser::make_LBRACE(loc); }
  "}"        { if (--scope == 0) { BEGIN(INITIAL); }
               return Choreo::Parser::make_RBRACE(loc);
             }
  "|"        { return Choreo::Parser::make_PIPE(loc); }
  {id}       { return Choreo::Parser::make_IDENTIFIER(yytext, loc); }
  .          { return Choreo::Parser::make_CHAR(yytext[0], loc); } // Any other single character within the function

} // state <CO_FUNC>

%%

// Any helper functions or C code goes here.
void handle_line_directive(const char* text) {
  // Parse the line number and optional filename
  int new_line;
  char filename[256] = {0};
  if (sscanf(text, "#line %d \"%[^\"]\"", &new_line, filename) == 2) {
    // Update location with new line and filename
	Choreo::location::filename_type fname(filename); // Convert to filename_type
	loc.initialize(fname, new_line - 1, 0); // Use correct type
  } else if (sscanf(text, "#line %d", &new_line) == 1) {
    // Update only line number if no filename is provided
    loc.initialize("", new_line - 1, 0); // Reset to line-1
  }
}

void update_location(const char *text) {
  const char* it = text;
  loc.step(); // set begin to current end
  while (*it) {
    if (*it == '\n') {
      loc.lines();
    } else {
      loc.columns();
    }
    ++it;
  }
}

int yywrap() {
  return 1;
}

bool Choreo::Scanner::debug = false;
bool Choreo::Scanner::keep_comments = true;
void Choreo::Scanner::Debug(std::string s) { if (debug) dbgs() << "[scanner] " << s << std::endl; }

