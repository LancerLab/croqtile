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
#include "options.hpp"
#include "command_line.hpp"

typedef Choreo::Parser::token::token_kind_type token_type;
static int scope = 0;
static int device_param_paren_scope = 0;
static int device_param_brace_scope = 0;
static int device_template_scope = 0;

extern Choreo::location loc;
extern Choreo::Option<bool> analyze_device_functions;
static std::string kernel_buffer; // buffering the kernel code
static std::string rawstr_delimiter;
static std::string rawstr_buffer;
static std::string analyze_free_buffer;
static Choreo::location rawstr_loc;
static std::string device_expr_buffer; // buffering the device expression
static std::string device_token_buffer;
static std::string device_template_buffer;

static token_type device_token_id = token_type::END;
static token_type token_id;

void update_location(const char *text);
void handle_line_directive(const char* text);
bool accumulate_buffer(const char* text, token_type current_token_id);
Choreo::Parser::symbol_type make_device_symbol();

// Original yyterminate() macro returns int. Since we're using Bison 3 variants
// as tokens, we must redefine it to change type from `int` to `Parser::semantic_type`
#define yyterminate() { Choreo::Parser::make_END(loc); break; }

/* update location on each matching */
#define YY_USER_ACTION update_location(yytext);

static bool double_bracket = false;
static bool no_line_update = false;
static bool accumulate_expr_buffer = false;

%}

%x CO_FUNC DEVICE_FUNC KERNEL REAL_KERNEL MDSPAN_SUGER RAW_STRING TEMPLATE

id          [a-zA-Z_][a-zA-Z0-9_]*
num         (0|[1-9][0-9]*)[uU]?[lL]{0,2}
hex_num     0[xX][0-9a-fA-F]+[uU]?[lL]{0,2}
fpnum       -?[0-9]+\.[0-9]+
fpnum_f     -?[0-9]+\.[0-9]+[fF]
cstr        \"([^"\\]|\\.)*\"
blank       [ \t\r]
attr_id     (__{id}__|\[\[{blank}*{id}{blank}*(?:\({blank}*{cstr}{blank}*\){blank}*)?\]\])
val         ({num}|{hex_num}|{fpnum}|{fpnum_f}|{cstr})
comments    \/\*([^*]|\*+[^*/])*\*+\/
device_attr (__device__|__co_device__|__attribute__\(\(device\)\))


%%
<INITIAL,KERNEL,CO_FUNC,DEVICE_FUNC>{
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
  {comments} {
    if (Choreo::Scanner::KeepComments())
      return Choreo::Parser::make_HOST_CODE(yytext, loc);
  }
  "__co__" {
    BEGIN(CO_FUNC);
    Choreo::Scanner::Debug("Entering Choreo Function");
    return Choreo::Parser::make_ATTR_CO(yytext, loc);
  }
  "__real_cok__" {
    BEGIN(REAL_KERNEL);
    if (scope != 0)
      Error(loc, "unexpected kernel code (mustn't be inside of existing scope).");
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
               auto res = Choreo::Parser::make_DEVICE_CODE(kernel_buffer, loc);
               kernel_buffer = "";  // reset the buffer
               return res;
             } else
               kernel_buffer += yytext;
           }
  {blank}+ { if (scope > 0) kernel_buffer += yytext; }
  .        { if (scope > 0) kernel_buffer += yytext;
             else Error(loc, "unexpected character outside of kernel code.");
           }
  \n+      { if (scope > 0) kernel_buffer += yytext; }
  <<EOF>>  { if (scope > 0)
                Choreo::Scanner::Error(loc, "missing '}' for the kernel code.");
             Choreo::Scanner::Debug("token (char): " + std::string(yytext));
             auto res = Choreo::Parser::make_DEVICE_CODE(kernel_buffer, loc);
             kernel_buffer = "";  // reset the buffer
             return res;
           }
} // state <KERNEL>

<REAL_KERNEL>{
  "{"         { scope++; }
  "}"         { if (--scope == 0) BEGIN(INITIAL); }
  {device_attr} { if (analyze_device_functions) {
                    BEGIN(DEVICE_FUNC);
                    device_expr_buffer = "";
                  }
                  if (!device_template_buffer.empty()) {
                    return Choreo::Parser::make_DEVICE_TEMPLATE(device_template_buffer, loc);
                  }
                }
  "template"  { device_template_buffer = "";
                device_template_scope = 0;
                if (analyze_device_functions) {
                  BEGIN(TEMPLATE);
                  device_template_buffer += yytext;
                }
              }
  <<EOF>>     { if (scope > 0)
                 Choreo::Scanner::Error(loc, "missing '}' for the kernel code.");
                Choreo::Scanner::Debug("token (char): " + std::string(yytext));
              }
  "//".*      {}
  {comments}  {}
  {cstr}      {}
  R\"\(       {}
  R\"{id}\(   {}
  .           {}
  \n+         {}
} // state <KERNEL>

<TEMPLATE>{
  {blank}+    { device_template_buffer += " "; }
  \n+         {}
  "//".*      {}
  {comments}  {}
  "<"         { if (device_template_scope < 0)
                  Choreo::Scanner::Error(loc, "unmatched '<' in device template declaration.");
                device_template_scope++;
                device_template_buffer += yytext;
              }
  ">"         { if (device_template_scope <= 0)
                  Choreo::Scanner::Error(loc, "unmatched '>' in device template declaration.");
                device_template_scope--;
                device_template_buffer += yytext;
                if (device_template_scope == 0)
                  BEGIN(REAL_KERNEL);
              }
  .           { if (device_template_scope > 0) device_template_buffer += yytext; }
}

<DEVICE_FUNC>{
  {blank}+    {}
  \n+         {}
  "//".*      {}
  {comments}  {}
  "<"         { if (accumulate_buffer(yytext, token_type::LT)) return make_device_symbol(); }
  ">"         { if (accumulate_buffer(yytext, token_type::GT)) return make_device_symbol(); }
  "("         { device_param_paren_scope++;
                if (accumulate_buffer(yytext, token_type::LPAREN)) return make_device_symbol(); }
  ")"         { if (accumulate_buffer(yytext, token_type::RPAREN)) {
                  device_param_paren_scope--;
                  return make_device_symbol();
                } else
                  device_param_paren_scope--;
              }
  "{"         { scope++; device_param_brace_scope++;
                if (accumulate_buffer(yytext, token_type::LBRACE)) {
                  if (device_param_paren_scope == 0) {
                    if (device_param_brace_scope != 1)
                      Choreo::Scanner::Error(loc, "unmatched '{' in device function declaration.");
                    BEGIN(REAL_KERNEL);
                    device_param_brace_scope = 0;
                    device_param_paren_scope = 0;
                    device_token_id = token_type::END;
                    device_template_buffer = "";
                  }
                  return make_device_symbol();
                }
              }
  "}"         { scope--;
                if (accumulate_buffer(yytext, token_type::RBRACE)) {
                  device_param_brace_scope--;
                  return make_device_symbol();
                } else
                  device_param_brace_scope--;
                if (device_param_brace_scope < 0)
                  Choreo::Scanner::Error(loc, "unmatched '}' in device function declaration.");
              }
  ";"         { if (accumulate_buffer(yytext, token_type::END)) {
                  BEGIN(REAL_KERNEL);
                  device_param_brace_scope = 0;
                  device_param_paren_scope = 0;
                  device_token_id = token_type::END;
                  device_template_buffer = "";
                  return make_device_symbol();
                }
              }
  "::"        { if (accumulate_buffer(yytext, token_type::DCOLS)) return make_device_symbol(); }
  "*"         { if (accumulate_buffer(yytext, token_type::STAR)) return make_device_symbol(); }
  "="         { if (accumulate_buffer(yytext, token_type::ASSIGN)) {
                  if (device_param_paren_scope == 1 && device_param_brace_scope == 0)
                    accumulate_expr_buffer = true;
                  return make_device_symbol();
                }
              }
  "&"         { if (accumulate_buffer(yytext, token_type::AMP)) return make_device_symbol(); }
  "&&"        { if (accumulate_buffer(yytext, token_type::AND)) return make_device_symbol(); }
  ","         { if (accumulate_buffer(yytext, token_type::COMMA)) return make_device_symbol(); }
  "signed"    { if (accumulate_buffer(yytext, token_type::SIGNED)) return make_device_symbol(); }
  "unsigned"  { if (accumulate_buffer(yytext, token_type::UNSIGNED)) return make_device_symbol(); }
  "auto"      { if (accumulate_buffer(yytext, token_type::AUTO)) return make_device_symbol(); }
  "void"      { if (accumulate_buffer(yytext, token_type::VOID)) return make_device_symbol(); }
  "bool"      { if (accumulate_buffer(yytext, token_type::BOOL)) return make_device_symbol(); }
  "char"      { if (accumulate_buffer(yytext, token_type::S8)) return make_device_symbol(); }
  "short"     { if (accumulate_buffer(yytext, token_type::S16)) return make_device_symbol(); }
  "int"       { if (accumulate_buffer(yytext, token_type::INT)) return make_device_symbol(); }
  "long long" { if (accumulate_buffer(yytext, token_type::S64)) return make_device_symbol(); }
  "float"     { if (accumulate_buffer(yytext, token_type::F32)) return make_device_symbol(); }
  "double"    { if (accumulate_buffer(yytext, token_type::F64)) return make_device_symbol(); }
  "const"     { if (accumulate_buffer(yytext, token_type::CONST)) return make_device_symbol(); }
  "constexpr" { if (accumulate_buffer(yytext, token_type::CONST)) return make_device_symbol(); }
  "constval"  { if (accumulate_buffer(yytext, token_type::CONST)) return make_device_symbol(); }
  "static"    { if (accumulate_buffer(yytext, token_type::STATIC)) return make_device_symbol(); }
  "extern"    { if (accumulate_buffer(yytext, token_type::EXTERN)) return make_device_symbol(); }
  "inline"    { if (accumulate_buffer(yytext, token_type::INLINE)) return make_device_symbol(); }
  "typename" { if (accumulate_buffer(yytext, token_type::TYPENAME)) return make_device_symbol(); }
  "__attribute__" { if (accumulate_buffer(yytext, token_type::ATTRIBUTE)) return make_device_symbol(); }
  {attr_id}   { if (accumulate_buffer(yytext, token_type::ATTR_ID)) return make_device_symbol(); }
  {id}        { if (accumulate_buffer(yytext, token_type::IDENTIFIER)) return make_device_symbol(); }
  {val}       { if (accumulate_buffer(yytext, token_type::VAL)) return make_device_symbol(); }
  .           { if (accumulate_buffer(yytext, token_type::CHAR)) return make_device_symbol(); }
} // state <DEVICE_FUNC>

<MDSPAN_SUGER>{
  "["/[^\[\],]*,[^\[\]]*"]" { // sugar form: emit '[['
                  BEGIN(CO_FUNC);
                  assert(double_bracket == false);
                  double_bracket = true;
                  no_line_update = true;
                  yyless(0);
                  return Choreo::Parser::make_LBRAKT (loc);
               }
  "]"          {  // Double-brackets is required, emit ']]'
                  BEGIN(CO_FUNC);
                  assert(double_bracket == true);
                  double_bracket = false;
                  no_line_update = true;
                  return Choreo::Parser::make_RBRAKT (loc);
               }
  /* Other tokens reset to CO_FUNC */
  [ \t\n]   ; /* Ignore whitespace */
  .         { BEGIN(CO_FUNC); yyless(0); } /* Rewind and reprocess in NORMAL */
}

<RAW_STRING>{

  \){id}\"  { std::string a {yytext};
              if (a.substr(1, a.size() - 2) == rawstr_delimiter) {
                auto rstr = rawstr_buffer;
                rawstr_buffer = "";
                rawstr_delimiter = "";
                BEGIN(CO_FUNC);
                return Choreo::Parser::make_STRING(rstr, rawstr_loc);
              } else
                rawstr_buffer = rawstr_buffer + yytext;
            }

  \)\"      {
              if (rawstr_delimiter == "") {
                auto rstr = rawstr_buffer;
                rawstr_buffer = "";
                BEGIN(CO_FUNC);
                return Choreo::Parser::make_STRING(rstr, rawstr_loc);
              } else
                rawstr_buffer = rawstr_buffer + yytext;
            }

  .         { rawstr_buffer += yytext; }

} // state <RAW_STRING>


<CO_FUNC>{
  {blank}+   {}
  \n+        {}
  {num}      {
               bool is_unsigned = strchr(yytext, 'u') || strchr(yytext, 'U');
               bool is_long_long = strstr(yytext, "ll") || strstr(yytext, "LL");
               errno = 0;
               std::string text = std::string(yytext);
               if (is_unsigned) {
                 unsigned long long val = strtoull(yytext, nullptr, 10);
                 if (errno == ERANGE) {
                   std::string msg = "unsigned integer literal '" + text + "' out of range [0, 18446744073709551615].";
                   Choreo::Scanner::Error(loc, msg);
                 }
                 if (!is_long_long && val <= std::numeric_limits<uint32_t>::max()) {
                   // 42U → unsigned int
                   return Choreo::Parser::make_U32_LITERAL(static_cast<uint32_t>(val), loc);
                 } else {
                   // 42ULL → unsigned long long
                   return Choreo::Parser::make_U64_LITERAL(val, loc);
                 }
               } else {
                 long long val = strtoll(yytext, nullptr, 10);
                 if (errno == ERANGE) {
                   std::string msg = "signed integer literal '" + text + "' out of range [0, 9223372036854775807].";
                   Choreo::Scanner::Error(loc, msg);
                 }
                 if (!is_long_long && val <= std::numeric_limits<int>::max()) {
                   // 42 → int
                   return Choreo::Parser::make_NUM(static_cast<int>(val), loc);
                 } else {
                   // 42LL → long long
                   return Choreo::Parser::make_S64_LITERAL(val, loc);
                 }
               }

             }
  {hex_num}  {
               bool is_unsigned = strchr(yytext, 'u') || strchr(yytext, 'U');
               bool is_long_long = strstr(yytext, "ll") || strstr(yytext, "LL");
               errno = 0;
               std::string text = std::string(yytext);
               if (is_unsigned) {
                 unsigned long long val = strtoull(yytext, nullptr, 16);
                 if (errno == ERANGE) {
                   std::string msg = "unsigned integer literal '" + text + "' out of range [0, 0xffffffffffffffff].";
                   Choreo::Scanner::Error(loc, msg);
                 }
                 if (!is_long_long && val <= std::numeric_limits<uint32_t>::max()) {
                   return Choreo::Parser::make_U32_LITERAL(static_cast<uint32_t>(val), loc);
                 } else {
                   return Choreo::Parser::make_U64_LITERAL(val, loc);
                 }
               } else {
                 long long val = strtoll(yytext, nullptr, 16);
                 if (errno == ERANGE) {
                   std::string msg = "signed integer literal '" + text + "' out of range [0, 0x7fffffffffffffff].";
                   Choreo::Scanner::Error(loc, msg);
                 }
                 if (!is_long_long && val <= std::numeric_limits<int>::max()) {
                   return Choreo::Parser::make_NUM(static_cast<int>(val), loc);
                 } else {
                   return Choreo::Parser::make_S64_LITERAL(val, loc);
                 }
               }
             }
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
  {comments} {}  // Ignore comments
  "auto"     return Choreo::Parser::make_AUTO(yytext, loc);
  "mdspan"   return Choreo::Parser::make_MDSPAN(yytext, loc);
  "ituple"   return Choreo::Parser::make_ITUPLE(yytext, loc);
  "stream"   return Choreo::Parser::make_STREAM(yytext, loc);
  "event"    return Choreo::Parser::make_EVENT(yytext, loc);
  "print"    return Choreo::Parser::make_PRINT(yytext, loc);
  "println"  return Choreo::Parser::make_PRINTLN(yytext, loc);
  "__acos"     return Choreo::Parser::make_ACOS(yytext, loc);
  "__asin"     return Choreo::Parser::make_ASIN(yytext, loc);
  "__atan"     return Choreo::Parser::make_ATAN(yytext, loc);
  "__atan2"    return Choreo::Parser::make_ATAN2(yytext, loc);
  "__ceil"     return Choreo::Parser::make_CEIL(yytext, loc);
  "__cos"      return Choreo::Parser::make_COS(yytext, loc);
  "__cosh"     return Choreo::Parser::make_COSH(yytext, loc);
  "__exp"      return Choreo::Parser::make_EXP(yytext, loc);
  "__expm1"    return Choreo::Parser::make_EXPM1(yytext, loc);
  "__floor"    return Choreo::Parser::make_FLOOR(yytext, loc);
  "__gelu"     return Choreo::Parser::make_GELU(yytext, loc);
  "__isfinite" return Choreo::Parser::make_ISFINITE(yytext, loc);
  "__round"    return Choreo::Parser::make_ROUND(yytext, loc);
  "__rsqrt"    return Choreo::Parser::make_RSQRT(yytext, loc);
  "__sigmoid"  return Choreo::Parser::make_SIGMOID(yytext, loc);
  "__sinh"     return Choreo::Parser::make_SINH(yytext, loc);
  "__softplus" return Choreo::Parser::make_SOFTPLUS(yytext, loc);
  "__sqrt"     return Choreo::Parser::make_SQRT(yytext, loc);
  "__tan"      return Choreo::Parser::make_TAN(yytext, loc);
  "__log1p"    return Choreo::Parser::make_LOG1P(yytext, loc);
  "__log"      return Choreo::Parser::make_LOG(yytext, loc);
  "__pow"      return Choreo::Parser::make_POW(yytext, loc);
  "__sign"     return Choreo::Parser::make_SIGN(yytext, loc);
  "__sin"      return Choreo::Parser::make_SIN(yytext, loc);
  "__tanh"     return Choreo::Parser::make_TANH(yytext, loc);
  "__alignup"  return Choreo::Parser::make_ALIGNUP(yytext, loc);
  "__aligndown" return Choreo::Parser::make_ALIGNDOWN(yytext, loc);
  "local"    return Choreo::Parser::make_STORAGE(Storage::LOCAL, loc);
  "shared"   return Choreo::Parser::make_STORAGE(Storage::SHARED, loc);
  "global"   return Choreo::Parser::make_STORAGE(Storage::GLOBAL, loc);
  "term"     return Choreo::Parser::make_PBLEVEL(ParallelLevel::TERM, loc);
  "device"   return Choreo::Parser::make_PBLEVEL(ParallelLevel::DEVICE, loc);
  "block"    return Choreo::Parser::make_PBLEVEL(ParallelLevel::BLOCK, loc);
  "group-4"  return Choreo::Parser::make_PBLEVEL(ParallelLevel::GROUPx4, loc);
  "group"    return Choreo::Parser::make_PBLEVEL(ParallelLevel::GROUP, loc);
  "thread"   return Choreo::Parser::make_PBLEVEL(ParallelLevel::THREAD, loc);
  "mutable"  return Choreo::Parser::make_MUTABLE(yytext, loc);
  "swizzle"  return Choreo::Parser::make_SWIZZLE(yytext, loc);
  "f64"      { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F64(BaseType::F64, loc); }
  "f32"      { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F32(BaseType::F32, loc); }
  "tf32"      { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_TF32(BaseType::TF32, loc); }
  "f16"      { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F16(BaseType::F16, loc); }
  "bf16"     { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_BF16(BaseType::BF16, loc); }
  "f8"       { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F8_E4M3(BaseType::F8_E4M3, loc); }
  "f8_e4m3"  { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F8_E4M3(BaseType::F8_E4M3, loc); }
  "f8_e5m2"  { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F8_E5M2(BaseType::F8_E5M2, loc); }
  "f8_ue4m3"  { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F8_UE4M3(BaseType::F8_UE4M3, loc); }
  "f8_ue8m0"  { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F8_UE8M0(BaseType::F8_UE8M0, loc); }
  "f6_e3m2"  { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F6_E3M2(BaseType::F6_E3M2, loc); }
  "f6_e2m3"  { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F6_E2M3(BaseType::F6_E2M3, loc); }
  "f4_e2m1"  { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_F4_E2M1(BaseType::F4_E2M1, loc); }
  "u32"      { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_U32(BaseType::U32, loc); }
  "s32"      { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_S32(BaseType::S32, loc); }
  "u64"      { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_U64(BaseType::U64, loc); }
  "s64"      { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_S64(BaseType::S64, loc); }
  "u16"      { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_U16(BaseType::U16, loc); }
  "s16"      { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_S16(BaseType::S16, loc); }
  "u8"       { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_U8(BaseType::U8, loc); }
  "s8"       { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_S8(BaseType::S8, loc); }
  "u6"       { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_U6(BaseType::U6, loc); }
  "s6"       { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_S6(BaseType::S6, loc); }
  "u4"       { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_U4(BaseType::U4, loc); }
  "s4"       { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_S4(BaseType::S4, loc); }
  "u2"       { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_U2(BaseType::U2, loc); }
  "s2"       { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_S2(BaseType::S2, loc); }
  "bin1"     { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_BIN1(BaseType::BIN1, loc); }
  "u1"       { BEGIN(MDSPAN_SUGER); return Choreo::Parser::make_U1(BaseType::U1, loc); }
  "int"      return Choreo::Parser::make_INT(BaseType::S32, loc);
  "bool"     return Choreo::Parser::make_BOOL(BaseType::BOOL, loc);
  "half"     return Choreo::Parser::make_F16(BaseType::F16, loc);
  "bfp16"    return Choreo::Parser::make_BF16(BaseType::BF16, loc);
  "float"    return Choreo::Parser::make_F32(BaseType::F32, loc);
  "double"   return Choreo::Parser::make_F64(BaseType::F64, loc);
  "void"     return Choreo::Parser::make_VOID(BaseType::VOID, loc);
  "true"     return Choreo::Parser::make_TRUE(yytext, loc);
  "false"    return Choreo::Parser::make_FALSE(yytext, loc);
  "dma"      return Choreo::Parser::make_DMA(yytext, loc);
  "tma"      return Choreo::Parser::make_TMA(yytext, loc);
  ".copy"    return Choreo::Parser::make_COPY(yytext, loc);
  ".transp"  return Choreo::Parser::make_TRANSPOSE(yytext, loc);
  ".pad"     return Choreo::Parser::make_PAD(yytext, loc);
  ".any"     return Choreo::Parser::make_NONE(yytext, loc);
  ".async"   return Choreo::Parser::make_ASYNC(yytext, loc);
  ".span"    return Choreo::Parser::make_FNSPAN(yytext, loc);
  ".span_as" return Choreo::Parser::make_FNSPANAS(yytext, loc);
  ".mdata"   return Choreo::Parser::make_FNMDATA(yytext, loc);
  ".data"    return Choreo::Parser::make_FNDATA(yytext, loc);
  ".chunkat" return Choreo::Parser::make_CHUNKAT(yytext, loc);
  ".chunk"   return Choreo::Parser::make_CHUNK(yytext, loc);
  ".subspan" return Choreo::Parser::make_SUBSPAN(yytext, loc);
  ".modspan" return Choreo::Parser::make_MODSPAN(yytext, loc);
  ".stride"  return Choreo::Parser::make_STRIDE(yytext, loc);
  ".at"      return Choreo::Parser::make_AT(yytext, loc);
  "=>"       return Choreo::Parser::make_TRANS(loc);
  "swap"     return Choreo::Parser::make_SWAP(yytext, loc);
  "rotate"   return Choreo::Parser::make_ROTATE(yytext, loc);
  "mma"      return Choreo::Parser::make_MMA(yytext, loc);
  ".fill"    return Choreo::Parser::make_FILL(yytext, loc);
  ".load"    return Choreo::Parser::make_LOAD(yytext, loc);
  ".store"   return Choreo::Parser::make_STORE(yytext, loc);
  ".row"     return Choreo::Parser::make_ROW(yytext, loc);
  ".col"     return Choreo::Parser::make_COLUMN(yytext, loc);
  ".sp("    return Choreo::Parser::make_SPLPAREN(yytext, loc);
  ".sp"      return Choreo::Parser::make_SP(yytext, loc);
  "assert"   return Choreo::Parser::make_ASSERT (yytext,loc);
  "sync"     return Choreo::Parser::make_SYNC(yytext, loc);
  "trigger"  return Choreo::Parser::make_TRIGGER(yytext, loc);
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
  "break"    return Choreo::Parser::make_BREAK(yytext, loc);
  "vectorize" return Choreo::Parser::make_VECTORIZE(yytext, loc);
  "continue" return Choreo::Parser::make_CONTINUE(yytext, loc);
  "<->"      return Choreo::Parser::make_BIND   (loc);
  "-"        return Choreo::Parser::make_MINUS  (loc);
  "--"       return Choreo::Parser::make_MMINUS (loc);
  "+"        return Choreo::Parser::make_PLUS   (loc);
  "++"       return Choreo::Parser::make_PPLUS   (loc);
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
  "<<"       return Choreo::Parser::make_LSHIFT (loc);
  ">>"       return Choreo::Parser::make_RSHIFT (loc);
  "?"        return Choreo::Parser::make_QES    (loc);
  "??"       return Choreo::Parser::make_DQES   (loc);
  ","        return Choreo::Parser::make_COMMA  (loc);
  ";"        return Choreo::Parser::make_SEMCOL (loc);
  "::"       return Choreo::Parser::make_DCOLS  (loc);
  ":"        return Choreo::Parser::make_COL    (loc);
  "."        return Choreo::Parser::make_DOT    (loc);
  "#"        return Choreo::Parser::make_UBOUND (loc);
  "&"        return Choreo::Parser::make_AMP    (loc);
  "#+"       return Choreo::Parser::make_UBPLUS (loc);
  "#-"       return Choreo::Parser::make_UBMINUS (loc);
  "#*"       return Choreo::Parser::make_UBSTAR (loc);
  "#/"       return Choreo::Parser::make_UBSLASH (loc);
  "#%"       return Choreo::Parser::make_UBPECET (loc);
  "("        return Choreo::Parser::make_LPAREN (loc);
  ")"        return Choreo::Parser::make_RPAREN (loc);
  "["        return Choreo::Parser::make_LBRAKT (loc);
  "]"        { if (double_bracket) {
                 BEGIN(MDSPAN_SUGER);
                 yyless(0);  // rewind to emit one more
               }
               return Choreo::Parser::make_RBRAKT (loc);
             }
  "{"        { scope++; return Choreo::Parser::make_LBRACE(loc); }
  "}"        { if (--scope == 0) { BEGIN(INITIAL); }
               return Choreo::Parser::make_RBRACE(loc);
             }
  "|"        { return Choreo::Parser::make_PIPE(loc); }
  "^"        { return Choreo::Parser::make_CARET(loc); }
  "~"        { return Choreo::Parser::make_TILDE(loc); }
  "__cpp__"  { return Choreo::Parser::make_INLCPP(loc); }
  {id}       { return Choreo::Parser::make_IDENTIFIER(yytext, loc); }
  R\"\(      { BEGIN(RAW_STRING); }
  R\"{id}\(  { std::string a {yytext};
               rawstr_delimiter = a.substr(2, a.size() - 3);
               rawstr_loc = loc;
               BEGIN(RAW_STRING);
             }
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
  if (!Choreo::Scanner::LocationUpdate())
    return;
  if (no_line_update) {
    no_line_update = false;
    return;
  }
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

bool accumulate_buffer(const char* text, token_type current_token_id) {
  if (device_expr_buffer == "")
    device_expr_buffer = text;

  device_token_buffer = device_expr_buffer;
  token_id = device_token_id;

  if (device_token_id == token_type::END) {
    device_token_id = current_token_id;
    return false;
  }
  device_token_id = current_token_id;

  // reset buffer if we are not accumulating
  if (!accumulate_expr_buffer) {
    device_expr_buffer = text;
  }

  // accumlate current device expression
  if (accumulate_expr_buffer) {
    // stop accumulating on comma or closing parenthesis
    if (device_param_paren_scope == 1 && device_param_brace_scope == 0 && (text[0] == ',' || text[0] == ')')) {
      accumulate_expr_buffer = false;
      token_id = token_type::DEVICE_EXPR;
      Choreo::Scanner::Debug("accumulate device expression: " + device_expr_buffer);
      return true;
    }
    if (text[0] != '=')
      device_expr_buffer += text;

    if (token_id == token_type::ASSIGN) {
      return true;
    }
    return false; // continue accumulating
  }

  return true;
}

typename Choreo::BaseType STR2BT(std::string s) {
  if (s == "void") return Choreo::BaseType::VOID;
  if (s == "float") return Choreo::BaseType::F32;
  if (s == "double") return Choreo::BaseType::F64;
  if (s == "long long") return Choreo::BaseType::S64;
  if (s == "int") return Choreo::BaseType::S32;
  if (s == "short") return Choreo::BaseType::S16;
  if (s == "char") return Choreo::BaseType::S8;
  if (s == "bool") return Choreo::BaseType::BOOL;

  return Choreo::BaseType::UNKNOWN; // default to void
}

Choreo::Parser::symbol_type make_device_symbol() {
  if (token_id >= token_type::ASSIGN && token_id <= token_type::INLCPP) {
    return Choreo::Parser::symbol_type (token_id, std::move(loc));
  } else if (token_id >= token_type::F64 && token_id <= token_type::INT) {
    return Choreo::Parser::symbol_type (token_id, STR2BT(device_token_buffer), std::move(loc));
  }
  return Choreo::Parser::symbol_type (token_id, std::move(device_token_buffer), std::move(loc));
}

int yywrap() {
  return 1;
}

bool Choreo::Scanner::debug = false;
bool Choreo::Scanner::keep_comments = true;
bool Choreo::Scanner::loc_update = true;
void Choreo::Scanner::Debug(std::string s) { if (debug) dbgs() << "[scanner] " << s << std::endl; }

