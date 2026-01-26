%skeleton "lalr1.cc" // -*- C++ -*-
%require "3.8"

%define api.token.constructor
//%define parse.trace
%define api.parser.class { Parser }
%parse-param { PContext &pctx }
%define parse.error verbose
%define parse.assert
%define api.namespace { Choreo }
%locations
%define api.location.type {Choreo::location}

// guard the conflict count
%expect 0

%code requires {

#include "loc.hpp"
#include <string>
#include <fstream>

namespace Choreo { class Scanner; }

template<typename T>
struct SymbolWithInitVal {
  std::string name;
  T init_val;
  SymbolWithInitVal(const std::string & n, T i) : name(n), init_val(i) {}
};

struct DMAOperationWithSwizzle {
  std::string operation;
  int swizzle_value;
  bool sparse;
  int sparse_n;
  int sparse_m;
  DMAOperationWithSwizzle()
      : operation(""), swizzle_value(0), sparse(false), sparse_n(0),
        sparse_m(0) {}
  DMAOperationWithSwizzle(const std::string& op, int swizzle, bool sp = false,
                          int sp_n = 0, int sp_m = 0)
      : operation(op), swizzle_value(swizzle), sparse(sp), sparse_n(sp_n),
        sparse_m(sp_m) {}
};

class PContext {
private:
  size_t error_count = 0;

public:
  size_t GetErrorCount() { return error_count; }
  bool HasError() { return error_count > 0; }
  void recordError() { error_count++; }
};
}

%code top {

#include <stdio.h>
#include <iostream>
#include <string>
#include <cstring>
#include <memory>
#include "ast.hpp"
#include "symtab.hpp"
#include "scanner.hpp"
#include <unistd.h>

using namespace Choreo;

extern char* yytext;
extern location loc;

extern AST::Program root;
extern Choreo::SymbolTable symtab;

const char* color_red = "\033[31m";
const char* color_reset = "\033[0m";
const char* color_green = "\033[32m";

static inline bool shell_supports_colors() {
	const char* term = getenv("TERM");
	return term && (strcmp(term, "xterm-256color") == 0
							 || strcmp(term, "xterm") == 0);
}

static inline bool should_use_colors() {
  return isatty(fileno(stdout)) && shell_supports_colors();
}


static Parser::symbol_type yylex(Scanner &scanner) {
  return scanner.get_next_token();
}

// The flag is used to disambiguate syntax sugar:
//
//   a {(0), (1), 3} represents { a(0), a(1), 3 }
//
// where:
//
//   {(0), (1), 3} represents {0, 1, 3}
//
static bool parsing_derivation_decl = false;
static bool ignore_fndata = false;

std::pair<ptr<AST::Identifier>, ptr<AST::MultiValues>> ElementMultiValues(const ptr<AST::Expr>&);
std::set<std::string> paraby_symbols;

inline ptr<AST::SpannedOperation> OptSpannedOperation(const ptr<AST::SpannedOperation> &);

}

%{
#include <stdio.h>
extern int yylex();

void choreo_info(const char *message) {
    // fprintf(stderr, "Error: %s\n", s);
  errs() << ((should_use_colors()) ? color_green : "") << "Info: "
         << ((should_use_colors()) ? color_reset : "");
  errs() << message << "\n";
  errs() << "Info location: " << ::loc << "\n";
}

%}

// make yylex() expects one parameter of type 'Choreo::Scanner &'
%lex-param { Choreo::Scanner &scanner  }
// make yyparse() expects one parameter of type 'Choreo::Scanner &'.
%parse-param { Choreo::Scanner &scanner  }

%token
  ASSIGN  "="
  MINUS   "-"
  MMINUS  "--"
  PLUS    "+"
  PPLUS   "++"
  STAR    "*"
  SLASH   "/"
  PECET   "%"
  LPAREN  "("
  RPAREN  ")"
  LBRACE  "{"
  RBRACE  "}"
  LBRAKT  "["
  RBRAKT  "]"
  COMMA   ","
  SEMCOL  ";"
  COL     ":"
  DCOLS   "::"
  DOT     "."
  LT      "<"
  GT      ">"
  EQ      "=="
  NE      "!="
  LE      "<="
  GE      ">="
  AND     "&&"
  OR      "||"
  NOT     "!"
  LSHIFT  "<<"
  RSHIFT  ">>"
  QES     "?"
  DQES    "??"
  TRANS   "=>"
  BIND    "<->"
  PIPE    "|"
  UBOUND  "#"
  AMP     "&"
  CARET   "^"
  TILDE   "~"
  UBPLUS  "#+"
  UBMINUS "#-"
  UBSTAR  "#*"
  UBSLASH "#/"
  UBPECET "#%"
  DONTCARE"_"
  CDIV    "cdiv"
  CHAIN   "after"
  INLCPP  "__cpp__"
;

// instead of union, using c++17 variant for terminal and non-terminals
%define api.value.type variant

// terminals
%token END 0 "end of file"
%token <char> CHAR
%token <int> NUM
%token <uint32_t> U32_LITERAL
%token <uint64_t> U64_LITERAL
/* %token <int32_t>  S32_LITERAL */  /* which is NUM */
%token <int64_t>  S64_LITERAL
%token <float> FPVAL
%token <double> DFPVAL
%token <std::string> TRUE FALSE
%token <std::string> STRING VAL
%token <std::string> HOST_CODE DEVICE_CODE
%token <std::string> IDENTIFIER ATTR_CO DEVICE_EXPR
%token <std::string> CONST STATIC EXTERN INLINE ATTR_ID ATTRIBUTE SIGNED UNSIGNED TYPENAME DEVICE_TEMPLATE
// type related
%token <std::string> MDSPAN ITUPLE EVENT MUTABLE STREAM
%token <Choreo::Storage> STORAGE
%token <Choreo::ParallelLevel> PBLEVEL
%token <Choreo::BaseType> F64 TF32 F32 F16 BF16 F8_E4M3 F8_E5M2 F8_UE4M3 F8_UE8M0 F6_E2M3 F6_E3M2 F4_E2M1
%token <Choreo::BaseType> BIN1 U1 U2 S2 U4 S4 U6 S6 U8 S8 U16 S16  U32 S32 U64 S64 BOOL VOID INT
// builtin operations
%token <std::string> DMA TMA COPY PAD TRANSPOSE NONE ASYNC FNSPAN FNDATA FNMDATA FNSPANAS CHUNKAT CHUNK SUBSPAN MODSPAN STRIDE AT WAIT CALL AUTO SELECT SWAP ROTATE SYNC CHUNKINBOUND ASSERT TRIGGER PRINT PRINTLN SWIZZLE SP SPLPAREN
// MMA related builtin operations
%token <std::string> MMA FILL LOAD STORE ROW COLUMN
%token <std::string> ACOS ASIN ATAN ATAN2 CEIL COS COSH EXP EXPM1 FLOOR GELU ISFINITE ROUND RSQRT SIGMOID SINH SOFTPLUS SQRT TAN LOG1P LOG POW SIGN SIN TANH ALIGNUP ALIGNDOWN BIF_MMA
// control related
%token <std::string> INTHDS IF ELSE PARA BY WITH IN FOREACH INCR RET WHERE WHILE BREAK CONTINUE
%token <std::string> VECTORIZE

// non-terminals
%nterm <std::string> builtin_print_func arith_operation spanid cstrings arith_builtin_func align_func id_with_namespace
%nterm <ptr<DMAConfig>> dma_config swizzle_config
%nterm <DMAOperationWithSwizzle> dma_operation_swizzle
%nterm <bool> bool_value sync_type pass_by_ref tdma
%nterm <int> integer_value index_or_none const_sizeof swizzle_value
%nterm <std::vector<size_t>> optional_array_dims
%nterm <Choreo::BaseType> fundamental_type
%nterm <AST::ptr<AST::CppSourceCode>> host_code inlcpp_stmt

%nterm <AST::ptr<AST::DeviceFunctionDecl>> device_function_decl
%nterm <std::string> device_attr device_attr_lists
%nterm <std::vector<AST::ptr<Choreo::DeviceDataType>>> device_params
%nterm <AST::ptr<Choreo::DeviceDataType>> device_type device_base_type device_complex_type device_param device_nested_type_list device_nested_type
%nterm <AST::ptr<AST::Memory>> storage_qual
%nterm <AST::ptr<AST::SpanAs>> span_as
%nterm <AST::ptr<AST::IntLiteral>> num_expr
%nterm <AST::ptr<AST::Call>> call_stmt
%nterm <AST::ptr<AST::Node>> any_code device_code foreach_block simple_val template_val int_or_id device_passable declaration statement assignment dma_stmt mma_stmt wait_stmt trigger_stmt swap_stmt break_stmt continue_stmt range_expr param_mdspan_val chunkat_or_storage_or_select returnable span_init_val
%nterm <AST::ptr<AST::MultiNodes>> statements declarations assignments withins where_binds where_clause multi_decls named_spanned_decls spanned_decls named_scalar_decls scalar_decls named_event_decls event_decls stmts_block
%nterm <AST::ptr<AST::MultiValues>> value_list g_value_list template_value_list param_mdspan_list range_exprs iv_list id_list with_matchers device_passables template_params ids_list subscriptions data_indices suffix_exprs
%nterm <AST::ptr<AST::Expr>> s_expr g_expr template_value_expr mdspan_expr mdspan_operator mdspan_val_expr ids_expr bound_expr subscript_like_expr dataid_expr call_expr ituple_derivation internal_sizeof_expr sizeof_expr
%nterm <AST::ptr<AST::AttributeExpr>> suffix_expr
%nterm <AST::ptr<AST::DataType>> scalar_type void_type auto_type param_type return_type mdspan_as_type
%nterm <AST::ptr<AST::DataAccess>> data_element
%nterm <AST::ptr<AST::ParamList>> parameter_list
%nterm <AST::ptr<AST::Parameter>> parameter
%nterm <AST::ptr<AST::ChoreoFunction>> dsl_function
%nterm <AST::ptr<AST::MultiDimSpans>> unnamed_mdspan_decl mdspan_list mdspan_derivation param_mdspan
%nterm <AST::ptr<AST::NamedTypeDecl>> named_mdspan_decl
%nterm <AST::ptr<AST::NamedVariableDecl>> named_ituple_decl spanned_decl scalar_decl event_decl
%nterm <AST::ptr<AST::IntTuple>> ituple_list
%nterm <AST::ptr<AST::WithBlock>> within_block
%nterm <AST::ptr<AST::InThreadsBlock>> inthreads_block
%nterm <AST::ptr<AST::WhileBlock>> while_block
%nterm <AST::ptr<AST::IfElseBlock>> if_else_block
%nterm <AST::ptr<AST::WithIn>> within
%nterm <AST::ptr<AST::WhereBind>> where_bind
%nterm <AST::ptr<AST::ParallelBy>> paraby_block parabys paraby
%nterm <AST::ptr<AST::Return>> return_stmt
%nterm <AST::ptr<AST::Synchronize>> sync_stmt
%nterm <std::vector<ptr<AST::SpannedOperation>>> spanned_ops
%nterm <ptr<AST::SpannedOperation>> spanned_op
%nterm <AST::ptr<AST::ChunkAt>> chunkat_expr subdata_expr
%nterm <AST::ptr<AST::Select>> select_expr
%nterm <AST::MMAOperation::ExecMethod> mma_exec_method
%nterm <Choreo::Storage> param_storage
%nterm <Choreo::ParallelLevel> note_pl

// resolving the ambiguity of dangling ELSE
%nonassoc IF_PREC
%nonassoc ELSE

// precedence (low to high) and associativity
%right ASSIGN
%right QES COL
%left OR
%left AND
%left EQ NE
%left LE GE GT LT
%left LSHIFT RSHIFT
%left PLUS MINUS
%left STAR SLASH PECET
%left UBMINUS UBPLUS
%left UBSTAR UBSLASH UBPECET
%right NOT PPLUS MMINUS
%left AMP CARET PIPE
%left UBOUND
%left TILDE
%left DOT
%nonassoc LPAREN RPAREN
%nonassoc LBRAKT RBRAKT
%left FNSPAN

%nonassoc HOST_CODE_REDUCE
%left HOST_CODE_SHIFT
%left HOST_CODE

%%

program
    : /* Empty */ {}
    | program any_code { if ($2 != nullptr) root.nodes->Append($2); }
    ;

any_code
    : host_code %prec HOST_CODE_REDUCE { $$ = $1; }
    | dsl_function { $$ = $1; }
    | device_code { $$ = $1; }
    ;

device_code
    : DEVICE_CODE /* can not be empty */ {
        $$ = AST::Make<AST::CppSourceCode>(@1, $1, AST::CppSourceCode::Device);
      }
    | device_function_decl { $$ = $1; }
    ;

host_code
    : HOST_CODE /* can not be empty */ {
        $$ = AST::Make<AST::CppSourceCode>(@1, $1, AST::CppSourceCode::Host);
      }
    | host_code HOST_CODE %prec HOST_CODE_SHIFT {
        $1->code += $2;
        $$ = $1;
      }
    ;

device_base_type
    : BOOL { $$ = MakeDeviceDataType("bool",$1); }
    | S8 { $$ = MakeDeviceDataType("char", $1); }
    | S16 { $$ = MakeDeviceDataType("short", $1); }
    | INT { $$ = MakeDeviceDataType("int", $1); }
    | S64 { $$ = MakeDeviceDataType("long long", $1); }
    | SIGNED S8 { $$ = MakeDeviceDataType($1 + " char", BaseType::S8); }
    | SIGNED S16 { $$ = MakeDeviceDataType($1 + " short", BaseType::S16); }
    | SIGNED INT { $$ = MakeDeviceDataType($1 + " int", BaseType::S32); }
    | SIGNED S64 { $$ = MakeDeviceDataType($1 + " long long", BaseType::S64); }
    | UNSIGNED S8 { $$ = MakeDeviceDataType($1 + " char", BaseType::U8); }
    | UNSIGNED S16 { $$ = MakeDeviceDataType($1 + " short", BaseType::U16); }
    | UNSIGNED INT { $$ = MakeDeviceDataType($1 + " int", BaseType::U32); }
    | UNSIGNED S64 { $$ = MakeDeviceDataType($1 + " long long", BaseType::U64); }
    | F32 { $$ = MakeDeviceDataType("float", $1); }
    | F64 { $$ = MakeDeviceDataType("double", $1); }
    | VOID { $$ = MakeDeviceDataType("void", $1); }
    | IDENTIFIER { $$ = MakeDeviceDataType($1, BaseType::UNKNOWN); }
    | SIGNED IDENTIFIER { $$ = MakeDeviceDataType($1, BaseType::UNKNOWN); }
    | UNSIGNED IDENTIFIER { $$ = MakeDeviceDataType($1, BaseType::UNKNOWN); }
    ;

device_nested_type
    : IDENTIFIER LT device_nested_type_list GT {
        auto type_str = $1 + "<" + $3->GetTypeStr() + ">";
        $3->SetTypeStr(type_str);
        $3->SetDataType(BaseType::UNKNOWN);
        $$ = $3;
      }
    | device_base_type {
        $$ = $1;
      }
    ;

device_nested_type_list
    : /* Empty */ { $$ = MakeDeviceDataType("", BaseType::UNKNOWN); }
    | device_nested_type_list COMMA device_complex_type {
        auto type_str = $3->GetTypeStr() + ", " + $3->GetTypeStr();
        $3->SetTypeStr(type_str);
        $3->SetDataType(BaseType::UNKNOWN);
        $$ = $3;
      }
    | device_type { $$ = $1; }
    ;

device_complex_type
    :  device_complex_type DCOLS device_nested_type  {
        auto type_str = $1->GetTypeStr() + "::" + $3->GetTypeStr();
        $1->SetTypeStr(type_str);
        $1->SetDataType(BaseType::UNKNOWN);
        $$ = $1;
      }
    | device_nested_type {
      $$ = $1;
    }
    ;

device_type
    : device_complex_type { $$ = $1; }
    | device_type STAR {
        auto type_str = $1->GetTypeStr() + " *";
        $1->SetTypeStr(type_str);
        if (!$1->IsNaiveType() || $1->IsPointerType()) {
          $1->SetDataType(BaseType::UNKNOWN);
        }
        $1->pointer_count++;
        $1->SetPointerType(true);
        $$ = $1;
      }
    | device_type AMP {
        auto type_str = $1->GetTypeStr() + " &";
        $1->SetTypeStr(type_str);
        $$ = $1;
      }
    | device_type AND {
        auto type_str = $1->GetTypeStr() + " &&";
        $1->SetTypeStr(type_str);
        $$ = $1;
      }
    | device_type CONST STAR {
        auto type_str =$1->GetTypeStr() + " const *";
        $1->SetTypeStr(type_str);
        if (!$1->IsNaiveType() || $1->IsPointerType()) {
          $1->SetDataType(BaseType::UNKNOWN);
        }
        $1->SetPointerType(true);
        $$ = $1;
      }
    | CONST device_complex_type {
        auto type_str = $2->GetTypeStr() + " " + $1;
        $2->SetTypeStr(type_str);
        $$ = $2;
      }
    | TYPENAME device_complex_type {
        auto type_str = $2->GetTypeStr() + " " + $1;
        $2->SetTypeStr(type_str);
        $$ = $2;
      }
    ;

device_params
    : /* Empty */  { $$ = std::vector<AST::ptr<Choreo::DeviceDataType>>(); }
    | device_param { $$ = std::vector<AST::ptr<Choreo::DeviceDataType>>({$1}); }
    | device_params COMMA device_param { $1.push_back($3); $$ = $1; }
    ;

device_param
    : device_type { $$ = $1; }
    | device_type IDENTIFIER { $$ = $1; }
    | device_type IDENTIFIER ASSIGN DEVICE_EXPR { $$ = $1; $$->init_expr = $4; }
    | device_attr device_param { $$ = $2; $$->attr = $1; }
    ;

device_attr_lists
    : LPAREN device_attr_lists RPAREN {
        $$ = "(" + $2 + ")";
      }
    | IDENTIFIER COMMA device_attr_lists {
        $$ = $1 + ", " + $3;
      }
    | IDENTIFIER LPAREN device_attr_lists RPAREN {
        $$ = $1 + "(" + $3 + ")";
      }
    | IDENTIFIER {
        $$ = $1;
      }
    | VAL { $$ = $1; }
    ;

device_attr
    : ATTR_ID { $$ = $1; }
    | STATIC { $$ = $1; }
    | INLINE { $$ = $1; }
    | EXTERN { $$ = $1; }
    | EXTERN VAL { $$ = $1 + " " + $2; }
    | ATTRIBUTE device_attr_lists {
        $$ = $1 + $2;
      }
    ;

device_function_decl
    : device_type IDENTIFIER LPAREN device_params RPAREN {
        $$ = AST::Make<AST::DeviceFunctionDecl>(@2);
        $$->name = $2;
        $$->ret_type = $1;
        $$->param_types = $4;
      }
    | device_type device_attr IDENTIFIER LPAREN device_params RPAREN {
        $$ = AST::Make<AST::DeviceFunctionDecl>(@2);
        $$->name = $3;
        $$->ret_type = $1;
        $$->param_types = $5;
      }
    | device_attr device_function_decl {
        $$ = $2;
      }
    | DEVICE_TEMPLATE device_function_decl {
        $$ = $2;
        $$->templates = $1;
      }
    ;

// choreo function declaration
dsl_function
    : ATTR_CO return_type IDENTIFIER LPAREN parameter_list RPAREN LBRACE statements RBRACE {
        $$ = AST::Make<AST::ChoreoFunction>(@1);
        $$->name = $3;
        $$->f_decl.name = $3;
        $$->f_decl.ret_type = $2;
        $$->f_decl.params = $5;
        $$->stmts = $8;
        symtab.Reset();
      }
    ;

return_type
    : param_type { $$ = $1; }
    | void_type  { $$ = $1; }
    | auto_type  { $$ = $1; }
    ;

param_type
    : scalar_type { $$ = $1; }
    | STREAM { $$ = AST::Make<AST::DataType>(@1, BaseType::STREAM); }
    | fundamental_type param_mdspan {
        $$ = AST::Make<AST::DataType>(@1, $1, $2);
      }
    ;

param_mdspan
    : MDSPAN LT NUM GT {
        $$ = AST::Make<AST::MultiDimSpans>(@2, "", $3);
      }
    | LBRAKT param_mdspan_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@2, "", $2);
      }
    | LBRAKT LBRAKT param_mdspan_list RBRAKT RBRAKT  {
        $$ = AST::Make<AST::MultiDimSpans>(@3, "", $3);
      }
    | MDSPAN LBRAKT param_mdspan_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, "", $3);
      }
    | MDSPAN LBRAKT LBRAKT param_mdspan_list RBRAKT RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, "", $4);
      }
    ;

param_mdspan_list
    : param_mdspan_list COMMA param_mdspan_val {
        $1->Append($3);
        $$ = $1;
      }
    | param_mdspan_val {
        $$ = AST::Make<AST::MultiValues>(@1);
        $$->Append($1);
      }
    ;

param_mdspan_val
    : QES   { $$ = AST::Make<AST::IntLiteral>(@1); }
    | DQES   { $$ = AST::Make<AST::NoValue>(@1); }
    | num_expr { $$ = $1; }
    | IDENTIFIER {
        $$ = AST::Make<AST::Identifier>(@1, $1);
        if (!symtab.Exists($1)) // allows same dim name
          symtab.AddSymbol($1, MakeIntegerType());
      }
    ;

num_expr
    : NUM   { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    /* related to promotion */
    | num_expr PLUS num_expr { $$ = AST::Make<AST::IntLiteral>(@1, $1->Val() + $3->Val()); }
    | num_expr MINUS num_expr { $$ = AST::Make<AST::IntLiteral>(@1, $1->Val() - $3->Val()); }
    | num_expr STAR num_expr { $$ = AST::Make<AST::IntLiteral>(@1, $1->Val() * $3->Val()); }
    | num_expr SLASH num_expr { $$ = AST::Make<AST::IntLiteral>(@1, $1->Val() / $3->Val()); }
    | num_expr PECET num_expr { $$ = AST::Make<AST::IntLiteral>(@1, $1->Val() % $3->Val()); }
    | LPAREN num_expr RPAREN { $$ = AST::Make<AST::IntLiteral>(@1, $2->Val()); }

void_type
    : VOID  { $$ = AST::Make<AST::DataType>(@1, $1); }
    ;

auto_type
    : AUTO  { $$ = AST::Make<AST::DataType>(@1, BaseType::UNKNOWN); }
    ;

scalar_type
    : fundamental_type { $$ = AST::Make<AST::DataType>(@1, $1, true);}
    | BOOL   { $$ = AST::Make<AST::DataType>(@1, $1, false); }
    | INT { $$ = AST::Make<AST::DataType>(@1, $1, false); }
    ;

mdspan_as_type
    : fundamental_type LBRAKT g_value_list RBRAKT {
        // if it contains a single mdspan, use it directly
        ptr<AST::MultiDimSpans> mds = nullptr;
        if ($3->Count() == 1)
          if (auto e = dyn_cast<AST::Expr>($3->ValueAt(0)))
            if (auto mdss = dyn_cast<AST::MultiDimSpans>(e->GetReference()))
              mds = mdss;

        if (!mds) mds = AST::Make<AST::MultiDimSpans>(@3, "", $3);
        $$ = AST::Make<AST::DataType>(@1, $1, mds);
      }
    | fundamental_type LBRAKT AUTO RBRAKT {
        $$ = AST::Make<AST::DataType>(@1, $1);
        $$->infer_span = true;
      }
    ;

fundamental_type
    : F64   { $$ = $1; }
    | TF32   { $$ = $1; }
    | F32   { $$ = $1; }
    | F16   { $$ = $1; }
    | BF16  { $$ = $1; }
    | F8_E4M3 { $$ = $1; }
    | F8_E5M2 { $$ = $1; }
    | F8_UE4M3 { $$ = $1; }
    | F8_UE8M0 { $$ = $1; }
    | F6_E2M3 { $$ = $1; }
    | F6_E3M2 { $$ = $1; }
    | F4_E2M1 { $$ = $1; }
    | U1 { $$ = $1; }
    | BIN1 { $$ = $1; }
    | S2 { $$ = $1; }
    | U2 { $$ = $1; }
    | S4 { $$ = $1; }
    | U4 { $$ = $1; }
    | S6 { $$ = $1; }
    | U6 { $$ = $1; }
    | U16   { $$ = $1; }
    | S16   { $$ = $1; }
    | U8    { $$ = $1; }
    | S8    { $$ = $1; }
    | U32   { $$ = $1; }
    | S32   { $$ = $1; }
    | U64   { $$ = $1; }
    | S64   { $$ = $1; }
    ;

simple_val
    : integer_value { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | U32_LITERAL { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | S64_LITERAL { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | U64_LITERAL { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | bool_value { $$ = AST::Make<AST::BoolLiteral>(@1, $1); }
    | FPVAL  { $$ = AST::Make<AST::FloatLiteral>(@1, $1); }
    | DFPVAL { $$ = AST::Make<AST::FloatLiteral>(@1, $1); }
    | cstrings { $$ = AST::Make<AST::StringLiteral>(@1, $1); }
    | IDENTIFIER { $$ = AST::Make<AST::Identifier>(@1, $1); }
    | IDENTIFIER FNSPAN { $$ = AST::Make<AST::Identifier>(@1, $1 + $2); }
    ;

int_or_id
    : integer_value { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | IDENTIFIER {
        if (!symtab.Exists($1))
          Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");
        $$ = AST::Make<AST::Identifier>(@1, $1);
      }
    ;

bool_value
    : TRUE { $$ = true; }
    | FALSE { $$ = false; }
    ;

parameter_list
    : /* Empty */ {
        $$ = AST::Make<AST::ParamList>(loc);
      }
    | parameter_list COMMA parameter {
        $1->values.push_back($3);
        $$ = $1;
      }
    | parameter {
        $$ = AST::Make<AST::ParamList>(@1);
        $$->values.push_back($1);
      }
    ;

param_storage
    : STORAGE {
        if ($1 == Storage::SHARED)
          Parser::error(@1, "the shared data can not be used as a parameter.");
        else if ($1 == Storage::LOCAL)
          Parser::error(@1, "the local data can not be used as a parameter.");
        else
          $$ = $1;
      }
    ;

parameter
    : param_type pass_by_ref IDENTIFIER { /* handle parameter type and name here */
        symtab.AddSymbol($3, $1->GetType());
        $$ = AST::Make<AST::Parameter>(@1, $1, AST::Make<AST::Identifier>(@3, $3), $2);
      }
    | param_type pass_by_ref {
        $$ = AST::Make<AST::Parameter>(@1, $1, AST::Make<AST::Identifier>(@1), $2);
      }
    | param_storage param_type pass_by_ref IDENTIFIER { /* handle parameter type and name here */
        symtab.AddSymbol($4, $2->GetType());
        $$ = AST::Make<AST::Parameter>(@1, $2, AST::Make<AST::Identifier>(@4, $4), $3, ParamAttr::GLOBAL_INPUT);
        
      }
    | param_storage param_type pass_by_ref {
        $$ = AST::Make<AST::Parameter>(@1, $2, AST::Make<AST::Identifier>(@2), $3, ParamAttr::GLOBAL_INPUT);
      }
    ;

pass_by_ref
    : /* Empty */ { $$ = false; }
    | AMP { $$ = true; }
    ;

statements
    : /* no statement */ { $$ = AST::Make<AST::MultiNodes>(loc); }
    | statements statement {
        if (auto mstmts = dyn_cast<AST::MultiNodes>($2)) {
          for (auto stmt : mstmts->AllSubs())
            $1->Append(stmt); // append multi-satements
        } else
          $1->Append($2);

        $$ = $1;
      }
    | statements SEMCOL { $$ = $1; }
    ;

statement
    : declarations SEMCOL        { $$ = $1; }
    | assignments  SEMCOL        { $$ = $1; }
    | dma_stmt     SEMCOL        { $$ = $1; }
    | mma_stmt     SEMCOL        { $$ = $1; }
    | wait_stmt    SEMCOL        { $$ = $1; }
    | trigger_stmt SEMCOL        { $$ = $1; }
    | call_stmt    SEMCOL        { $$ = $1; }
    | swap_stmt    SEMCOL        { $$ = $1; }
    | return_stmt  SEMCOL        { $$ = $1; }
    | sync_stmt    SEMCOL        { $$ = $1; }
    | inlcpp_stmt  SEMCOL        { $$ = $1; }
    | break_stmt   SEMCOL        { $$ = $1; }
    | continue_stmt  SEMCOL      { $$ = $1; }
    | paraby_block               { $$ = $1; }
    | within_block               { $$ = $1; }
    | inthreads_block            { $$ = $1; }
    | while_block                { $$ = $1; }
    | if_else_block              { $$ = $1; }
    | foreach_block              { $$ = $1; }
    ;

sync_stmt
    : SYNC DOT STORAGE {
        $$ = AST::Make<AST::Synchronize>(@1, $3);
      }
    ;

return_stmt
    : RET            { $$ = AST::Make<AST::Return>(@1); }
    | RET returnable { $$ = AST::Make<AST::Return>(@1, $2); }
    ;

stmts_block
    : LBRACE statements RBRACE { $$ = $2; }
    | statement {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    | SEMCOL { $$ = AST::Make<AST::MultiNodes>(@1); }
    ;

paraby_block
    : PARA sync_type {
        paraby_symbols.clear();
      } parabys stmts_block {
        $4->SetAsync($2);
        // attach statement to the inner-most pb
        auto pb = $4;
        while (!pb->stmts->None() && isa<AST::ParallelBy>(pb->stmts->SubAt(0)))
          pb = cast<AST::ParallelBy>(pb->stmts->SubAt(0));
        assert(pb->stmts->None() && "expect no statement.");
        pb->stmts = $5;
        $$ = $4;
      }
    ;

parabys
    : parabys COMMA paraby note_pl {
        // add the paraby as the first stmt of inner-most parallel-by
        $1->SetLevel($4);
        auto pb = $1;
        while (!pb->stmts->None() && isa<AST::ParallelBy>(pb->stmts->SubAt(0)))
          pb = cast<AST::ParallelBy>(pb->stmts->SubAt(0));
        pb->stmts->Append($3);
        $$ = $1;
      }
    | paraby note_pl {
        $1->SetLevel($2);
        $$ = $1;
      }
    ; /* do not allow empty paraby */

paraby
    : BY s_expr {
        auto anon_id = AST::Make<AST::Identifier>(@1, SymbolTable::GetAnonName());
        $$ = AST::Make<AST::ParallelBy>(@1, anon_id, $2);
        $$->SetEnforced();
      }
    | IDENTIFIER BY s_expr {
        if (paraby_symbols.find($1) != paraby_symbols.end()) {
          Parser::error(@1, "The parallel variable '" + $1 + "' has been used in the same parallelby block.");
          YYERROR;
        }
        paraby_symbols.insert($1);
        symtab.AddSymbol($1, MakeUnknownType());
        // tolerate 'parallel p by [2]'
        if ($3->IsReference()) {
          if (auto mds = dyn_cast<AST::MultiDimSpans>($3->GetReference())) {
            if (auto mv = dyn_cast<AST::MultiValues>(mds->list)) {
              if (mv->Count() > 1) {
                Parser::error(@1, "The parallel variable '" + $1 + "' can not be assigned multiple bounds.");
                YYERROR;
              } else
                $$ = AST::Make<AST::ParallelBy>(@1, AST::Make<AST::Identifier>(@1, $1), cast<AST::Expr>(mv->ValueAt(0)));
            } else {
              auto expr = dyn_cast<AST::Expr>(mds->list);
              assert(expr);
              $$ = AST::Make<AST::ParallelBy>(@1, AST::Make<AST::Identifier>(@1, $1), expr);
            }
          } else
            $$ = AST::Make<AST::ParallelBy>(@1, AST::Make<AST::Identifier>(@1, $1), $3);
        } else
          $$ = AST::Make<AST::ParallelBy>(@1, AST::Make<AST::Identifier>(@1, $1), $3);
        $$->SetEnforced();
      }
    | IDENTIFIER ASSIGN LBRACE id_list RBRACE BY LBRAKT value_list RBRAKT {
        if (paraby_symbols.find($1) != paraby_symbols.end()) {
          Parser::error(@1, "The parallel variable '" + $1 + "' has been used in the same parallelby block.");
          //YYERROR;
        }
        paraby_symbols.insert($1);
        symtab.AddSymbol($1, MakeUnknownType());
        for (auto id : $4->AllValues()) {
          auto name = cast<AST::Identifier>(id)->name;
          if (paraby_symbols.find(name) != paraby_symbols.end()) {
            Parser::error(@1, "The parallel variable '" + name + "' has been used in the same parallelby block.");
            //YYERROR;
          }
          paraby_symbols.insert(name);
          symtab.AddSymbol(name, MakeUnknownType());
        }
        $8->SetDelimiter(", ");
        $$ = AST::Make<AST::ParallelBy>(@1, AST::Make<AST::Identifier>(@1, $1), nullptr, $4, $8);
        $$->SetBracketed(true);
        $$->SetEnforced();
      }
    | LBRACE id_list RBRACE BY LBRAKT value_list RBRAKT {
        for (auto id : $2->AllValues()) {
          auto name = cast<AST::Identifier>(id)->name;
          if (paraby_symbols.find(name) != paraby_symbols.end()) {
            Parser::error(@1, "The parallel variable '" + name + "' has been used in the same parallelby block.");
            //YYERROR;
          }
          paraby_symbols.insert(name);
          symtab.AddSymbol(name, MakeUnknownType());
        }
        auto anon_id = AST::Make<AST::Identifier>(@1, SymbolTable::GetAnonName());
        $6->SetDelimiter(", ");
        $$ = AST::Make<AST::ParallelBy>(@1, anon_id, nullptr, $2, $6);
        $$->SetBracketed(true);
        $$->SetEnforced();
      }
    ;

assignments
    : assignments COMMA assignment {
        $1->Append($3);
        $$ = $1;
      }
    | assignment {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ;

declarations
    : declaration {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    | multi_decls { $$ = $1; }
    ;

declaration
    : named_mdspan_decl   { $$ = $1; }
    | named_ituple_decl   { $$ = $1; }
    ;

multi_decls
    : named_spanned_decls { $$ = $1; }
    | named_scalar_decls  { $$ = $1; }
    | named_event_decls   { $$ = $1; }
    ;

named_scalar_decls
    : MUTABLE scalar_type scalar_decls {
        assert($2->isScalar() && "Not a scalar type.");
        $2->SetMutable(true);
        $2->ReGenSemaType();
        for (auto sub : $3->AllSubs()) {
          auto decl = cast<AST::NamedVariableDecl>(sub);
          decl->type = cast<AST::DataType>($2->Clone());
          decl->SetMutable(true);
          symtab.AddSymbol(decl->name_str, $2->GetType()->Clone());
          // override the data type
        }
        $$ = $3;
      }
    | scalar_type scalar_decls {
        assert($1->isScalar() && "Not a scalar type.");
        for (auto sub : $2->AllSubs()) {
          auto decl = cast<AST::NamedVariableDecl>(sub);
          decl->type = cast<AST::DataType>($1->Clone());
          decl->SetMutable($1->IsMutable());
          symtab.AddSymbol(decl->name_str, $1->GetType()->Clone());
          // override the data type
        }
        $$ = $2;
      }
    | MUTABLE scalar_decls {
        // must apply type inference
        for (auto sub : $2->AllSubs()) {
          auto decl = cast<AST::NamedVariableDecl>(sub);
          decl->SetMutable(true);
          symtab.AddSymbol(decl->name_str, MakeUnknownType());
          // update the mutable specifier for scalar types
          decl->type->SetMutable(true);
        }
        $$ = $2;
      }
    | STORAGE named_scalar_decls {
        for (auto sub : $2->AllSubs()) {
          auto decl = cast<AST::NamedVariableDecl>(sub);
          decl->SetMemory(AST::Make<AST::Memory>(@1, $1));
        }
        $$ = $2;
    }
    ;

scalar_decls
    : scalar_decls COMMA scalar_decl {
        $1->Append($3);
        $$ = $1;
      }
    | scalar_decl {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ;

scalar_decl
    : IDENTIFIER ASSIGN s_expr {
        $$ = AST::Make<AST::NamedVariableDecl>(@1, $1,
             AST::Make<AST::DataType>(@1, BaseType::UNKNOWN), nullptr, $3);
      }
    | IDENTIFIER ASSIGN call_stmt {
        $3->SetExpr();
        $$ = AST::Make<AST::NamedVariableDecl>(@1, $1,
             AST::Make<AST::DataType>(@1, BaseType::UNKNOWN), nullptr, $3);
      }
    | IDENTIFIER ASSIGN LBRAKT {
        Parser::error(@3, "must use '{' and '}' to define an ituple.");
        YYERROR;
      }
    | IDENTIFIER {
        $$ = AST::Make<AST::NamedVariableDecl>(@1, $1,
            AST::Make<AST::DataType>(@1, BaseType::UNKNOWN));
      }
    | IDENTIFIER LBRACE s_expr RBRACE {
      $$ = AST::Make<AST::NamedVariableDecl>(@1, $1,
            AST::Make<AST::DataType>(@1, BaseType::UNKNOWN), nullptr, $3);
      }
    ;

named_event_decls
    : storage_qual EVENT event_decls {
        for (auto sub : $3->AllSubs()) {
          auto decl = cast<AST::NamedVariableDecl>(sub);
          auto sym_name = decl->name_str;
          symtab.AddSymbol(sym_name, MakeEventType($1->Get()));
          // override the data type
          decl->mem = cast<AST::Memory>(cast<AST::Memory>($1->Clone()));
        }
        $$ = $3;
      }
    ;

event_decls
    : event_decls COMMA event_decl {
        $1->Append($3);
        $$ = $1;
      }
    | event_decl {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ;

event_decl
    : IDENTIFIER optional_array_dims {
        $$ = AST::Make<AST::NamedVariableDecl>(@1, $1,
             AST::Make<AST::DataType>(@1, BaseType::EVENT, $2), nullptr, $2);
      }
    ;

named_spanned_decls
    : STORAGE mdspan_as_type spanned_decls {
        auto mem = AST::Make<AST::Memory>(@1, $1);
        for (auto item : $3->AllSubs()) {
          auto decl = cast<AST::NamedVariableDecl>(item);
          symtab.AddSymbol(decl->name_str, $2->GetType()->Clone());
          decl->type = cast<AST::DataType>($2->Clone());
          if (decl->IsArray()) {
            decl->type->array_dims = decl->ArrayDimensions();
            decl->type->ReGenSemaType();
          }
          decl->mem = cast<AST::Memory>(mem->Clone());
        }
        $3->SetLOC(@1);
        $$ = $3;
      }
    | mdspan_as_type spanned_decls {
        auto mem = AST::Make<AST::Memory>(loc);
        for (auto item : $2->AllSubs()) {
          auto decl = cast<AST::NamedVariableDecl>(item);
          symtab.AddSymbol(decl->name_str, $1->GetType()->Clone());
          decl->type = cast<AST::DataType>($1->Clone());
          if (decl->IsArray()) {
            decl->type->array_dims = decl->ArrayDimensions();
            decl->type->ReGenSemaType();
          }
          decl->mem = cast<AST::Memory>(mem->Clone());
        }
        $2->SetLOC(@1);
        $$ = $2;
      }
    ;

spanned_decls
    : spanned_decl {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    | spanned_decls COMMA spanned_decl {
        $1->Append($3);
        $$ = $1;
      }
    ;

spanned_decl
    : IDENTIFIER optional_array_dims {
        $$ = AST::Make<AST::NamedVariableDecl>(@1, $1, nullptr, nullptr, nullptr, $2);
      }
    | IDENTIFIER optional_array_dims span_init_val {
        $$ = AST::Make<AST::NamedVariableDecl>(@1, $1, nullptr, nullptr, nullptr, $2, $3);
      }
    ;

optional_array_dims
    : /*empty*/ {}
    | optional_array_dims LBRAKT NUM RBRAKT {
        $1.push_back($3);
        $$ = $1;
      }
    ;

subscriptions
    : LBRAKT s_expr RBRAKT {
        $$ = AST::Make<AST::MultiValues>(loc);
        $$->Append($2);
      }
    | subscriptions LBRAKT s_expr RBRAKT {
        $1->Append($3);
        $$ = $1;
      }
    ;

span_init_val
    : LBRACE s_expr RBRACE {
        $$ = $2;
      }
    ;

g_expr /* any expression like 'a[...]' are interpreted as derivation */
    : { parsing_derivation_decl = true; } s_expr {
        $$ = $2;
        parsing_derivation_decl = false;
      }
    ;

value_list /* contains at least one value */
    : value_list COMMA s_expr {
        $1->Append($3);
        $$ = $1;
      }
    | s_expr {
        $$ = AST::Make<AST::MultiValues>(@1);
        $$->Append($1);
      }
    ;

g_value_list /* contains at least two value */
    : g_value_list COMMA g_expr {
        $1->Append($3);
        $$ = $1;
      }
    | g_expr {
        $$ = AST::Make<AST::MultiValues>(@1);
        $$->Append($1);
      }
    ;

template_val
    : NUM { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | MINUS NUM { $$ = AST::Make<AST::IntLiteral>(@1, -$2); }
    | spanid { $$ = AST::Make<AST::Identifier>(@1, $1); }
    | scalar_type { $$ = $1; }
    ;

template_value_expr
    : template_val      { $$ = AST::Make<AST::Expr>(@1, $1); }
    | UBOUND IDENTIFIER {
        $$ = AST::Make<AST::Expr>(@1, "ubound", AST::Make<AST::Identifier>(@2, $2));
      }
    | template_value_expr PLUS template_value_expr { $$ = AST::Make<AST::Expr>(@1, "+", $1, $3); }
    | template_value_expr MINUS template_value_expr { $$ = AST::Make<AST::Expr>(@1, "-", $1, $3); }
    | template_value_expr STAR template_value_expr { $$ = AST::Make<AST::Expr>(@1, "*", $1, $3); }
    | template_value_expr SLASH template_value_expr { $$ = AST::Make<AST::Expr>(@1, "/", $1, $3); }
    | template_value_expr PECET template_value_expr { $$ = AST::Make<AST::Expr>(@1, "%", $1, $3); }
    | CDIV LPAREN template_value_expr COMMA template_value_expr RPAREN { $$ = AST::Make<AST::Expr>(@1, "cdiv", $3, $5); }
    | template_value_expr UBOUND template_value_expr {$$ = AST::Make<AST::Expr>(@1, "#", $1, $3); }
    | template_value_expr LPAREN int_or_id RPAREN {
        $$ = AST::Make<AST::Expr>(@1, "dimof", $1, AST::Make<AST::IntIndex>(@3, $3));
      }
    ;

template_value_list
    : /* Empty list */ {
        $$ = AST::Make<AST::MultiValues>(loc);
      }
    | template_value_list COMMA template_value_expr {
        $1->Append($3);
        $$ = $1;
      }
    | template_value_expr {
        $$ = AST::Make<AST::MultiValues>(@1);
        $$->Append($1);
      }
    ;

template_params
    : LT template_value_list GT {
        $$ = $2;
      }
    ;

spanid
    : IDENTIFIER FNSPAN { $$ = $1 + $2; }
    | IDENTIFIER { $$ = $1; }
    ;

mdspan_derivation
    : IDENTIFIER LBRAKT { parsing_derivation_decl = true; } value_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, $1, $4);
        parsing_derivation_decl = false;
      }
    | IDENTIFIER FNSPAN LBRAKT { parsing_derivation_decl = true; } value_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, $1, $5);
        parsing_derivation_decl = false;
      }
    ;

mdspan_list
    : LBRAKT { parsing_derivation_decl = true; } value_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, "", $3);
        parsing_derivation_decl = false;
      }
    ;

named_mdspan_decl
    : MDSPAN IDENTIFIER COL { parsing_derivation_decl = true; } s_expr {
        symtab.AddSymbol($2, MakeUninitMDSpanType());
        $$ = AST::Make<AST::NamedTypeDecl>(@2, $2, $5);
        parsing_derivation_decl = false;
      }
    | MDSPAN LT NUM GT IDENTIFIER COL { parsing_derivation_decl = true; } s_expr {
        symtab.AddSymbol($5, MakeRankedMDSpanType($3));
        $$ = AST::Make<AST::NamedTypeDecl>(@5, $5, $8, $3);
        parsing_derivation_decl = false;
      }
    | IDENTIFIER COL { parsing_derivation_decl = true; } s_expr {
        symtab.AddSymbol($1, MakeUninitMDSpanType());
        $$ = AST::Make<AST::NamedTypeDecl>(@1, $1, $4);
        parsing_derivation_decl = false;
      }
    | MDSPAN IDENTIFIER ASSIGN {
        Parser::error(@3, "must use ':' to initialize a mdspan.");
        YYERROR;
      }
    | MDSPAN LT NUM GT IDENTIFIER ASSIGN {
        Parser::error(@3, "must use ':' to initialize a mdspan.");
        YYERROR;
      }
    ;

ituple_list
    : LBRACE g_value_list RBRACE {
        $2->SetDelimiter(", ");
        $$ = AST::Make<AST::IntTuple>(@1, "", $2);
      }
    ;

named_ituple_decl
    : ITUPLE IDENTIFIER ASSIGN g_expr {
        symtab.AddSymbol($2, MakeUninitITupleType());
        $$ = AST::Make<AST::NamedVariableDecl>(@2,
              $2, AST::Make<AST::DataType>(@1, BaseType::ITUPLE), nullptr, $4);
      }
    | ITUPLE LT NUM GT IDENTIFIER ASSIGN g_expr {
        symtab.AddSymbol($5, MakeUninitITupleType());
        $$ = AST::Make<AST::NamedVariableDecl>(@5,
              $5, AST::Make<AST::DataType>(@1, BaseType::ITUPLE, $3), nullptr, $7);
      }
    | ITUPLE IDENTIFIER COL {
        Parser::error(@3, "must use '=' to initialize an ituple.");
        YYERROR;
      }
    | ITUPLE IDENTIFIER LBRAKT s_expr RBRAKT {
        Parser::error(@2, "no ituple array is allowed.");
        YYERROR;
      }
    | ITUPLE IDENTIFIER {
        Parser::error(@2, "an ituple must be initialized.");
        YYERROR;
      }
    ; // do not allow uninitialized ituple

/*
spanas_spanned_decl
    : IDENTIFIER ASSIGN { ignore_fndata = true; } ids_expr span_as {
        symtab.AddSymbol($1, MakeUnknownType());
        auto ide = ElementMultiValues($4);
        $5->id = ide.first;
        auto expr = AST::Make<AST::Expr>(@1, $5);
        $$ = AST::Make<AST::NamedVariableDecl>(@1,
              $1, AST::Make<AST::DataType>(@1, BaseType::UNKNOWN), nullptr, expr);
        // TODO: make it a named variable instead of expr assignment
        // $$ = AST::Make<AST::NamedVariableDecl>(@1, $1, nullptr, nullptr, $5, $4);
        ignore_fndata = false;
      }
    ;
*/

note_pl
    : COL PBLEVEL { $$ = $2; }
    | /* empty */ { $$ = ParallelLevel::NONE; }
    ;

storage_qual
    : /* Empty */ { $$ = AST::Make<AST::Memory>(loc); }
    | STORAGE { $$ = AST::Make<AST::Memory>(@1, $1); }
    ;

arith_operation
    : PLUS  { $$ = "+"; }
    | MINUS { $$ = "-"; }
    | STAR  { $$ = "*"; }
    | SLASH { $$ = "/"; }
    | PECET { $$ = "%"; }
    ;

assignment
    : IDENTIFIER ASSIGN s_expr {
        // Note: It checks the symbol existance without considering its scope.
        //       As a result, it generates NamedVariableDecls conservatively,
        //       where some symbol with a same name of outer scope ones is
        //       treated as Assignment. AST visitors must take care of this.
        if (!symtab.Exists($1)) {
          // since the symbol is not defined, it is a declaration without type annotation
          symtab.AddSymbol($1, MakeUnknownType());
          $$ = AST::Make<AST::NamedVariableDecl>(@1,
                $1, AST::Make<AST::DataType>(@1, BaseType::UNKNOWN), nullptr, $3);
          break;
        } else {
          $$ = AST::Make<AST::Assignment>(@2, $1, $3);
        }
      }
    | IDENTIFIER ASSIGN ids_expr span_as {
        symtab.AddSymbol($1, MakeUnknownType());
        auto ide = ElementMultiValues($3);
        $4->id = ide.first;
        auto expr = AST::Make<AST::Expr>(@1, $4);
        $$ = AST::Make<AST::NamedVariableDecl>(@1,
              $1, AST::Make<AST::DataType>(@1, BaseType::UNKNOWN), nullptr, expr);
        // TODO: make it a named variable instead of expr assignment
        // $$ = AST::Make<AST::NamedVariableDecl>(@1, $1, nullptr, nullptr, $5, $4);
      }
    | IDENTIFIER ASSIGN ituple_derivation {
        $$ = AST::Make<AST::NamedVariableDecl>(@1,
              $1, AST::Make<AST::DataType>(@1, BaseType::ITUPLE), nullptr, $3);
      }
    | IDENTIFIER ASSIGN call_stmt {
        $3->SetExpr();
        $$ = AST::Make<AST::Assignment>(@1, $1, $3);
      }
    | IDENTIFIER arith_operation ASSIGN s_expr {
        if (!symtab.Exists($1)) {
          Parser::error(@1, "The symbol '" + $1 + "` has not been defined.");
        } else {
          auto da = AST::Make<AST::DataAccess>(@1, AST::Make<AST::Identifier>(@1, $1));
          auto da_expr = AST::Make<AST::Expr>(@1, da);
          da_expr->AddNote("update");
          $$ = AST::Make<AST::Assignment>(@3, da, AST::Make<AST::Expr>(@2, $2, da_expr, $4));
          $$->AddNote("update");
        }
      }
    | IDENTIFIER ASSIGN select_expr {
        if (!symtab.Exists($1)) {
          // since the symbol is not defined, it is a declaration without type annotation
          symtab.AddSymbol($1, MakeUnknownType());
          $$ = AST::Make<AST::NamedVariableDecl>(@1, $1,
                 AST::Make<AST::DataType>(@1, BaseType::UNKNOWN), nullptr, $3);
          break;
        } else {
          $$ = AST::Make<AST::Assignment>(@2, $1, $3);
        }
      }
    | data_element ASSIGN s_expr {
        $$ = AST::Make<AST::Assignment>(@1, $1, $3);
      }
    | data_element ASSIGN call_stmt {
        $3->SetExpr();
        $$ = AST::Make<AST::Assignment>(@1, $1, $3);
      }
    | data_element arith_operation ASSIGN s_expr {
        std::string dname = $1->GetDataName();
        if (SuffixedWith(dname, ".data")) dname = dname.substr(0, dname.size() - 5);
        if (!symtab.Exists(dname)) {
          Parser::error(@1, "The symbol '" + dname + "` has not been defined.");
        } else {
          $$ = AST::Make<AST::Assignment>(@3, cast<AST::DataAccess>($1->Clone()),
                AST::Make<AST::Expr>(@2, $2, AST::Make<AST::Expr>(@1, $1), $4));
        }
      }
    | IDENTIFIER ASSIGN LBRAKT {
        Parser::error(@3, "must use '{' and '}' to define an ituple.");
        YYERROR;
      }
    ;

/* simple expression: no ambiguity with [] */
s_expr
    : s_expr PLUS s_expr { $$ = AST::Make<AST::Expr>(@1, "+", $1, $3); }
    | s_expr MINUS s_expr { $$ = AST::Make<AST::Expr>(@1, "-", $1, $3); }
    | s_expr STAR s_expr { $$ = AST::Make<AST::Expr>(@1, "*", $1, $3); }
    | s_expr SLASH s_expr { $$ = AST::Make<AST::Expr>(@1, "/", $1, $3); }
    | s_expr PECET s_expr { $$ = AST::Make<AST::Expr>(@1, "%", $1, $3); }
    | CDIV LPAREN s_expr COMMA s_expr RPAREN { $$ = AST::Make<AST::Expr>(@1, "cdiv", $3, $5); }
    | s_expr OR s_expr { $$ = AST::Make<AST::Expr>(@1, "||", $1, $3); }
    | s_expr PIPE s_expr { $$ = AST::Make<AST::Expr>(@1, "|", $1, $3); }
    | s_expr AMP s_expr { $$ = AST::Make<AST::Expr>(@1, "&", $1, $3); }
    | s_expr CARET s_expr { $$ = AST::Make<AST::Expr>(@1, "^", $1, $3); }
    | s_expr LSHIFT s_expr { $$ = AST::Make<AST::Expr>(@1, "<<", $1, $3); }
    | s_expr RSHIFT s_expr { $$ = AST::Make<AST::Expr>(@1, ">>", $1, $3); }
    | s_expr AND s_expr { $$ = AST::Make<AST::Expr>(@1, "&&", $1, $3); }
    | s_expr UBOUND s_expr {$$ = AST::Make<AST::Expr>(@1, "#", $1, $3); }
    | s_expr UBPLUS s_expr { $$ = AST::Make<AST::Expr>(@1, "#+", $1, $3); }
    | s_expr UBMINUS s_expr { $$ = AST::Make<AST::Expr>(@1, "#-", $1, $3); }
    | s_expr UBSTAR s_expr { $$ = AST::Make<AST::Expr>(@1, "#*", $1, $3); }
    | s_expr UBSLASH s_expr { $$ = AST::Make<AST::Expr>(@1, "#/", $1, $3); }
    | s_expr UBPECET s_expr { $$ = AST::Make<AST::Expr>(@1, "#%", $1, $3); }
    | NOT s_expr { $$ = AST::Make<AST::Expr>(@1, "!", $2); }
    | TILDE s_expr { $$ = AST::Make<AST::Expr>(@1, "~", $2); }
    | LPAREN s_expr RPAREN {
        // Does String "(0)" represent an indexing operation or an arithmetic operation
        if (!parsing_derivation_decl) {
          $$ = $2;
          break;
        }

        if (auto ref = $2->GetReference()) {
          if (isa<AST::IntLiteral>(ref.get())) {
            $$ = AST::Make<AST::Expr>(@2, AST::Make<AST::IntIndex>(@2, ref));
            break;
          } else if (auto id = dyn_cast<AST::Identifier>(ref.get())) {
            if (!symtab.Exists(id->name))
              Parser::error(id->LOC(),
                "The symbol `" + id->name + "' has not been defined.");

            $$ =  AST::Make<AST::Expr>(@2, AST::Make<AST::IntIndex>(@2, ref));
            break;
          }
        }

        // or else, it is arithmetical
        $$ = $2;
      }
    | LPAREN s_expr RPAREN QES s_expr COL s_expr {
        $$ = AST::Make<AST::Expr>(@1, "?", $2, $5, $7);
      }
    | s_expr LT s_expr { $$ = AST::Make<AST::Expr>(@1, "<", $1, $3); }
    | s_expr GT s_expr { $$ = AST::Make<AST::Expr>(@1, ">", $1, $3); }
    | s_expr EQ s_expr { $$ = AST::Make<AST::Expr>(@1, "==", $1, $3); }
    | s_expr NE s_expr { $$ = AST::Make<AST::Expr>(@1, "!=", $1, $3); }
    | s_expr LE s_expr { $$ = AST::Make<AST::Expr>(@1, "<=", $1, $3); }
    | s_expr GE s_expr { $$ = AST::Make<AST::Expr>(@1, ">=", $1, $3); }
    | simple_val       { $$ = AST::Make<AST::Expr>(@1, $1); }
    | ituple_list      { $$ = AST::Make<AST::Expr>(@1, $1); }
    | mdspan_list      { $$ = AST::Make<AST::Expr>(@1, $1); }
    | dataid_expr      { $$ = $1; }
    | subscript_like_expr { $$ = $1; }
    | s_expr LPAREN int_or_id RPAREN {
        $$ = AST::Make<AST::Expr>(@1, "dimof", $1, AST::Make<AST::IntIndex>(@3, $3));
      }
    | UBOUND IDENTIFIER {
        $$ = AST::Make<AST::Expr>(@1, "ubound", AST::Make<AST::Identifier>(@2, $2));
      }
    | AMP IDENTIFIER {
        $$ = AST::Make<AST::Expr>(@1, "addrof", AST::Make<AST::Identifier>(@2, $2));
      }
    | AMP data_element {
        $$ = AST::Make<AST::Expr>(@1, "addrof", $2);
      }
    | PPLUS IDENTIFIER {
        $$ = AST::Make<AST::Expr>(@1, "++", AST::Make<AST::Identifier>(@1, $2));
      }
    | MMINUS IDENTIFIER {
        $$ = AST::Make<AST::Expr>(@1, "--", AST::Make<AST::Identifier>(@1, $2));
      }
    | data_element { $$ = AST::Make<AST::Expr>(@1, $1); }
    | call_expr { $$ = $1; }
    | const_sizeof { $$ = AST::MakeIntExpr(@1, $1); }
    | sizeof_expr { $$ = $1; }
    ;

internal_sizeof_expr
    : IDENTIFIER FNSPAN { $$ = AST::Make<AST::Expr>(@1, AST::Make<AST::Identifier>(@1, $1 + $2)); }
    | int_or_id { $$ = AST::Make<AST::Expr>(@1, $1); }
    | internal_sizeof_expr PLUS internal_sizeof_expr { $$ = AST::Make<AST::Expr>(@1, "+", $1, $3); }
    | internal_sizeof_expr MINUS internal_sizeof_expr { $$ = AST::Make<AST::Expr>(@1, "-", $1, $3); }
    | internal_sizeof_expr STAR internal_sizeof_expr { $$ = AST::Make<AST::Expr>(@1, "*", $1, $3); }
    | internal_sizeof_expr SLASH internal_sizeof_expr { $$ = AST::Make<AST::Expr>(@1, "/", $1, $3); }
    | internal_sizeof_expr PECET internal_sizeof_expr { $$ = AST::Make<AST::Expr>(@1, "%", $1, $3); }
    | CDIV LPAREN internal_sizeof_expr COMMA internal_sizeof_expr RPAREN { $$ = AST::Make<AST::Expr>(@1, "cdiv", $3, $5); }
    ;

sizeof_expr
    : PIPE internal_sizeof_expr PIPE { $$ = AST::Make<AST::Expr>(@1, "sizeof", $2); }
    ;

const_sizeof /* make it immediate values */
    : PIPE S64 PIPE { $$ = 8; }
    | PIPE U64 PIPE { $$ = 8; }
    | PIPE S32 PIPE { $$ = 4; }
    | PIPE U32 PIPE { $$ = 4; }
    | PIPE S16 PIPE { $$ = 2; }
    | PIPE U16 PIPE { $$ = 2; }
    | PIPE S8 PIPE { $$ = 1; }
    | PIPE U8 PIPE { $$ = 1; }
    | PIPE U6 PIPE { $$ = 1; }
    | PIPE S6 PIPE { $$ = 1; }
    | PIPE U4 PIPE { $$ = 1; }
    | PIPE S4 PIPE { $$ = 1; }
    | PIPE U2 PIPE { $$ = 1; }
    | PIPE S2 PIPE { $$ = 1; }
    | PIPE BIN1 PIPE { $$ = 1; }
    | PIPE U1 PIPE { $$ = 1; }
    | PIPE F64 PIPE { $$ = 8; }
    | PIPE TF32 PIPE { $$ = 4; }
    | PIPE F32 PIPE { $$ = 4; }
    | PIPE BF16 PIPE { $$ = 2; }
    | PIPE F16 PIPE { $$ = 2; }
    | PIPE F8_E4M3 PIPE { $$ = 1; }
    | PIPE F8_E5M2 PIPE { $$ = 1; }
    | PIPE F8_UE4M3 PIPE { $$ = 1; }
    | PIPE F8_UE8M0 PIPE { $$ = 1; }
    | PIPE F6_E2M3 PIPE { $$ = 1; }
    | PIPE F6_E3M2 PIPE { $$ = 1; }
    | PIPE F4_E2M1 PIPE { $$ = 1; }
    ;

mdspan_expr
    : mdspan_expr PLUS  mdspan_expr { $$ = AST::Make<AST::Expr>(@1, "+", $1, $3); }
    | mdspan_expr MINUS mdspan_expr { $$ = AST::Make<AST::Expr>(@1, "-", $1, $3); }
    | mdspan_expr STAR  mdspan_expr { $$ = AST::Make<AST::Expr>(@1, "*", $1, $3); }
    | mdspan_expr SLASH mdspan_expr { $$ = AST::Make<AST::Expr>(@1, "/", $1, $3); }
    | mdspan_expr PECET mdspan_expr { $$ = AST::Make<AST::Expr>(@1, "%", $1, $3); }
    | mdspan_operator { $$ = $1; }
    ;

mdspan_operator
    : mdspan_val_expr { $$ = $1; }
    | ituple_list { $$ = AST::Make<AST::Expr>(@1, $1); }
    | integer_value { $$ = AST::Make<AST::Expr>(@1, AST::Make<AST::IntLiteral>(@1, $1)); }
/*    | spanid LPAREN int_or_id RPAREN {
        auto id_expr = AST::MakeIdExpr(@1, $1);
        auto idx = AST::Make<AST::IntIndex>(@3, $3);
        $$ = AST::Make<AST::Expr>(@1, "dimof", id_expr, idx);
      }
    | UBOUND IDENTIFIER {
        $$ = AST::Make<AST::Expr>(@1, "ubound", AST::Make<AST::Identifier>(@2, $2));
      } */
    ;

unnamed_mdspan_decl
    : mdspan_list       { $$ = $1; }
    | mdspan_derivation { $$ = $1; }
    ;

mdspan_val_expr
    : unnamed_mdspan_decl { $$ = AST::Make<AST::Expr>(@1, $1); }
    | spanid { $$ = AST::MakeIdExpr(@1, $1); }
    ;

dataid_expr
    : IDENTIFIER FNDATA {
        if (ignore_fndata) {
          $$ = AST::MakeIdExpr(@1, $1);
        } else {
          $$ = AST::Make<AST::Expr>(@1, "dataof", AST::MakeIdExpr(@1, $1));
        }
      }
    | IDENTIFIER FNMDATA {
        if (ignore_fndata) {
          $$ = AST::MakeIdExpr(@1, $1);
        } else {
          $$ = AST::Make<AST::Expr>(@1, "mdataof", AST::MakeIdExpr(@1, $1));
        }
      }
    ;

within_block
    : WITH withins stmts_block {
        $$ = AST::Make<AST::WithBlock>(@1);
        $$->withins = $2;
        $$->stmts = $3;
      }
    | WITH withins where_clause stmts_block {
        $$ = AST::Make<AST::WithBlock>(@1);
        $$->withins = $2;
        $$->reqs = $3;
        $$->stmts = $4;
      }
    | FOREACH withins stmts_block {
        $$ = AST::Make<AST::WithBlock>(@1);
        $$->withins = $2;
        // compose the range expression for 'foreach'
        auto mn = AST::Make<AST::MultiValues>(@2);
        for (auto item : $2->AllSubs()) {
          auto wi = cast<AST::WithIn>(item);
          if (wi->with == nullptr) {
            assert(wi->with_matchers != nullptr);
            for (auto wm : wi->with_matchers->AllValues()) {
              auto id = cast<AST::Identifier>(wm);
              mn->Append(AST::Make<AST::LoopRange>(id->LOC(), id));
            }
          } else
            mn->Append(AST::Make<AST::LoopRange>(wi->LOC(), wi->with));
        }
        auto fe = AST::Make<AST::ForeachBlock>(@1, mn, $3);
        $$->stmts = AST::Make<AST::MultiNodes>(@3);
        $$->stmts->Append(fe);
      }
      | FOREACH withins COL suffix_exprs stmts_block {
        $$ = AST::Make<AST::WithBlock>(@1);
        $$->withins = $2;
        // compose the range expression for 'foreach'
        auto mv = AST::Make<AST::MultiValues>(@2);
        for (auto item : $2->AllSubs()) {
          auto wi = cast<AST::WithIn>(item);
          if (wi->with == nullptr) {
            assert(wi->with_matchers != nullptr);
            for (auto wm : wi->with_matchers->AllValues()) {
              auto id = cast<AST::Identifier>(wm);
              mv->Append(AST::Make<AST::LoopRange>(id->LOC(), id));
            }
          } else
            mv->Append(AST::Make<AST::LoopRange>(wi->LOC(), wi->with));
        }
        auto fe = AST::Make<AST::ForeachBlock>(@1, mv, $4, $5);
        $$->stmts = AST::Make<AST::MultiNodes>(@5);
        $$->stmts->Append(fe);
      }
    ;

inthreads_block
    : INTHDS sync_type LPAREN s_expr RPAREN stmts_block {
        $$ = AST::Make<AST::InThreadsBlock>(@1, $4, $6, $2);
      }
    ;

while_block
    : WHILE LPAREN s_expr RPAREN stmts_block {
        $$ = AST::Make<AST::WhileBlock>(@1, $3, $5);
      }
    ;

if_else_block
    : IF LPAREN s_expr RPAREN stmts_block %prec IF_PREC {
        $$ = AST::Make<AST::IfElseBlock>(@1, $3, $5, nullptr);
      }
    | IF LPAREN call_stmt RPAREN stmts_block %prec IF_PREC {
        $$ = AST::Make<AST::IfElseBlock>(@1, $3, $5, nullptr);
      }
    | IF LPAREN s_expr RPAREN stmts_block ELSE stmts_block {
        $$ = AST::Make<AST::IfElseBlock>(@1, $3, $5, $7);
      }
    | IF LPAREN call_stmt RPAREN stmts_block ELSE stmts_block {
        $$ = AST::Make<AST::IfElseBlock>(@1, $3, $5, $7);
      }
    | ELSE {
        Parser::error(@1, "'else' without a previous 'if'.");
      }
    ;

withins
    : withins COMMA within {
        $1->Append($3);
        $$ = $1;
      }
    | within {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; /* do not allow empty within */

within
    : IDENTIFIER IN mdspan_expr {
        symtab.AddSymbol($1, MakeUnknownType()/*Need inference*/);
        $$ = AST::Make<AST::WithIn>(@1, AST::Make<AST::Identifier>(@1,$1), $3);
      }
    | IDENTIFIER ASSIGN LBRACE with_matchers RBRACE IN mdspan_expr {
        symtab.AddSymbol($1, MakeUnknownType()/*Need inference*/);
        $$ = AST::Make<AST::WithIn>(@1, AST::Make<AST::Identifier>(@1,$1), $7, $4);
        parsing_derivation_decl = false;
      }
    | LBRACE with_matchers RBRACE IN mdspan_expr {
        $$ = AST::Make<AST::WithIn>(@1, $5, $2); // no identifier
        parsing_derivation_decl = false;
      }
    ;

where_clause
    : WHERE where_binds { $$ = $2; }
    ;

where_binds
    : where_binds COMMA where_bind {
        $1->Append($3);
        $$ = $1;
      }
    | where_bind {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; /* do not allow empty where_bind */

where_bind
    : IDENTIFIER BIND IDENTIFIER {
        if (!symtab.Exists($1))
          Parser::error(@1, "The symbol '" + $1 + "` has not been defined.");
        if (!symtab.Exists($3))
          Parser::error(@3, "The symbol '" + $1 + "` has not been defined.");

        $$ = AST::Make<AST::WhereBind>(@1, AST::Make<AST::Identifier>(@1, $1),
                                           AST::Make<AST::Identifier>(@3, $3));
      }
    ;

foreach_block
    : FOREACH range_exprs stmts_block {
        $$ = AST::Make<AST::ForeachBlock>(@1, $2, $3);
      }
    | FOREACH range_exprs COL suffix_exprs stmts_block {
        $$ = AST::Make<AST::ForeachBlock>(@1, $2, $4, $5);
      }
    ;

range_exprs
    : range_exprs COMMA range_expr  {
        $1->Append($3);
        $$ = $1;
      }
    | range_expr {
        $$ = AST::Make<AST::MultiValues>(@1);
        $$->Append($1);
      }
    ; /* do not allow the empty ivs */

integer_value
    : NUM { $$ = $1; }
    | MINUS NUM { $$ = -$2; }
    ;

index_or_none
    : integer_value { $$ = $1; }
    | /*nothing*/ { $$ = GetInvalidBound(); }
    ;

bound_expr
    : s_expr { $$ = $1; }
    | /*nothing*/ { $$ = nullptr; }
    ;

range_expr
    : IDENTIFIER { $$ = AST::Make<AST::LoopRange>(@1, AST::Make<AST::Identifier>(@1, $1)); }
    | IDENTIFIER LPAREN bound_expr COL bound_expr RPAREN {
        $$ = AST::Make<AST::LoopRange>(@1, AST::Make<AST::Identifier>(@1, $1), $3, $5);
      }
    | IDENTIFIER LPAREN bound_expr COL bound_expr COL index_or_none RPAREN {
        $$ = AST::Make<AST::LoopRange>(@1, AST::Make<AST::Identifier>(@1, $1), $3, $5, $7);
      }
    ;

tdma
    : DMA { $$ = false; }
    | TMA { $$ = true; }
    ;

dma_stmt
    : IDENTIFIER ASSIGN tdma dma_operation_swizzle sync_type dma_config chunkat_expr TRANS chunkat_or_storage_or_select {
        symtab.AddSymbol($1, MakeDummyFutureType($5));
        auto dma = AST::Make<AST::DMA>(@3, $4.operation, $1, $7, $9, $5, $6);
        dma->SetTMA($3);
        // Handle swizzle from dma_operation_swizzle (only if explicitly specified in syntax)
        if ($4.swizzle_value > 0) {
          dma->SetSwizzleValue($4.swizzle_value);
          dma->SetSwizzleExplicit(true);
        }
        if ($4.sparse) {
          dma->SetSparse(true);
          dma->SetSparsePattern($4.sparse_n, $4.sparse_m);
        }
        // If swizzle is not specified in dma_operation_swizzle, swizzle_explicit remains false
        $$ = dma;
      }
    | IDENTIFIER ASSIGN tdma dma_operation_swizzle sync_type dma_config chunkat_expr TRANS chunkat_or_storage_or_select CHAIN IDENTIFIER {
        symtab.AddSymbol($1, MakeDummyFutureType($5));
        auto dma = AST::Make<AST::DMA>(@3, $4.operation, $1, $11, $7, $9, $5, $6);
        dma->SetTMA($3);
        // Handle swizzle from dma_operation_swizzle (only if explicitly specified in syntax)
        if ($4.swizzle_value > 0) {
          dma->SetSwizzleValue($4.swizzle_value);
          dma->SetSwizzleExplicit(true);
        }
        if ($4.sparse) {
          dma->SetSparse(true);
          dma->SetSparsePattern($4.sparse_n, $4.sparse_m);
        }
        // If swizzle is not specified in dma_operation_swizzle, swizzle_explicit remains false
        $$ = dma;
      }
    | tdma dma_operation_swizzle sync_type dma_config chunkat_expr TRANS chunkat_or_storage_or_select {
        auto dma = AST::Make<AST::DMA>(@1, $2.operation, "", $5, $7, $3, $4);
        dma->SetTMA($1);
        // Handle swizzle from dma_operation_swizzle (only if explicitly specified in syntax)
        if ($2.swizzle_value > 0) {
          dma->SetSwizzleValue($2.swizzle_value);
          dma->SetSwizzleExplicit(true);
        }
        if ($2.sparse) {
          dma->SetSparse(true);
          dma->SetSparsePattern($2.sparse_n, $2.sparse_m);
        }
        // If swizzle is not specified in dma_operation_swizzle, swizzle_explicit remains false
        $$ = dma;
      }
    | tdma dma_operation_swizzle sync_type dma_config chunkat_expr TRANS chunkat_or_storage_or_select CHAIN IDENTIFIER {
        auto dma = AST::Make<AST::DMA>(@1, $2.operation, "", $9, $5, $7, $3, $4);
        dma->SetTMA($1);
        // Handle swizzle from dma_operation_swizzle (only if explicitly specified in syntax)
        if ($2.swizzle_value > 0) {
          dma->SetSwizzleValue($2.swizzle_value);
          dma->SetSwizzleExplicit(true);
        }
        if ($2.sparse) {
          dma->SetSparse(true);
          dma->SetSparsePattern($2.sparse_n, $2.sparse_m);
        }
        // If swizzle is not specified in dma_operation_swizzle, swizzle_explicit remains false
        $$ = dma;
      }
    | IDENTIFIER ASSIGN tdma NONE {
        symtab.AddSymbol($1, MakePlaceHolderFutureType());
        auto dma = AST::Make<AST::DMA>(@1, $1);
        dma->SetTMA($3);
        $$ = dma;
      }
    ;

dma_operation_swizzle
    : COPY {
        $$ = DMAOperationWithSwizzle($1, 0);  // 0 means no swizzle specified
      }
    | COPY DOT SWIZZLE LPAREN swizzle_value RPAREN {
        $$ = DMAOperationWithSwizzle($1, $5);  // explicit swizzle value
      }
    | COPY SP {
        $$ = DMAOperationWithSwizzle($1, 0, true, 2, 4);
      }
    | COPY SPLPAREN integer_value COL integer_value RPAREN {
        $$ = DMAOperationWithSwizzle($1, 0, true, $3, $5);
      }
    | PAD {
        $$ = DMAOperationWithSwizzle($1, 0);
      }
    | TRANSPOSE {
        $$ = DMAOperationWithSwizzle($1, 0);
      }
    ;

dma_config
    : LT LBRACE value_list RBRACE COMMA LBRACE value_list RBRACE COMMA LBRACE value_list RBRACE COMMA simple_val GT {
        auto pc = AST::Make<PadConfig>();
        $3->SetDelimiter(", ");
        pc->pad_low = $3;
        $7->SetDelimiter(", ");
        pc->pad_high = $7;
        $11->SetDelimiter(", ");
        pc->pad_mid = $11;
        pc->SetPadValue(AST::Make<AST::Expr>($14->LOC(), $14));
        $$ = pc;
      }
    | LT iv_list GT {
        auto tc = AST::Make<TransposeConfig>();
        for (auto value : $2->values)
          tc->dim_values.push_back(cast<AST::IntLiteral>(value)->Val());
        $$ = tc;
    }
    | /* Empty for no config */ { $$ = nullptr; }
    ;

swizzle_value
    : NUM {
        if ($1 != 128 && $1 != 64 && $1 != 32) {
          Parser::error(@1, "swizzle value must be 128, 64, or 32");
          YYERROR;
        }
        $$ = $1;
      }
    ;

swizzle_config
    : SWIZZLE LPAREN swizzle_value RPAREN {
        $$ = AST::Make<SwizzleConfig>($3);
      }
    | /* Empty for no swizzle config */ { $$ = nullptr; }
    ;

sync_type
    : { $$ = false; }
    | ASYNC { $$ = true; }
    ;

chunkat_or_storage_or_select
    : { ignore_fndata = true; } chunkat_expr { $$ = $2; ignore_fndata = false; }
    | STORAGE { $$ = AST::Make<AST::Memory>(@1, $1); }
    | select_expr  { $$ = $1; }
    ;

span_as
    : FNSPANAS LPAREN g_value_list RPAREN {
        $$ = AST::Make<AST::SpanAs>(@1, nullptr/*fill later*/, $3);
        parsing_derivation_decl = false;
      }
    ;

chunkat_expr
    : subdata_expr { $$ = $1; }
    | ids_expr {
        auto ide = ElementMultiValues($1);
        $$ = AST::Make<AST::ChunkAt>(@1, ide.first, ide.second);
      }
    ;

spanned_ops
    : spanned_ops spanned_op {
        if ($2 != nullptr) $1.push_back($2);
        $$ = $1;
      }
    | spanned_op {
        $$ = std::vector<ptr<AST::SpannedOperation>>();
        if ($1 != nullptr) $$.push_back($1);
      }
    ;

spanned_op
    : CHUNKAT LPAREN value_list RPAREN {
        $3->SetDelimiter(", ");
        $$ = OptSpannedOperation(AST::Make<AST::SpannedOperation>(@1, $3, AST::SpannedOperation::TILING));
      }
    | CHUNK LPAREN value_list RPAREN AT LPAREN value_list RPAREN {
        $3->SetDelimiter(", ");
        $7->SetDelimiter(", ");
        $$ = OptSpannedOperation(AST::Make<AST::SpannedOperation>(@1, $7, $3, AST::SpannedOperation::TILEAT));
      }
    | SUBSPAN LPAREN value_list RPAREN {
        $3->SetDelimiter(", ");
        auto p = AST::Make<AST::MultiValues>(@3, ", ");
        for (size_t i = 0; i < $3->Count(); ++i) p->Append(AST::Make<AST::Expr>(@3, AST::Make<AST::IntLiteral>(@3, 0)));
        $$ = OptSpannedOperation(AST::Make<AST::SpannedOperation>(@1, p, $3, AST::SpannedOperation::SUBSPAN));
      }
    | SUBSPAN LPAREN value_list RPAREN AT LPAREN value_list RPAREN {
        $3->SetDelimiter(", ");
        $7->SetDelimiter(", ");
        $$ = OptSpannedOperation(AST::Make<AST::SpannedOperation>(@1, $7, $3, AST::SpannedOperation::SUBSPAN));
      }
    | MODSPAN LPAREN value_list RPAREN AT LPAREN value_list RPAREN {
        $3->SetDelimiter(", ");
        $7->SetDelimiter(", ");
        $$ = OptSpannedOperation(AST::Make<AST::SpannedOperation>(@1, $7, $3, AST::SpannedOperation::MODSPAN));
      }
    | SUBSPAN LPAREN value_list RPAREN STRIDE LPAREN value_list RPAREN AT LPAREN value_list RPAREN {
        $3->SetDelimiter(", ");
        $7->SetDelimiter(", ");
        $11->SetDelimiter(", ");
        $$ = OptSpannedOperation(AST::Make<AST::SpannedOperation>(@1, $11, $3, $7, AST::SpannedOperation::SUBSPAN));
      }
    | MODSPAN LPAREN value_list RPAREN STRIDE LPAREN value_list RPAREN AT LPAREN value_list RPAREN {
        $3->SetDelimiter(", ");
        $7->SetDelimiter(", ");
        $11->SetDelimiter(", ");
        $$ = OptSpannedOperation(AST::Make<AST::SpannedOperation>(@1, $11, $3, $7, AST::SpannedOperation::MODSPAN));
      }
    | FNSPANAS LPAREN g_value_list RPAREN {
        $3->SetDelimiter(", ");
        $$ = AST::Make<AST::SpannedOperation>(@1, $3, AST::SpannedOperation::RESHAPE);
      }
    ;

subdata_expr
    : ids_expr spanned_ops {
        auto ide = ElementMultiValues($1);
        $$ = AST::Make<AST::ChunkAt>(@1, ide.first, ide.second, $2);
      }
    ;

mma_stmt
    : IDENTIFIER ASSIGN MMA FILL s_expr {
        auto op = AST::Make<AST::MMAOperation>($1, $5);
        symtab.AddSymbol($1, MakeUnknownType());
        $$ = AST::Make<AST::MMA>(@1, op);
      }
    | IDENTIFIER ASSIGN MMA FILL DOT fundamental_type s_expr {
        auto op = AST::Make<AST::MMAOperation>($1, $7, $6);
        symtab.AddSymbol($1, MakeUnknownType());
        $$ = AST::Make<AST::MMA>(@1, op);
      }
    | IDENTIFIER ASSIGN MMA LOAD sync_type chunkat_expr {
        auto op = AST::Make<AST::MMAOperation>($6, $1, $5);
        symtab.AddSymbol($1, MakeUnknownType());
        $$ = AST::Make<AST::MMA>(@1, op);
      }
    | IDENTIFIER ASSIGN MMA LOAD DOT SWIZZLE LPAREN swizzle_value RPAREN sync_type chunkat_expr {
        auto op = AST::Make<AST::MMAOperation>($11, $1, $10, $8);
        symtab.AddSymbol($1, MakeUnknownType());
        $$ = AST::Make<AST::MMA>(@1, op);
      }
    | MMA mma_exec_method IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER {
        auto op = AST::Make<AST::MMAOperation>($2, $3, $5, $7);
        $$ = AST::Make<AST::MMA>(@1, op);
      }
    | MMA mma_exec_method SP IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER {
        auto op = AST::Make<AST::MMAOperation>($2, $4, $6, $8, true);
        $$ = AST::Make<AST::MMA>(@1, op);
      }
    | MMA mma_exec_method SP IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER COMMA IDENTIFIER {
        auto op = AST::Make<AST::MMAOperation>($2, $4, $6, $8, $10, true);
        $$ = AST::Make<AST::MMA>(@1, op);
      }
    | MMA STORE IDENTIFIER COMMA chunkat_expr {
        auto op = AST::Make<AST::MMAOperation>($3, $5);
        $$ = AST::Make<AST::MMA>(@1, op);
      }
    ;

mma_exec_method
    : ROW COLUMN    { $$ = AST::MMAOperation::ROW_COL; }
    | ROW ROW       { $$ = AST::MMAOperation::ROW_ROW; }
    | COLUMN COLUMN { $$ = AST::MMAOperation::COL_COL; }
    | COLUMN ROW    { $$ = AST::MMAOperation::COL_ROW; }
    ;

data_element
    : IDENTIFIER AT LPAREN data_indices RPAREN {
        $$ = AST::Make<AST::DataAccess>(@1, AST::Make<AST::Identifier>(@1, $1), $4);
      }
    | IDENTIFIER FNDATA AT LPAREN data_indices RPAREN {
        $$ = AST::Make<AST::DataAccess>(@1, AST::Make<AST::Identifier>(@1, $1+$2), $5);
      }
    | IDENTIFIER FNMDATA AT LPAREN data_indices RPAREN {
        $$ = AST::Make<AST::DataAccess>(@1, AST::Make<AST::Identifier>(@1, $1+$2), $5);
      }
    ;

data_indices
    : s_expr {
        $$ = AST::Make<AST::MultiValues>(@1);
        $$->Append($1);
      }
    | data_indices COMMA s_expr {
        $1->Append($3);
        $$ = $1;
      }
    ;

select_expr
    : SELECT LPAREN s_expr COMMA value_list RPAREN {
        $5->SetDelimiter(", ");
        $$ = AST::Make<AST::Select>(@1, $3, $5);
      }
    ;

iv_list
    : iv_list COMMA NUM {
        $1->Append(AST::Make<AST::IntLiteral>(@3, $3));
        $$ = $1;
      }
    | NUM {
        $$ = AST::Make<AST::MultiValues>(@1, ", ");
        $$->Append(AST::Make<AST::IntLiteral>(@1, $1));
      }
    ;

id_list
    : id_list COMMA IDENTIFIER {
        $1->Append(AST::Make<AST::Identifier>(@3, $3));
        $$ = $1;
      }
    | IDENTIFIER {
        $$ = AST::Make<AST::MultiValues>(@1, ", ");
        $$->Append(AST::Make<AST::Identifier>(@1, $1));
      }
    ;

device_passables
    : /* Empty */ {
        $$ = AST::Make<AST::MultiValues>(loc, ", ");
      }
    | device_passable {
        $$ = AST::Make<AST::MultiValues>(@1, ", ");
        $$->Append($1);
      }
    | device_passables COMMA device_passable {
        $1->Append($3);
        $$ = $1;
      }
    ;

device_passable
    : s_expr { $$ = $1; }
    | subdata_expr { $$ = AST::Make<AST::Expr>(@1, $1); }
    ;

returnable
    : s_expr { $$ = $1; }
    ;

with_matchers /* TODO: this special case is pattern-match ids for with-block */
    : with_matchers COMMA IDENTIFIER {
        $1->Append(AST::Make<AST::Identifier>(@3, $3));
        symtab.AddSymbol($3, MakeIntegerType()); /* in withins, this values should be int only */
        $$ = $1;
      }
    | IDENTIFIER {
        $$ = AST::Make<AST::MultiValues>(@1, ", ");
        symtab.AddSymbol($1, MakeIntegerType()); /* in withins, this values should be int only */
        $$->Append(AST::Make<AST::Identifier>(@1, $1));
      }
    ;

ids_expr /* enforce: either a ref to id or a subscription */
    : s_expr {
        if ($1->IsReference()) {
          if (!$1->GetSymbol())
            Parser::error($1->LOC(), "expect a symbol but got a " + $1->GetR()->TypeNameString() + ".");
          $$ = $1;
        } else if ($1->op == "dataof") {  // ignore the dataof
          auto er = cast<AST::Expr>($1->GetR());
          if ((er->op == "elemof") || (er->GetSymbol()))
            $$ = er;
          else
            Parser::error($1->LOC(), "expect a subscription but got a " + $1->op + ".");
        } else if ($1->op == "mdataof") {  // ignore the mdataof
          auto er = cast<AST::Expr>($1->GetR());
          if ((er->op == "elemof") || (er->GetSymbol()))
            $$ = er;
          else
            Parser::error($1->LOC(), "expect a subscription but got a " + $1->op + ".");
        } else if ($1->op != "elemof") {
          Parser::error($1->LOC(), "expect a subscription but got a " + $1->op + ".");
        } else
          $$ = $1;
      }
    ;

subscript_like_expr /* for multi-dim element accesses and also mdspan derivations */
    : IDENTIFIER LBRAKT s_expr RBRAKT {
        if (parsing_derivation_decl) {  /* mdspan */
          $$ = AST::Make<AST::Expr>(@1, AST::Make<AST::MultiDimSpans>(@1, $1, $3));
        } else {
          $$ = AST::Make<AST::Expr>(@1, "elemof", AST::Make<AST::Identifier>(@1, $1), $3);
        }
      }
    | IDENTIFIER LBRAKT s_expr RBRAKT subscriptions {
        if (parsing_derivation_decl)
          Parser::error(@3, "must use 'a[(1), (2), ...]' for mdspan derivations.");

        assert ($5->Count() > 0);
        $5->Insert($3, 0);
        AST::ptr<AST::Expr> expr = nullptr;
        for (auto e : $5->AllValues()) {
          if (expr == nullptr)
            expr = AST::Make<AST::Expr>(@1, "elemof", AST::Make<AST::Identifier>(@1, $1), e);
          else
            expr = AST::Make<AST::Expr>(@1, "elemof", expr, e);
        }
        $$ = expr;
      }
    | IDENTIFIER FNSPAN LBRAKT s_expr RBRAKT subscriptions {
        Parser::error(@3, "must use 'a.span[(1), (2), ...]' for mdspan derivations.");
        YYERROR;
      }
    | IDENTIFIER LBRAKT s_expr COMMA value_list RBRAKT {
        if (!parsing_derivation_decl)
          Parser::error(@3, "must use '[a][2]...' for multi-dimensional array access.");
        $5->Insert($3, 0);
        $$ = AST::Make<AST::Expr>(@1, AST::Make<AST::MultiDimSpans>(@1, $1, $5));
      }
    | IDENTIFIER FNSPAN LBRAKT s_expr COMMA value_list RBRAKT {
        if (!parsing_derivation_decl)
          Parser::error(@3, "must use '[a][2]...' for multi-dimensional array access.");
        $6->Insert($4, 0);
        $$ = AST::Make<AST::Expr>(@1, AST::Make<AST::MultiDimSpans>(@1, $1, $6));
      }
/*
    | IDENTIFIER LBRACE s_expr RBRACE {
        if (!parsing_derivation_decl)
          Parser::error(@3, "must use 'a{(1)}' for ituple derivations.");
        $$ = AST::Make<AST::Expr>(@1, AST::Make<AST::MultiDimSpans>(@1, $1, $3));
      }
*/

ituple_derivation
    : IDENTIFIER LBRACE g_value_list RBRACE {
        $3->SetDelimiter(", ");
        $$ = AST::Make<AST::Expr>(@1, AST::Make<AST::IntTuple>(@1, $1, $3));
      }
    ;

ids_list
    : ids_list COMMA ids_expr {
        $1->Append($3);
        $$ = $1;
      }
    | ids_expr {
        $$ = AST::Make<AST::MultiValues>(@1, ", ");
        $$->Append($1);
      }
    ;

wait_stmt
    : WAIT ids_list { $$ = AST::Make<AST::Wait>(@1, $2); }
    ;

trigger_stmt
    : TRIGGER ids_list { $$ = AST::Make<AST::Trigger>(@1, $2); }
    ;

align_func
    : ALIGNUP { $$ = $1; }
    | ALIGNDOWN { $$ = $1; }
    ;

builtin_print_func
    : PRINT   { $$ = $1; }
    | PRINTLN { $$ = $1; }
    ;

arith_builtin_func
    : ACOS     { $$ = $1; }
    | ASIN     { $$ = $1; }
    | ATAN     { $$ = $1; }
    | ATAN2    { $$ = $1; }
    | CEIL     { $$ = $1; }
    | COS      { $$ = $1; }
    | COSH     { $$ = $1; }
    | EXP      { $$ = $1; }
    | EXPM1    { $$ = $1; }
    | FLOOR    { $$ = $1; }
    | GELU     { $$ = $1; }
    | ISFINITE { $$ = $1; }
    | ROUND    { $$ = $1; }
    | RSQRT    { $$ = $1; }
    | SIGMOID  { $$ = $1; }
    | SIN      { $$ = $1; }
    | SINH     { $$ = $1; }
    | SOFTPLUS { $$ = $1; }
    | SQRT     { $$ = $1; }
    | TAN      { $$ = $1; }
    | TANH     { $$ = $1; }
    | LOG1P    { $$ = $1; }
    | LOG      { $$ = $1; }
    | POW      { $$ = $1; }
    | SIGN     { $$ = $1; }
    ;

suffix_exprs
    : suffix_exprs COMMA suffix_expr {
        $1->Append($3);
        $$ = $1;
      }
    | suffix_expr {
        $$ = AST::Make<AST::MultiValues>(@1, ", ");
        $$->Append($1);
      }
    ;

suffix_expr
    : VECTORIZE LPAREN IDENTIFIER COMMA s_expr RPAREN {
        auto mv = AST::Make<AST::MultiValues>(@1, ", ");
        mv->Append(AST::Make<AST::Expr>(@1, AST::Make<AST::Identifier>(@3, $3)));
        mv->Append($5);
        $$ = AST::Make<AST::AttributeExpr>(@1, $1, mv);
    }
  ;

call_expr
    : arith_builtin_func LPAREN value_list RPAREN {
        $$ = AST::Make<AST::Expr>(@1,
             AST::Make<AST::Call>(@1,
             AST::Make<AST::Identifier>(@1, $1), $3, AST::Call::BIF | AST::Call::ARITH | AST::Call::EXPR));
      }
    | align_func LPAREN s_expr COMMA s_expr RPAREN {
        auto mn = AST::Make<AST::MultiValues>(@1, ", ");
        mn->Append($3);
        mn->Append($5);
        $$ = AST::Make<AST::Expr>(@1,
             AST::Make<AST::Call>(@1,
             AST::Make<AST::Identifier>(@1, $1), mn, AST::Call::BIF | AST::Call::EXPR));
      }
    ;

cstrings /* concatenate strings */
    : cstrings STRING { $$ = $1 + $2; }
    | STRING { $$ = $1; }
    ;

id_with_namespace
    : id_with_namespace DCOLS IDENTIFIER { $$ = $1 + "::" + $3; }
    | IDENTIFIER { $$ = $1; }

inlcpp_stmt
    : INLCPP LPAREN cstrings RPAREN {
        $$ = AST::Make<AST::CppSourceCode>(@3, $3, AST::CppSourceCode::Inline);
      }

call_stmt
    : CALL id_with_namespace LPAREN device_passables RPAREN {
        $$ = AST::Make<AST::Call>(@1,
                AST::Make<AST::Identifier>(@2, $2), $4);
      }
    | CALL id_with_namespace template_params LPAREN device_passables RPAREN {
        $$ = AST::Make<AST::Call>(@1,
                AST::Make<AST::Identifier>(@2, $2), $5, $3);
      }
    | ASSERT LPAREN s_expr COMMA cstrings RPAREN {
        auto mv = AST::Make<AST::MultiValues>(@1, ", ");
        mv->Append($3);
        mv->Append(AST::Make<AST::StringLiteral>(@5, $5));
        $$ = AST::Make<AST::Call>(@1, AST::Make<AST::Identifier>(@1, $1), mv, AST::Call::BIF);
      }
    | builtin_print_func LPAREN value_list RPAREN {
        $3->SetDelimiter(", ");
        $$ = AST::Make<AST::Call>(@1, AST::Make<AST::Identifier>(@1, $1), $3, AST::Call::BIF);
      }
    | builtin_print_func NOT LPAREN value_list RPAREN {
        $4->SetDelimiter(", ");
        $$ = AST::Make<AST::Call>(@1, AST::Make<AST::Identifier>(@1, $1), $4,
                                  AST::Call::BIF | AST::Call::COMPTIME);
      }
    ;

swap_stmt
    : SWAP LPAREN id_list RPAREN {
        if ($3->Count() != 2)
          Parser::error(@3, "Builtin function 'swap' accept exact two parameters.");
        $$ = AST::Make<AST::Rotate>(@1, $3);
      }
    | SWAP id_list {
        if ($2->Count() != 2)
          Parser::error(@2, "Builtin function 'swap' accept exact two parameters.");
        $$ = AST::Make<AST::Rotate>(@1, $2);
      }
    | ROTATE LPAREN id_list RPAREN {
        if ($3->Count() < 2)
          Parser::error(@3, "Builtin function 'rotate' accept two or more parameters.");
        $$ = AST::Make<AST::Rotate>(@1, $3);
      }
    | ROTATE id_list {
        if ($2->Count() < 2)
          Parser::error(@2, "Builtin function 'rotate' accept two or more parameters.");
        $$ = AST::Make<AST::Rotate>(@1, $2);
      }
    ;

break_stmt : BREAK { $$ = AST::Make<AST::Break>(@1); }

continue_stmt : CONTINUE { $$ = AST::Make<AST::Continue>(@1); }

%%

std::pair<ptr<AST::Identifier>, ptr<AST::MultiValues>>
ElementMultiValues(const ptr<AST::Expr>&e) {
  auto mv = AST::Make<AST::MultiValues>(e->LOC());
  ptr<AST::Identifier> id = nullptr;
  if (e->IsReference()) {
    id = cast<AST::Identifier>(e->GetR());
  } else {
    auto we = e;
    while (true) {
      if (we->op != "elemof")
        choreo_unreachable("unable to handle this expr: " + PSTR(we));
      mv->Insert(we->GetR(), 0);
      if (id = dyn_cast<AST::Identifier>(we->GetL()))
        break;
      else
        we = cast<AST::Expr>(we->GetL());
    }
  }
  return std::make_pair(id, mv);
}

inline ptr<AST::SpannedOperation> OptSpannedOperation(const ptr<AST::SpannedOperation> &tsi) {
  if (tsi == nullptr) return nullptr;

  bool not_tiled = true;

  auto no_tiling_norm = [&](ptr<AST::MultiValues> mv) -> void {
    if (mv == nullptr) return;
    for (auto v : mv->AllValues()) {
      if (auto id = AST::GetIdentifier(*v))
        if (id->name == "_") {
          id->name = "__choreo_no_tiling__";
          continue;
        }
      not_tiled = false;
    }
  };

  switch (tsi->OpCode()) {
  case AST::SpannedOperation::Kind::TILING: no_tiling_norm(tsi->Positions()); break;
  case AST::SpannedOperation::Kind::TILEAT: [[fallthrough]];
  case AST::SpannedOperation::Kind::SUBSPAN: [[fallthrough]];
  case AST::SpannedOperation::Kind::MODSPAN: no_tiling_norm(tsi->TFSS()); no_tiling_norm(tsi->GetStrides()); break;
  default: choreo_unreachable("Unexpect SpannedOperation Kind");
  }

  if (not_tiled) return nullptr;

  return tsi;
}

// Bison expects us to provide implementation - otherwise linker complains
void Parser::error(const location &loc , const std::string &message) {
  errs() << loc << ": ";
  errs() << ((should_use_colors()) ? color_red : "") << "error: "
         << ((should_use_colors()) ? color_reset : "");
  errs() << message << "\n";

  if (!CCtx().ShowSourceLocation()) return;

  // Retrieve the line that caused the error
  std::string error_line = CCtx().GetSourceLine(loc.begin.line);
  if (!error_line.empty()) {
    errs() << "  " << error_line << "\n"; // Print the source line

    // Print caret (^) under the error position
    errs() << "  ";
    for (int i = 1; i < loc.begin.column; ++i)
      errs() << " "; // Align the caret with the exact error position

    errs() << "^" << "\n";
  }

  pctx.recordError();
}
