%skeleton "lalr1.cc" // -*- C++ -*-
%require "3.8"

%define api.token.constructor
//%define parse.trace
%define api.parser.class { Parser }
%define parse.error verbose
%define parse.assert
%define api.namespace { Choreo }
//%define api.token.prefix {TOK_}
%locations

%code requires {

#include <string>
#include <fstream>

namespace Choreo { class Scanner; }

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

const char* red = "\033[31m";
const char* reset = "\033[0m";

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

// The flag is used to disambiguate syntax suger:
//
//   a {(0), (1), 3} represents { a(0), a(1), 3 }
//
// where:
//
//   {(0), (1), 3} represents {0, 1, 3}
//
static bool parsing_prefixed_list = false;

}

%{
#include <stdio.h>
extern int yylex();

void choreo_info(const char *message) {
    // fprintf(stderr, "Error: %s\n", s);
  const char* GREEN = "\033[32m";
  if (should_use_colors())
      std::cerr << GREEN;
  std::cerr << "Info: ";
  if (should_use_colors())
      std::cerr << reset;
  std::cerr << message << std::endl;
  std::cerr << "Info location: " << ::loc << std::endl;
}
%}

// make yylex() expects one parameter of type 'Choreo::Scanner &'
%lex-param { Choreo::Scanner &scanner  }
// make yyparse() expects one parameter of type 'Choreo::Scanner &'.
%parse-param { Choreo::Scanner &scanner  }

%token
  ASSIGN  "="
  MINUS   "-"
  PLUS    "+"
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
  LT      "<"
  GT      ">"
  EQ      "=="
  NE      "!="
  LE      "<="
  GE      ">="
  AND     "&&"
  OR      "||"
  NOT     "!"
  QES     "?"
  TRANS   "=>"
  BIND    "<->"
  PIPE    "|"
  UBOUND  "#"
  CDIV    "cdiv"
;

// instead of union, using c++17 variant for terminal and non-terminals
%define api.value.type variant

// terminals
%token END 0 "end of file"
%token <char> CHAR
%token <int> NUM
%token <std::string> HOST_CODE KERNEL_CODE
%token <std::string> IDENTIFIER ATTR_CO
// type related
%token <std::string> MDSPAN ITUPLE
%token <Choreo::Storage> LOCAL SHARED GLOBAL
%token <Choreo::BaseType> F32 F16 BF16 U16 S16 U8 S8 U32 S32 INT BOOL VOID
// builtin operations
%token <std::string> DMA COPY SLICE PAD ASYNC FNSPAN FNDATA CHUNKAT WAIT CALL AUTO
// control related
%token <std::string> IF ELSE PARA BY WITH IN FOREACH RET WHERE
%token <std::string> TRUE FALSE

// non-terminals
%nterm <std::string> dma_operation
%nterm <ptr<DMAConfig>> dma_config
%nterm <bool> sync_type
%nterm <Choreo::Storage> storage
%nterm <Choreo::BaseType> fundamental_type
%nterm <AST::ptr<AST::CppSourceCode>> pass_by host_code
%nterm <AST::ptr<AST::Memory>> storage_qual
%nterm <AST::ptr<AST::Node>> foreach_block general_val simple_int span_val direct_ituple_val bool_literal passable declaration statement assignment dma_stmt wait_stmt call_stmt index_or_value iv_expr if_else_block optional_scalar_init param_mdspan_val chunkat_or_storage
%nterm <AST::ptr<AST::MultiNodes>> statements declarations assignments withins where_binds where_clause else_block named_spanned_decls
%nterm <AST::ptr<AST::MultiValues>> index_value_list value_list param_mdspan_list iv_exprs iv_list id_list with_matchers passables
%nterm <AST::ptr<AST::Expr>> s_expr span_expr
%nterm <AST::ptr<AST::DataType>> scalar_type void_type auto_type param_type return_type spanned_type
%nterm <AST::ptr<AST::ParamList>> parameter_list
%nterm <AST::ptr<AST::Parameter>> parameter
%nterm <AST::ptr<AST::ChoreoFunction>> dsl_function
%nterm <AST::ptr<AST::MultiDimSpans>> unnamed_mdspan_decl param_mdspan
%nterm <AST::ptr<AST::NamedTypeDecl>> named_mdspan_decl
%nterm <AST::ptr<AST::NamedVariableDecl>> named_ituple_decl named_scalar_decl
%nterm <AST::ptr<AST::IntTuple>> unnamed_ituple_decl sugar_unnamed_ituple_decl sugarless_unnamed_ituple_decl
%nterm <AST::ptr<AST::IntIndex>> s_index
%nterm <AST::ptr<AST::WithBlock>> within_block
%nterm <AST::ptr<AST::WithIn>> within
%nterm <AST::ptr<AST::WhereBind>> where_bind
%nterm <AST::ptr<AST::ParallelBy>> paraby_block
%nterm <AST::ptr<AST::Return>> return_stmt
%nterm <AST::ptr<AST::ChunkAt>> chunkat_expr

// precedence (low to high) and associativity
%right LBRACE
%right ASSIGN
%right QES COL
%left OR
%left AND
%right NOT
%nonassoc LT GT LE GE EQ NE
%left PLUS MINUS
%left STAR SLASH PECET
%nonassoc UBOUND
%nonassoc LPAREN RPAREN
//%left HOST_CODE

%%

program
    : /* Empty */ {}
    | program pass_by       { root.nodes.push_back($2); }
    | program dsl_function  { root.nodes.push_back($2); }

pass_by
    : host_code   { $$ = $1; }
    | KERNEL_CODE {
        $$ = AST::Make<AST::CppSourceCode>(@1, $1, false);
      }
    ;

host_code
    : host_code HOST_CODE {
        $$ = AST::Make<AST::CppSourceCode>(@1, $1->code + $2);
      }
    | HOST_CODE /* can not be empty */ {
        $$ = AST::Make<AST::CppSourceCode>(@1, $1);
      }
    ;

dsl_function
    : ATTR_CO return_type IDENTIFIER LPAREN parameter_list RPAREN LBRACE statements RBRACE {
        $$ = AST::Make<AST::ChoreoFunction>(@1);
        $$->name = $3;
        $$->f_decl.name = $3;
        $$->f_decl.ret_type = $2;
        $$->f_decl.params = $5;
        $$->stmts = $8;
      }
    ;

return_type
    : param_type { $$ = $1; }
    | void_type  { $$ = $1; }
    | auto_type  { $$ = $1; }
    ;

param_type
    : scalar_type { $$ = $1; }
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
    | MDSPAN LBRAKT param_mdspan_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@2, "", $3);
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
    : NUM   { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | QES   { $$ = AST::Make<AST::IntLiteral>(@1); }
    ;

void_type
    : VOID  { $$ = AST::Make<AST::DataType>(@1, $1); }
    ;

auto_type
    : AUTO  { $$ = AST::Make<AST::DataType>(@1, BaseType::UNKNOWN); }
    ;

scalar_type
    : INT   { $$ = AST::Make<AST::DataType>(@1, $1); }
    | BOOL  { $$ = AST::Make<AST::DataType>(@1, $1); }
    ;

spanned_type
    : fundamental_type LBRAKT value_list RBRAKT {
        $$ = AST::Make<AST::DataType>(@1, $1, AST::Make<AST::MultiDimSpans>(@3, "", $3));
      }
    ;

fundamental_type
    : F32   { $$ = $1; }
    | F16   { $$ = $1; }
    | BF16  { $$ = $1; }
    | U16   { $$ = $1; }
    | S16   { $$ = $1; }
    | U8    { $$ = $1; }
    | S8    { $$ = $1; }
    | U32   { $$ = $1; }
    | S32   { $$ = $1; }
    ;

general_val
    : NUM { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | IDENTIFIER {
        if (!symtab.Exists($1))
          Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

        $$ = AST::Make<AST::Identifier>(@1, $1);
      }
    | IDENTIFIER FNSPAN { $$ = AST::Make<AST::Identifier>(@1, $1 + $2); }
    | unnamed_mdspan_decl { $$ = $1; }
    | unnamed_ituple_decl { $$ = $1; }
    | bool_literal { $$ = $1; }
    ;

simple_int
    : NUM { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | IDENTIFIER {
        if (!symtab.Exists($1))
          Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

        auto ty = symtab.GetSymbol($1)->GetType();
        if (!isa<IntegerType>(ty.get()))
          Parser::error(@1, "expecting symbol `" + $1 + "' (type: " + ty->Name()
                            + ") of a integer type.");

        $$ = AST::Make<AST::Identifier>(@1, $1);
    	}
    ;

bool_literal
    : TRUE { $$ = AST::Make<AST::Boolean>(@1, std::string("true")); }
    | FALSE { $$ = AST::Make<AST::Boolean>(@1, std::string("false")); }
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

parameter
    : param_type IDENTIFIER { /* handle parameter type and name here */
        symtab.AddSymbol($2, $1->GetType());
        $$ = AST::Make<AST::Parameter>(@1, $1, AST::Make<AST::Identifier>(@2, $2));
      }
    | param_type {
        $$ = AST::Make<AST::Parameter>(@1, $1, AST::Make<AST::Identifier>(@1));
      }
    ;

statements
    : /* no statement */ { $$ = AST::Make<AST::MultiNodes>(loc); }
    | statements statement {
        $1->Append($2);
        $$ = $1;
      }
    | statements SEMCOL { $$ = $1; }
    ;

statement
    : declarations SEMCOL { $$ = $1; }
    | assignments  SEMCOL { $$ = $1; }
    | dma_stmt     SEMCOL { $$ = $1; }
    | wait_stmt    SEMCOL { $$ = $1; }
    | call_stmt    SEMCOL { $$ = $1; }
    | return_stmt  SEMCOL { $$ = $1; }
    | paraby_block        { $$ = $1; }
    | within_block        { $$ = $1; }
    | if_else_block       { $$ = $1; }
    | foreach_block       { $$ = $1; }
    ;

return_stmt
    : RET          { $$ = AST::Make<AST::Return>(@1);}
    | RET passable { $$ = AST::Make<AST::Return>(@1, $2); }
    ;

paraby_block
    : PARA IDENTIFIER BY NUM {
        symtab.AddSymbol($2, MakeBoundedIntegerType($4));
      } LBRACE statements RBRACE {
        $$ = AST::Make<AST::ParallelBy>(@1, $2, $4);
        $$->stmts = $7;
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
    | named_spanned_decls {
        $$ = $1;
      }
    ;

declaration
    : named_mdspan_decl  { $$ = $1; }
    | named_ituple_decl  { $$ = $1; }
    | named_scalar_decl  { $$ = $1; }
    ;

named_scalar_decl
    : scalar_type IDENTIFIER optional_scalar_init {
        assert($1->isScalar() && "Not a scalar type.");
        symtab.AddSymbol($2, $1->GetType());
        if (!$3)
          $$ = AST::Make<AST::NamedVariableDecl>(@2, $2, $1);
        else
          $$ = AST::Make<AST::NamedVariableDecl>(@2, $2, $1, nullptr, $3);
      }
    ;

optional_scalar_init
    : /*Empty */     { $$ = nullptr; }
    | ASSIGN s_expr  { $$ = $2; }
    ;

named_spanned_decls
    : named_spanned_decls COMMA IDENTIFIER {
        const auto& node = std::dynamic_pointer_cast<AST::NamedVariableDecl>($1->values[0]);
        symtab.AddSymbol($3, node->GetType());
        $1->Append(AST::Make<AST::NamedVariableDecl>(@3, $3, node->type, node->mem));
        $$ = $1;
      }
    | storage_qual spanned_type IDENTIFIER {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append(AST::Make<AST::NamedVariableDecl>(@3, $3, $2, $1));
      }
    ;

s_index
    : LPAREN simple_int RPAREN {
        $$ = AST::Make<AST::IntIndex>(@2, $2);
      }
    ;

index_or_value
    : s_expr      { $$ = $1; }
    | QES         { $$ = AST::Make<AST::IntLiteral>(@1); }
    ; // do not allow non-element

index_value_list
    : index_value_list COMMA index_or_value {
        $1->Append($3);
        $$ = $1;
      }
    | index_or_value {
        $$ = AST::Make<AST::MultiValues>(@1);
        $$->Append($1);
      }
    ; // do not allow an empty list

value_list
    : /* Empty list */ {
        $$ = AST::Make<AST::MultiValues>(loc);
      }
    | value_list COMMA s_expr {
        $1->Append($3);
        $$ = $1;
      }
    | s_expr {
        $$ = AST::Make<AST::MultiValues>(@1);
        $$->Append($1);
      }
    ;

unnamed_mdspan_decl
    : IDENTIFIER FNSPAN LBRAKT { parsing_prefixed_list = true; }
      index_value_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, $1, $5);
        parsing_prefixed_list = false;
      }
    | IDENTIFIER LBRAKT { parsing_prefixed_list = true; }
      index_value_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, $1, $4);
        parsing_prefixed_list = false;
      }
    | LBRAKT value_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, "", $2);
      }
    ;

span_val
    : unnamed_mdspan_decl { $$ = $1; }
    | IDENTIFIER { $$ = AST::Make<AST::Identifier>(@1, $1); }
    | IDENTIFIER FNSPAN { $$ = AST::Make<AST::Identifier>(@1, $1 + $2); }
    ;

named_mdspan_decl
    : MDSPAN IDENTIFIER COL s_expr {
        symtab.AddSymbol($2, MakeUninitMDSpanType());
        $$ = AST::Make<AST::NamedTypeDecl>(@2, $2, $4);
      }
    | MDSPAN LT NUM GT IDENTIFIER COL s_expr {
        symtab.AddSymbol($5, MakeRankedMDSpanType($3));
        $$ = AST::Make<AST::NamedTypeDecl>(@5, $5, $7, $3);
      }
    | IDENTIFIER COL s_expr {
        symtab.AddSymbol($1, MakeUninitMDSpanType());
        $$ = AST::Make<AST::NamedTypeDecl>(@1, $1, $3);
      }
    ;

unnamed_ituple_decl
    : sugarless_unnamed_ituple_decl { $$ = $1; }
    | sugar_unnamed_ituple_decl     { $$ = $1; }
    ;

sugar_unnamed_ituple_decl
    : IDENTIFIER LBRACE { parsing_prefixed_list = true; }
      index_value_list RBRACE {
        // anchor
        if (!symtab.Exists($1))
          Parser::error(@1, "The symbol '" + $1 + "` has not been defined.");

        $4->SetDelimiter(", ");
        $$ = AST::Make<AST::IntTuple>(@1, $1, $4);
        parsing_prefixed_list = true;
      }
    ;

sugarless_unnamed_ituple_decl
    : LBRACE value_list RBRACE {
        $2->SetDelimiter(", ");
        $$ = AST::Make<AST::IntTuple>(@1, "", $2);
      }
    ;

direct_ituple_val
    : sugarless_unnamed_ituple_decl { $$ = $1; }
    | IDENTIFIER {
        $$ = AST::Make<AST::Identifier>(@1, $1);
      }
    ;

named_ituple_decl
    : ITUPLE IDENTIFIER ASSIGN s_expr {
        symtab.AddSymbol($2, MakeUninitITupleType());
        $$ = AST::Make<AST::NamedVariableDecl>(@2,
              $2, AST::Make<AST::DataType>(@1, BaseType::ITUPLE), nullptr, $4);
      }
    | ITUPLE LT NUM GT IDENTIFIER ASSIGN s_expr {
        symtab.AddSymbol($5, MakeUninitITupleType());
        $$ = AST::Make<AST::NamedVariableDecl>(@5,
              $5, AST::Make<AST::DataType>(@1, BaseType::ITUPLE, $3), nullptr, $7);
      }
    ; // do not allow uninitialized ituple

storage
    : LOCAL   { $$ = $1; }
    | SHARED  { $$ = $1; }
    | GLOBAL  { $$ = $1; }
    ;

storage_qual
    : /* Empty */ { $$ = AST::Make<AST::Memory>(loc); }
    | storage { $$ = AST::Make<AST::Memory>(@1, $1); }
    ;

assignment
    : IDENTIFIER ASSIGN s_expr {
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
    | IDENTIFIER PLUS ASSIGN s_expr {
        if (!symtab.Exists($1)) {
          Parser::error(@1, "The symbol '" + $1 + "` has not been defined.");
        } else {
          $$ = AST::Make<AST::Assignment>(@1,
              $1, AST::Make<AST::Expr>(@1, "+", $4, AST::Make<AST::Identifier>(@1, $1)));
        }
      }
    ;

s_expr
    : s_expr PLUS s_expr { $$ = AST::Make<AST::Expr>(@1, "+", $1, $3); }
    | s_expr MINUS s_expr { $$ = AST::Make<AST::Expr>(@1, "-", $1, $3); }
    | s_expr STAR s_expr { $$ = AST::Make<AST::Expr>(@1, "*", $1, $3); }
    | s_expr SLASH s_expr { $$ = AST::Make<AST::Expr>(@1, "/", $1, $3); }
    | s_expr PECET s_expr { $$ = AST::Make<AST::Expr>(@1, "%", $1, $3); }
    | CDIV LPAREN s_expr COMMA s_expr RPAREN { $$ = AST::Make<AST::Expr>(@1, "cdiv", $3, $5); }
    | s_expr OR s_expr { $$ = AST::Make<AST::Expr>(@1, "||", $1, $3); }
    | s_expr AND s_expr { $$ = AST::Make<AST::Expr>(@1, "&&", $1, $3); }
    | NOT s_expr { $$ = AST::Make<AST::Expr>(@1, "!", $2); }
    | LPAREN s_expr RPAREN {
        // Does String "(0)" represent an indexing operation or an arithmetic operation 
        if (!parsing_prefixed_list) {
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
        $$ = AST::Make<AST::Expr>(@1, "$", $2, $5, $7);
      }
    | s_expr LT s_expr { $$ = AST::Make<AST::Expr>(@1, "<", $1, $3); }
    | s_expr GT s_expr { $$ = AST::Make<AST::Expr>(@1, ">", $1, $3); }
    | s_expr EQ s_expr { $$ = AST::Make<AST::Expr>(@1, "==", $1, $3); }
    | s_expr NE s_expr { $$ = AST::Make<AST::Expr>(@1, "!=", $1, $3); }
    | s_expr LE s_expr { $$ = AST::Make<AST::Expr>(@1, "<=", $1, $3); }
    | s_expr GE s_expr { $$ = AST::Make<AST::Expr>(@1, ">=", $1, $3); }
    | general_val      { $$ = AST::Make<AST::Expr>(@1, $1); }
    | PIPE s_expr PIPE { $$ = AST::Make<AST::Expr>(@1, "sizeof", $2); }
    | s_expr s_index   { $$ = AST::Make<AST::Expr>(@1, "dimof", $1, $2); }
    | UBOUND IDENTIFIER {
        $$ = AST::Make<AST::Expr>(@1, "ubound", AST::Make<AST::Identifier>(@2, $2));
      }
    ;

span_expr
    : span_expr PLUS  direct_ituple_val { $$ = AST::Make<AST::Expr>(@1, "+", $1, $3); }
    | span_expr MINUS direct_ituple_val { $$ = AST::Make<AST::Expr>(@1, "-", $1, $3); }
    | span_expr STAR  direct_ituple_val { $$ = AST::Make<AST::Expr>(@1, "*", $1, $3); }
    | span_expr SLASH direct_ituple_val { $$ = AST::Make<AST::Expr>(@1, "/", $1, $3); }
    | span_expr PECET direct_ituple_val { $$ = AST::Make<AST::Expr>(@1, "%", $1, $3); }
    | span_val { $$ = AST::Make<AST::Expr>(@1, $1); }
    ;

if_else_block
    : IF LPAREN s_expr RPAREN LBRACE statements RBRACE else_block {
        $$ = AST::Make<AST::IfElse>(@1, $3, $6, $8);
      }
    ;

else_block
    : ELSE LBRACE statements RBRACE { $$ = $3; }
    | /* empty */ { $$ = AST::Make<AST::MultiNodes>(loc); }
    ;

within_block
    : WITH withins LBRACE statements RBRACE {
        $$ = AST::Make<AST::WithBlock>(@1);
        $$->withins = $2;
        $$->stmts = $4;
      }
    | WITH withins where_clause LBRACE statements RBRACE {
        $$ = AST::Make<AST::WithBlock>(@1);
        $$->withins = $2;
        $$->reqs = $3;
        $$->stmts = $5;
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
    : IDENTIFIER IN span_expr {
        symtab.AddSymbol($1, MakeUnknownType()/*Need inference*/);
        $$ = AST::Make<AST::WithIn>(@1, AST::Make<AST::Identifier>(@1,$1), $3);
      }
    | IDENTIFIER ASSIGN LBRACE with_matchers RBRACE IN span_expr {
        symtab.AddSymbol($1, MakeUnknownType()/*Need inference*/);
        $$ = AST::Make<AST::WithIn>(@1, AST::Make<AST::Identifier>(@1,$1), $7, $4);
      }
    | LBRACE with_matchers RBRACE IN span_expr {
        $$ = AST::Make<AST::WithIn>(@1, $5, $2); // no identifier
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
    : FOREACH iv_exprs LBRACE statements RBRACE {
        $$ = AST::Make<AST::ForeachBlock>(@1, $2, $4);
      }
    ;

iv_exprs
    : iv_exprs COMMA iv_expr  {
        $1->Append($3);
        $$ = $1;
      }
    | iv_expr {
        $$ = AST::Make<AST::MultiValues>(@1);
        $$->Append($1);
      }
    ; /* do not allow the empty ivs */

iv_expr
    : IDENTIFIER { $$ = AST::Make<AST::Identifier>(@1, $1); }
    ;

dma_stmt
    : IDENTIFIER ASSIGN DMA dma_operation sync_type dma_config chunkat_expr TRANS chunkat_or_storage {
        symtab.AddSymbol($1, MakeDummyFutureType($5));
        $$ = AST::Make<AST::DMA>(@3, $4, $1, $7, $9, $5, $6);
      }
    | DMA dma_operation sync_type dma_config chunkat_expr TRANS chunkat_or_storage {
        $$ = AST::Make<AST::DMA>(@1, $2, "", $5, $7, $3, $4);
      }
    ;

dma_operation
    : COPY   { $$ = $1; }
    | SLICE  { $$ = $1; }
    | PAD    { $$ = $1; }
    ;

dma_config
    : LT LBRACE iv_list RBRACE COMMA LBRACE iv_list RBRACE COMMA LBRACE iv_list RBRACE COMMA NUM GT {
        auto pc = std::make_shared<PadConfig>();
        for (auto high : $3->values)
          pc->pad_high.push_back(cast<AST::IntLiteral>(high)->Val());
        for (auto low : $7->values)
          pc->pad_low.push_back(cast<AST::IntLiteral>(low)->Val());
        for (auto mid : $11->values)
          pc->pad_mid.push_back(cast<AST::IntLiteral>(mid)->Val());
        pc->SetPadValue($14);
        $$ = pc;
      }
    | /* Empty for no config */ { $$ = nullptr; }
    ;

sync_type
    : { $$ = false; }
    | ASYNC { $$ = true; }
    ;

chunkat_or_storage
    : chunkat_expr { $$ = $1; }
    | storage      { $$ = AST::Make<AST::Memory>(@1, $1); }
    ;

chunkat_expr
    : IDENTIFIER CHUNKAT LPAREN id_list RPAREN {
        $$ = AST::Make<AST::ChunkAt>(@1, AST::Make<AST::Identifier>(@1,$1), $4);
      }
    | IDENTIFIER FNDATA CHUNKAT LPAREN id_list RPAREN {
        $$ = AST::Make<AST::ChunkAt>(@1, AST::Make<AST::Identifier>(@1,$1), $5);
      }
    /*| IDENTIFIER CHUNKAT LPAREN value_list RPAREN {
        $$ = AST::Make<AST::ChunkAt>(@1, AST::Make<AST::Identifier>(@1,$1), $4);
      }*/
    | IDENTIFIER { $$ = AST::Make<AST::ChunkAt>(@1, AST::Make<AST::Identifier>(@1,$1)); }
    | IDENTIFIER FNDATA { $$ = AST::Make<AST::ChunkAt>(@1, AST::Make<AST::Identifier>(@1,$1)); }
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

passables
    : /* Empty */ {
        $$ = AST::Make<AST::MultiValues>(loc);
      }
    | passable {
        $$ = AST::Make<AST::MultiValues>(@1);
        $$->Append($1);
      }
    | passables COMMA passable {
        $1->Append($3);
        $$ = $1;
      }
    ;

passable
    : s_expr { $$ = $1; }
    | s_expr FNDATA {
      $$ = AST::Make<AST::Expr>(@1, "dataof", $1);
    }
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

wait_stmt
    : WAIT id_list {
        $$ = AST::Make<AST::Wait>(@1, $2);
      }
    ;

call_stmt
    : CALL IDENTIFIER LPAREN passables RPAREN {
        $$ = AST::Make<AST::Call>(@1,
                AST::Make<AST::Identifier>(@2, $2), $4);
      }
    ;

%%


// Bison expects us to provide implementation - otherwise linker complains
void Parser::error(const location &loc , const std::string &message) {
  std::cerr << loc << ": ";
  if (should_use_colors())
      std::cerr << red;
  std::cerr << "error: ";
  if (should_use_colors())
      std::cerr << reset;
  std::cerr << message << std::endl;
}
