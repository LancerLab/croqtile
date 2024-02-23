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

extern char* yytext;
extern Choreo::location loc;

extern AST::Program root;
extern AST::SymbolTable symtab;
extern AST::ITupleTable ituple_symtab;

const char* red = "\033[31m";
const char* reset = "\033[0m";

static inline bool shell_supports_colors() {
	const char* term = getenv("TERM");
	return term && (strcmp(term, "xterm-256color") == 0
							 || strcmp(term, "xterm") == 0);
}

static Choreo::Parser::symbol_type yylex(Choreo::Scanner &scanner) {
  return scanner.get_next_token();
}

}

%{
#include <stdio.h>
extern int yylex();

void choreo_info(const char *message) {
    // fprintf(stderr, "Error: %s\n", s);
  const char* GREEN = "\033[32m";
  if (shell_supports_colors())
      std::cerr << GREEN;
  std::cerr << "Info: ";
  if (shell_supports_colors())
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
  TRANS   "=>"
  BIND    "<->"
;


// instead of union, using c++17 variant for terminal and non-terminals
%define api.value.type variant

// terminals
%token END 0 "end of file"
%token <char> CHAR
%token <int> NUM
%token <std::string> CPP_CODE
%token <std::string> IDENTIFIER ATTR_CO
// type related
%token <std::string> MDSPAN ITUPLE
%token <AST::Storage> LOCAL SHARED GLOBAL
%token <AST::BaseType> F32 F16 BF16 U16 S16 U8 S8 U32 S32 INT BOOL
// builtin operations
%token <std::string> DMA DLIN DSLICE DPAD COPY FNSPAN FNDATA CHUNKAT WAIT CALL
// control related
%token <std::string> IF ELSE PARA BY WITH IN ITER RET REQUIRE
%token <std::string> TRUE FALSE

// non-terminals
%nterm <std::string> dma_operation
%nterm <AST::Storage> storage
%nterm <AST::BaseType> base_type
%nterm <AST::ptr<AST::Node>> pass_by foreach_block simple_val span_val ituple_val int_val bool_val declaration statement assignment pb_statement w_statement dma_statement wait_statement call_statement span_elem mixed_span_elem iv_expr
%nterm <AST::ptr<AST::MultiNodes>> statements declarations assignments pb_statements w_statements iterate_statements span_list mixed_span_list withins iv_exprs require_binds require_clause id_list with_matchers
%nterm <AST::ptr<AST::NodeRef>> if_else
%nterm <AST::ptr<AST::IntList>> int_list
%nterm <AST::ptr<AST::SValList>> sval_list
%nterm <AST::ptr<AST::Expr>> term expr span_expr span_term
%nterm <AST::ptr<AST::DataType>> general_type param_type aggregate_type
%nterm <AST::ptr<AST::ParamList>> parameter_list
%nterm <AST::ptr<AST::ParamType>> parameter
%nterm <AST::ptr<AST::ChoreoFunction>> dsl_function
%nterm <AST::ptr<AST::MultiSpans>> unnamed_span_decl
%nterm <AST::ptr<AST::NamedDecl>> named_span_decl named_tuple_decl scalar_decl
%nterm <AST::ptr<AST::IntTuple>> unnamed_tuple_decl tuple_assign
%nterm <AST::ptr<AST::IntIndex>> s_index
%nterm <AST::ptr<AST::IntIndexList>> s_index_list
%nterm <AST::ptr<AST::WithBlock>> with_block
%nterm <AST::ptr<AST::WithIn>> within
%nterm <AST::ptr<AST::RequireBind>> require_bind 
%nterm <AST::ptr<AST::ParallelBy>> para_by
%nterm <AST::ptr<AST::ChunkAt>> chunkat_expr

%%

program
    : /* Empty */ {}
    | program pass_by      { root.nodes.push_back($2); }
    | program dsl_function { root.nodes.push_back($2); }
    | END {}
    ;

pass_by
    : CPP_CODE {
        $$ = std::make_shared<AST::CppSourceCode>($1);
      }
    ;

dsl_function
    : ATTR_CO param_type IDENTIFIER LPAREN parameter_list RPAREN LBRACE statements RBRACE {
        $$ = std::make_shared<AST::ChoreoFunction>();
        $$->name = $3;
        $$->f_decl.name = $3;
        $$->f_decl.ret_type = $2;
        $$->f_decl.params = $5;
        $$->statms = $8;
      }
    ;

param_type
    : base_type { $$ = std::make_shared<AST::DataType>($1); }
    | base_type MDSPAN LT NUM GT { $$ = std::make_shared<AST::DataType>($1, false); }
    ;

general_type
    : base_type { $$ = std::make_shared<AST::DataType>($1); }
    | aggregate_type { $$ = $1; }
    ;

aggregate_type
    : base_type MDSPAN LBRAKT span_list RBRAKT {
        $$ = std::make_shared<AST::DataType>($1,
              std::make_shared<AST::MultiSpans>("", $4));
      }
    | base_type LT IDENTIFIER GT {
        if (!symtab.exists($3))
          Choreo::Parser::error(@3, "The symbol `" + $3 + "' has not been defined.");

        if (!symtab.getSymbol($3)->isAggregate())
          Choreo::Parser::error(@3, "expecting a symbol of aggregate type.");

        $$ = std::make_shared<AST::DataType>($1, std::make_shared<AST::Identifier>($3));
      }
    ;

base_type
    : F32   { $$ = $1; }
    | F16   { $$ = $1; }
    | BF16  { $$ = $1; }
    | U16   { $$ = $1; }
    | S16   { $$ = $1; }
    | U8    { $$ = $1; }
    | S8    { $$ = $1; }
    | U32   { $$ = $1; }
    | S32   { $$ = $1; }
    | INT   { $$ = $1; }
    | BOOL  { $$ = $1; }
    ;

int_list
    : /* allows the empty list */ {
        $$ = std::make_shared<AST::IntList>();
      }
    | int_list COMMA NUM {
        $1->Append(std::make_shared<AST::IntLiteral>($3));
        $$ = $1;
      }
    | NUM {
        $$ = std::make_shared<AST::IntList>();
        $$->values.push_back(std::make_shared<AST::IntLiteral>($1));
      }
    ;

sval_list
    : /* allows the empty list */ {
        $$ = std::make_shared<AST::SValList>();
      }
    | sval_list COMMA simple_val {
        $1->Append($3);
        $$ = $1;
      }
    | simple_val {
        $$ = std::make_shared<AST::SValList>();
        $$->Append($1);
      }
    ;

simple_val
    : NUM { $$ = std::make_shared<AST::IntLiteral>($1); }
    | IDENTIFIER {
        if (!symtab.exists($1))
          Choreo::Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

#if 0
        if (symtab.getSymbol($1)->isAggregate())
          Choreo::Parser::error(@1, "expecting symbol `" + $1 +
                                "' of a scalar type.");
#endif

        if (symtab.getSymbol($1)->getType() != AST::BaseType::INT)
          Choreo::Parser::error(@1, "expecting symbol `" + $1 +
                                "' of an integer type.");

        $$ = std::make_shared<AST::Identifier>($1);
    	}
    ;

int_val
    : IDENTIFIER s_index {
        if (!symtab.exists($1)) {
          Choreo::Parser::error(@1, ": the symbol '" + $1 + "` is not defined.");
          exit(1);
        }

        $$ = std::make_shared<AST::NthBound>(std::make_shared<AST::Identifier>($1), $2);
      }
    | simple_val { $$ = $1; }
    ;

bool_val
    : TRUE { $$ = std::make_shared<AST::Boolean>(std::string("true")); }
    | FALSE { $$ = std::make_shared<AST::Boolean>(std::string("false")); }
    ;

parameter_list
    : /* Empty */ {
        $$ = std::make_shared<AST::ParamList>();
      }
    | parameter_list COMMA parameter {
        $1->values.push_back($3);
        $$ = $1;
      }
    | parameter {
        $$ = std::make_shared<AST::ParamList>();
        $$->values.push_back($1);
      }
    ;

parameter
    : param_type IDENTIFIER { /* handle parameter type and name here */
        $$ = std::make_shared<AST::ParamType>(std::pair($1, std::make_shared<AST::Identifier>($2)));
        if (symtab.exists($2)) {
          Choreo::Parser::error(@2, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.addSymbol($2, $1->getBaseType(), $1->isAggregate());
      }
    | param_type {
        $$ = std::make_shared<AST::ParamType>(std::pair($1, std::make_shared<AST::Identifier>(AST::SymbolTable::getAnonName())));
      }
    ;

statements
    : /* no statement */ { $$ = std::make_shared<AST::MultiNodes>(); }
    | statements statement {
        $1->Append($2);
        $$ = $1;
      }
    ;

statement
    : declarations SEMCOL { $$ = $1; }
    | assignments  SEMCOL { $$ = $1; }
    | para_by             { $$ = $1; }
    ;

para_by
    : PARA IDENTIFIER BY NUM LBRACE pb_statements RBRACE {
        $$ = std::make_shared<AST::ParallelBy>($2, $4);
        $$->statms = $6;
      }
    ;

pb_statements
    : /* Empty */ { $$ = std::make_shared<AST::MultiNodes>(); }
    | pb_statements pb_statement {
        $1->Append($2);
        $$ = $1;
      }
    ;

pb_statement
    : declarations SEMCOL { $$ = $1; }
    | assignments  SEMCOL { $$ = $1; }
    | with_block { $$ = $1; }
    | if_else
    ;

assignments
    : assignments COMMA assignment {
        $1->Append($3);
        $$ = $1;
      }
    | assignment {
        $$ = std::make_shared<AST::MultiNodes>();
        $$->Append($1);
      }
    ;

declarations
    : declarations COMMA declaration {
        $1->Append($3);
        $$ = $1;
      }
    | declaration {
        $$ = std::make_shared<AST::MultiNodes>();
        $$->Append($1);
      }
    ;

declaration
    : named_span_decl  { $$ = $1; }
    | named_tuple_decl { $$ = $1;  }
    | scalar_decl      { $$ = $1;  }
    ;

scalar_decl
    : INT IDENTIFIER ASSIGN expr {
        if (symtab.exists($2)) {
          Choreo::Parser::error(@2, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.addSymbol($2, $1);
        $$ = std::make_shared<AST::NamedDecl>($2, "int", $4);
      }
    | BOOL IDENTIFIER ASSIGN expr {
        if (symtab.exists($2)) {
          Choreo::Parser::error(@2, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.addSymbol($2, $1);
        $$ = std::make_shared<AST::NamedDecl>($2, "bool", $4);
      }
    ;

s_index
    : LPAREN simple_val RPAREN {
        $$ = std::make_shared<AST::IntIndex>($2);
      }
    ;

s_index_list
    : s_index_list COMMA s_index {
        $1->Append($3);
        $$ = $1;
      }
    | s_index {
        $$ = std::make_shared<AST::IntIndexList>();
        $$->indices.push_back($1);
      }
    ; /* do not allow empty list */

span_elem
    : s_index     { $$ = $1; }
    | simple_val  { $$ = $1; }
    ; // do not allow non-element

span_list
    : span_list COMMA span_elem {
        $1->Append($3);
        $$ = $1;
      }
    | span_elem {
        $$ = std::make_shared<AST::MultiNodes>();
        $$->Append($1);
      }
    ; // do not allow an empty list

mixed_span_elem
    : IDENTIFIER s_index {
        if (!symtab.exists($1))
          Choreo::Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

        if (!symtab.getSymbol($1)->isAggregate())
          Choreo::Parser::error(@1, "expecting a symbol of aggregate type.");

        $$ = std::make_shared<AST::NthBound>(
            std::make_shared<AST::Identifier>($1), $2);
      }
    | IDENTIFIER FNSPAN s_index {
        if (!symtab.exists($1))
          Choreo::Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

        if (!symtab.getSymbol($1)->isAggregate())
          Choreo::Parser::error(@1, "expecting a symbol of aggregate type.");

        $$ = std::make_shared<AST::NthBound>(
            std::make_shared<AST::Identifier>($1+$2), $3);

      }
    | simple_val { $$ = $1; }
    ; // do not allow an empty item

mixed_span_list
    : /* Empty list */ {
        $$ = std::make_shared<AST::MultiNodes>();
      }
    | mixed_span_list COMMA mixed_span_elem {
        $1->Append($3);
        $$ = $1;
      }
    | mixed_span_elem {
        $$ = std::make_shared<AST::MultiNodes>();
        $$->Append($1);
      }
    ;

unnamed_span_decl
    : IDENTIFIER FNSPAN LBRAKT span_list RBRAKT {
        $$ = std::make_shared<AST::MultiSpans>($1, $4);
      }
    | IDENTIFIER LBRAKT span_list RBRAKT {
        $$ = std::make_shared<AST::MultiSpans>($1, $3);
      }
    | LBRAKT mixed_span_list RBRAKT {
        $$ = std::make_shared<AST::MultiSpans>("", $2);
      }
    ;

span_val
    : unnamed_span_decl { $$ = $1; }
    | IDENTIFIER { $$ = std::make_shared<AST::Identifier>($1); }
    | IDENTIFIER FNSPAN { $$ = std::make_shared<AST::Identifier>($1 + $2); }
    ;

named_span_decl
    : MDSPAN IDENTIFIER COL span_expr {
        if (symtab.exists($2)) {
          Choreo::Parser::error(@2, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.addSymbol($2, AST::BaseType::INT, true);
        $$ = std::make_shared<AST::NamedDecl>($2, "span", $4, "-");
      }
    | MDSPAN LT NUM GT IDENTIFIER COL span_expr {
        // TODO: check if the span defined aligned with declaration
        #if 0
        if ($3 != $7->list.size())
          Choreo::Parser::error(@3,
            "The rank of mdspan is not consistent with its decleration.");
            #endif
        if (symtab.exists($5)) {
          Choreo::Parser::error(@5, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.addSymbol($5, AST::BaseType::INT, true);
        $$ = std::make_shared<AST::NamedDecl>($5, "span", $7, "-");
      }
    | IDENTIFIER COL span_expr {
        if (symtab.exists($1)) {
          Choreo::Parser::error(@1, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.addSymbol($1, AST::BaseType::INT, true);
        $$ = std::make_shared<AST::NamedDecl>($1, "span", $3, "-");
      }
    ;

unnamed_tuple_decl
    : LBRACE sval_list RBRACE {
        $$ = std::make_shared<AST::IntTuple>("", $2);
      }
    | IDENTIFIER LBRACE s_index_list RBRACE {
        // anchor
        if (!symtab.exists($1))
          Choreo::Parser::error(@1, "The symbol has not been defined.");

        $$ = std::make_shared<AST::IntTuple>($1, $3);
        #if 0
        auto src_ituple = ituple_symtab.getSymbol($1);
        auto ret_tuple = std::make_shared<AST::SValList>();
        for (AST::ptr<AST::IntIndex> idx : $3->indices) {
          // TODO: evaluate index later
          ret_tuple->Append(src_ituple->value->values[dynamic_cast<AST::IntLiteral*>(&*idx->value)->value]);
        }
        $$ = std::make_shared<AST::IntTuple>($1, std::move(ret_tuple));
        #endif
      }
    ;

ituple_val
    : unnamed_tuple_decl { $$ = $1; }
    | IDENTIFIER {
        $$ = std::make_shared<AST::Identifier>($1);
      }
    ;

named_tuple_decl
    : ITUPLE IDENTIFIER ASSIGN ituple_val {
        /* TODO: workaround: use INT for ituple's base type use a dedicated type for ituple in symboltable */
        //ituple_symtab.addITupleSymbol($2, $4);
        symtab.addSymbol($2, AST::BaseType::INT, true);
        $$ = std::make_shared<AST::NamedDecl>($2, "ituple", $4);
      }
    | IDENTIFIER ASSIGN unnamed_tuple_decl {
        //ituple_symtab.addITupleSymbol($1, $3);
        symtab.addSymbol($1, AST::BaseType::INT, true);
        $$ = std::make_shared<AST::NamedDecl>($1, "ituple", $3);
      }
    ;

storage
    : LOCAL   { $$ = $1; }
    | SHARED  { $$ = $1; }
    | GLOBAL  { $$ = $1; };

assignment
    : IDENTIFIER ASSIGN expr {
        #if 0
        // The assignment is overrided to be initilization of ituple
        auto ref = std::dynamic_pointer_cast<AST::NodeRef>($3->value_r);
        if (ref) {
          auto sym = std::dynamic_pointer_cast<AST::Identifier>(ref->value);
          if (sym) {
            auto src_ituple = ituple_symtab.getSymbol(sym->name);
            auto ret_ituple = std::make_shared<AST::IntTuple>($1, src_ituple->value);
            ituple_symtab.addITupleSymbol($1, ret_ituple);
            $$ = std::make_shared<AST::NodeRef>(ret_ituple);
            break;
          }
        }
        #endif

        if (!symtab.exists($1)) {
          // since the symbol is not defined, it is a declaration without type annotation
          symtab.addSymbol($1, AST::BaseType::INT, true);
          $$ = std::make_shared<AST::NamedDecl>($1, "ituple", $3);
          break;
        }

        $$ = std::make_shared<AST::Assignment>($1, $3);
      }
    ;

expr
    : expr PLUS term { $$ = std::make_shared<AST::Expr>("+", $1, $3); }
    | expr MINUS term { $$ = std::make_shared<AST::Expr>("-", $1, $3); }
    | term { $$ = $1; }
    ;

term
    : term STAR int_val { $$ = std::make_shared<AST::Expr>("*", $1, $3); }
    | term SLASH int_val { $$ = std::make_shared<AST::Expr>("/", $1, $3); }
    | term PECET int_val { $$ = std::make_shared<AST::Expr>("%", $1, $3); }
    | LPAREN expr RPAREN { $$ = $2; }
    | int_val { $$ = std::make_shared<AST::Expr>($1); }
    | bool_val { $$ = std::make_shared<AST::Expr>($1); }
    ;

span_expr
    : span_expr PLUS span_term { $$ = std::make_shared<AST::Expr>("+", $1, $3); }
    | span_expr MINUS span_term { $$ = std::make_shared<AST::Expr>("-", $1, $3); }
    | span_term { $$ = $1; }
    ;

span_term
    : span_term STAR ituple_val { $$ = std::make_shared<AST::Expr>("*", $1, $3); }
    | span_term SLASH ituple_val { $$ = std::make_shared<AST::Expr>("/", $1, $3); }
    | span_term PECET ituple_val { $$ = std::make_shared<AST::Expr>("%", $1, $3); }
    | LPAREN span_expr RPAREN { $$ = $2; }
    | span_val { $$ = std::make_shared<AST::Expr>($1); }
    ;

if_else
    : if_clause
    | if_clause else_clause
    ;

if_clause
    : IF LBRAKT cmp_expr RBRAKT LBRACE RBRACE
    ;

else_clause:
       ELSE LBRACE RBRACE
    ;

cmp_expr
    : expr LT expr
    | expr GT expr
    | expr EQ expr
    | expr NE expr
    | expr LE expr
    | expr GE expr
    ;

with_block
    : WITH withins LBRACE w_statements RBRACE {
        $$ = std::make_shared<AST::WithBlock>();
        $$->withins = $2;
        $$->statms = $4;
      }
    | WITH withins require_clause LBRACE w_statements RBRACE {
        $$ = std::make_shared<AST::WithBlock>();
        $$->withins = $2;
        $$->reqs = $3;
        $$->statms = $5;
      }
    ;

withins
    : withins COMMA within {
        $1->Append($3);
        $$ = $1;
      }
    | within {
        $$ = std::make_shared<AST::MultiNodes>();
        $$->Append($1);
      }
    ; /* do not allow empty within */

within
    : IDENTIFIER IN IDENTIFIER {
        if (!symtab.exists($3))
          Choreo::Parser::error(@3, "The symbol has not been defined.");

        symtab.addSymbol($1, AST::BaseType::INT, true);
        $$ = std::make_shared<AST::WithIn>(std::make_shared<AST::Identifier>($1),
                                           std::make_shared<AST::Identifier>($3));
      }
    | IDENTIFIER ASSIGN LBRACE with_matchers RBRACE IN IDENTIFIER {
        if (!symtab.exists($7))
          Choreo::Parser::error(@7, "The symbol has not been defined.");

        symtab.addSymbol($1, AST::BaseType::INT, true);
        $$ = std::make_shared<AST::WithIn>(std::make_shared<AST::Identifier>($1),
                                           std::make_shared<AST::Identifier>($7));
        $$->with_matchers = $4;
      }
    ;

require_clause
    : REQUIRE require_binds {
        /* $$ = std::make_shared<AST::RequireClause>();
        $$->binds = $2;*/
        $$ = $2;
      }
    ;

require_binds
    : require_binds COMMA require_bind {
        $1->Append($3);
        $$ = $1;
      }
    | require_bind {
        $$ = std::make_shared<AST::MultiNodes>();
        $$->Append($1);
      }
    ; /* do not allow empty require_bind */

require_bind
    : IDENTIFIER BIND IDENTIFIER {
        if (!symtab.exists($1))
          Choreo::Parser::error(@1, "The symbol has not been defined.");
        if (!symtab.exists($3))
          Choreo::Parser::error(@3, "The symbol has not been defined.");

        $$ = std::make_shared<AST::RequireBind>(std::make_shared<AST::Identifier>($1),
                                           std::make_shared<AST::Identifier>($3));
      }
    ;

w_statements
    : /*Empty statement */ { $$ = std::make_shared<AST::MultiNodes>(); }
    | w_statements w_statement {
        $1->Append($2);
        $$ = $1;
      }
    ;

w_statement
    : declarations SEMCOL   { $$ = $1; }
    | assignments  SEMCOL   { $$ = $1; }
    | dma_statement SEMCOL  { $$ = $1; }
    | wait_statement SEMCOL { $$ = $1; }
    | call_statement SEMCOL { $$ = $1; }
    | if_else               { $$ = $1; }
    | foreach_block         { $$ = $1; }
    ;

foreach_block
    : ITER iv_exprs LBRACE w_statements RBRACE {
        $$ = std::make_shared<AST::ForeachBlock>($2, $4);
      }
    ;

iv_exprs
    : iv_exprs COMMA iv_expr  {
        $1->Append($3);
        $$ = $1;
      }
    | iv_expr {
        $$ = std::make_shared<AST::MultiNodes>();
        $$->Append($1);
      }
    ; /* do not allow the empty ivs */

iv_expr
    : IDENTIFIER { $$ = std::make_shared<AST::Identifier>($1); }
    ;

dma_statement
    : IDENTIFIER ASSIGN DMA dma_operation chunkat_expr TRANS storage {
        $$ = std::make_shared<AST::DMA>(
              $4,
              std::make_shared<AST::Identifier>($1),
              $5,
              std::make_shared<AST::Memory>($7));
      }
    | IDENTIFIER ASSIGN DMA dma_operation IDENTIFIER TRANS chunkat_expr {
        $$ = std::make_shared<AST::DMA>(
              $4,
              std::make_shared<AST::Identifier>($1),
              std::make_shared<AST::Identifier>($5),
              $7);
      }
    ;

dma_operation
    : DLIN    { $$ = $1; }
    | DSLICE  { $$ = $1; }
    | DPAD    { $$ = $1; }
    ;

chunkat_expr
    : IDENTIFIER CHUNKAT LPAREN id_list RPAREN {
        $$ = std::make_shared<AST::ChunkAt>(
              std::make_shared<AST::Identifier>($1), $4);
      }
    ;

id_list
    : id_list COMMA IDENTIFIER {
        $1->Append(std::make_shared<AST::Identifier>($3));
        $$ = $1;
      }
    | IDENTIFIER {
        $$ = std::make_shared<AST::MultiNodes>();
        $$->Append(std::make_shared<AST::Identifier>($1));
      }
    ;

with_matchers /* TODO: this special case is pattern-match ids for with-block */
    : with_matchers COMMA IDENTIFIER {
        $1->Append(std::make_shared<AST::Identifier>($3));
        std::cout << symtab.exists($3) << std::endl;
        symtab.addSymbol($3, AST::BaseType::INT, false); /* in withins, this values should be int only */
        std::cout << symtab.exists($3) << std::endl;
        std::cout << "here" << $3 << std::endl;
        $$ = $1;
      }
    | IDENTIFIER {
        $$ = std::make_shared<AST::MultiNodes>();
        std::cout << symtab.exists($1) << std::endl;
        symtab.addSymbol($1, AST::BaseType::INT, false); /* in withins, this values should be int only */
        std::cout << symtab.exists($1) << std::endl;
        std::cout << "here" << $1 << std::endl;
        $$->Append(std::make_shared<AST::Identifier>($1));
      }
    ;

wait_statement
    : WAIT IDENTIFIER {
        $$ = std::make_shared<AST::Wait>(
                std::make_shared<AST::Identifier>($2));
      }
    ;

call_statement
    : CALL IDENTIFIER LPAREN id_list RPAREN {
        $$ = std::make_shared<AST::Call>(
                std::make_shared<AST::Identifier>($2), $4);
      }
    | storage IDENTIFIER ASSIGN CALL IDENTIFIER LPAREN id_list RPAREN  {

      }
    ;

%%


// Bison expects us to provide implementation - otherwise linker complains
void Choreo::Parser::error(const location &loc , const std::string &message) {
  std::cerr << loc << ": ";
  if (shell_supports_colors())
      std::cerr << red;
  std::cerr << "Error: ";
  if (shell_supports_colors())
      std::cerr << reset;
  std::cerr << message << std::endl;
}
