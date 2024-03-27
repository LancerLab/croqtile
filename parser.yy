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

using namespace Choreo;

extern char* yytext;
extern location loc;

extern AST::Program root;
extern AST::SymbolTable symtab;

const char* red = "\033[31m";
const char* reset = "\033[0m";

static inline bool shell_supports_colors() {
	const char* term = getenv("TERM");
	return term && (strcmp(term, "xterm-256color") == 0
							 || strcmp(term, "xterm") == 0);
}

static Parser::symbol_type yylex(Scanner &scanner) {
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
  AND     "&&"
  OR      "||"
  NOT     "!"
  QES     "?"
  TRANS   "=>"
  BIND    "<->"
  PIPE    "|"
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
%token <Choreo::BaseType> F32 F16 BF16 U16 S16 U8 S8 U32 S32 INT BOOL
// builtin operations
%token <std::string> DMA DLIN DSLICE DPAD COPY FNSPAN FNDATA CHUNKAT WAIT CALL
// control related
%token <std::string> IF ELSE PARA BY WITH IN ITER RET REQUIRE
%token <std::string> TRUE FALSE

// non-terminals
%nterm <std::string> dma_operation
%nterm <AST::Storage> storage
%nterm <Choreo::BaseType> fundamental_type
%nterm <AST::ptr<AST::Memory>> storage_qual
%nterm <AST::ptr<AST::Node>> pass_by foreach_block simple_val span_val ituple_val int_val bool_literal passable declaration statement assignment pb_statement w_statement dma_statement wait_statement call_statement span_elem mixed_span_elem iv_expr if_else optional_scalar_init param_mdspan_val
%nterm <AST::ptr<AST::MultiNodes>> statements declarations assignments pb_statements w_statements span_list mixed_span_list param_mdspan_list withins iv_exprs require_binds require_clause id_list with_matchers else_clause futures passables
%nterm <AST::ptr<AST::SValList>> sval_list
%nterm <AST::ptr<AST::Expr>> s_expr span_expr
%nterm <AST::ptr<AST::DataType>> scalar_type param_type spanned_type
%nterm <AST::ptr<AST::ParamList>> parameter_list
%nterm <AST::ptr<AST::Parameter>> parameter
%nterm <AST::ptr<AST::ChoreoFunction>> dsl_function
%nterm <AST::ptr<AST::MultiDimSpans>> unnamed_mdspan_decl param_mdspan
%nterm <AST::ptr<AST::NamedTypeDecl>> named_mdspan_decl
%nterm <AST::ptr<AST::NamedVariableDecl>> named_ituple_decl named_scalar_decl named_spanned_decl
%nterm <AST::ptr<AST::IntTuple>> unnamed_ituple_decl
%nterm <AST::ptr<AST::IntIndex>> s_index
%nterm <AST::ptr<AST::IntIndexList>> s_index_list
%nterm <AST::ptr<AST::WithBlock>> with_block
%nterm <AST::ptr<AST::WithIn>> within
%nterm <AST::ptr<AST::RequireBind>> require_bind
%nterm <AST::ptr<AST::ParallelBy>> para_by
%nterm <AST::ptr<AST::Return>> return
%nterm <AST::ptr<AST::ChunkAt>> chunkat_expr

// precedence (low to high) and associativity
%right ASSIGN
%right QES COL
%left OR
%left AND
%right NOT
%nonassoc LT GT LE GE EQ NE
%left PLUS MINUS
%left STAR SLASH PECET

%%

program
    : /* Empty */ {}
    | program pass_by      { root.nodes.push_back($2); }
    | program dsl_function { root.nodes.push_back($2); }
    ;

pass_by
    : CPP_CODE {
        $$ = AST::Make<AST::CppSourceCode>(@1, $1);
      }
    ;

dsl_function
    : ATTR_CO param_type IDENTIFIER LPAREN parameter_list RPAREN LBRACE statements RBRACE {
        $$ = AST::Make<AST::ChoreoFunction>(@1);
        $$->name = $3;
        $$->f_decl.name = $3;
        $$->f_decl.ret_type = $2;
        $$->f_decl.params = $5;
        $$->statms = $8;
      }
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
    ;

param_mdspan_list
    : param_mdspan_list COMMA param_mdspan_val {
        $1->Append($3);
        $$ = $1;
      }
    | param_mdspan_val {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ;

param_mdspan_val
    : NUM   { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | QES   { $$ = AST::Make<AST::IntLiteral>(@1); }
    ;

scalar_type
    : INT   { $$ = AST::Make<AST::DataType>(@1, $1); }
    | BOOL  { $$ = AST::Make<AST::DataType>(@1, $1); }
    ;

spanned_type
    : fundamental_type unnamed_mdspan_decl {
        $$ = AST::Make<AST::DataType>(@1, $1, $2);
      }
    | fundamental_type LBRAKT span_expr RBRAKT {
        printf("1");
        $$ = AST::Make<AST::DataType>(@1, $1, $3);
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

sval_list
    : /* allows the empty list */ {
        $$ = AST::Make<AST::SValList>(loc);
      }
    | sval_list COMMA s_expr {
        $1->Append($3);
        $$ = $1;
      }
    | s_expr {
        $$ = AST::Make<AST::SValList>(@1);
        $$->Append($1);
      }
    ;

simple_val
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

int_val
    : IDENTIFIER s_index {
        if (!symtab.Exists($1)) {
          Parser::error(@1, ": the symbol '" + $1 + "` is not defined.");
          exit(1);
        }

        $$ = AST::Make<AST::NthBound>(@1, AST::Make<AST::Identifier>(@1, $1), $2);
      }
    | simple_val { $$ = $1; }
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
        if (symtab.Exists($2)) {
          Parser::error(@2, "ODR violation: the symbol '" + $2 + "` is already defined.");
          exit(1);
        }
        symtab.AddSymbol($2, $1->MakeSemaType());
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
    ;

statement
    : declarations SEMCOL { $$ = $1; }
    | assignments  SEMCOL { $$ = $1; }
    | para_by             { $$ = $1; }
    | if_else             { $$ = $1; }
    | return SEMCOL       { $$ = $1; }
    ;

return
    : RET          { $$ = AST::Make<AST::Return>(@1);}
    | RET passable { $$ = AST::Make<AST::Return>(@1, $2); }
    ;

para_by
    : PARA IDENTIFIER BY NUM LBRACE pb_statements RBRACE {
        $$ = AST::Make<AST::ParallelBy>(@1, $2, $4);
        $$->statms = $6;
      }
    ;

pb_statements
    : /* Empty */ { $$ = AST::Make<AST::MultiNodes>(loc); }
    | pb_statements pb_statement {
        $1->Append($2);
        $$ = $1;
      }
    ;

pb_statement
    : declarations SEMCOL { $$ = $1; }
    | assignments  SEMCOL { $$ = $1; }
    | with_block { $$ = $1; }
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
    : declarations COMMA declaration {
        $1->Append($3);
        $$ = $1;
      }
    | declaration {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ;

declaration
    : named_mdspan_decl  { $$ = $1; }
    | named_ituple_decl  { $$ = $1; }
    | named_scalar_decl  { $$ = $1; }
    | named_spanned_decl { $$ = $1; }
    ;

named_scalar_decl
    : scalar_type IDENTIFIER optional_scalar_init {
        if (symtab.Exists($2)) {
          Parser::error(@2, "ODR violation: the symbol '" + $2 + "` is already defined.");
          exit(1);
        }
        assert($1->isScalar() && "Not a scalar type.");
        symtab.AddSymbol($2, $1->MakeSemaType());
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

named_spanned_decl
    : storage_qual spanned_type IDENTIFIER {
        if (symtab.Exists($3)) {
          Parser::error(@3, "ODR violation: the symbol '" + $3 + "` is already defined.");
          exit(1);
        }
        symtab.AddSymbol($3, $2->MakeSemaType());
        $$ = AST::Make<AST::NamedVariableDecl>(@3, $3, $2, $1);
      }
    ;

s_index
    : LPAREN simple_val RPAREN {
        $$ = AST::Make<AST::IntIndex>(@2, $2);
      }
    ;

s_index_list
    : s_index_list COMMA s_index {
        $1->Append($3);
        $$ = $1;
      }
    | s_index {
        $$ = AST::Make<AST::IntIndexList>(@1);
        $$->indices.push_back($1);
      }
    ; /* do not allow empty list */

span_elem
    : s_index     { $$ = $1; }
    | simple_val  { $$ = $1; }
    | QES         { $$ = AST::Make<AST::IntLiteral>(@1); }
    ; // do not allow non-element

span_list
    : span_list COMMA span_elem {
        $1->Append($3);
        $$ = $1;
      }
    | span_elem {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; // do not allow an empty list

mixed_span_elem
    : IDENTIFIER s_index {
        if (!symtab.Exists($1))
          Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

//        if (!symtab.GetSymbol($1)->IsComposite())
//          Parser::error(@1, "expecting a symbol of composite type.");

        $$ = AST::Make<AST::NthBound>(@1,
            AST::Make<AST::Identifier>(@1, $1), $2);
      }
    | IDENTIFIER FNSPAN s_index {
        if (!symtab.Exists($1))
          Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

//        if (!symtab.GetSymbol($1)->IsComposite())
//          Parser::error(@1, "expecting a symbol of composite type.");

        $$ = AST::Make<AST::NthBound>(@1,
            AST::Make<AST::Identifier>(@1, $1+$2), $3);

      }
    | simple_val { $$ = $1; }
    ; // do not allow an empty item

mixed_span_list
    : /* Empty list */ {
        $$ = AST::Make<AST::MultiNodes>(loc);
      }
    | mixed_span_list COMMA mixed_span_elem {
        $1->Append($3);
        $$ = $1;
      }
    | mixed_span_elem {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ;

unnamed_mdspan_decl
    : IDENTIFIER FNSPAN LBRAKT span_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, $1, $4);
      }
    | IDENTIFIER LBRAKT span_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, $1, $3);
      }
    | LBRAKT mixed_span_list RBRAKT {
        $$ = AST::Make<AST::MultiDimSpans>(@1, "", $2);
      }
    ;

span_val
    : unnamed_mdspan_decl { $$ = $1; }
    | IDENTIFIER { $$ = AST::Make<AST::Identifier>(@1, $1); }
    | IDENTIFIER FNSPAN { $$ = AST::Make<AST::Identifier>(@1, $1 + $2); }
    ;

named_mdspan_decl
    : MDSPAN IDENTIFIER COL span_expr {
        if (symtab.Exists($2)) {
          Parser::error(@2, "ODR violation: the symbol '" + $2 + "`  is already defined.");
          exit(1);
        }
        symtab.AddSymbol($2, MakeUninitMDSpanType());
        $$ = AST::Make<AST::NamedTypeDecl>(@2, $2, $4);
      }
    | MDSPAN LT NUM GT IDENTIFIER COL span_expr {
        // TODO: check if the span defined aligned with declaration
        #if 0
        if ($3 != $7->list.size())
          Parser::error(@3,
            "The rank of mdspan is not consistent with its decleration.");
            #endif
        if (symtab.Exists($5)) {
          Parser::error(@5, "ODR violation: the symbol '" + $5 + "` is already defined.");
          exit(1);
        }
        symtab.AddSymbol($5, MakeDimSizedMDSpanType($3));
        $$ = AST::Make<AST::NamedTypeDecl>(@5, $5, $7);
      }
    | IDENTIFIER COL span_expr {
        if (symtab.Exists($1)) {
          Parser::error(@1, "ODR violation: the symbol '" + $1 + "` is already defined.");
          exit(1);
        }
        symtab.AddSymbol($1, MakeUninitMDSpanType());
        $$ = AST::Make<AST::NamedTypeDecl>(@1, $1, $3);
      }
    ;

unnamed_ituple_decl
    : LBRACE sval_list RBRACE {
        $$ = AST::Make<AST::IntTuple>(@1, "", $2);
      }
    | IDENTIFIER LBRACE s_index_list RBRACE {
        // anchor
        if (!symtab.Exists($1))
          Parser::error(@1, "The symbol '" + $1 + "` has not been defined.");

        $$ = AST::Make<AST::IntTuple>(@1, $1, $3);
      }
    ;

ituple_val
    : unnamed_ituple_decl { $$ = $1; }
    | IDENTIFIER {
        $$ = AST::Make<AST::Identifier>(@1, $1);
      }
    ;

named_ituple_decl
    : ITUPLE IDENTIFIER ASSIGN ituple_val {
        /* TODO: workaround: use INT for ituple's base type use a dedicated type for ituple in symboltable */
        symtab.AddSymbol($2, MakeUninitITupleType());
        $$ = AST::Make<AST::NamedVariableDecl>(@2,
              $2, AST::Make<AST::DataType>(@1, BaseType::ITUPLE), nullptr, $4);
      }
    | IDENTIFIER ASSIGN unnamed_ituple_decl {
        symtab.AddSymbol($1, MakeUninitITupleType());
        $$ = AST::Make<AST::NamedVariableDecl>(@1,
              $1, AST::Make<AST::DataType>(@1, BaseType::ITUPLE), nullptr, $3);
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
                $1, AST::Make<AST::DataType>(@1, BaseType::ITUPLE), nullptr, $3);
          break;
        }
        $$ = AST::Make<AST::Assignment>(@2, $1, $3);
      }
    | IDENTIFIER PLUS ASSIGN s_expr {
        if (!symtab.Exists($1))
          Parser::error(@1, "The symbol '" + $1 + "` has not been defined.");
        $$ = AST::Make<AST::Assignment>(@1,
              $1, AST::Make<AST::Expr>(@1, "+", $4, AST::Make<AST::Identifier>(@1, $1)));
      }
    ;

s_expr
    : s_expr PLUS s_expr { $$ = AST::Make<AST::Expr>(@1, "+", $1, $3); }
    | s_expr MINUS s_expr { $$ = AST::Make<AST::Expr>(@1, "-", $1, $3); }
    | s_expr STAR s_expr { $$ = AST::Make<AST::Expr>(@1, "*", $1, $3); }
    | s_expr SLASH s_expr { $$ = AST::Make<AST::Expr>(@1, "/", $1, $3); }
    | s_expr PECET s_expr { $$ = AST::Make<AST::Expr>(@1, "%", $1, $3); }
    | s_expr OR s_expr { $$ = AST::Make<AST::Expr>(@1, "||", $1, $3); }
    | s_expr AND s_expr { $$ = AST::Make<AST::Expr>(@1, "&&", $1, $3); }
    | NOT s_expr { $$ = AST::Make<AST::Expr>(@1, "!", $2); }
    | LPAREN s_expr RPAREN { $$ = $2; }
    | LPAREN s_expr RPAREN QES s_expr COL s_expr { $$ = AST::Make<AST::Expr>(@1, "$", $2, $5, $7); }
    | s_expr LT s_expr { $$ = AST::Make<AST::Expr>(@1, "<", $1, $3); }
    | s_expr GT s_expr { $$ = AST::Make<AST::Expr>(@1, ">", $1, $3); }
    | s_expr EQ s_expr { $$ = AST::Make<AST::Expr>(@1, "==", $1, $3); }
    | s_expr NE s_expr { $$ = AST::Make<AST::Expr>(@1, "!=", $1, $3); }
    | s_expr LE s_expr { $$ = AST::Make<AST::Expr>(@1, "<=", $1, $3); }
    | s_expr GE s_expr { $$ = AST::Make<AST::Expr>(@1, ">=", $1, $3); }
    | bool_literal { $$ = AST::Make<AST::Expr>(@1, $1); }
    | int_val { $$ = AST::Make<AST::Expr>(@1, $1); }
    | PIPE span_expr PIPE { $$ = AST::Make<AST::Expr>(@1, "sizeof", $2); }
    | span_expr s_index { $$ = AST::Make<AST::Expr>(@1, "dimof", $1, $2); }
    ;

span_expr
    : span_expr PLUS span_expr { $$ = AST::Make<AST::Expr>(@1, "+", $1, $3); }
    | span_expr MINUS span_expr { $$ = AST::Make<AST::Expr>(@1, "-", $1, $3); }
    | span_expr STAR ituple_val { $$ = AST::Make<AST::Expr>(@1, "*", $1, $3); }
    | span_expr SLASH ituple_val { $$ = AST::Make<AST::Expr>(@1, "/", $1, $3); }
    | span_expr PECET ituple_val { $$ = AST::Make<AST::Expr>(@1, "%", $1, $3); }
    | LPAREN span_expr RPAREN { $$ = $2; }
    | span_val { $$ = AST::Make<AST::Expr>(@1, $1); }
    ;

if_else
    : IF s_expr LBRACE statements RBRACE else_clause { $$ = AST::Make<AST::IfElse>(@1, $2, $4, $6);}
    ;

else_clause
    : ELSE LBRACE statements RBRACE { $$ = $3; }
    | /* empty */ { $$ = AST::Make<AST::MultiNodes>(loc); }
    ;

with_block
    : WITH withins LBRACE w_statements RBRACE {
        $$ = AST::Make<AST::WithBlock>(@1);
        $$->withins = $2;
        $$->statms = $4;
      }
    | WITH withins require_clause LBRACE w_statements RBRACE {
        $$ = AST::Make<AST::WithBlock>(@1);
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
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; /* do not allow empty within */

within
    : IDENTIFIER IN span_val {
        symtab.AddSymbol($1, MakeUnknownType()/*TODO*/);
        $$ = AST::Make<AST::WithIn>(@1, AST::Make<AST::Identifier>(@1, $1), $3);
      }
    | IDENTIFIER ASSIGN LBRACE with_matchers RBRACE IN span_val {
        symtab.AddSymbol($1, MakeUnknownType()/*TODO*/);
        $$ = AST::Make<AST::WithIn>(@1, AST::Make<AST::Identifier>(@1, $1), $7);
        $$->with_matchers = $4;
      }
    ;

require_clause
    : REQUIRE require_binds {
        /* $$ = AST::Make<AST::RequireClause>();
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
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; /* do not allow empty require_bind */

require_bind
    : IDENTIFIER BIND IDENTIFIER {
        if (!symtab.Exists($1))
          Parser::error(@1, "The symbol '" + $1 + "` has not been defined.");
        if (!symtab.Exists($3))
          Parser::error(@3, "The symbol '" + $1 + "` has not been defined.");

        $$ = AST::Make<AST::RequireBind>(@1, AST::Make<AST::Identifier>(@1, $1),
                                           AST::Make<AST::Identifier>(@3, $3));
      }
    ;

w_statements
    : /*Empty statement */ { $$ = AST::Make<AST::MultiNodes>(loc); }
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
        $$ = AST::Make<AST::ForeachBlock>(@1, $2, $4);
      }
    ;

iv_exprs
    : iv_exprs COMMA iv_expr  {
        $1->Append($3);
        $$ = $1;
      }
    | iv_expr {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; /* do not allow the empty ivs */

iv_expr
    : IDENTIFIER { $$ = AST::Make<AST::Identifier>(@1, $1); }
    ;

dma_statement
    : IDENTIFIER ASSIGN DMA dma_operation chunkat_expr TRANS storage {
        $$ = AST::Make<AST::DMA>(@3,
              $4,
              AST::Make<AST::Identifier>(@1, $1),
              $5,
              AST::Make<AST::Memory>(@7, $7));
      }
    | IDENTIFIER ASSIGN DMA dma_operation IDENTIFIER TRANS chunkat_expr {
        $$ = AST::Make<AST::DMA>(@3,
              $4,
              AST::Make<AST::Identifier>(@1, $1),
              AST::Make<AST::Identifier>(@5, $5),
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
        $$ = AST::Make<AST::ChunkAt>(@1,
              AST::Make<AST::Identifier>(@1,$1), $4);
      }
    ;

id_list
    : id_list COMMA IDENTIFIER {
        $1->Append(AST::Make<AST::Identifier>(@3, $3));
        $$ = $1;
      }
    | IDENTIFIER {
        $$ = AST::Make<AST::MultiNodes>(@1, ", ");
        $$->Append(AST::Make<AST::Identifier>(@1, $1));
      }
    ;

passables
    : /* Empty */ {
        $$ = AST::Make<AST::MultiNodes>(loc);
      }
    | passable {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    | passables COMMA passable {
        $1->Append($3);
        $$ = $1;
      }
    ;

passable
    : s_expr { $$ = $1; }
    | span_expr FNDATA { $$ = AST::Make<AST::Expr>(@1, ".data", $1); }
    ;

with_matchers /* TODO: this special case is pattern-match ids for with-block */
    : with_matchers COMMA IDENTIFIER {
        $1->Append(AST::Make<AST::Identifier>(@3, $3));
        symtab.AddSymbol($3, MakeIntegerType()); /* in withins, this values should be int only */
        $$ = $1;
      }
    | IDENTIFIER {
        $$ = AST::Make<AST::MultiNodes>(@1);
        symtab.AddSymbol($1, MakeIntegerType()); /* in withins, this values should be int only */
        $$->Append(AST::Make<AST::Identifier>(@1, $1));
      }
    ;

wait_statement
    : WAIT futures {
        $$ = AST::Make<AST::Wait>(@1, $2);
      }
    ;

futures
    : futures COMMA IDENTIFIER {
        $1->Append(AST::Make<AST::Identifier>(@3, $3));
        $$ = $1;
      }
    | IDENTIFIER {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append(AST::Make<AST::Identifier>(@1, $1));
      }
    ;

call_statement
    : CALL IDENTIFIER LPAREN passables RPAREN {
        $$ = AST::Make<AST::Call>(@1,
                AST::Make<AST::Identifier>(@2, $2), $4);
      }
    ;

%%


// Bison expects us to provide implementation - otherwise linker complains
void Parser::error(const location &loc , const std::string &message) {
  std::cerr << loc << ": ";
  if (shell_supports_colors())
      std::cerr << red;
  std::cerr << "Error: ";
  if (shell_supports_colors())
      std::cerr << reset;
  std::cerr << message << std::endl;
}
