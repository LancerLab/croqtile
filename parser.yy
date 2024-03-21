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
extern AST::ITupleTable ituple_symtab;

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
%nterm <AST::ptr<AST::Node>> pass_by foreach_block simple_val span_val ituple_val int_val bool_literal passable declaration statement assignment pb_statement w_statement dma_statement wait_statement call_statement span_elem mixed_span_elem iv_expr if_else optional_scalar_init
%nterm <AST::ptr<AST::MultiNodes>> statements declarations assignments pb_statements w_statements span_list mixed_span_list withins iv_exprs require_binds require_clause id_list with_matchers else_clause futures passables
%nterm <AST::ptr<AST::IntList>> int_list
%nterm <AST::ptr<AST::SValList>> sval_list
%nterm <AST::ptr<AST::Expr>> expr span_expr
%nterm <AST::ptr<AST::DataType>> scalar_type param_type spanned_type
%nterm <AST::ptr<AST::ParamList>> parameter_list
%nterm <AST::ptr<AST::ParamType>> parameter
%nterm <AST::ptr<AST::ChoreoFunction>> dsl_function
%nterm <AST::ptr<AST::MultiDimSpans>> unnamed_span_decl
%nterm <AST::ptr<AST::NamedTypeDecl>> named_span_decl
%nterm <AST::ptr<AST::NamedVariableDecl>> named_tuple_decl named_scalar_decl named_spanned_decl
%nterm <AST::ptr<AST::IntTuple>> unnamed_tuple_decl
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
        $$ = std::make_shared<AST::CppSourceCode>(@1, $1);
      }
    ;

dsl_function
    : ATTR_CO param_type IDENTIFIER LPAREN parameter_list RPAREN LBRACE statements RBRACE {
        $$ = std::make_shared<AST::ChoreoFunction>(@1);
        $$->name = $3;
        $$->f_decl.name = $3;
        $$->f_decl.ret_type = $2;
        $$->f_decl.params = $5;
        $$->statms = $8;
      }
    ;

param_type
    : scalar_type { $$ = $1; }
    | fundamental_type MDSPAN LT NUM GT {
        $$ = std::make_shared<AST::DataType>(@1, $1,
                std::make_shared<AST::MultiDimSpans>(@2, PartialTypeTable::getAnonName(),
                $4));
      }
    ;

scalar_type
    : INT   { $$ = std::make_shared<AST::DataType>(@1, $1); }
    | BOOL  { $$ = std::make_shared<AST::DataType>(@1, $1); }
    ;

spanned_type
    : fundamental_type unnamed_span_decl {
        $$ = std::make_shared<AST::DataType>(@1, $1, $2);
      }
    | fundamental_type LBRAKT span_expr RBRAKT {
        $$ = std::make_shared<AST::DataType>(@1, $1, $3);
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

int_list
    : /* allows the empty list */ {
        $$ = std::make_shared<AST::IntList>({});
      }
    | int_list COMMA NUM {
        $1->Append(std::make_shared<AST::IntLiteral>(@3, $3));
        $$ = $1;
      }
    | NUM {
        $$ = std::make_shared<AST::IntList>(@1);
        $$->values.push_back(std::make_shared<AST::IntLiteral>(@1, $1));
      }
    ;

sval_list
    : /* allows the empty list */ {
        $$ = std::make_shared<AST::SValList>(loc);
      }
    | sval_list COMMA simple_val {
        $1->Append($3);
        $$ = $1;
      }
    | simple_val {
        $$ = std::make_shared<AST::SValList>(@1);
        $$->Append($1);
      }
    ;

simple_val
    : NUM { $$ = std::make_shared<AST::IntLiteral>(@1, $1); }
    | IDENTIFIER {
        if (!symtab.Exists($1))
          Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

#if 0
        if (symtab.GetSymbol($1)->IsComposite())
          Parser::error(@1, "expecting symbol `" + $1 +
                                "' of a scalar type.");
#endif

        if (symtab.GetSymbol($1)->GetTypeCategory() != TypeCategory::INT)
          Parser::error(@1, "expecting symbol `" + $1 +
                                "' of an integer type.");

        $$ = std::make_shared<AST::Identifier>(@1, $1);
    	}
    ;

int_val
    : IDENTIFIER s_index {
        if (!symtab.Exists($1)) {
          Parser::error(@1, ": the symbol '" + $1 + "` is not defined.");
          exit(1);
        }

        $$ = std::make_shared<AST::NthBound>(@1, std::make_shared<AST::Identifier>(@1, $1), $2);
      }
    | simple_val { $$ = $1; }
    ;

bool_literal
    : TRUE { $$ = std::make_shared<AST::Boolean>(@1, std::string("true")); }
    | FALSE { $$ = std::make_shared<AST::Boolean>(@1, std::string("false")); }
    ;

parameter_list
    : /* Empty */ {
        $$ = std::make_shared<AST::ParamList>(loc);
      }
    | parameter_list COMMA parameter {
        $1->values.push_back($3);
        $$ = $1;
      }
    | parameter {
        $$ = std::make_shared<AST::ParamList>(@1);
        $$->values.push_back($1);
      }
    ;

parameter
    : param_type IDENTIFIER { /* handle parameter type and name here */
        $$ = std::make_shared<AST::ParamType>(std::pair($1, std::make_shared<AST::Identifier>(@2, $2)));
        if (symtab.Exists($2)) {
          Parser::error(@2, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.AddSymbol($2, $1->MakeSemaType());
      }
    | param_type {
        $$ = std::make_shared<AST::ParamType>(
              std::pair($1, std::make_shared<AST::Identifier>(@1, AST::SymbolTable::getAnonName())));
      }
    ;

statements
    : /* no statement */ { $$ = std::make_shared<AST::MultiNodes>(loc); }
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
    : RET          { $$ = std::make_shared<AST::Return>(@1);}
    | RET passable { $$ = std::make_shared<AST::Return>(@1, $2); }
    ;

para_by
    : PARA IDENTIFIER BY NUM LBRACE pb_statements RBRACE {
        $$ = std::make_shared<AST::ParallelBy>(@1, $2, $4);
        $$->statms = $6;
      }
    ;

pb_statements
    : /* Empty */ { $$ = std::make_shared<AST::MultiNodes>(loc); }
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
        $$ = std::make_shared<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ;

declarations
    : declarations COMMA declaration {
        $1->Append($3);
        $$ = $1;
      }
    | declaration {
        $$ = std::make_shared<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ;

declaration
    : named_span_decl    { $$ = $1; }
    | named_tuple_decl   { $$ = $1; }
    | named_scalar_decl  { $$ = $1; }
    | named_spanned_decl { $$ = $1; }
    ;

named_scalar_decl
    : scalar_type IDENTIFIER optional_scalar_init {
        if (symtab.Exists($2)) {
          Parser::error(@2, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        assert($1->isScalar() && "Not a scalar type.");
        symtab.AddSymbol($2, $1->MakeSemaType());
        if (!$3)
          $$ = std::make_shared<AST::NamedVariableDecl>(@2, $2, $1);
        else
          $$ = std::make_shared<AST::NamedVariableDecl>(@2, $2, $1, nullptr, $3);
      }
    ;

optional_scalar_init
    : /*Empty */        { $$ = nullptr; }
    | ASSIGN expr       { $$ = $2; }
    ;

named_spanned_decl
    : storage_qual spanned_type IDENTIFIER {
        if (symtab.Exists($3)) {
          Parser::error(@3, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.AddSymbol($3, $2->MakeSemaType());
        $$ = std::make_shared<AST::NamedVariableDecl>(@3, $3, $2, $1);
      }
    ;

s_index
    : LPAREN simple_val RPAREN {
        $$ = std::make_shared<AST::IntIndex>(@2, $2);
      }
    ;

s_index_list
    : s_index_list COMMA s_index {
        $1->Append($3);
        $$ = $1;
      }
    | s_index {
        $$ = std::make_shared<AST::IntIndexList>(@1);
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
        $$ = std::make_shared<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; // do not allow an empty list

mixed_span_elem
    : IDENTIFIER s_index {
        if (!symtab.Exists($1))
          Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

        if (!symtab.GetSymbol($1)->IsComposite())
          Parser::error(@1, "expecting a symbol of composite type.");

        $$ = std::make_shared<AST::NthBound>(@1,
            std::make_shared<AST::Identifier>(@1, $1), $2);
      }
    | IDENTIFIER FNSPAN s_index {
        if (!symtab.Exists($1))
          Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

        if (!symtab.GetSymbol($1)->IsComposite())
          Parser::error(@1, "expecting a symbol of composite type.");

        $$ = std::make_shared<AST::NthBound>(@1,
            std::make_shared<AST::Identifier>(@1, $1+$2), $3);

      }
    | simple_val { $$ = $1; }
    ; // do not allow an empty item

mixed_span_list
    : /* Empty list */ {
        $$ = std::make_shared<AST::MultiNodes>(loc);
      }
    | mixed_span_list COMMA mixed_span_elem {
        $1->Append($3);
        $$ = $1;
      }
    | mixed_span_elem {
        $$ = std::make_shared<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ;

unnamed_span_decl
    : IDENTIFIER FNSPAN LBRAKT span_list RBRAKT {
        $$ = std::make_shared<AST::MultiDimSpans>(@1, $1, $4);
      }
    | IDENTIFIER LBRAKT span_list RBRAKT {
        $$ = std::make_shared<AST::MultiDimSpans>(@1, $1, $3);
      }
    | LBRAKT mixed_span_list RBRAKT {
        $$ = std::make_shared<AST::MultiDimSpans>(@1, "", $2);
      }
    ;

span_val
    : unnamed_span_decl { $$ = $1; }
    | IDENTIFIER { $$ = std::make_shared<AST::Identifier>(@1, $1); }
    | IDENTIFIER FNSPAN { $$ = std::make_shared<AST::Identifier>(@1, $1 + $2); }
    ;

named_span_decl
    : MDSPAN IDENTIFIER COL span_expr {
        if (symtab.Exists($2)) {
          Parser::error(@2, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.AddSymbol($2, MakeUninitMDSpanType());
        $$ = std::make_shared<AST::NamedTypeDecl>(@2, $2, $4);
      }
    | MDSPAN LT NUM GT IDENTIFIER COL span_expr {
        // TODO: check if the span defined aligned with declaration
        #if 0
        if ($3 != $7->list.size())
          Parser::error(@3,
            "The rank of mdspan is not consistent with its decleration.");
            #endif
        if (symtab.Exists($5)) {
          Parser::error(@5, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.AddSymbol($5, MakeDimSizedMDSpanType($3));
        $$ = std::make_shared<AST::NamedTypeDecl>(@5, $5, $7);
      }
    | IDENTIFIER COL span_expr {
        if (symtab.Exists($1)) {
          Parser::error(@1, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.AddSymbol($1, MakeUninitMDSpanType());
        $$ = std::make_shared<AST::NamedTypeDecl>(@1, $1, $3);
      }
    ;

unnamed_tuple_decl
    : LBRACE sval_list RBRACE {
        $$ = std::make_shared<AST::IntTuple>(@1, "", $2);
      }
    | IDENTIFIER LBRACE s_index_list RBRACE {
        // anchor
        if (!symtab.Exists($1))
          Parser::error(@1, "The symbol has not been defined.");

        $$ = std::make_shared<AST::IntTuple>(@1, $1, $3);
        #if 0
        auto src_ituple = ituple_symtab.GetSymbol($1);
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
        $$ = std::make_shared<AST::Identifier>(@1, $1);
      }
    ;

named_tuple_decl
    : ITUPLE IDENTIFIER ASSIGN ituple_val {
        /* TODO: workaround: use INT for ituple's base type use a dedicated type for ituple in symboltable */
        //ituple_symtab.addITupleSymbol($2, $4);
        symtab.AddSymbol($2, MakeUninitITupleType());
        $$ = std::make_shared<AST::NamedVariableDecl>(@2,
              $2, std::make_shared<AST::DataType>(@1, BaseType::ITUPLE), nullptr, $4);
      }
    | IDENTIFIER ASSIGN unnamed_tuple_decl {
        //ituple_symtab.addITupleSymbol($1, $3);
        symtab.AddSymbol($1, MakeUninitITupleType());
        $$ = std::make_shared<AST::NamedVariableDecl>(@1,
              $1, std::make_shared<AST::DataType>(@1, BaseType::ITUPLE), nullptr, $3);
      }
    ; // do not allow uninitialized ituple

storage
    : LOCAL   { $$ = $1; }
    | SHARED  { $$ = $1; }
    | GLOBAL  { $$ = $1; }
    ;

storage_qual
    : /* Empty */ { $$ = std::make_shared<AST::Memory>(loc); }
    | storage { $$ = std::make_shared<AST::Memory>(@1, $1); }
    ;

assignment
    : IDENTIFIER ASSIGN expr {
        if (!symtab.Exists($1)) {
          // since the symbol is not defined, it is a declaration without type annotation
          symtab.AddSymbol($1, MakeUnknownType());
          $$ = std::make_shared<AST::NamedVariableDecl>(@1,
                $1, std::make_shared<AST::DataType>(@1, BaseType::ITUPLE), nullptr, $3);
          break;
        }
        $$ = std::make_shared<AST::Assignment>(@2, $1, $3);
      }
    | IDENTIFIER PLUS ASSIGN expr {
        if (!symtab.Exists($1))
          Parser::error(@1, "The symbol has not been defined.");
        $$ = std::make_shared<AST::Assignment>(@1,
              $1, std::make_shared<AST::Expr>(@1, "+", $4, std::make_shared<AST::Identifier>(@1, $1)));
      }
    ;

expr
    : expr PLUS expr { $$ = std::make_shared<AST::Expr>(@1, "+", $1, $3); }
    | expr MINUS expr { $$ = std::make_shared<AST::Expr>(@1, "-", $1, $3); }
    | expr STAR expr { $$ = std::make_shared<AST::Expr>(@1, "*", $1, $3); }
    | expr SLASH expr { $$ = std::make_shared<AST::Expr>(@1, "/", $1, $3); }
    | expr PECET expr { $$ = std::make_shared<AST::Expr>(@1, "%", $1, $3); }
    | expr OR expr { $$ = std::make_shared<AST::Expr>(@1, "||", $1, $3); }
    | expr AND expr { $$ = std::make_shared<AST::Expr>(@1, "&&", $1, $3); }
    | NOT expr { $$ = std::make_shared<AST::Expr>(@1, "!", $2); }
    | LPAREN expr RPAREN { $$ = $2; }
    | LPAREN expr RPAREN QES expr COL expr { $$ = std::make_shared<AST::Expr>(@1, "$", $2, $5, $7); }
    | expr LT expr { $$ = std::make_shared<AST::Expr>(@1, "<", $1, $3); }
    | expr GT expr { $$ = std::make_shared<AST::Expr>(@1, ">", $1, $3); }
    | expr EQ expr { $$ = std::make_shared<AST::Expr>(@1, "==", $1, $3); }
    | expr NE expr { $$ = std::make_shared<AST::Expr>(@1, "!=", $1, $3); }
    | expr LE expr { $$ = std::make_shared<AST::Expr>(@1, "<=", $1, $3); }
    | expr GE expr { $$ = std::make_shared<AST::Expr>(@1, ">=", $1, $3); }
    | bool_literal { $$ = std::make_shared<AST::Expr>(@1, $1); }
    | int_val { $$ = std::make_shared<AST::Expr>(@1, $1); }
    | PIPE span_expr PIPE { $$ = std::make_shared<AST::Expr>(@1, "sizeof", $2); }
    ;

span_expr
    : span_expr PLUS span_expr { $$ = std::make_shared<AST::Expr>(@1, "+", $1, $3); }
    | span_expr MINUS span_expr { $$ = std::make_shared<AST::Expr>(@1, "-", $1, $3); }
    | span_expr STAR ituple_val { $$ = std::make_shared<AST::Expr>(@1, "*", $1, $3); }
    | span_expr SLASH ituple_val { $$ = std::make_shared<AST::Expr>(@1, "/", $1, $3); }
    | span_expr PECET ituple_val { $$ = std::make_shared<AST::Expr>(@1, "%", $1, $3); }
    | LPAREN span_expr RPAREN { $$ = $2; }
    | span_val { $$ = std::make_shared<AST::Expr>(@1, $1); }
    ;

if_else
    : IF expr LBRACE statements RBRACE else_clause { $$ = std::make_shared<AST::IfElse>(@1, $2, $4, $6);}
    ;

else_clause
    : ELSE LBRACE statements RBRACE { $$ = $3; }
    | /* empty */ { $$ = std::make_shared<AST::MultiNodes>(loc); }
    ;

with_block
    : WITH withins LBRACE w_statements RBRACE {
        $$ = std::make_shared<AST::WithBlock>(@1);
        $$->withins = $2;
        $$->statms = $4;
      }
    | WITH withins require_clause LBRACE w_statements RBRACE {
        $$ = std::make_shared<AST::WithBlock>(@1);
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
        $$ = std::make_shared<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; /* do not allow empty within */

within
    : IDENTIFIER IN span_val {
        symtab.AddSymbol($1, MakeUnknownType()/*TODO*/);
        $$ = std::make_shared<AST::WithIn>(@1, std::make_shared<AST::Identifier>(@1, $1), $3);
      }
    | IDENTIFIER ASSIGN LBRACE with_matchers RBRACE IN span_val {
        symtab.AddSymbol($1, MakeUnknownType()/*TODO*/);
        $$ = std::make_shared<AST::WithIn>(@1, std::make_shared<AST::Identifier>(@1, $1), $7);
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
        $$ = std::make_shared<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; /* do not allow empty require_bind */

require_bind
    : IDENTIFIER BIND IDENTIFIER {
        if (!symtab.Exists($1))
          Parser::error(@1, "The symbol has not been defined.");
        if (!symtab.Exists($3))
          Parser::error(@3, "The symbol has not been defined.");

        $$ = std::make_shared<AST::RequireBind>(@1, std::make_shared<AST::Identifier>(@1, $1),
                                           std::make_shared<AST::Identifier>(@3, $3));
      }
    ;

w_statements
    : /*Empty statement */ { $$ = std::make_shared<AST::MultiNodes>(loc); }
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
        $$ = std::make_shared<AST::ForeachBlock>(@1, $2, $4);
      }
    ;

iv_exprs
    : iv_exprs COMMA iv_expr  {
        $1->Append($3);
        $$ = $1;
      }
    | iv_expr {
        $$ = std::make_shared<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; /* do not allow the empty ivs */

iv_expr
    : IDENTIFIER { $$ = std::make_shared<AST::Identifier>(@1, $1); }
    ;

dma_statement
    : IDENTIFIER ASSIGN DMA dma_operation chunkat_expr TRANS storage {
        $$ = std::make_shared<AST::DMA>(@3,
              $4,
              std::make_shared<AST::Identifier>(@1, $1),
              $5,
              std::make_shared<AST::Memory>(@7, $7));
      }
    | IDENTIFIER ASSIGN DMA dma_operation IDENTIFIER TRANS chunkat_expr {
        $$ = std::make_shared<AST::DMA>(@3,
              $4,
              std::make_shared<AST::Identifier>(@1, $1),
              std::make_shared<AST::Identifier>(@5, $5),
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
        $$ = std::make_shared<AST::ChunkAt>(@1,
              std::make_shared<AST::Identifier>(@1,$1), $4);
      }
    ;

id_list
    : id_list COMMA IDENTIFIER {
        $1->Append(std::make_shared<AST::Identifier>(@3, $3));
        $$ = $1;
      }
    | IDENTIFIER {
        $$ = std::make_shared<AST::MultiNodes>(@1, ", ");
        $$->Append(std::make_shared<AST::Identifier>(@1, $1));
      }
    ;

passables
    : /* Empty */ {
        $$ = std::make_shared<AST::MultiNodes>(loc);
      }
    | passable {
        $$ = std::make_shared<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    | passables COMMA passable {
        $1->Append($3);
        $$ = $1;
      }
    ;

passable
    : expr { $$ = $1; }
    | span_expr FNDATA { $$ = std::make_shared<AST::Expr>(@1, ".data", $1); }
    ;

with_matchers /* TODO: this special case is pattern-match ids for with-block */
    : with_matchers COMMA IDENTIFIER {
        $1->Append(std::make_shared<AST::Identifier>(@3, $3));
        symtab.AddSymbol($3, MakeIntegerType()); /* in withins, this values should be int only */
        $$ = $1;
      }
    | IDENTIFIER {
        $$ = std::make_shared<AST::MultiNodes>(@1);
        symtab.AddSymbol($1, MakeIntegerType()); /* in withins, this values should be int only */
        $$->Append(std::make_shared<AST::Identifier>(@1, $1));
      }
    ;

wait_statement
    : WAIT futures {
        $$ = std::make_shared<AST::Wait>(@1, $2);
      }
    ;

futures
    : futures COMMA IDENTIFIER {
        $1->Append(std::make_shared<AST::Identifier>(@3, $3));
        $$ = $1;
      }
    | IDENTIFIER {
        $$ = std::make_shared<AST::MultiNodes>(@1);
        $$->Append(std::make_shared<AST::Identifier>(@1, $1));
      }
    ;

call_statement
    : CALL IDENTIFIER LPAREN passables RPAREN {
        $$ = std::make_shared<AST::Call>(@1,
                std::make_shared<AST::Identifier>(@2, $2), $4);
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
