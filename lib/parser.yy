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

// The flag is used to disambiguate syntax suger:
//
//   a {(0), (1), 3} represents { a(0), a(1), 3 }
//
// where:
//
//   {(0), (1), 3} represents {0, 1, 3}
//
static bool parsing_derivation_decl = false;
static bool ignore_fndata = false;

ptr<AST::MultiNodes> ConstructPBRecursively(size_t idx,
                                            const ptr<AST::MultiNodes>& ps,
                                            const ptr<AST::MultiNodes>& stmts,
                                            bool);
std::pair<ptr<AST::Identifier>, ptr<AST::MultiValues>> ElementMultiValues(const ptr<AST::Expr>&);
std::set<std::string> paraby_symbols;

inline ptr<AST::ChunkAt> ReformChunkAt(const ptr<AST::ChunkAt> &);

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
  QES     "?"
  TRANS   "=>"
  BIND    "<->"
  PIPE    "|"
  UBOUND  "#"
  UBPLUS  "#+"
  UBMINUS "#-"
  UBSTAR  "#*"
  UBSLASH "#/"
  UBPECET "#%"
  DONTCARE"_"
  CDIV    "cdiv"
  CHAIN   "after"
;

// instead of union, using c++17 variant for terminal and non-terminals
%define api.value.type variant

// terminals
%token END 0 "end of file"
%token <char> CHAR
%token <int> NUM
%token <float> FPVAL
%token <double> DFPVAL
%token <std::string> TRUE FALSE
%token <std::string> STRING
%token <std::string> HOST_CODE KERNEL_CODE
%token <std::string> IDENTIFIER ATTR_CO
// type related
%token <std::string> MDSPAN ITUPLE EVENT MUTABLE
%token <Choreo::Storage> LOCAL SHARED GLOBAL
%token <Choreo::BaseType> F32 F16 BF16 U16 S16 U8 S8 U32 S32 INT HALF8 HALF BFP16 FLOAT DOUBLE BOOL VOID
// builtin operations
%token <std::string> DMA COPY PAD TRANSPOSE NONE ASYNC FNSPAN FNDATA FNSPANAS CHUNKAT CHUNK AT WAIT CALL AUTO SELECT SWAP ROTATE SYNC CHUNKINBOUND ASSERT TRIGGER PRINT PRINTLN
// control related
%token <std::string> INTHDS IF ELSE PARA BY WITH IN FOREACH INCR RET WHERE WHILE

// non-terminals
%nterm <std::string> dma_operation builtin_print_func arith_operation spanid
%nterm <ptr<DMAConfig>> dma_config
%nterm <bool> bool_value sync_type optional_mutable
%nterm <int> integer_value index_or_none
%nterm <std::vector<size_t>> optional_array_dims
%nterm <Choreo::Storage> storage
%nterm <Choreo::BaseType> fundamental_type
%nterm <AST::ptr<AST::CppSourceCode>> host_code
%nterm <AST::ptr<AST::Memory>> storage_qual
%nterm <AST::ptr<AST::SpanAs>> span_as
%nterm <AST::ptr<AST::IntLiteral>> num_expr
%nterm <AST::ptr<AST::Node>> any_code foreach_block simple_val template_val int_or_id device_passable declaration statement assignment dma_stmt wait_stmt trigger_stmt call_stmt swap_stmt range_expr param_mdspan_val chunkat_or_storage_or_select returnable span_init_val
%nterm <AST::ptr<AST::MultiNodes>> statements declarations assignments withins parabys paraby where_binds where_clause multi_decls named_spanned_decls spanned_decls named_scalar_decls scalar_decls named_event_decls event_decls stmts_block
%nterm <AST::ptr<AST::MultiValues>> value_list g_value_list template_value_list param_mdspan_list range_exprs iv_list id_list with_matchers device_passables template_params idt_list ids_list subscriptions data_indices
%nterm <AST::ptr<AST::Expr>> s_expr g_expr template_value_expr mdspan_expr mdspan_operator mdspan_val_expr ids_expr bound_expr subscript_like_expr dataid_expr
%nterm <AST::ptr<AST::DataType>> scalar_type void_type auto_type param_type return_type mdspan_as_type
%nterm <AST::ptr<AST::DataAccess>> data_element
%nterm <AST::ptr<AST::ParamList>> parameter_list
%nterm <AST::ptr<AST::Parameter>> parameter
%nterm <AST::ptr<AST::ChoreoFunction>> dsl_function
%nterm <AST::ptr<AST::MultiDimSpans>> unamed_mdspan_decl mdspan_list mdspan_derivation param_mdspan
%nterm <AST::ptr<AST::NamedTypeDecl>> named_mdspan_decl
%nterm <AST::ptr<AST::NamedVariableDecl>> named_ituple_decl spanned_decl scalar_decl event_decl
%nterm <AST::ptr<AST::IntTuple>> ituple_list
%nterm <AST::ptr<AST::WithBlock>> within_block
%nterm <AST::ptr<AST::InThreadsBlock>> inthreads_block
%nterm <AST::ptr<AST::WhileBlock>> while_block
%nterm <AST::ptr<AST::IfElseBlock>> if_else_block
%nterm <AST::ptr<AST::WithIn>> within
%nterm <AST::ptr<AST::WhereBind>> where_bind
%nterm <AST::ptr<AST::ParallelBy>> paraby_block
%nterm <AST::ptr<AST::Return>> return_stmt
%nterm <AST::ptr<AST::Synchronize>> sync_stmt
%nterm <AST::ptr<AST::ChunkAt>> chunkat_expr subdata_expr
%nterm <AST::ptr<AST::Select>> select_expr

// precedence (low to high) and associativity
%right ASSIGN
%right QES COL
%left OR
%left GT LT
%left AND
%right NOT
%nonassoc LE GE EQ NE
%left PLUS MINUS
%left STAR SLASH PECET
%left UBMINUS UBPLUS
%left UBSTAR UBSLASH UBPECET
%right PPLUS MMINUS
%left UBOUND
%nonassoc LPAREN RPAREN
%nonassoc LBRAKT RBRAKT
%left FNSPAN

%left HOST_CODE

%%

program
    : /* Empty */ {}
    | program any_code { if ($2 != nullptr) root.nodes.push_back($2); }
    ;

any_code
    : host_code { $$ = $1; }
    | dsl_function { $$ = $1; }
    | KERNEL_CODE {
        $$ = AST::Make<AST::CppSourceCode>(@1, $1, false);
      }
    ;

host_code
    : HOST_CODE /* can not be empty */ {
        $$ = AST::Make<AST::CppSourceCode>(@1, $1);
      }
    | host_code HOST_CODE {
        $1->code += $2;
        $$ = $1;
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

optional_mutable
    : /* empty */ { $$ = false; }
    | MUTABLE { $$ = true; }
    ;

param_mdspan_val
    : QES   { $$ = AST::Make<AST::IntLiteral>(@1); }
    | num_expr { $$ = $1; }
    | IDENTIFIER {
        $$ = AST::Make<AST::Identifier>(@1, $1);
        if (!symtab.Exists($1)) // allows same dim name
          symtab.AddSymbol($1, MakeIntegerType());
      }
    ;

num_expr
    : NUM   { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | num_expr PLUS num_expr { $$ = AST::Make<AST::IntLiteral>(@1, $1->value + $3->value); }
    | num_expr MINUS num_expr { $$ = AST::Make<AST::IntLiteral>(@1, $1->value - $3->value); }
    | num_expr STAR num_expr { $$ = AST::Make<AST::IntLiteral>(@1, $1->value * $3->value); }
    | num_expr SLASH num_expr { $$ = AST::Make<AST::IntLiteral>(@1, $1->value / $3->value); }
    | num_expr PECET num_expr { $$ = AST::Make<AST::IntLiteral>(@1, $1->value % $3->value); }
    | LPAREN num_expr RPAREN { $$ = AST::Make<AST::IntLiteral>(@1, $2->value);}

void_type
    : VOID  { $$ = AST::Make<AST::DataType>(@1, $1); }
    ;

auto_type
    : AUTO  { $$ = AST::Make<AST::DataType>(@1, BaseType::UNKNOWN); }
    ;

scalar_type
    : INT    { $$ = AST::Make<AST::DataType>(@1, $1); }
    | HALF8  { $$ = AST::Make<AST::DataType>(@1, $1); }
    | HALF   { $$ = AST::Make<AST::DataType>(@1, $1); }
    | BFP16  { $$ = AST::Make<AST::DataType>(@1, $1); }
    | FLOAT  { $$ = AST::Make<AST::DataType>(@1, $1); }
    | DOUBLE { $$ = AST::Make<AST::DataType>(@1, $1); }
    | BOOL   { $$ = AST::Make<AST::DataType>(@1, $1); }
    ;

mdspan_as_type
    : fundamental_type LBRAKT g_value_list RBRAKT {
        // if it contains a single mdspan, use it directly
        bool direct_mdspan = false;
        ptr<AST::MultiDimSpans> mds = nullptr;
        if ($3->Count() == 1)
          if (auto e = dyn_cast<AST::Expr>($3->ValueAt(0)))
            if (auto mdss = dyn_cast<AST::MultiDimSpans>(e->GetReference()))
              mds = mdss;

        if (!mds) mds = AST::Make<AST::MultiDimSpans>(@3, "", $3);
        $$ = AST::Make<AST::DataType>(@1, $1, mds);
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

simple_val
    : integer_value { $$ = AST::Make<AST::IntLiteral>(@1, $1); }
    | bool_value { $$ = AST::Make<AST::Boolean>(@1, $1); }
    | FPVAL  { $$ = AST::Make<AST::FloatLiteral>(@1, $1); }
    | DFPVAL { $$ = AST::Make<AST::FloatLiteral>(@1, $1); }
    | STRING { $$ = AST::Make<AST::StringLiteral>(@1, $1); }
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

idt_list /* list of integers or ids */
    : idt_list COMMA int_or_id {
        $1->Append($3);
        $$ = $1;
      }
    | int_or_id {
        $$ = AST::Make<AST::MultiValues>(@1, ", ");
        $$->Append($1);
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
    | wait_stmt    SEMCOL        { $$ = $1; }
    | trigger_stmt SEMCOL        { $$ = $1; }
    | call_stmt    SEMCOL        { $$ = $1; }
    | swap_stmt    SEMCOL        { $$ = $1; }
    | return_stmt  SEMCOL        { $$ = $1; }
    | sync_stmt    SEMCOL        { $$ = $1; }
    | paraby_block               { $$ = $1; }
    | within_block               { $$ = $1; }
    | inthreads_block            { $$ = $1; }
    | while_block                { $$ = $1; }
    | if_else_block              { $$ = $1; }
    | foreach_block              { $$ = $1; }
    ;

sync_stmt
    : SYNC DOT storage {
        $$ = AST::Make<AST::Synchronize>(@1, AST::Make<AST::Memory>(@3, $3));
      }
    ;

return_stmt
    : RET          { $$ = AST::Make<AST::Return>(@1);}
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
        $$ = AST::Make<AST::ParallelBy>(@1, cast<AST::MultiNodes>($4->AllSubs()[0]), $5, $2);
        if ($4->Count() > 1)
          $$->stmts = ConstructPBRecursively(1, $4, $5, $2);
      }
    ;

parabys
    : parabys COMMA paraby {
        $1->Append($3);
        $$ = $1;
      }
    | paraby {
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($1);
      }
    ; /* do not allow empty paraby */

paraby
    : BY NUM {
        $$ = AST::Make<AST::MultiNodes>(@2);
        $$->Append(AST::Make<AST::Identifier>(@1, SymbolTable::GetAnonName()));
        $$->Append(AST::Make<AST::IntLiteral>(@2, $2));
      }
    | IDENTIFIER BY int_or_id {
        if (paraby_symbols.find($1) != paraby_symbols.end())
          Parser::error(@1, "The symbol '" + $1 + "' has been used in the same parallelby block.");
        paraby_symbols.insert($1);
        symtab.AddSymbol($1, MakeUnknownType());
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append(AST::Make<AST::Identifier>(@1, $1));
        $$->Append($3);
      }
    | IDENTIFIER ASSIGN LBRACE id_list RBRACE BY LBRAKT idt_list RBRAKT {
        if (paraby_symbols.find($1) != paraby_symbols.end())
          Parser::error(@1, "The symbol '" + $1 + "' has been used in the same parallelby block.");
        paraby_symbols.insert($1);
        symtab.AddSymbol($1, MakeUnknownType());
        if ($4->Count() != $8->Count())
          Parser::error(@4, "The number of arguments in parallel bound config "
                        "should be consistent.");
        for (auto id : $4->AllValues()) {
          auto name = cast<AST::Identifier>(id)->name;
          if (paraby_symbols.find(name) != paraby_symbols.end())
            Parser::error(@1, "The symbol '" + name + "' has been used in the same parallelby block.");
          paraby_symbols.insert(name);
          symtab.AddSymbol(name, MakeUnknownType());
        }
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append(AST::Make<AST::Identifier>(@1, $1));
        $$->Append($4);
        $$->Append($8);
      }
    | LBRACE id_list RBRACE BY LBRAKT idt_list RBRAKT {
        if ($2->Count() != $6->Count())
          Parser::error(@2, "The number of arguments in parallel bound config "
                        "should be consistent.");
        for (auto id : $2->AllValues()) {
          auto name = cast<AST::Identifier>(id)->name;
          if (paraby_symbols.find(name) != paraby_symbols.end())
            Parser::error(@1, "The symbol '" + name + "' has been used in the same parallelby block.");
          paraby_symbols.insert(name);
          symtab.AddSymbol(name, MakeUnknownType());
        }
        $$ = AST::Make<AST::MultiNodes>(@1);
        $$->Append($2);
        $$->Append($6);
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
    : optional_mutable scalar_type scalar_decls {
        assert($2->isScalar() && "Not a scalar type.");
        $2->SetMutable($1);
        $2->ReGenSemaType();
        for (auto sub : $3->AllSubs()) {
          auto decl = cast<AST::NamedVariableDecl>(sub);
          decl->type = $2;
          decl->SetMutable($1);
          symtab.AddSymbol(decl->name_str, $2->GetType());
          // override the data type
        }
        $$ = $3;
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
    | IDENTIFIER ASSIGN LBRAKT {
        Parser::error(@3, "must use '{' and '}' to define an ituple.");
        YYERROR;
      }
    ;

named_event_decls
    : storage_qual EVENT event_decls {
        for (auto sub : $3->AllSubs()) {
          auto decl = cast<AST::NamedVariableDecl>(sub);
          auto sym_name = decl->name_str;
          symtab.AddSymbol(sym_name, MakeEventType($1->Get()));
          // override the data type
          decl->mem = $1;
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
    : storage_qual mdspan_as_type spanned_decls {
        for (auto item : $3->AllSubs()) {
          auto decl = cast<AST::NamedVariableDecl>(item);
          symtab.AddSymbol(decl->name_str, $2->GetType());
          decl->type = $2;
          decl->mem = $1;
        }
        $3->SetLOC(@1);
        $$ = $3;
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
    : LBRACE NUM RBRACE {
        $$ = AST::Make<AST::IntLiteral>(@2, $2);
      }
    | LBRACE MINUS NUM RBRACE {
        $$ = AST::Make<AST::IntLiteral>(@2, -$3);
      }
    | LBRACE FPVAL RBRACE {
        $$ = AST::Make<AST::FloatLiteral>(@2, $2);
      }
    | LBRACE MINUS FPVAL RBRACE {
        $$ = AST::Make<AST::FloatLiteral>(@2, -$3);
      }
    | LBRACE DFPVAL RBRACE {
        $$ = AST::Make<AST::FloatLiteral>(@2, $2);
      }
    | LBRACE MINUS DFPVAL RBRACE {
        $$ = AST::Make<AST::FloatLiteral>(@2, -$3);
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
    | spanid { $$ = AST::Make<AST::Identifier>(@1, $1); }
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

storage
    : LOCAL   { $$ = $1; }
    | SHARED  { $$ = $1; }
    | GLOBAL  { $$ = $1; }
    ;

storage_qual
    : /* Empty */ { $$ = AST::Make<AST::Memory>(loc); }
    | storage { $$ = AST::Make<AST::Memory>(@1, $1); }
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
        // note: the symbol is not scoped. therefore, an assignment could result in initialization
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
    | IDENTIFIER arith_operation ASSIGN s_expr {
        if (!symtab.Exists($1)) {
          Parser::error(@1, "The symbol '" + $1 + "` has not been defined.");
        } else {
          auto da = AST::Make<AST::DataAccess>(@1, AST::Make<AST::Identifier>(@1, $1));
          $$ = AST::Make<AST::Assignment>(@3,
              da, AST::Make<AST::Expr>(@2, $2, AST::Make<AST::Expr>(@1, da), $4));
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
    | data_element arith_operation ASSIGN s_expr {
        if (!symtab.Exists($1->GetDataName())) {
          Parser::error(@1, "The symbol '" + $1->GetDataName() + "` has not been defined.");
        } else {
          $$ = AST::Make<AST::Assignment>(@3, $1,
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
    | s_expr AND s_expr { $$ = AST::Make<AST::Expr>(@1, "&&", $1, $3); }
    | s_expr UBOUND s_expr {$$ = AST::Make<AST::Expr>(@1, "#", $1, $3); }
    | s_expr UBPLUS s_expr { $$ = AST::Make<AST::Expr>(@1, "#+", $1, $3); }
    | s_expr UBMINUS s_expr { $$ = AST::Make<AST::Expr>(@1, "#-", $1, $3); }
    | s_expr UBSTAR s_expr { $$ = AST::Make<AST::Expr>(@1, "#*", $1, $3); }
    | s_expr UBSLASH s_expr { $$ = AST::Make<AST::Expr>(@1, "#/", $1, $3); }
    | s_expr UBPECET s_expr { $$ = AST::Make<AST::Expr>(@1, "#%", $1, $3); }
    | NOT s_expr { $$ = AST::Make<AST::Expr>(@1, "!", $2); }
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
    | PIPE s_expr PIPE { $$ = AST::Make<AST::Expr>(@1, "sizeof", $2); }
    | s_expr LPAREN int_or_id RPAREN {
        $$ = AST::Make<AST::Expr>(@1, "dimof", $1, AST::Make<AST::IntIndex>(@3, $3));
      }
    | UBOUND IDENTIFIER {
        $$ = AST::Make<AST::Expr>(@1, "ubound", AST::Make<AST::Identifier>(@2, $2));
      }
    | PPLUS IDENTIFIER {
        $$ = AST::Make<AST::Expr>(@1, "++", AST::Make<AST::Identifier>(@1, $2));
      }
    | MMINUS IDENTIFIER {
        $$ = AST::Make<AST::Expr>(@1, "--", AST::Make<AST::Identifier>(@1, $2));
      }
    | data_element { $$ = AST::Make<AST::Expr>(@1, $1); }
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

unamed_mdspan_decl
    : mdspan_list       { $$ = $1; }
    | mdspan_derivation { $$ = $1; }
    ;

mdspan_val_expr
    : unamed_mdspan_decl { $$ = AST::Make<AST::Expr>(@1, $1); }
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
    : IF LPAREN s_expr RPAREN stmts_block {
        $$ = AST::Make<AST::IfElseBlock>(@1, $3, $5, nullptr);
      }
    | IF LPAREN call_stmt RPAREN stmts_block {
        $$ = AST::Make<AST::IfElseBlock>(@1, $3, $5, nullptr);
      }
      /* TODO: handle if-else */
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

dma_stmt
    : IDENTIFIER ASSIGN DMA dma_operation sync_type dma_config chunkat_expr TRANS chunkat_or_storage_or_select {
        symtab.AddSymbol($1, MakeDummyFutureType($5));
        $$ = AST::Make<AST::DMA>(@3, $4, $1, $7, $9, $5, $6);
      }
    | IDENTIFIER ASSIGN DMA dma_operation sync_type dma_config chunkat_expr TRANS chunkat_or_storage_or_select CHAIN IDENTIFIER {
        symtab.AddSymbol($1, MakeDummyFutureType($5));
        $$ = AST::Make<AST::DMA>(@3, $4, $1, $11, $7, $9, $5, $6);
      }
    | DMA dma_operation sync_type dma_config chunkat_expr TRANS chunkat_or_storage_or_select {
        $$ = AST::Make<AST::DMA>(@1, $2, "", $5, $7, $3, $4);
      }
    | DMA dma_operation sync_type dma_config chunkat_expr TRANS chunkat_or_storage_or_select CHAIN IDENTIFIER {
        $$ = AST::Make<AST::DMA>(@1, $2, "", $9, $5, $7, $3, $4);
      }
    | IDENTIFIER ASSIGN DMA NONE {
        symtab.AddSymbol($1, MakePlaceHolderFutureType());
        $$ = AST::Make<AST::DMA>(@1, $1);
      }
    ;

dma_operation
    : COPY      { $$ = $1; }
    | PAD       { $$ = $1; }
    | TRANSPOSE { $$ = $1; }
    ;

dma_config
    : LT LBRACE iv_list RBRACE COMMA LBRACE iv_list RBRACE COMMA LBRACE iv_list RBRACE COMMA NUM GT {
        auto pc = AST::Make<PadConfig>();
        for (auto high : $3->values)
          pc->pad_high.push_back(cast<AST::IntLiteral>(high)->Val());
        for (auto low : $7->values)
          pc->pad_low.push_back(cast<AST::IntLiteral>(low)->Val());
        for (auto mid : $11->values)
          pc->pad_mid.push_back(cast<AST::IntLiteral>(mid)->Val());
        pc->SetPadValue($14);
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

sync_type
    : { $$ = false; }
    | ASYNC { $$ = true; }
    ;

chunkat_or_storage_or_select
    : { ignore_fndata = true; } chunkat_expr { $$ = $2; ignore_fndata = false; }
    | storage      { $$ = AST::Make<AST::Memory>(@1, $1); }
    | select_expr  { $$ = $1; }
    ;

span_as
    : FNSPANAS LPAREN LBRAKT g_value_list RBRAKT RPAREN {
        $$ = AST::Make<AST::SpanAs>(@1, nullptr/*fill later*/, $4);
        parsing_derivation_decl = false;
      }
    ;

chunkat_expr
    : subdata_expr { $$ = $1; }
    | ids_expr {
        auto ide = ElementMultiValues($1);
        $$ = ReformChunkAt(
          AST::Make<AST::ChunkAt>(@1, ide.first, ide.second));
      }
    | ids_expr span_as {
        auto ide = ElementMultiValues($1);
        $2->id = ide.first;
        $$ = ReformChunkAt(AST::Make<AST::ChunkAt>(@1, $2, ide.second));
      }
    ;

subdata_expr
    : ids_expr CHUNKAT LPAREN value_list RPAREN {
        $4->SetDelimiter(", ");
        auto ide = ElementMultiValues($1);
        $$ = ReformChunkAt(AST::Make<AST::ChunkAt>(@1, ide.first, ide.second, $4));
      }
    | ids_expr CHUNK LPAREN value_list RPAREN AT LPAREN value_list RPAREN {
        $4->SetDelimiter(", ");
        $8->SetDelimiter(", ");
        auto ide = ElementMultiValues($1);
        $$ = ReformChunkAt(AST::Make<AST::ChunkAt>(@1, ide.first, ide.second, $8, $4));
      }
    | ids_expr span_as CHUNKAT LPAREN value_list RPAREN {
        // note: normalize will hoist span_as
        $5->SetDelimiter(", ");
        auto ide = ElementMultiValues($1);
        $2->id = ide.first;
        $$ = ReformChunkAt(AST::Make<AST::ChunkAt>(@1, $2, ide.second, $5));
      }
    ;

data_element
    : IDENTIFIER AT LPAREN data_indices RPAREN {
        $$ = AST::Make<AST::DataAccess>(@1, AST::Make<AST::Identifier>(@1, $1), $4);
      }
    | IDENTIFIER FNDATA AT LPAREN data_indices RPAREN {
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
    | IDENTIFIER LBRACE g_value_list RBRACE {
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

builtin_print_func
    : PRINT   { $$ = $1; }
    | PRINTLN { $$ = $1; }
    ;

call_stmt
    : CALL IDENTIFIER LPAREN device_passables RPAREN {
        $$ = AST::Make<AST::Call>(@1,
                AST::Make<AST::Identifier>(@2, $2), $4);
      }
    | CALL IDENTIFIER template_params LPAREN device_passables RPAREN {
        $$ = AST::Make<AST::Call>(@1,
                AST::Make<AST::Identifier>(@2, $2), $5, $3);
      }
    | ASSERT LPAREN s_expr COMMA STRING RPAREN {
        auto mv = AST::Make<AST::MultiValues>(@1, ", ");
        mv->Append($3);
        mv->Append(AST::Make<AST::StringLiteral>(@5, $5));
        $$ = AST::Make<AST::Call>(@1, AST::Make<AST::Identifier>(@1, $1), mv, true);
      }
    | builtin_print_func LPAREN value_list RPAREN {
        $$ = AST::Make<AST::Call>(@1, AST::Make<AST::Identifier>(@1, $1), $3, true);
      }
    ;

swap_stmt
    : SWAP LPAREN id_list RPAREN {
        if ($3->Count() != 2)
          Parser::error(@3, "Builtin function 'swap' accept exact two parameters.");
        $$ = AST::Make<AST::Rotate>(@1, $3);
      }
    | ROTATE LPAREN id_list RPAREN {
        if ($3->Count() < 2)
          Parser::error(@3, "Builtin function 'rotate' accept two or more parameters.");
        $$ = AST::Make<AST::Rotate>(@1, $3);
      }
    ;

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
        choreo_unreachable("unable to handle this expr: " + PSTR(e));
      mv->Insert(e->GetR(), 0);
      if (id = dyn_cast<AST::Identifier>(e->GetL()))
        break;
      else
        we = cast<AST::Expr>(e->GetL());
    }
  }
  return std::make_pair(id, mv);
}

ptr<AST::MultiNodes> ConstructPBRecursively(size_t idx,
                                            const ptr<AST::MultiNodes>& ps,
                                            const ptr<AST::MultiNodes>& stmts,
                                            bool async) {
  auto pb = AST::Make<AST::ParallelBy>(ps->LOC(),
                                       cast<AST::MultiNodes>(ps->AllSubs()[idx]),
                                       stmts, async);
  if (idx < ps->Count() - 1)
    pb->stmts = ConstructPBRecursively(idx + 1, ps, stmts, async);
  auto mn = AST::Make<AST::MultiNodes>(ps->LOC());
  mn->Append(pb);
  return mn;
}

inline ptr<AST::ChunkAt> ReformChunkAt(const ptr<AST::ChunkAt> &ca) {
  // normalize "_" list
  if (!ca->positions) return ca;

  bool not_tiled = true;
  for (auto pos : ca->positions->values) {
    if (auto bpv = AST::GetIdentifier(*pos)) {
      if (bpv->name == "_") {
        bpv->name = "__choreo_no_tiling__";
        continue;
      }
    }
    not_tiled = false;
  }

  if (not_tiled)
    ca->positions = nullptr;

  return ca;
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
