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
#include <cstdlib>
#include <getopt.h>

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

AST::Program root;
AST::SymbolTable symtab;
AST::ITupleTable ituple_symtab(symtab);

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
%token <std::string> MDSPAN ITUPLE LOCAL SHARED GLOBAL
%token <AST::BaseType> F32 F16 BF16 U16 S16 U8 S8 U32 S32 INT
// builtin operations
%token <std::string> DMA COPY FNSPAN FNDATA
// control related
%token <std::string> IF ELSE PARA BY WITH IN ITER RET

// non-terminals
%nterm <AST::BaseType> base_type
%nterm <AST::ptr<AST::Node>> pass_by sval_ref para_by
%nterm <AST::ptr<AST::MultiNodes>> statements declarations assignments pb_statements with_statements iterate_statements span_list mixed_span_list
%nterm <AST::ptr<AST::NodeRef>> statement declaration pb_statement span_elem mixed_span_elem if_else for_each assignment
%nterm <AST::ptr<AST::IntList>> int_list
%nterm <AST::ptr<AST::SValList>> sval_list
%nterm <AST::ptr<AST::Expr>> term expr
%nterm <AST::ptr<AST::DataType>> general_type param_type aggregate_type
%nterm <AST::ptr<AST::ParamList>> parameter_list
%nterm <AST::ptr<AST::ParamType>> parameter
%nterm <AST::ptr<AST::ChoreoFunction>> dsl_function
%nterm <AST::ptr<AST::MultiSpans>> named_span_decl unnamed_span_decl
%nterm <AST::ptr<AST::IntTuple>> named_tuple_decl unnamed_tuple_decl tuple_assign
%nterm <AST::ptr<AST::IntVal>> scalar_decl
%nterm <AST::ptr<AST::IntIndex>> s_index
%nterm <AST::ptr<AST::IntIndexList>> s_index_list

%%

program
    : /* Empty */ {}
    | program pass_by      { root.nodes.push_back($2); }
    | program dsl_function { root.nodes.push_back($2); }
    | END {  }
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

        $$ = std::make_shared<AST::DataType>($1, AST::DataType::getSpanType($3));
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
    | sval_list COMMA sval_ref {
        $1->Append($3);
        $$ = $1;
      }
    | sval_ref {
        $$ = std::make_shared<AST::SValList>();
        $$->Append($1);
      }
    ;

sval_ref
    : NUM { $$ = std::make_shared<AST::IntLiteral>($1); }
    | IDENTIFIER {
        if (!symtab.exists($1))
          Choreo::Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

        if (symtab.getSymbol($1)->isAggregate())
          Choreo::Parser::error(@1, "expecting symbol `" + $1 +
                                "' of a scalar type.");

        if (symtab.getSymbol($1)->getType() != AST::BaseType::INT)
          Choreo::Parser::error(@1, "expecting symbol `" + $1 +
                                "' of an integer type.");

        $$ = std::make_shared<AST::Identifier>($1);
    	}
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
    : declarations SEMCOL { $$ = std::make_shared<AST::NodeRef>($1); }
    | assignments  SEMCOL { $$ = std::make_shared<AST::NodeRef>($1); }
    | para_by             { $$ = std::make_shared<AST::NodeRef>($1); }
    ;

para_by
    : PARA IDENTIFIER BY NUM LBRACE pb_statements RBRACE {
      $$ = std::make_shared<AST::ParallelBy>($2, $4);
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
    : declarations SEMCOL { $$ = std::make_shared<AST::NodeRef>($1); }
    | assignments  SEMCOL { $$ = std::make_shared<AST::NodeRef>($1); }
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

declaration
    : named_span_decl { $$ = std::make_shared<AST::NodeRef>($1); }
    | named_tuple_decl { $$ = std::make_shared<AST::NodeRef>($1);  }
    | scalar_decl { $$ = std::make_shared<AST::NodeRef>($1);  }
    ;

scalar_decl
    : INT IDENTIFIER ASSIGN expr {
        if (symtab.exists($2)) {
          Choreo::Parser::error(@2, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.addSymbol($2, $1);
        $$ = std::make_shared<AST::IntVal>($2, $4);
      }
    ;

s_index
    : LPAREN sval_ref RPAREN {
        $$ = std::make_shared<AST::IntIndex>($2);
      }
    ;

s_index_list
    : /* allows the empty list */ {
        $$ = std::make_shared<AST::IntIndexList>();
      }
    | s_index_list COMMA s_index {
        $1->Append($3);
        $$ = $1;
      }
    | s_index {
        $$ = std::make_shared<AST::IntIndexList>();
        $$->indices.push_back($1);
      }
    ;

span_elem
    : s_index { $$ = std::make_shared<AST::NodeRef>($1); }
    | sval_ref { $$ = std::make_shared<AST::NodeRef>($1); }
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

        auto nth_elem = std::make_shared<AST::NthBound>(
            std::make_shared<AST::Identifier>($1), $2);

        $$ = std::make_shared<AST::NodeRef>(nth_elem);
      }
    | IDENTIFIER FNSPAN s_index {
        if (!symtab.exists($1))
          Choreo::Parser::error(@1,
            "The symbol `" + $1 + "' has not been defined.");

        if (!symtab.getSymbol($1)->isAggregate())
          Choreo::Parser::error(@1, "expecting a symbol of aggregate type.");

        auto nth_elem = std::make_shared<AST::NthBound>(
            std::make_shared<AST::Identifier>($1+$2), $3);

        $$ = std::make_shared<AST::NodeRef>(nth_elem);
      }
    | sval_ref { $$ = std::make_shared<AST::NodeRef>($1); }
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
        $$ = std::make_shared<AST::MultiSpans>("", $1, $4);
      }
    | IDENTIFIER LBRAKT span_list RBRAKT {
        $$ = std::make_shared<AST::MultiSpans>("", $1, $3);
      }
    | LBRAKT mixed_span_list RBRAKT {
        $$ = std::make_shared<AST::MultiSpans>("", $2);
      }
    ;

named_span_decl
    : MDSPAN IDENTIFIER COL unnamed_span_decl {
        if (symtab.exists($2)) {
          Choreo::Parser::error(@2, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.addSymbol($2, AST::BaseType::INT, true);
        $4->name = $2;
        $$ = $4;
      }
    | MDSPAN LT NUM GT IDENTIFIER COL unnamed_span_decl {
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
        $7->name = $5;
        $$ = $7;
      }
    | IDENTIFIER COL unnamed_span_decl {
        if (symtab.exists($1)) {
          Choreo::Parser::error(@1, "ODR violation: the symbol is already defined.");
          exit(1);
        }
        symtab.addSymbol($1, AST::BaseType::INT, true);
        $3->name = $1;
        $$ = $3;
      }
    ;


unnamed_tuple_decl
    : LBRACE sval_list RBRACE {
      $$ = std::make_shared<AST::IntTuple>("", $2);
    }
    ;

named_tuple_decl
    : ITUPLE IDENTIFIER ASSIGN unnamed_tuple_decl {
        $4 -> name = $2;
        /* TODO: workaround: use INT for ituple's base type use a dedicated type for ituple in symboltable */
        ituple_symtab.addITupleSymbol($2, $4);
        $$ = $4;
      }
    | IDENTIFIER ASSIGN unnamed_tuple_decl {
        $3 -> name = $1;
        ituple_symtab.addITupleSymbol($1, $3);
        $$ = $3;
      }
    | IDENTIFIER ASSIGN IDENTIFIER LBRACE s_index_list RBRACE {
        // anchor
        if (!ituple_symtab.exists($3))
          Choreo::Parser::error(@3, "The symbol has not been defined.");
        auto src_ituple = ituple_symtab.getSymbol($3);
        auto ret_tuple = std::make_shared<AST::SValList>();
        for (AST::ptr<AST::IntIndex> idx : $5->indices) {
          // TODO: evaluate index later
          ret_tuple->Append(src_ituple->value->values[dynamic_cast<AST::IntLiteral*>(&*idx->value)->value]);
        }
        $$ = std::make_shared<AST::IntTuple>($1, std::move(ret_tuple));
      }
    ;

storage_specifier: LOCAL | SHARED | GLOBAL;

assignment
    : IDENTIFIER ASSIGN expr {
        // The assignment is overrided to be initilization of ituple
        auto sym = std::dynamic_pointer_cast<AST::Identifier>($3->value_r);
        if (sym) {
          auto src_ituple = ituple_symtab.getSymbol(sym->name);
          auto ret_ituple = std::make_shared<AST::IntTuple>($1, src_ituple->value);
          ituple_symtab.addITupleSymbol($1, ret_ituple);
          $$ = std::make_shared<AST::NodeRef>(ret_ituple);
          break;
        }

        if (!symtab.exists($1)) {
          Choreo::Parser::error(@1, ": the symbol '" + $1 + "` is not defined.");
          exit(1);
        }

        $$ = std::make_shared<AST::NodeRef>(std::make_shared<AST::Assignment>($1, $3));
      }
    ;

expr
    : expr PLUS term { $$ = std::make_shared<AST::Expr>("+", $1, $3); }
    | expr MINUS term { $$ = std::make_shared<AST::Expr>("-", $1, $3); }
    | term { $$ = $1; }
    ;

term
    : term STAR sval_ref { $$ = std::make_shared<AST::Expr>("*", $1, $3); }
    | term SLASH sval_ref { $$ = std::make_shared<AST::Expr>("/", $1, $3); }
    | term PECET sval_ref { $$ = std::make_shared<AST::Expr>("%", $1, $3); }
    | LPAREN expr RPAREN { $$ = $2; }
    | sval_ref { $$ = std::make_shared<AST::Expr>($1); }
    ;

if_else
    : if_clause
    | if_clause else_clause

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

with_binding
    : WITH IDENTIFIER LBRACE with_statements RBRACE
    ;

with_statements
    : /*Empty*/
    | with_statements w_statement
    ;

w_statement
    : /*Empty */
    | declarations SEMCOL
    | assignments  SEMCOL
    | dma SEMCOL
    | if_else
    | for_each
    ;

iterate_statements
    : declarations
    | assignments
    ;

for_each:
    ITER IDENTIFIER LBRACE iterate_statements RBRACE
    ;

dma : DMA COL sval_ref source_to_dest
    | COPY COL sval_ref source_to_dest

source_to_dest:
      IDENTIFIER ind_expr TRANS IDENTIFIER ind_expr

ind_expr: LBRAKT indices RBRAKT

indices: IDENTIFIER
/*    | int_list */


%%


// Bison expects us to provide implementation - otherwise linker complains
void Choreo::Parser::error(const location &loc , const std::string &message) {
  if (shell_supports_colors())
      std::cerr << red;
  std::cerr << "Error: ";
  if (shell_supports_colors())
      std::cerr << reset;
  std::cerr << message << std::endl;
  std::cerr << "Error location: " << loc << std::endl;
}

int main(int argc, char* argv[]) {
	std::string filename;
	bool debugMode = false;
	bool dumpAST = false;

	// Define long options
	static struct option long_options[] = {
			{"debug", no_argument, 0, 'd'},
			{"dump-ast", no_argument, 0, 'e'},
			{0, 0, 0, 0}
	};

	// Parse command-line options
	int opt;
	int option_index = 0;
	while ((opt = getopt_long(argc, argv, "de", long_options, &option_index)) != -1) {
		switch (opt) {
			case 'd':
					debugMode = true;
					break;
			case 'e':
					dumpAST = true;
					break;
			case '?':
					// getopt_long already printed an error message
					return 1;
			default:
					break;
		}
	}

	if (optind >= argc) {
    std::cerr << "Usage: " << argv[0] << " <filename>\n";
    return 1;
	}

  filename = argv[optind];

  if (filename.empty()) {
    std::cerr << "Usage: " << argv[0] << " <filename>\n";
    return 1;
  }

	std::ifstream file(filename);
	if (!file) {
		std::cerr << "Could not open file: " << filename << std::endl;
		return 1;
	}

  loc.begin.filename = loc.end.filename = &filename;

  Choreo::Scanner s;
  s.yyrestart(file);
  Choreo::Parser p(s);

	if (debugMode) {
		std::cout << "Choreo: Debug of parsing is switched on." << std::endl;
		p.set_debug_level(1); // Enable Bison debugging
    Choreo::Scanner::SetDebug();
	}

  p.parse();

  if (dumpAST)
    root.Print(std::cout);

  return 0;
}

int AST::SymbolTable::anonymous_count = 0;
std::unordered_map<std::string, AST::ptr<AST::MultiSpans>> AST::DataType::namedSpans;
// tuple_decl
//     : LBRACE sval_list RBRACE {
//         $$ = std::make_shared<AST::IntTuple>("", $2);
//       }
//     ;
//
