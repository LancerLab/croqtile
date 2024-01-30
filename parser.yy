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

const char* red = "\033[31m";
const char* reset = "\033[0m";

static inline bool shell_supports_colors() {
	const char* term = getenv("TERM");
	return term && (strcmp(term, "xterm-256color") == 0 
							 || strcmp(term, "xterm") == 0);
}

static inline void print_fixed_header() {
  std::cout << "#include \"../../utils/utils.h\"\n"
            << "#include \"dtu/factor/factor.h\"\n"
            << "#include \"dtu/factor/program_experimental.h\"\n"
            << "#include \"llvm/ADT/ArrayRef.h\"\n"
            << "#include \"logging_api.h\"\n"
            << "#include \"tests/factor/api/base/fixture.h\"\n";
}

static inline void print_wrapper_begin() {
  std::cout << "TEST(DoradoBasicTest, SimpleAddtest) {\n";
  std::cout << "  using namespace factor;\n";
  std::cout << "  FACTOR_PROGRAM(p);\n\n";
  std::cout << "  p([&](auto target_name) {\n";
}

static inline void print_wrapper_end() {
  std::cout << "TEST(DoradoBasicTest, SimpleAddtest) {\n";
  std::cout << "  };\n";
  std::cout << "};\n";
}

static Choreo::Parser::symbol_type yylex(Choreo::Scanner &scanner) {
  return scanner.get_next_token();
}

//#define yylex(x) scanner.get_next_token()


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
  LBRAC   "["
  RBRAC   "]"
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
%token <std::string> IDENTIFIER ATTR_CO DMA COPY
// type related
%token <std::string> MDSPANS MINDS LOCAL SHARED GLOBAL
%token <AST::BaseType> F32 F16 BF16 U16 S16 U8 S8 U32 S32 INT
// control related
%token <std::string> IF ELSE WITH ITER RET

// non-terminals
//%nterm <std::shared_ptr<AST::Node>> mdspans_decl mindices_decl statements value expr
%nterm <std::shared_ptr<AST::Node>> statement value expr
%nterm <std::shared_ptr<AST::Statements>> statements
%nterm <std::shared_ptr<AST::MdimSpans>> init_list
%nterm <std::shared_ptr<AST::Node>> pass_by
%nterm <AST::BaseType> base_type
%nterm <std::shared_ptr<AST::DataType>> general_type aggregate_type
%nterm <std::shared_ptr<AST::ParamList>> parameter_list
%nterm <std::shared_ptr<AST::ParamType>> parameter
%nterm <std::shared_ptr<AST::ChoreoFunction>> dsl_function

%%

program:
    /* Empty */ {}
  | program pass_by      { root.nodes.push_back($2); }
  | program dsl_function { root.nodes.push_back($2); }
  | END {  }
  ;

pass_by:
    CPP_CODE {
      $$ = std::make_shared<AST::CppSourceCode>($1);
    }
    ;

dsl_function:
    ATTR_CO general_type IDENTIFIER LPAREN parameter_list RPAREN LBRACE statements RBRACE {
      $$ = std::make_shared<AST::ChoreoFunction>();
      $$->name = $3;
      $$->decls.name = $3;
      $$->decls.ret_type = $2;
      $$->decls.params = $5;
      $$->states = $8;
    }
    ;

general_type:
      base_type { $$ = std::make_shared<AST::DataType>($1); }
    | aggregate_type { $$ = $1; }
    ;

aggregate_type:
	    base_type LBRAC IDENTIFIER RBRAC { $$ = std::make_shared<AST::DataType>($1); }
    | base_type LBRAC init_list RBRAC  { $$ = std::make_shared<AST::DataType>($1, $3); }

base_type: F32   { $$ = $1; }
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

init_list:
      init_list COMMA NUM
      {  
				$1->values.push_back(std::make_shared<AST::IntLiteral>($3));
				$$ = $1;
      }
    | NUM
      { 
				$$ = std::make_shared<AST::MdimSpans>();
				$$->values.push_back(std::make_shared<AST::IntLiteral>($1));
      }


/* for now we only support init_list with constants
init_list: value
						{
								$$ = std::make_shared<AST::MdimSpans>();
								$$->values.push_back($1);
						}
         | init_list COMMA value
            {  
				        $1->values.push_back($3);
								$$ = $1;
            }
         ;

*/

value: 
      NUM { $$ = std::make_shared<AST::IntLiteral>($1); }
    | IDENTIFIER {
        $$ = std::make_shared<AST::Identifier>($1);
    	}
     ;

parameter_list:
    /* Empty */
      {  $$ = std::make_shared<AST::ParamList>();}
    | parameter_list COMMA parameter { 
        $$->values.push_back($3);
      }
    | parameter {
        $$ = std::make_shared<AST::ParamList>();
        $$->values.push_back($1);
      }
    ;

parameter:
    general_type IDENTIFIER { /* handle parameter type and name here */
      $$ = std::make_shared<AST::ParamType>(std::pair($1, std::make_shared<AST::Identifier>($2)));
    }
    ;

statements: /* Empty */
    { $$ = std::make_shared<AST::Statements>(); }
    | statements statement
    { $$->subs.push_back($2); }
    ;

statement:
      declaration SEMCOL
    | assignments SEMCOL
    | if_else
    | with_loop
    ;

assignments:
      assignments COMMA assignment
    | assignment
    ;

declaration:
      base_type IDENTIFIER LBRACE NUM RBRACE
      {
      }
    | storage_specifier MDSPANS IDENTIFIER LBRACE init_list RBRACE
      {
                //MdimSpansType dims;
                //dims.dimensions = $4;
//                store_dims($2, dims);  // Assuming you have this function to store the dimensions
      }
    | storage_specifier MINDS IDENTIFIER LBRAC init_list RBRAC
      {
      }
    ;

storage_specifier: LOCAL | SHARED | GLOBAL;

assignment: IDENTIFIER ASSIGN expr
      {
      }
    ;

expr:
	    expr PLUS value { std::cout << $1 << " + " << $3; }
    | expr MINUS value { std::cout << $1 << " - " << $3; }
    | expr STAR value { std::cout << $1 << " * " << $3; }
    | expr SLASH value { std::cout << $1 << " / " << $3; }
    | expr PECET value { std::cout << $1 << " % " << $3; }
    | value
    ;

if_else:
       if_clause
    |  if_clause else_clause

if_clause:
       IF LBRAC cmp_expr RBRAC LBRACE statements RBRACE
    ;

else_clause:
       ELSE LBRACE statements RBRACE
    ;

cmp_expr:
      expr LT expr
    | expr GT expr
    | expr EQ expr
    | expr NE expr
    | expr LE expr
    | expr GE expr
    ;

with_loop:
      WITH assignments LBRACE with_statements RBRACE
    ;

with_statements: /* Empty */
    |  with_statements w_statement
    {}
    ;

w_statement:
      declaration SEMCOL
    | assignments SEMCOL
    | data_move SEMCOL
    | if_else
    | iterate_clause
    ;

iterate_clause:
    ITER IDENTIFIER LBRACE statements RBRACE
    {}
    ;

data_move: /* Empty */
    | DMA COL value source_to_dest SEMCOL
    | COPY COL value source_to_dest SEMCOL

source_to_dest:
      IDENTIFIER ind_expr TRANS IDENTIFIER ind_expr

ind_expr: LBRAC indices RBRAC

indices: IDENTIFIER
    | init_list


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

  Choreo::Scanner s;
  s.yyrestart(file);
  Choreo::Parser p(s);

	if (debugMode) {
		std::cout << "Choreo: Debug is on." << std::endl;
		p.set_debug_level(1); // Enable Bison debugging
	}

  p.parse();

  if (dumpAST)
    root.Print(std::cout);

  return 0;
}
