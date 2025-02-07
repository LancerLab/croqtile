#include "visitor.hpp"

using namespace Choreo;

std::unordered_set<std::string> Choreo::Visitor::AllVisitors;

location loc;
AST::Program root(loc);
