#ifndef __CHOREO_AST_HPP__
#define __CHOREO_AST_HPP__

#include <vector>
#include <string>
#include <memory>

namespace AST {

// Base class for all AST nodes
struct Node {
    virtual ~Node() {}
};

// For types like f32, f16, etc.
enum class BaseType {
    F32, F16, BF16, U32, S32, U16, S16, U8, S8, INT
};

// For storage specifiers like local, global, shared
enum class StorageSpec {
    LOCAL, GLOBAL, SHARED
};

// Represents both dimensions and s like {3, 4, 5} or {1, 2, 1}
struct MdimSpans : public Node {
  std::vector<std::shared_ptr<Node>> values;
  explicit MdimSpans() {}
  MdimSpans(const std::vector<std::shared_ptr<Node>>& v) : values(v) {}
};

// Represents declarations like: mdimspans d{3, 4, 5};
struct MdimSpansDecl : public Node {
    std::string name;
    std::shared_ptr<MdimSpans> spans;
    MdimSpansDecl(const std::string& n, const std::shared_ptr<MdimSpans>& s) : name(n), spans(s) {}
};

// Represents data declarations like: global f32 data{d};
struct DataDecl : public Node {
    StorageSpec storage;
    BaseType type;
    std::string name;
    std::shared_ptr<MdimSpans> mdspans;  // Refers to a previously declared MdimSpans
    DataDecl(StorageSpec s, BaseType t, const std::string& n, const std::shared_ptr<MdimSpans> & spans)
        : storage(s), type(t), name(n), mdspans(spans) {}
};

struct DataType : public Node {
  bool scalar;
  BaseType type;
  std::shared_ptr<MdimSpans> mdspans;
  DataType(BaseType t) : scalar(true), type(t) {}
  DataType(BaseType t, const std::shared_ptr<MdimSpans> & spans) : scalar(false), type(t), mdspans(spans) {}
};

struct IntLiteral : public Node {
    int value;
    IntLiteral(int v) : value(v) {}
};

struct Identifier : public Node {
    std::string name;
    Identifier(const std::string &n) : name(n) {}
};

struct Declaration : public Node {
    std::vector<std::unique_ptr<Node>> subs;
};

struct CppSourceCode : public Node {
  std::string code;
  CppSourceCode(const std::string & c): code(c){}
};

struct Expression : public Node {
    std::vector<std::unique_ptr<Node>> exprs;
};

// Top-level program structure
struct Program : public Node {
    std::vector<std::unique_ptr<Node>> declarations;
};


}  // end of namespace AST

#endif // __CHOREO_AST_HPP__
