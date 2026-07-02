#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "ast.hpp"
#include "choreo_sdk.hpp"
#include "context.hpp"
#include "options.hpp"
#include "pipeline.hpp"
#include "target_registry.hpp"

namespace py = pybind11;

namespace C = Choreo;
namespace A = Choreo::AST;

template <typename T>
using Ptr = std::shared_ptr<T>;

static C::location pyloc() {
  return C::location(C::position("<python>", 1, 1));
}

extern A::Program root;

PYBIND11_MODULE(_core, m) {
  m.doc() = "CroqTile: low-level pybind11 bindings for CroqTile AST";

  // ---- Enums ------------------------------------------------

  py::enum_<C::BaseType>(m, "BaseType")
      .value("F64", C::BaseType::F64)
      .value("F32", C::BaseType::F32)
      .value("TF32", C::BaseType::TF32)
      .value("F16", C::BaseType::F16)
      .value("BF16", C::BaseType::BF16)
      .value("F8_E4M3", C::BaseType::F8_E4M3)
      .value("F8_E5M2", C::BaseType::F8_E5M2)
      .value("S64", C::BaseType::S64)
      .value("U64", C::BaseType::U64)
      .value("S32", C::BaseType::S32)
      .value("U32", C::BaseType::U32)
      .value("S16", C::BaseType::S16)
      .value("U16", C::BaseType::U16)
      .value("S8", C::BaseType::S8)
      .value("U8", C::BaseType::U8)
      .value("BOOL", C::BaseType::BOOL)
      .value("VOID", C::BaseType::VOID)
      .value("INDEX", C::BaseType::INDEX)
      .value("F6_E2M3", C::BaseType::F6_E2M3)
      .value("F6_E3M2", C::BaseType::F6_E3M2)
      .value("F4_E2M1", C::BaseType::F4_E2M1)
      .value("STREAM", C::BaseType::STREAM)
      .value("UNKNOWN", C::BaseType::UNKNOWN);

  py::enum_<C::Storage>(m, "Storage")
      .value("REG", C::Storage::REG)
      .value("LOCAL", C::Storage::LOCAL)
      .value("SHARED", C::Storage::SHARED)
      .value("GLOBAL", C::Storage::GLOBAL)
      .value("DEFAULT", C::Storage::DEFAULT)
      .value("NONE", C::Storage::NONE);

  // ---- Node (base) ------------------------------------------

  py::class_<A::Node, Ptr<A::Node>>(m, "Node")
      .def("print_str", [](const A::Node& self) {
        std::ostringstream os;
        self.Print(os);
        return os.str();
      })
      .def("type_name", [](const A::Node& self) {
        return self.TypeNameString();
      });

  // ---- Identifier -------------------------------------------

  py::class_<A::Identifier, A::Node, Ptr<A::Identifier>>(m, "Identifier")
      .def(py::init([](const std::string& name) {
        return A::Make<A::Identifier>(pyloc(), name);
      }), py::arg("name"))
      .def_readwrite("name", &A::Identifier::name);

  // ---- Literals ---------------------------------------------

  py::class_<A::IntLiteral, A::Node, Ptr<A::IntLiteral>>(m, "IntLiteral")
      .def(py::init([](int64_t v) {
        return A::Make<A::IntLiteral>(pyloc(), v);
      }), py::arg("value"))
      .def_readwrite("value", &A::IntLiteral::value);

  py::class_<A::FloatLiteral, A::Node, Ptr<A::FloatLiteral>>(m, "FloatLiteral")
      .def(py::init([](double v) {
        return A::Make<A::FloatLiteral>(pyloc(), v);
      }), py::arg("value"));

  py::class_<A::BoolLiteral, A::Node, Ptr<A::BoolLiteral>>(m, "BoolLiteral")
      .def(py::init([](bool v) {
        return A::Make<A::BoolLiteral>(pyloc(), v);
      }), py::arg("value"));

  // ---- Expr -------------------------------------------------

  py::class_<A::Expr, A::Node, Ptr<A::Expr>>(m, "Expr")
      .def("is_binary", &A::Expr::IsBinary)
      .def("is_unary", &A::Expr::IsUnary)
      .def("is_reference", &A::Expr::IsReference);

  m.def("make_id_expr", [](const std::string& name) -> Ptr<A::Expr> {
    return A::MakeIdExpr(pyloc(), name);
  }, py::arg("name"),
     "Create an identifier-reference expression.");

  m.def("make_int_expr", [](int val) -> Ptr<A::Expr> {
    return std::const_pointer_cast<A::Expr>(A::MakeIntExpr(pyloc(), val));
  }, py::arg("value"),
     "Create an integer literal expression.");

  m.def("make_float_expr", [](double val) -> Ptr<A::Expr> {
    auto lit = A::Make<A::FloatLiteral>(pyloc(), val);
    return A::Make<A::Expr>(pyloc(), static_cast<Ptr<A::Node>>(lit));
  }, py::arg("value"),
     "Create a float literal expression.");

  m.def("make_binary_expr",
        [](const std::string& op,
           const Ptr<A::Node>& lhs,
           const Ptr<A::Node>& rhs) -> Ptr<A::Expr> {
          return A::Make<A::Expr>(pyloc(), C::Opcode(op), lhs, rhs);
        },
        py::arg("op"), py::arg("lhs"), py::arg("rhs"),
        "Create a binary expression. op is e.g. +, -, *, /, <, >.");

  m.def("make_ternary_expr",
        [](const Ptr<A::Expr>& cond,
           const Ptr<A::Node>& true_val,
           const Ptr<A::Node>& false_val) -> Ptr<A::Expr> {
          return A::Make<A::Expr>(pyloc(), C::Opcode("?"), cond,
                                  true_val, false_val);
        },
        py::arg("cond"), py::arg("true_val"), py::arg("false_val"),
        "Create a ternary expression: cond ? true_val : false_val.");

  m.def("make_elemof_expr",
        [](const Ptr<A::Expr>& base,
           const Ptr<A::Expr>& index) -> Ptr<A::Expr> {
          return A::Make<A::Expr>(
              pyloc(), C::Opcode(C::Opcode::Kind::ElemOf),
              static_cast<Ptr<A::Node>>(base),
              static_cast<Ptr<A::Node>>(index));
        },
        py::arg("base"), py::arg("index"),
        "Create an element-access expression: base[index].");

  // ---- MultiNodes -------------------------------------------

  py::class_<A::MultiNodes, A::Node, Ptr<A::MultiNodes>>(m, "MultiNodes")
      .def(py::init([]() { return A::Make<A::MultiNodes>(pyloc()); }))
      .def("append", &A::MultiNodes::Append)
      .def("count", &A::MultiNodes::Count);

  // ---- MultiValues ------------------------------------------

  py::class_<A::MultiValues, A::Node, Ptr<A::MultiValues>>(m, "MultiValues")
      .def(py::init([]() {
        return A::Make<A::MultiValues>(pyloc(), std::string(", "));
      }))
      .def("append", &A::MultiValues::Append)
      .def("count", &A::MultiValues::Count);

  // ---- DataType ---------------------------------------------

  py::class_<A::DataType, A::Node, Ptr<A::DataType>>(m, "DataType")
      .def(py::init([](C::BaseType bt) {
        return A::Make<A::DataType>(pyloc(), bt);
      }), py::arg("base_type"),
         "Create a scalar DataType.")
      .def(py::init([](C::BaseType bt,
                       std::vector<int> shape) -> Ptr<A::DataType> {
        auto dims = A::Make<A::MultiValues>(pyloc(), std::string(", "));
        for (int d : shape)
          dims->Append(std::const_pointer_cast<A::Expr>(
              A::MakeIntExpr(pyloc(), d)));
        auto mds = A::Make<A::MultiDimSpans>(
            pyloc(), std::string(""),
            static_cast<Ptr<A::Node>>(dims));
        return A::Make<A::DataType>(
            pyloc(), bt, static_cast<Ptr<A::Node>>(mds));
      }), py::arg("base_type"), py::arg("shape"),
         "Create a shaped tensor DataType, e.g. DataType(S32, [6, 64]).");

  // ---- Memory -----------------------------------------------

  py::class_<A::Memory, A::Node, Ptr<A::Memory>>(m, "Memory")
      .def(py::init([](C::Storage s) {
        return A::Make<A::Memory>(pyloc(), s);
      }), py::arg("storage"));

  // ---- Parameter / ParamList --------------------------------

  py::class_<A::Parameter, A::Node, Ptr<A::Parameter>>(m, "Parameter")
      .def(py::init([](const Ptr<A::DataType>& type,
                       const std::string& name) {
        auto sym = A::Make<A::Identifier>(pyloc(), name);
        return A::Make<A::Parameter>(pyloc(), type, sym);
      }), py::arg("type"), py::arg("name"))
      .def(py::init([](const Ptr<A::DataType>& type,
                       const std::string& name,
                       C::Storage storage) {
        auto sym = A::Make<A::Identifier>(pyloc(), name);
        type->SetType(
            C::MakeUnRankedSpannedType(type->getBaseType(), storage));
        return A::Make<A::Parameter>(pyloc(), type, sym);
      }), py::arg("type"), py::arg("name"), py::arg("storage"));

  py::class_<A::ParamList, A::Node, Ptr<A::ParamList>>(m, "ParamList")
      .def(py::init([]() { return A::Make<A::ParamList>(pyloc()); }))
      .def("add", [](A::ParamList& self, const Ptr<A::Parameter>& p) {
        self.values.push_back(p);
      });

  // ---- NamedVariableDecl ------------------------------------

  py::class_<A::NamedVariableDecl, A::Node, Ptr<A::NamedVariableDecl>>(
      m, "NamedVariableDecl")
      .def(py::init([](const std::string& name,
                       const Ptr<A::DataType>& type) {
        return A::Make<A::NamedVariableDecl>(pyloc(), name, type);
      }), py::arg("name"), py::arg("type"))
      .def(py::init([](const std::string& name,
                       const Ptr<A::DataType>& type,
                       C::Storage storage) {
        auto mem = A::Make<A::Memory>(pyloc(), storage);
        return A::Make<A::NamedVariableDecl>(pyloc(), name, type, mem);
      }), py::arg("name"), py::arg("type"), py::arg("storage"))
      .def(py::init([](const std::string& name,
                       const Ptr<A::DataType>& type,
                       C::Storage storage, int init_val) {
        auto mem = A::Make<A::Memory>(pyloc(), storage);
        Ptr<A::Node> init_node = A::Make<A::IntLiteral>(pyloc(), init_val);
        return A::Make<A::NamedVariableDecl>(
            pyloc(), name, type, mem, nullptr, init_node);
      }), py::arg("name"), py::arg("type"), py::arg("storage"),
         py::arg("init_val"));

  // ---- DataAccess / Assignment ------------------------------

  py::class_<A::DataAccess, A::Node, Ptr<A::DataAccess>>(m, "DataAccess")
      .def(py::init([](const std::string& name) {
        return A::Make<A::DataAccess>(pyloc(), name);
      }), py::arg("name"))
      .def(py::init([](const std::string& name,
                       const Ptr<A::MultiValues>& indices) {
        auto id = A::Make<A::Identifier>(pyloc(), name);
        return A::Make<A::DataAccess>(pyloc(), id, indices);
      }), py::arg("name"), py::arg("indices"));

  py::class_<A::Assignment, A::Node, Ptr<A::Assignment>>(m, "Assignment")
      .def(py::init([](const std::string& name, const Ptr<A::Node>& value) {
        return A::Make<A::Assignment>(pyloc(), name, value);
      }), py::arg("name"), py::arg("value"))
      .def(py::init([](const Ptr<A::DataAccess>& da,
                       const Ptr<A::Node>& value) {
        return A::Make<A::Assignment>(pyloc(), da, value);
      }), py::arg("data_access"), py::arg("value"));

  // ---- Block / ParallelBy -----------------------------------

  py::class_<A::Block, A::Node, Ptr<A::Block>>(m, "Block");

  py::class_<A::ParallelBy, A::Block, Ptr<A::ParallelBy>>(m, "ParallelBy")
      .def("set_body", [](A::ParallelBy& self,
                          const Ptr<A::MultiNodes>& stmts) {
        self.stmts = stmts;
      })
      .def("get_body", [](A::ParallelBy& self) -> Ptr<A::MultiNodes> {
        return self.stmts;
      });

  auto apply_scope = [](Ptr<A::ParallelBy>& pb, const std::string& scope) {
    C::ParallelLevel lvl = C::ParallelLevel::NONE;
    if (scope == "block") lvl = C::ParallelLevel::BLOCK;
    else if (scope == "group") lvl = C::ParallelLevel::GROUP;
    else if (scope == "group-4") lvl = C::ParallelLevel::GROUPx4;
    else if (scope == "thread") lvl = C::ParallelLevel::THREAD;
    else if (scope == "cluster") lvl = C::ParallelLevel::CLUSTER;
    if (lvl != C::ParallelLevel::NONE) pb->SetLevel(lvl);
  };

  m.def("make_parallel_by",
        [&apply_scope](const std::string& var_name, int bound,
           const Ptr<A::MultiNodes>& body,
           const std::string& scope) -> Ptr<A::ParallelBy> {
          auto pv = A::Make<A::Identifier>(pyloc(), var_name);
          pv->SetType(C::MakeBoundedITupleType(C::Shape(1, bound)));

          auto spv = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          auto epv = A::Make<A::Identifier>(pyloc(), var_name);
          epv->SetType(C::MakeBoundedIntegerType(C::sbe::nu(bound)));
          spv->Append(epv);

          auto p_bound = std::const_pointer_cast<A::Expr>(
              A::MakeIntExpr(pyloc(), bound));
          p_bound->SetType(C::MakeIntegerType());
          auto spv_bounds =
              A::Make<A::MultiValues>(pyloc(), std::string(", "));
          spv_bounds->Append(p_bound->Clone());
          spv_bounds->SetType(C::MakeITupleType(1));

          auto pb = A::Make<A::ParallelBy>(pyloc(), pv, p_bound, spv,
                                           spv_bounds, body);
          pb->SetType(C::MakeBoundedIntegerType(C::sbe::nu(bound)));
          apply_scope(pb, scope);
          return pb;
        },
        py::arg("var_name"), py::arg("bound"), py::arg("body"),
        py::arg("scope") = "",
        "Create a ParallelBy node: parallel var by bound { body }.");

  m.def("make_parallel_by_expr",
        [&apply_scope](const std::string& var_name,
           const Ptr<A::Expr>& bound_expr,
           const Ptr<A::MultiNodes>& body,
           const std::string& scope) -> Ptr<A::ParallelBy> {
          auto pv = A::Make<A::Identifier>(pyloc(), var_name);
          pv->SetType(C::MakeITupleType(1));

          auto spv = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          auto epv = A::Make<A::Identifier>(pyloc(), var_name);
          epv->SetType(C::MakeIntegerType());
          spv->Append(epv);

          auto p_bound = std::const_pointer_cast<A::Expr>(bound_expr);
          p_bound->SetType(C::MakeIntegerType());
          auto spv_bounds =
              A::Make<A::MultiValues>(pyloc(), std::string(", "));
          spv_bounds->Append(p_bound->Clone());
          spv_bounds->SetType(C::MakeITupleType(1));

          auto pb = A::Make<A::ParallelBy>(pyloc(), pv, p_bound, spv,
                                           spv_bounds, body);
          pb->SetType(C::MakeIntegerType());
          apply_scope(pb, scope);
          return pb;
        },
        py::arg("var_name"), py::arg("bound_expr"), py::arg("body"),
        py::arg("scope") = "",
        "Create a ParallelBy node with expression bound.");

  // ---- Return -----------------------------------------------

  py::class_<A::Return, A::Node, Ptr<A::Return>>(m, "Return")
      .def(py::init([](const Ptr<A::Node>& value) {
        return A::Make<A::Return>(pyloc(), value);
      }), py::arg("value"));

  // ---- LoopRange / ForeachBlock ----------------------------

  py::class_<A::LoopRange, A::Node, Ptr<A::LoopRange>>(m, "LoopRange");

  py::class_<A::ForeachBlock, A::Block, Ptr<A::ForeachBlock>>(
      m, "ForeachBlock");

  m.def("make_foreach",
        [](const std::vector<std::pair<std::string, int>>& bindings,
           const Ptr<A::MultiNodes>& body) -> Ptr<A::ForeachBlock> {
          auto ranges = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          for (auto& [name, bound] : bindings) {
            auto iv = A::Make<A::Identifier>(pyloc(), name);
            auto ub = std::const_pointer_cast<A::Expr>(
                A::MakeIntExpr(pyloc(), bound));
            auto lr = A::Make<A::LoopRange>(pyloc(), iv, nullptr, ub, 1);
            ranges->Append(lr);
          }
          return A::Make<A::ForeachBlock>(pyloc(), ranges, body);
        },
        py::arg("bindings"), py::arg("body"),
        "Create a ForeachBlock: foreach {vars} in [bounds] { body }.");

  m.def("make_foreach_expr",
        [](const std::vector<std::pair<std::string, Ptr<A::Expr>>>& bindings,
           const Ptr<A::MultiNodes>& body) -> Ptr<A::ForeachBlock> {
          auto ranges = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          for (auto& [name, bound_expr] : bindings) {
            auto iv = A::Make<A::Identifier>(pyloc(), name);
            auto ub = std::const_pointer_cast<A::Expr>(bound_expr);
            auto lr = A::Make<A::LoopRange>(pyloc(), iv, nullptr, ub, 1);
            ranges->Append(lr);
          }
          return A::Make<A::ForeachBlock>(pyloc(), ranges, body);
        },
        py::arg("bindings"), py::arg("body"),
        "Create a ForeachBlock with expression bounds.");

  m.def("make_foreach_staged",
        [](const std::string& name, int start,
           const Ptr<A::MultiNodes>& body) -> Ptr<A::ForeachBlock> {
          auto ranges = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          auto iv = A::Make<A::Identifier>(pyloc(), name);
          auto lb = std::const_pointer_cast<A::Expr>(
              A::MakeIntExpr(pyloc(), start));
          auto lr = A::Make<A::LoopRange>(pyloc(), iv, lb, nullptr, 1);
          ranges->Append(lr);
          return A::Make<A::ForeachBlock>(pyloc(), ranges, body);
        },
        py::arg("name"), py::arg("start"), py::arg("body"),
        "Create staged ForeachBlock: foreach k(start:) { body }.");

  // ---- WithIn / WithBlock ----------------------------------

  py::class_<A::WithIn, A::Node, Ptr<A::WithIn>>(m, "WithIn");
  py::class_<A::WithBlock, A::Block, Ptr<A::WithBlock>>(m, "WithBlock");

  m.def("make_with_in",
        [](const std::vector<std::pair<std::string, int>>& bindings,
           const Ptr<A::MultiNodes>& body) -> Ptr<A::WithBlock> {
          auto withins = A::Make<A::MultiNodes>(pyloc());
          for (auto& [name, bound] : bindings) {
            auto sym = A::Make<A::Identifier>(pyloc(), name);
            sym->SetType(C::MakeBoundedIntegerType(C::sbe::nu(bound)));
            auto bound_expr =
                static_cast<Ptr<A::Node>>(A::MakeIntExpr(pyloc(), bound));
            auto matchers =
                A::Make<A::MultiValues>(pyloc(), std::string(", "));
            matchers->Append(sym->Clone());
            auto wi = A::Make<A::WithIn>(pyloc(), sym, bound_expr, matchers);
            withins->Append(wi);
          }
          return A::Make<A::WithBlock>(pyloc(), withins, nullptr, body);
        },
        py::arg("bindings"), py::arg("body"),
        "Create a WithBlock: with name in bound { body }.");

  m.def("make_with_in_expr",
        [](const std::vector<std::pair<std::string, Ptr<A::Expr>>>& bindings,
           const Ptr<A::MultiNodes>& body) -> Ptr<A::WithBlock> {
          auto withins = A::Make<A::MultiNodes>(pyloc());
          for (auto& [name, bound_expr] : bindings) {
            auto sym = A::Make<A::Identifier>(pyloc(), name);
            sym->SetType(C::MakeIntegerType());
            auto be = static_cast<Ptr<A::Node>>(
                std::const_pointer_cast<A::Expr>(bound_expr));
            auto matchers =
                A::Make<A::MultiValues>(pyloc(), std::string(", "));
            matchers->Append(sym->Clone());
            auto wi = A::Make<A::WithIn>(pyloc(), sym, be, matchers);
            withins->Append(wi);
          }
          return A::Make<A::WithBlock>(pyloc(), withins, nullptr, body);
        },
        py::arg("bindings"), py::arg("body"),
        "Create a WithBlock with expression bounds.");

  // ---- Wait --------------------------------------------------

  py::class_<A::Wait, A::Node, Ptr<A::Wait>>(m, "Wait");

  m.def("make_wait",
        [](const Ptr<A::MultiValues>& targets) -> Ptr<A::Wait> {
          return A::Make<A::Wait>(pyloc(), targets);
        },
        py::arg("targets"),
        "Create a Wait node for async DMA/TMA futures.");

  // ---- ChunkAt -----------------------------------------------

  py::class_<A::ChunkAt, A::Node, Ptr<A::ChunkAt>>(m, "ChunkAt")
      .def(py::init([](const std::string& name,
                       const Ptr<A::MultiValues>& indices) {
        auto id = A::Make<A::Identifier>(pyloc(), name);
        return A::Make<A::ChunkAt>(pyloc(), id, indices);
      }), py::arg("name"), py::arg("indices"));

  m.def("make_chunkat",
        [](const std::string& name,
           const std::vector<Ptr<A::Node>>& indices) -> Ptr<A::ChunkAt> {
          auto id = A::Make<A::Identifier>(pyloc(), name);
          auto mv = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          for (auto& idx : indices) mv->Append(idx);
          return A::Make<A::ChunkAt>(pyloc(), id, mv);
        },
        py::arg("name"), py::arg("indices"),
        "Create a ChunkAt node: data.chunkat(i, j).");

  m.def("make_chunkat_view",
        [](const std::string& name,
           const std::vector<Ptr<A::Node>>& shape,
           const std::vector<Ptr<A::Node>>& offsets) -> Ptr<A::ChunkAt> {
          auto id = A::Make<A::Identifier>(pyloc(), name);
          auto indices = A::Make<A::MultiValues>(pyloc(), std::string(", "));

          auto subspan = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          for (auto& s : shape) subspan->Append(s);

          auto off_mv = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          for (auto& o : offsets) off_mv->Append(o);

          auto view = A::Make<A::SOP::View>(pyloc(), subspan, off_mv);
          std::vector<std::shared_ptr<A::SpannedOperation>> ops;
          ops.push_back(std::static_pointer_cast<A::SpannedOperation>(view));
          return A::Make<A::ChunkAt>(pyloc(), id, indices, ops);
        },
        py::arg("name"), py::arg("shape"), py::arg("offsets"),
        "Create ChunkAt with View operation: data.view(shape).from(offsets).");

  m.def("make_chunkat_subspan",
        [](const std::string& name,
           const std::vector<Ptr<A::Node>>& shape,
           const std::vector<Ptr<A::Node>>& at_indices,
           const std::vector<Ptr<A::Node>>& chunkat_indices,
           const std::vector<Ptr<A::Node>>& steps) -> Ptr<A::ChunkAt> {
          auto id = A::Make<A::Identifier>(pyloc(), name);

          auto ca_mv = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          for (auto& ci : chunkat_indices) ca_mv->Append(ci);

          auto ss_shape = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          for (auto& s : shape) ss_shape->Append(s);

          auto ss_idx = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          for (auto& i : at_indices) ss_idx->Append(i);

          Ptr<A::MultiValues> ss_steps = nullptr;
          if (!steps.empty()) {
            ss_steps = A::Make<A::MultiValues>(pyloc(), std::string(", "));
            for (auto& st : steps) ss_steps->Append(st);
          }

          auto ss = A::Make<A::SOP::SubSpan>(
              pyloc(), ss_shape, ss_idx, ss_steps);
          std::vector<std::shared_ptr<A::SpannedOperation>> ops;
          ops.push_back(std::static_pointer_cast<A::SpannedOperation>(ss));
          return A::Make<A::ChunkAt>(pyloc(), id, ca_mv, ops);
        },
        py::arg("name"), py::arg("shape"), py::arg("at_indices"),
        py::arg("chunkat_indices") = std::vector<Ptr<A::Node>>{},
        py::arg("steps") = std::vector<Ptr<A::Node>>{},
        "Create ChunkAt with SubSpan: data.subspan(shape).at(indices).");

  // ---- Synchronize (sync.shared) --------------------------------

  py::class_<A::Synchronize, A::Node, Ptr<A::Synchronize>>(m, "Synchronize");

  m.def("make_sync",
        [](const std::string& kind) -> Ptr<A::Synchronize> {
          C::Storage s = C::Storage::SHARED;
          if (kind == "global") s = C::Storage::GLOBAL;
          return A::Make<A::Synchronize>(pyloc(), s);
        },
        py::arg("kind") = "shared",
        "Create a Synchronize barrier node: sync.shared or sync.global.");

  // ---- DMA ---------------------------------------------------

  py::class_<A::DMA, A::Node, Ptr<A::DMA>>(m, "DMA");

  m.def("make_dma_copy",
        [](const std::string& future_name,
           const Ptr<A::Node>& from,
           const Ptr<A::Node>& to,
           bool async, bool is_tma, int swizzle) -> Ptr<A::DMA> {
          A::DMAAsync da(async);
          C::SwizMode sm = C::SwizMode::NONE;
          if (swizzle == 32) sm = C::SwizMode::B32;
          else if (swizzle == 64) sm = C::SwizMode::B64;
          else if (swizzle == 128) sm = C::SwizMode::B128;
          A::DMAAttribute attr(sm, sm != C::SwizMode::NONE);
          return A::Make<A::DMA>(
              pyloc(), ".copy", future_name, from, to, da, attr, is_tma);
        },
        py::arg("future") = "",
        py::arg("from") = nullptr,
        py::arg("to") = nullptr,
        py::arg("async") = false,
        py::arg("is_tma") = false,
        py::arg("swizzle") = 0,
        "Create a DMA copy node.");

  m.def("make_dma_any",
        [](const std::string& future_name) -> Ptr<A::DMA> {
          A::DMAAsync da(false);
          return A::Make<A::DMA>(
              pyloc(), ".any", future_name,
              nullptr, nullptr, da);
        },
        py::arg("future"),
        "Create a DMA.any placeholder future.");

  // ---- Select (DMA target: => shared / => local) ---------------

  py::class_<A::Select, A::Node, Ptr<A::Select>>(m, "Select");

  m.def("make_select",
        [](const std::string& storage_name) -> Ptr<A::Select> {
          auto sf = A::MakeIdExpr(pyloc(), storage_name);
          auto list = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          return A::Make<A::Select>(pyloc(), sf, list);
        },
        py::arg("storage_name"),
        "Create a Select node for DMA targets (shared/local/global).");

  // ---- MMA ---------------------------------------------------

  py::class_<A::MMAOperation>(m, "MMAOperation");

  py::class_<A::MMA, A::Node, Ptr<A::MMA>>(m, "MMA");

  m.def("make_mma_fill",
        [](double val, const std::string& buffer_name,
           C::BaseType elem_type) -> Ptr<A::MMA> {
          auto fill_expr = A::Make<A::FloatLiteral>(pyloc(), val);
          Ptr<A::Expr> buffer = buffer_name.empty()
                            ? nullptr
                            : A::MakeIdExpr(pyloc(), buffer_name);
          bool is_decl = (buffer == nullptr);
          auto op = std::make_shared<A::MMAOperation>(
              buffer, std::static_pointer_cast<A::Expr>(
                  A::Make<A::Expr>(pyloc(),
                      static_cast<Ptr<A::Node>>(fill_expr))),
              is_decl, elem_type);
          return A::Make<A::MMA>(pyloc(), op);
        },
        py::arg("value") = 0.0,
        py::arg("buffer") = "",
        py::arg("elem_type") = C::BaseType::F32,
        "Create MMA fill: mc = mma.fill 0.0 or mma.fill mc, 0.0f.");

  m.def("make_mma_load",
        [](const Ptr<A::ChunkAt>& expr, int swizzle) -> Ptr<A::MMA> {
          C::SwizMode sm = C::SwizMode::NONE;
          if (swizzle == 32) sm = C::SwizMode::B32;
          else if (swizzle == 64) sm = C::SwizMode::B64;
          else if (swizzle == 128) sm = C::SwizMode::B128;
          auto op = std::make_shared<A::MMAOperation>(
              expr, nullptr, false, sm, false);
          return A::Make<A::MMA>(pyloc(), op);
        },
        py::arg("expr"), py::arg("swizzle") = 0,
        "Create MMA load: ma = mma.load data.chunkat(i, j).");

  m.def("make_mma_exec",
        [](const std::string& method,
           const Ptr<A::Expr>& acc,
           const Ptr<A::Expr>& lhs,
           const Ptr<A::Expr>& rhs) -> Ptr<A::MMA> {
          A::MMAOperation::ExecMethod em;
          if (method == "row.col" || method == "ROW_COL")
            em = A::MMAOperation::ROW_COL;
          else if (method == "row.row" || method == "ROW_ROW")
            em = A::MMAOperation::ROW_ROW;
          else if (method == "col.col" || method == "COL_COL")
            em = A::MMAOperation::COL_COL;
          else if (method == "col.row" || method == "COL_ROW")
            em = A::MMAOperation::COL_ROW;
          else
            throw std::runtime_error("Unknown MMA method: " + method);
          auto op = std::make_shared<A::MMAOperation>(em, acc, lhs, rhs);
          return A::Make<A::MMA>(pyloc(), op);
        },
        py::arg("method"), py::arg("acc"), py::arg("lhs"), py::arg("rhs"),
        "Create MMA exec: mma.row.col mc, ma, mb.");

  m.def("make_mma_store",
        [](const Ptr<A::Expr>& buffer,
           const Ptr<A::ChunkAt>& target,
           bool transpose) -> Ptr<A::MMA> {
          auto op = std::make_shared<A::MMAOperation>(buffer, target, transpose);
          return A::Make<A::MMA>(pyloc(), op);
        },
        py::arg("buffer"), py::arg("target"),
        py::arg("transpose") = false,
        "Create MMA store: mma.store mc, output.chunkat(m, n).");

  m.def("make_mma_store_var",
        [](const Ptr<A::Expr>& buffer,
           const std::string& target_name,
           bool transpose) -> Ptr<A::MMA> {
          auto id = A::Make<A::Identifier>(pyloc(), target_name);
          auto mv = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          auto ca = A::Make<A::ChunkAt>(pyloc(), id, mv);
          auto op = std::make_shared<A::MMAOperation>(buffer, ca, transpose);
          return A::Make<A::MMA>(pyloc(), op);
        },
        py::arg("buffer"), py::arg("target_name"),
        py::arg("transpose") = false,
        "Create MMA store to a named variable (no chunkat indices).");

  // ---- Call (builtins: __sqrt, __pow, etc.) ---------------------

  py::class_<A::Call, A::Node, Ptr<A::Call>>(m, "Call");

  m.def("make_call_expr",
        [](const std::string& func_name,
           const std::vector<Ptr<A::Node>>& args) -> Ptr<A::Expr> {
          auto id = A::Make<A::Identifier>(pyloc(), func_name);
          auto mv = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          for (auto& a : args) mv->Append(a);
          auto call = A::Make<A::Call>(
              pyloc(), id, mv,
              A::Call::BIF | A::Call::ARITH | A::Call::EXPR);
          return A::Make<A::Expr>(
              pyloc(), static_cast<Ptr<A::Node>>(call));
        },
        py::arg("func_name"), py::arg("args"),
        "Create a builtin function call expression: __sqrt(x), __pow(x, y).");

  m.def("make_println",
        [](const std::vector<Ptr<A::Node>>& args) -> Ptr<A::Call> {
          auto id = A::Make<A::Identifier>(pyloc(), "println");
          auto mv = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          for (auto& a : args) mv->Append(a);
          return A::Make<A::Call>(pyloc(), id, mv, A::Call::BIF);
        },
        py::arg("args"),
        "Create a println statement: println(a, b, ...).");

  // ---- IfElseBlock --------------------------------------------

  py::class_<A::IfElseBlock, A::Block, Ptr<A::IfElseBlock>>(m, "IfElseBlock");

  m.def("make_if",
        [](const Ptr<A::Expr>& cond,
           const Ptr<A::MultiNodes>& if_body,
           const Ptr<A::MultiNodes>& else_body) -> Ptr<A::IfElseBlock> {
          return A::Make<A::IfElseBlock>(pyloc(), cond, if_body, else_body);
        },
        py::arg("cond"), py::arg("if_body"),
        py::arg("else_body") = nullptr,
        "Create an if/else block.");

  // ---- Rotate (swap) -----------------------------------------

  py::class_<A::Rotate, A::Node, Ptr<A::Rotate>>(m, "Rotate");

  m.def("make_swap",
        [](const Ptr<A::Node>& a, const Ptr<A::Node>& b) -> Ptr<A::Rotate> {
          auto mv = A::Make<A::MultiValues>(pyloc(), std::string(", "));
          mv->Append(a);
          mv->Append(b);
          return A::Make<A::Rotate>(pyloc(), mv);
        },
        py::arg("a"), py::arg("b"),
        "Create a Swap (Rotate) node: swap(a, b).");

  // ---- FunctionDecl / ChoreoFunction ------------------------

  py::class_<A::FunctionDecl, A::Node, Ptr<A::FunctionDecl>>(m, "FunctionDecl");

  py::class_<A::ChoreoFunction, A::Block, Ptr<A::ChoreoFunction>>(
      m, "ChoreoFunction")
      .def_readwrite("name", &A::ChoreoFunction::name)
      .def("get_body", [](A::ChoreoFunction& self) -> Ptr<A::MultiNodes> {
        return self.stmts;
      });

  m.def("make_function",
        [](const std::string& name,
           const Ptr<A::DataType>& ret_type,
           const Ptr<A::ParamList>& params,
           const Ptr<A::MultiNodes>& body) -> Ptr<A::ChoreoFunction> {
          auto func = A::Make<A::ChoreoFunction>(pyloc());
          func->name = name;
          func->f_decl.name = name;
          func->f_decl.ret_type = ret_type;
          func->f_decl.params = params;
          func->stmts = body ? body : A::Make<A::MultiNodes>(pyloc());
          return func;
        },
        py::arg("name"), py::arg("ret_type"), py::arg("params"),
        py::arg("body") = nullptr,
        "Create a ChoreoFunction.");

  // ---- Program ----------------------------------------------

  py::class_<A::Program, A::Block, Ptr<A::Program>>(m, "Program")
      .def(py::init([]() { return A::Make<A::Program>(pyloc()); }))
      .def("append", [](A::Program& self, const Ptr<A::Node>& node) {
        self.stmts->Append(node);
      })
      .def("print_ast", [](const A::Program& self) {
        std::ostringstream os;
        self.Print(os);
        return os.str();
      })
      .def("count", [](const A::Program& self) {
        return self.stmts->Count();
      });

  // ---- CppSourceCode ----------------------------------------

  py::class_<A::CppSourceCode, A::Node, Ptr<A::CppSourceCode>>(
      m, "CppSourceCode")
      .def(py::init([](const std::string& code) {
        return A::Make<A::CppSourceCode>(
            pyloc(), code, A::CppSourceCode::Host);
      }), py::arg("code"));

  // ---- Pipeline & Compilation Context -----------------------

  m.def("setup_target",
        [](const std::string& target_name, const std::string& arch) {
          auto tgt = C::TargetRegistry::Create(target_name);
          if (!tgt)
            throw std::runtime_error("Unknown target: " + target_name);
          C::CCtx().SetTarget(std::move(tgt));
          if (!arch.empty()) C::CCtx().AddArch(arch);
        },
        py::arg("target") = "cute", py::arg("arch") = "",
        "Set the compilation target and architecture.");

  m.def("set_no_codegen",
        [](bool v) { C::CCtx().SetNoCodegen(v); },
        py::arg("value") = true);

  m.def("set_output_kind",
        [](const std::string& kind) {
          if (kind == "source")
            C::CCtx().SetOutputKind(C::OutputKind::TargetSourceCode);
        },
        py::arg("kind") = "source");

  m.def("run_pipeline",
        [](A::Program& prog) -> std::pair<bool, int> {
          auto& pl = C::ASTPipeline::Get().PlanAllRoutines();
          bool ok = pl.RunOnProgram(prog);
          return {ok, ok ? 0 : pl.Status()};
        },
        py::arg("program"),
        "Run the full compiler pipeline. Returns (success, status).");

  m.def("write_ast_to_file",
        [](const A::Program& prog, const std::string& path) {
          std::ofstream ofs(path);
          if (!ofs)
            throw std::runtime_error("Cannot write to: " + path);
          prog.Print(ofs);
          ofs.close();
        },
        py::arg("program"), py::arg("path"),
        "Write the AST representation to a file.");

  m.def("run_semantic_only",
        [](A::Program& prog) -> std::pair<bool, int> {
          C::CCtx().SetNoCodegen(true);
          auto& pl = C::ASTPipeline::Get().PlanSemanticRoutine();
          bool ok = pl.RunOnProgram(prog);
          return {ok, ok ? 0 : pl.Status()};
        },
        py::arg("program"),
        "Run semantic analysis only. Returns (success, status).");

  m.def("dump_ast",
        [](const A::Program& prog) -> std::string {
          std::ostringstream os;
          prog.Print(os);
          return os.str();
        },
        py::arg("program"),
        "Print the AST tree as a string.");

  m.def("list_targets",
        []() -> std::vector<std::string> {
          auto targets = C::TargetRegistry::List();
          std::vector<std::string> names;
          for (auto& t : targets) names.push_back(t.name);
          return names;
        },
        "List available compilation targets.");

  m.def("version",
        []() { return C::SDK::Version(); },
        "Return the CroqTile SDK version string.");
}
