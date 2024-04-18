#ifndef __CHOREO_VISUALIZE_DMA_HPP__
#define __CHOREO_VISUALIZE_DMA_HPP__

// This apply the type check and symbol table generation

#include <set>

#include "visitor.hpp"

namespace Choreo {

std::vector<std::string> pov_colors = {
    "Orange", "Pink",   "Magenta", "Gold",   "Cyan",  "Brown", "Blue", "Green",
    "Red",    "Yellow", "Violet",  "Silver", "Black", "White"
    // Add more colors as needed
};

struct Polyhedron {
  std::vector<int> points;
  std::vector<size_t> sizes;
  std::string color;

  Polyhedron(const std::vector<int> &p, const std::vector<size_t> &s,
             const std::string &c = "Red")
      : points(p), sizes(s), color(c) {}
};

struct ShapePolyhedron {
 private:
  std::ostream &os;
  bool debug = false;

  std::vector<Polyhedron> polyhedrons;
  std::vector<int> minimums;
  std::vector<int> maximums;

 public:
  ShapePolyhedron(std::ostream &o = std::cout, bool d = false)
      : os(o), debug(d) {}

  void Create(std::vector<int> &p, std::vector<size_t> &s, std::vector<int> &b,
              std::set<int> &pb, size_t index, std::vector<int> &currentPos,
              int colorIndex) {
    if (index == b.size()) {
      polyhedrons.emplace_back(currentPos, s, pov_colors[colorIndex]);
      // Here you can do something with the created polyhedron, like adding it
      // to a list
      if (debug) {
        os << "Created Polyhedron with position: ";
        for (int i : currentPos) {
          os << i << " ";
        }
        os << "(";
        for (auto i : s) os << i << " ";
        os << ")";
        os << std::endl;
      }
      return;
    }

    bool parallel_bound = pb.count(index);
    for (int i = 0; i < b[index]; ++i) {
      currentPos[index] = p[index] + i * s[index];
      if (parallel_bound)
        Create(p, s, b, pb, index + 1, currentPos, i % pov_colors.size());
      else
        Create(p, s, b, pb, index + 1, currentPos, colorIndex);
      maximums[index] = std::max(maximums[index], currentPos[index]);
      minimums[index] = std::min(minimums[index], currentPos[index]);
    }
  }

  // Create multiple polyhedrons from position 'p', each with size 's', and
  // repeating multi-dimensional with bound 'b'
  void Create(std::vector<int> &p, std::vector<size_t> &s, std::vector<int> &b,
              std::set<int> &pb) {
    std::vector<int> currentPos(b.size(), 0);
    maximums.resize(b.size(), std::numeric_limits<int>::min());
    minimums.resize(b.size(), std::numeric_limits<int>::max());
    Create(p, s, b, pb, 0, currentPos, 0);
  }

  void RenderPov(std::ostream &pov) {
    constexpr double scale = 1.5;
    for (const auto &polyhedron : polyhedrons) {
      int x2 = polyhedron.points[0] + polyhedron.sizes[0] * 0.9;
      int y2 = polyhedron.points[1] + polyhedron.sizes[1] * 0.9;
      int z2 = polyhedron.points[2] + polyhedron.sizes[2] * 0.9;

      // Output the box
      pov << "box {\n";
      pov << "<" << polyhedron.points[0] << ", " << polyhedron.points[1] << ", "
          << polyhedron.points[2] << ">, <" << x2 << ", " << y2 << ", " << z2
          << "> // Polyhedron Point\n";
      pov << "  pigment { " << polyhedron.color << " }\n";
      pov << "  finish {\n";
      pov << "    ambient 0.4\n";
      pov << "    diffuse 0.85\n";
      pov << "    specular 0.05\n";
      pov << "    roughness 0.3\n";
      pov << "  }\n";
      pov << "}\n";
    }

    std::vector<double> axis_max{
        (maximums[0] - minimums[0]) * scale + minimums[0],
        (maximums[1] - minimums[1]) * scale + minimums[1],
        (maximums[2] - minimums[2]) * scale + minimums[2]};

    auto CreateAxis = [&axis_max, &pov, this](const std::string &name,
                                              size_t index) {
      pov << "// " << name << " Axis\n";
      pov << "union {\n";
      pov << "  cylinder {\n";
      pov << "    <" << minimums[0] << ", " << minimums[1] << ", "
          << minimums[2] << ">,\n";
      pov << "    <";
      for (size_t i = 0; i < axis_max.size() - 1; ++i)
        pov << ((i == index) ? std::to_string(axis_max[index])
                             : std::to_string(minimums[i]))
            << ", ";
      pov << ((index == axis_max.size() - 1)
                  ? std::to_string(axis_max[index])
                  : std::to_string(minimums[axis_max.size() - 1]))
          << ">,\n";
      pov << "    Axis_Radius\n";
      pov << "    pigment { Blue }\n";
      pov << "    finish {\n";
      pov << "      ambient 1\n";
      pov << "      diffuse 0.9\n";
      pov << "      specular 0.2\n";
      pov << "      roughness 0.1\n";
      pov << "    }\n";
      pov << "  }\n";
      pov << "  cone {\n";
      pov << "    <";
      for (size_t i = 0; i < axis_max.size() - 1; ++i)
        pov << ((i == index) ? axis_max[index] : minimums[i]) << ", ";
      pov << ((index == axis_max.size() - 1) ? axis_max[index]
                                             : minimums[axis_max.size() - 1])
          << ">, Axis_Radius\n";
      pov << "    <";
      for (size_t i = 0; i < axis_max.size() - 1; ++i)
        pov << ((i == index)
                    ? std::to_string(axis_max[index]) + " - Arrowhead_Length"
                    : std::to_string(minimums[i]))
            << ", ";
      pov << ((index == axis_max.size() - 1)
                  ? std::to_string(axis_max[index]) + " - Arrowhead_Length"
                  : std::to_string(minimums[axis_max.size() - 1]))
          << ">, Arrowhead_Radius * Axis_Radius\n";
      pov << "    pigment { Blue }\n";
      pov << "    finish { ambient 1 }\n";
      pov << "  }\n";
      pov << "  text {\n";
      pov << "    internal 1, \"" << name << "\", 0.25, 0\n";
      pov << "    scale 80\n";
      pov << "    translate <";
      for (size_t i = 0; i < axis_max.size() - 1; ++i)
        pov << ((i == index)
                    ? std::to_string(axis_max[index]) + " + Label_Distance"
                    : std::to_string(minimums[i]))
            << ", ";
      pov << ((index == axis_max.size() - 1)
                  ? std::to_string(axis_max[index]) + " + Label_Distance"
                  : std::to_string(minimums[axis_max.size() - 1]))
          << ">\n";
      // pov << "    " << direction << ",\n";
      pov << "    pigment {color Blue}\n";
      pov << "  }\n";
      pov << "}\n";
    };

    CreateAxis("X", 0);
    CreateAxis("Y", 1);
    CreateAxis("Z", 2);
  }
};

struct Visualizer : public VisitorWithSymTab {
 private:
  std::ostream &os;
  std::vector<std::unique_ptr<ShapePolyhedron>> shape_polyhedrons;
  bool debug = false;

  void GeneratePov(std::ostream &pov) {
    pov << "// POV-Ray Scene Description Language File\n";
    pov << "#version 3.7;\n";
    pov << "#include \"colors.inc\"\n";
    pov << "#declare Camera_Distance = 2000.0;\n";
    pov << "#declare Axis_Radius = 3;\n";
    pov << "#declare Arrowhead_Length = 10;\n";
    pov << "#declare Arrowhead_Radius = 3;\n";
    pov << "#declare Label_Distance = 20;\n";
    pov << "global_settings { assumed_gamma 1.0 }\n";
    pov << "background { color Gray50 }\n";
    pov << "camera {\n";
    pov << "  location <Camera_Distance, Camera_Distance, -Camera_Distance>\n";
    pov << "  look_at 0\n";
    pov << "}\n";
    pov << "light_source { <Camera_Distance+10, Camera_Distance+10, "
           "-Camera_Distance-10>, color White }\n";
    for (auto &polyhedron : shape_polyhedrons) polyhedron->RenderPov(pov);
  }

 public:
  Visualizer(const ptr<SymbolTable> s_tab, std::ostream &o = std::cout,
             bool d = false)
      : VisitorWithSymTab(s_tab), os(o), debug(d) {}
  ~Visualizer() {}

  // derived class must call this to incorporate with symbol table

  bool Visit(AST::MultiNodes &) override { return true; }
  bool Visit(AST::MultiValues &) override { return true; }
  bool Visit(AST::IntLiteral &) override { return true; }
  bool Visit(AST::Expr &) override { return true; }
  bool Visit(AST::MultiDimSpans &) override { return true; }
  bool Visit(AST::NamedTypeDecl &) override { return true; }
  bool Visit(AST::NamedVariableDecl &) override { return true; }
  bool Visit(AST::IntTuple &) override { return true; }
  bool Visit(AST::Assignment &) override { return true; }
  bool Visit(AST::IntIndex &) override { return true; }
  bool Visit(AST::DataType &) override { return true; }
  bool Visit(AST::Identifier &) override { return true; }
  bool Visit(AST::Parameter &) override { return true; }
  bool Visit(AST::ParamList &) override { return true; }
  bool Visit(AST::ParallelBy &) override { return true; }
  bool Visit(AST::RequireBind &) override { return true; }
  bool Visit(AST::WithIn &) override { return true; }
  bool Visit(AST::WithBlock &) override { return true; }
  bool Visit(AST::Memory &) override { return true; }

  bool Visit(AST::DMA &n) override {
    auto *from = n.from.get();
    // auto *to = n.to.get();

    int last_x = 0;
    if (auto ca = dyn_cast<AST::ChunkAt>(from)) {
      if (ca->positions) {
        std::vector<int> bounds;
        std::set<int> parallel_bounds;  // map bound to colors
        for (auto pos : ca->positions->values) {
          auto id = dyn_cast<AST::Identifier>(pos);
          assert(id && "unhandled value.");
          auto ty = GetSymbolType(id->name);
          if (auto bivs = dyn_cast<BoundedITupleType>(ty)) {
            bool parallel = (!bivs->GetNote().empty());
            for (auto b : bivs->GetBounds().Value()) {
              if (auto pint = dyn_cast<int>(&b)) {
                bounds.push_back(*pint);
              } else {
                Warning(n.LOC(), "unable to handle '" + *cast<ValueExpr>(&b) +
                                     "' (with runtime value).");
                return false;
              }
              if (parallel) parallel_bounds.insert(bounds.size() - 1);
            }
          } else if (auto biv = dyn_cast<BoundedIntegerType>(ty)) {
            assert(false && "not expected.");
            if (auto pint = dyn_cast<int>(&biv->bound)) {
              bounds.push_back(*pint);
            } else {
              Warning(n.LOC(), "unable to handle '" +
                                   *cast<ValueExpr>(&biv->bound) +
                                   "' (with runtime value).");
              return false;
            }
          } else {
            os << STR(*ty) << " is not expected.\n";
            choreo_unreachable("unable to handle the type.");
          }
        }
        Shape shape = cast<SpannedType>(ca->GetType())->GetShape();

        assert(shape.Dims() > 1 && "unexpected shape dimensions.");
        assert(shape.Dims() == bounds.size() &&
               "inconsistence between shape bounds and tiling");

        if (shape.Dims() > 3) {
          Warning(n.LOC(), "unable to visualize tensors with high dimensions.");
          return false;
        }

        auto pdata_type = GetSymbolType(ca->data->name);
        Shape data_shape = cast<SpannedType>(pdata_type)->GetShape();
        last_x += data_shape.NthInteger(0) + 100;

        if (debug) {
          os << "fullsize: " << STR(*pdata_type) << "\n";
          os << "shape dims: " << shape.Dims() << "\n";
        }
        std::vector<size_t> sizes;
        for (auto si : shape.Value()) {
          if (auto pint = dyn_cast<int>(&si))
            sizes.push_back(*pint);
          else {
            Warning(n.LOC(), "unable to handle '" + ca->data->name +
                                 "' with runtime shape.");
            return false;
          }
        }
        if (debug) {
          os << "bounds: [ ";
          for (auto b : bounds) os << b << " ";
          os << "]\nsize: [";
          for (auto s : sizes) os << s << " ";
          os << "]\n";
        }

        std::vector<int> positions(bounds.size(), 0);
        positions[0] = last_x;
        auto sp = std::make_unique<ShapePolyhedron>(os);
        sp->Create(positions, sizes, bounds, parallel_bounds);
        shape_polyhedrons.emplace_back(std::move(sp));
      }
    }
    return true;
  }

  bool Visit(AST::ChunkAt &) override { return true; }
  bool Visit(AST::Wait &) override { return true; }
  bool Visit(AST::Call &) override { return true; }
  bool Visit(AST::Return &) override { return true; }
  bool Visit(AST::ForeachBlock &) override { return true; }
  bool Visit(AST::FunctionDecl &) override { return true; }
  bool Visit(AST::ChoreoFunction &) override { return true; }
  bool Visit(AST::CppSourceCode &) override { return true; }
  bool Visit(AST::Program &) override { return true; }

 public:
  bool BeforeVisitImpl(AST::Node &) override { return true; }
  bool AfterVisitImpl(AST::Node &n) override {
    if (!isa<AST::Program>(&n)) return true;
    // render as .dot format
    GeneratePov(os);
    return true;
  }
};

}  // end namespace Choreo

#endif  // __CHOREO_VISUALIZE_DMA_HPP__
