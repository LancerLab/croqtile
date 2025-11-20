#ifndef __CHOREO_VISUALIZE_DMA_HPP__
#define __CHOREO_VISUALIZE_DMA_HPP__

// This apply the type check and symbol table generation

#include <set>

#include "visitor.hpp"

namespace Choreo {

constexpr int label_distance = 20;
constexpr double axis_scale[] = {1.3, 1.5, 3.2};

std::vector<std::string> pov_colors = {
    "Orange", "Pink",   "Magenta", "Gold",   "Cyan",  "Brown", "Blue", "Green",
    "Red",    "Yellow", "Violet",  "Silver", "Black", "White"
    // Add more colors as needed
};

struct Polyhedron {
  std::vector<int> points;
  std::vector<size_t> sizes;
  std::string color;

  Polyhedron(const std::vector<int>& p, const std::vector<size_t>& s,
             const std::string& c = "Red")
      : points(p), sizes(s), color(c) {}
};

struct ShapePolyhedron {
private:
  bool debug = false;

  std::vector<Polyhedron> polyhedrons;
  std::vector<int> minimums;
  std::vector<int> maximums;
  std::vector<std::string> axes_labels;

  std::string expr;

  int text_scale = 1;

public:
  ShapePolyhedron(const std::string& e, bool d = false) : debug(d), expr(e) {}

  int MaxPointValue() {
    int max = std::numeric_limits<int>::min();
    for (auto m : maximums) max = (max > m) ? max : m;
    return max;
  }

  std::vector<int> MaxPoints() { return maximums; }
  std::vector<int> MinPoints() { return maximums; }

  void SetAxesLabels(const std::vector<std::string>& axes) {
    axes_labels = axes;
  }

  void SetTextScale(int ts) { text_scale = ts; }

  void Create(std::vector<int>& p, std::vector<size_t>& s, std::vector<int>& b,
              std::set<int>& pb, size_t index, std::vector<int>& currentPos,
              int colorIndex) {
    if (index == b.size()) {
      polyhedrons.emplace_back(currentPos, s, pov_colors[colorIndex]);
      // Here you can do something with the created polyhedron, like adding it
      // to a list
      if (debug) {
        dbgs() << "Created Polyhedron with position: ";
        for (int i : currentPos) { dbgs() << i << " "; }
        dbgs() << "(";
        for (auto i : s) dbgs() << i << " ";
        dbgs() << ")";
        dbgs() << std::endl;
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
      maximums[index] =
          std::max(maximums[index], currentPos[index] + (int)s[index]);
      minimums[index] = std::min(minimums[index], currentPos[index]);
    }
  }

  // Create multiple polyhedrons from position 'p', each with size 's', and
  // repeating multi-dimensional with bound 'b'
  void Create(std::vector<int>& p, std::vector<size_t>& s, std::vector<int>& b,
              std::set<int>& pb) {
    std::vector<int> currentPos(b.size(), 0);
    maximums.resize(b.size(), std::numeric_limits<int>::min());
    minimums.resize(b.size(), std::numeric_limits<int>::max());
    Create(p, s, b, pb, 0, currentPos, 0);
  }

  void RenderToPov(std::ostream& pov) {
    if (debug) {
      dbgs() << "min: [" << minimums[0] << ", " << minimums[1] << ", "
             << minimums[2] << "]\n";
      dbgs() << "max: [" << maximums[0] << ", " << maximums[1] << ", "
             << maximums[2] << "]\n";
    }
    for (const auto& polyhedron : polyhedrons) {
      int x2 = polyhedron.points[0] + polyhedron.sizes[0] * 0.9;
      int y2 = polyhedron.points[1] + polyhedron.sizes[1] * 0.9;
      int z2 = polyhedron.points[2] + polyhedron.sizes[2] * 0.9;

      // Output the box
      pov << "box {\n";
      pov << "  <" << polyhedron.points[0] << ", " << polyhedron.points[1]
          << ", " << polyhedron.points[2] << ">, <" << x2 << ", " << y2 << ", "
          << z2 << "> // Polyhedron Point\n";
      pov << "  pigment { " << polyhedron.color << " }\n";
      pov << "  finish {\n";
      pov << "    ambient 0.4\n";
      pov << "    diffuse 0.85\n";
      pov << "    specular 0.9\n";
      pov << "    roughness 0.001\n";
      pov << "    reflection { 0.5 metallic }\n";
      pov << "  }\n";
      pov << "}\n";
    }

    std::vector<double> axis_max{
        (maximums[0] - minimums[0]) * axis_scale[0] + minimums[0],
        (maximums[1] - minimums[1]) * axis_scale[1] + minimums[1],
        (maximums[2] - minimums[2]) * axis_scale[2] + minimums[2]};

    // handle corner cases to avoid povray fails
    for (size_t i = 0; i < axis_max.size(); ++i)
      if (axis_max[i] < 10) axis_max[i] = 30;

    auto CreateAxis = [&axis_max, &pov, this](const std::string& name,
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
      pov << "    internal 1, \"" << name << "\", 0.05, 0\n";
      int font_scale = (maximums[index] - minimums[index]) / 5;
      if (font_scale > 1) pov << "    scale " << font_scale << "\n";
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
      pov << "    pigment {color Blue}\n";
      pov << "  }\n";
      pov << "}\n";
    };

    for (size_t i = 0; i < axes_labels.size(); ++i)
      CreateAxis(axes_labels[i], i);

    // now render the expression
    pov << "  text {\n";
    pov << "    internal 1, \"" << expr << "\", 0.05, 0\n";
    pov << "    scale " << text_scale << "\n";
    pov << "    translate <" << minimums[0] << ", "
        << (maximums[1] - minimums[1]) + minimums[1] + text_scale << ", "
        << (minimums[2] + maximums[2]) / 2 << ">\n";
    pov << "    pigment {color Brown}\n";
    pov << "  }\n";
  }
};

struct DMAPolyhedron {
  std::unique_ptr<ShapePolyhedron> from;
  std::unique_ptr<ShapePolyhedron> to;

  std::string futname; // future name
  std::string expr;

public:
  DMAPolyhedron(const std::string& n, const std::string& e)
      : futname(n), expr(e) {}

public:
  void GeneratePov() {
    auto filename = futname + ".pov";
    std::replace(filename.begin(), filename.end(), ':', '_');
    std::ofstream pov{filename};

    // control the camera distance
    int camera = std::numeric_limits<int>::min();
    camera = (camera > to->MaxPointValue()) ? camera : to->MaxPointValue();
    int max_y = (from->MaxPoints()[1] > to->MaxPoints()[1])
                    ? from->MaxPoints()[1]
                    : to->MaxPoints()[1];

    assert(camera > 10 && "unreasonable camera value.");

    pov << "// POV-Ray Scene Description Language File\n";
    pov << "#version 3.7;\n";
    pov << "#include \"colors.inc\"\n";
    pov << "#declare Camera_Distance = " << camera << ";\n";
    pov << "#declare Axis_Radius = 0.5;\n";
    pov << "#declare Arrowhead_Length = 5;\n";
    pov << "#declare Arrowhead_Radius = 2;\n";
    pov << "#declare Label_Distance = " << label_distance << ";\n";
    pov << "global_settings { assumed_gamma 1.0 }\n";
    pov << "background { color rgb <0.8, 0.8, 0.8> }\n";
    pov << "camera {\n";
    pov << "  location <Camera_Distance, (Camera_Distance + " << max_y
        << ")/2, -Camera_Distance>\n";
    pov << "  look_at <Camera_Distance/2, " << max_y / 2 << ", 0>\n";
    pov << "}\n";
    pov << "light_source { <Camera_Distance*1.2, Camera_Distance*1.2, "
           "-Camera_Distance*0.2>, color White spotlight point_at <0, 0, 0> "
           "}\n";

    int shapes_length = to->MaxPoints()[0] - from->MinPoints()[0];
    int text_scale = shapes_length / 10;
    if (text_scale < 1) text_scale = 1;

    from->SetTextScale(text_scale);
    to->SetTextScale(text_scale);

    from->RenderToPov(pov);
    to->RenderToPov(pov);

    // now render the expression
    pov << "  text {\n";
    pov << "    internal 1, \"" << expr << "\", 0.05, 0\n";
    pov << "    scale " << text_scale << "\n";
    pov << "    translate <0, -" << text_scale * 1.5 << ", 0>\n";
    pov << "    pigment {color Brown}\n";
    pov << "  }\n";

    dbgs() << "Generated POV: " << filename << "\n";
  }
};

struct Visualizer : public VisitorWithSymTab {
private:
  std::vector<std::unique_ptr<DMAPolyhedron>> dma_polyhedrons;

private:
  int parallel_factor = 1;

private:
  int start_x = 0;
  int start_y = 0;

public:
  Visualizer()
      : VisitorWithSymTab("visual"), parallel_factor(1), start_x(0),
        start_y(0) {}
  ~Visualizer() {}

  bool Visit(AST::ParallelBy& pb) override {
    if (auto b = VIInt(pb.BoundValue()))
      parallel_factor *= *b;
    else
      choreo_unreachable("symbolic bound is not supported in visulize yet.");
    return true;
  }
  bool Visit(AST::WhereBind&) override { return true; }
  bool Visit(AST::WithIn&) override { return true; }
  bool Visit(AST::WithBlock&) override { return true; }
  bool Visit(AST::Memory&) override { return true; }
  bool Visit(AST::SpanAs&) override { return true; }

  bool Visit(AST::DMA& n) override {
    start_x = 0;
    auto dp = std::make_unique<DMAPolyhedron>(InScopeName(n.future),
                                              n.SourceString());
    auto caf = cast<AST::ChunkAt>(n.from);
    if (auto sp = HandleChunkAt(*caf, parallel_factor))
      dp->from = std::move(sp);

    if (auto cat = dyn_cast<AST::ChunkAt>(n.to)) {
      if (auto sp = HandleChunkAt(*cat, parallel_factor))
        dp->to = std::move(sp);
    } else {
      auto m = dyn_cast<AST::Memory>(n.to);
      if (auto sp = HandleImplicit(
              *m, cast<FutureType>(n.GetType())->GetShape(), parallel_factor))
        dp->to = std::move(sp);
    }
    dma_polyhedrons.emplace_back(std::move(dp));
    return true;
  }

public:
  bool BeforeVisitImpl(AST::Node&) override { return true; }
  bool AfterVisitImpl(AST::Node& n) override {
    if (auto pb = dyn_cast<AST::ParallelBy>(&n)) {
      if (auto b = VIInt(pb->BoundValue()))
        parallel_factor /= *b;
      else
        choreo_unreachable("symbolic bound is not supported in visulize yet.");
      return true;
    }
    if (!isa<AST::Program>(&n)) return true;

    for (auto& dp : dma_polyhedrons) dp->GeneratePov();

    return true;
  }

private:
  constexpr static size_t p_dim = 1;

  std::unique_ptr<ShapePolyhedron> HandleChunkAt(AST::ChunkAt& ca,
                                                 int parallel_count = 1) {
    std::string data_name = ca.data->name;
    std::string expr = AST::STR(ca);
    auto pdata_type = GetSymbolType(data_name);
    Shape data_shape = cast<SpannedType>(pdata_type)->GetShape();

    std::vector<size_t> data_sizes;
    if (auto ilist = data_shape.PosValList())
      data_sizes = *ilist;
    else {
      Warning(ca.LOC(),
              "unable to handle '" + ca.data->name + "' with runtime shape.");
      return nullptr;
    }

    std::vector<int> positions{start_x, start_y, 0};
    start_x += data_sizes[0] * axis_scale[0] + label_distance + 100;

    if (ca.HasOperation() && ca.AllOperations().size() == 1) {
      // use whole data as a single chunk

      // dimension 1 is repeated parallel_count times
      std::vector<int> cmpt_bounds(data_shape.Rank(), 1);
      cmpt_bounds[p_dim] = parallel_count;
      std::set<int> parallel_bounds;
      parallel_bounds.insert(p_dim);

      auto sp = std::make_unique<ShapePolyhedron>(expr, debug_visit);
      sp->Create(positions, data_sizes, cmpt_bounds, parallel_bounds);
      return sp;
    }

    std::vector<int>
        cmpt_bounds; // specify the repeating count for each dimension
    std::vector<std::string> bv_names; // the name of the bounded variable
    std::set<int>
        parallel_bounds; // specify which dimension is executed in parallel

    if (ca.HasOperation() && ca.AllOperations().size() == 1) {
      for (auto pos : ca.AllOperations()[0]->GetIndices()) {
        auto id = dyn_cast<AST::Identifier>(pos);
        assert(id && "node other than identifier is not handled.");
        auto ty = GetSymbolType(id->name);
        if (auto bpvs = dyn_cast<BoundedITupleType>(ty)) {
          bool parallel = bpvs->HasNote("pv"); // TODO: fix this
          auto vlist = bpvs->GetUpperBounds().Value();
          for (size_t i = 0; i < vlist.size(); ++i) {
            if (auto pint = VIInt(vlist[i])) {
              cmpt_bounds.push_back(*pint);
            } else {
              Warning(ca.LOC(), "unable to handle '" + PSTR(vlist[i]) +
                                    "' (with runtime value).");
              return nullptr;
            }
            if (vlist.size() == 1)
              bv_names.push_back(id->name);
            else
              bv_names.push_back(id->name + "(" + std::to_string(i) + ")");
            if (parallel) parallel_bounds.insert(cmpt_bounds.size() - 1);
          }
        } else {
          dbgs() << STR(*ty) << " is not expected.\n";
          choreo_unreachable("unable to handle the type.");
        }
      }
    }
    // this calculate the tiled blocks
    Shape block_shape = cast<SpannedType>(ca.GetType())->GetShape();

    assert(block_shape.Rank() > 1 && "unexpected shape dimensions.");
    assert(block_shape.Rank() == cmpt_bounds.size() &&
           "inconsistency between shape cmpt_bounds and tiling");

    if (block_shape.Rank() > 3) {
      Warning(ca.LOC(), "unable to visualize tensors with high dimensions.");
      return nullptr;
    }

    auto psizes = block_shape.PosValList();
    if (!psizes) {
      Warning(ca.LOC(), "unable to visualize tensors.");
      return nullptr;
    }

    if (debug_visit) {
      dbgs() << "data shape: " << STR(data_shape) << "\n";
      dbgs() << "block shape: " << STR(block_shape) << "\n";
      dbgs() << "tiling factors: [ ";
      for (auto b : cmpt_bounds) dbgs() << b << " ";
      dbgs() << "]\n";
    }

    auto sp = std::make_unique<ShapePolyhedron>(expr, debug_visit);
    sp->Create(positions, *psizes, cmpt_bounds, parallel_bounds);
    sp->SetAxesLabels(bv_names);
    return sp;
  }

  std::unique_ptr<ShapePolyhedron> HandleImplicit(AST::Memory& s, Shape shape,
                                                  int parallel_count = 1) {
    std::string mem = STR(s.st);

    std::vector<size_t> sizes;
    if (auto ilist = shape.PosValList())
      sizes = *ilist;
    else {
      Warning(s.LOC(), "unable to handle '" + mem + "' with runtime shape.");
      return nullptr;
    }

    std::vector<int> positions{start_x, start_y, 0};
    start_x += sizes[0] * axis_scale[0] + label_distance + 100;

    // the dimension representing parallelism is repeated parallel_count times
    std::vector<int> cmpt_bounds(shape.Rank(), 1);
    cmpt_bounds[p_dim] = parallel_count;
    std::set<int> parallel_bounds;
    parallel_bounds.insert(p_dim);

    auto sp = std::make_unique<ShapePolyhedron>(mem, debug_visit);
    sp->Create(positions, sizes, cmpt_bounds, parallel_bounds);
    return sp;
  }
};

} // end namespace Choreo

#endif // __CHOREO_VISUALIZE_DMA_HPP__
