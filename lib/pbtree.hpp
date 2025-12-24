#ifndef __CHOREO_PARALLEL_BY_TREE_HPP__
#define __CHOREO_PARALLEL_BY_TREE_HPP__

#include "ast.hpp"

namespace Choreo {

class PBTree {
public:
  // Constructor/Destructor
  PBTree() = default;
  ~PBTree() { Clear(); }

  // Core tree operations
  bool AddSingle(AST::ParallelBy*);
  bool AddChild(AST::ParallelBy*, AST::ParallelBy*);
  const std::vector<AST::ParallelBy*> GetChildren(AST::ParallelBy* node) const;
  AST::ParallelBy* GetParent(AST::ParallelBy* node) const;
  bool HasSiblings(AST::ParallelBy* node) const;
  const std::vector<AST::ParallelBy*> GetSiblings(AST::ParallelBy* node) const;
  void Clear();

  // Additional utility methods
  bool IsRoot(AST::ParallelBy* node) const;
  bool IsLeaf(AST::ParallelBy* node) const;
  bool IsEmpty() const;
  size_t GetSize() const;
  size_t GetDepth(AST::ParallelBy* node) const;
  size_t GetHeight(AST::ParallelBy* node) const;
  size_t GetRootCount() const;
  const std::vector<AST::ParallelBy*> GetAllNodes() const;

  // Tree traversal methods
  const std::vector<AST::ParallelBy*>
  GetDescendants(AST::ParallelBy* node) const;
  bool IsAncestor(AST::ParallelBy* ancestor, AST::ParallelBy* descendant) const;

  void Print(std::ostream& os) const;

private:
  // Internal data structures
  std::unordered_map<AST::ParallelBy*, std::vector<AST::ParallelBy*>>
      children_map_;
  std::unordered_map<AST::ParallelBy*, AST::ParallelBy*> parent_map_;
  std::unordered_set<AST::ParallelBy*> all_nodes_;

  // Helper methods
  bool ValidateNode(AST::ParallelBy* node) const;
  void RemoveFromMaps(AST::ParallelBy* node);
  void PrintRecursive(AST::ParallelBy* node, int depth, int max_depth,
                      std::ostream& os) const;
};

inline bool PBTree::AddSingle(AST::ParallelBy* pb) {
  if (!pb) choreo_unreachable("invalid null parallel-by.");
  all_nodes_.insert(pb);
  return true;
}

inline bool PBTree::AddChild(AST::ParallelBy* parent, AST::ParallelBy* child) {
  // Validate input parameters
  if (!parent || !child) { return false; }

  // Prevent adding a node as its own child
  if (parent == child) { return false; }

  // std::cout << "new p-c: ";  parent->InlinePrint(std::cout); std::cout << "
  // -- "; child->InlinePrint(std::cout); std::cout << "\n";
  //  Check if child already has a parent
  if (parent_map_.find(child) != parent_map_.end()) {
    // Child already has a parent, remove it from current parent's children
    AST::ParallelBy* current_parent = parent_map_[child];
    auto& current_children = children_map_[current_parent];
    current_children.erase(
        std::remove(current_children.begin(), current_children.end(), child),
        current_children.end());
  }

  // Add child to parent's children list
  children_map_[parent].push_back(child);

  // Set parent reference for child
  parent_map_[child] = parent;

  // Add both nodes to the all_nodes_ set
  all_nodes_.insert(parent);
  all_nodes_.insert(child);

  return true;
}

inline const std::vector<AST::ParallelBy*>
PBTree::GetChildren(AST::ParallelBy* node) const {
  if (!ValidateNode(node)) choreo_unreachable("invalid parallel-by.");

  if (!children_map_.count(node)) return {};

  return children_map_.at(node);
}

inline AST::ParallelBy* PBTree::GetParent(AST::ParallelBy* node) const {
  if (!ValidateNode(node)) choreo_unreachable("invalid parallel-by.");

  auto it = parent_map_.find(node);
  return (it != parent_map_.end()) ? it->second : nullptr;
}

inline bool PBTree::HasSiblings(AST::ParallelBy* node) const {
  return !GetSiblings(node).empty();
}

inline const std::vector<AST::ParallelBy*>
PBTree::GetSiblings(AST::ParallelBy* node) const {
  if (!ValidateNode(node)) { return {}; }

  AST::ParallelBy* parent = GetParent(node);
  if (!parent) {
    // Root node has no siblings
    return {};
  }

  std::vector<AST::ParallelBy*> siblings = GetChildren(parent);

  // Remove the node itself from the siblings list
  siblings.erase(std::remove(siblings.begin(), siblings.end(), node),
                 siblings.end());

  return siblings;
}

inline void PBTree::Clear() {
  children_map_.clear();
  parent_map_.clear();
  all_nodes_.clear();
}

inline bool PBTree::IsLeaf(AST::ParallelBy* node) const {
  if (!ValidateNode(node)) choreo_unreachable("invalid parallel-by.");

  return children_map_.find(node) == children_map_.end();
}

inline bool PBTree::IsRoot(AST::ParallelBy* node) const {
  if (!ValidateNode(node)) choreo_unreachable("invalid parallel-by.");

  return parent_map_.find(node) == parent_map_.end();
}

inline bool PBTree::IsEmpty() const { return all_nodes_.empty(); }

inline size_t PBTree::GetSize() const { return all_nodes_.size(); }

inline size_t PBTree::GetDepth(AST::ParallelBy* node) const {
  if (!ValidateNode(node)) choreo_unreachable("invalid parallel-by.");

  size_t depth = 0;
  AST::ParallelBy* current = node;

  while (current && !IsRoot(current)) {
    current = GetParent(current);
    depth++;
  }

  return depth;
}

inline size_t PBTree::GetHeight(AST::ParallelBy* node) const {
  if (!node) return 0;
  if (!ValidateNode(node)) choreo_unreachable("invalid parallel-by.");

  // Find the maximum depth among all children
  size_t max_child_height = 0;
  for (auto* child : GetChildren(node)) {
    size_t child_height = GetHeight(child);
    if (child_height > max_child_height) max_child_height = child_height;
  }

  // Add 1 for the edge to the deepest child
  return max_child_height + 1;
}

inline const std::vector<AST::ParallelBy*> PBTree::GetAllNodes() const {
  return std::vector<AST::ParallelBy*>(all_nodes_.begin(), all_nodes_.end());
}

inline const std::vector<AST::ParallelBy*>
PBTree::GetDescendants(AST::ParallelBy* node) const {
  std::vector<AST::ParallelBy*> descendants;

  if (!ValidateNode(node)) { return descendants; }

  // Get direct children
  std::vector<AST::ParallelBy*> children =
      GetChildren(const_cast<AST::ParallelBy*>(node));

  // Add direct children to descendants
  for (auto* child : children) {
    descendants.push_back(child);
    // Recursively get descendants of each child
    std::vector<AST::ParallelBy*> child_descendants = GetDescendants(child);
    descendants.insert(descendants.end(), child_descendants.begin(),
                       child_descendants.end());
  }

  return descendants;
}

inline bool PBTree::IsAncestor(AST::ParallelBy* ancestor,
                               AST::ParallelBy* descendant) const {
  if (!ValidateNode(ancestor) || !ValidateNode(descendant)) { return false; }

  if (ancestor == descendant) {
    return false; // A node is not considered its own ancestor
  }

  AST::ParallelBy* current = descendant;

  while (current) {
    if (current == ancestor) { return true; }
    current = GetParent(const_cast<AST::ParallelBy*>(current));
  }

  return false;
}

inline bool PBTree::ValidateNode(AST::ParallelBy* node) const {
  return node != nullptr && all_nodes_.find(node) != all_nodes_.end();
}

inline void PBTree::RemoveFromMaps(AST::ParallelBy* node) {
  if (!node) return;

  // Remove from children map
  auto it = children_map_.find(node);
  if (it != children_map_.end()) { children_map_.erase(it); }

  // Remove from parent map
  parent_map_.erase(node);

  // Remove from all_nodes set
  all_nodes_.erase(node);
}

inline void PBTree::Print(std::ostream& os) const {
  if (IsEmpty()) {
    os << "Tree is empty." << std::endl;
    return;
  }

  // Find all root nodes
  std::vector<AST::ParallelBy*> roots;
  for (auto* node : all_nodes_)
    if (IsRoot(node)) roots.push_back(node);

  // Print each root and its subtree
  for (size_t i = 0; i < roots.size(); ++i) {
    if (i > 0) os << "\n";
    int max_depth = static_cast<int>(GetHeight(roots[i]));
    PrintRecursive(roots[i], 0, max_depth, os);
  }
}

inline void PBTree::PrintRecursive(AST::ParallelBy* node, int depth,
                                   int max_depth, std::ostream& os) const {
  if (!node) return;

  // Calculate indentation based on depth
  std::string indent(depth * 2, ' ');

  // Print node with appropriate prefix
  os << indent;

  if (depth > 0) os << "+-- ";

  // Print the node using InlinePrint
  node->InlinePrint(os);
  os << "\n";

  // Get and print children
  auto& children = GetChildren(node);
  for (size_t i = 0; i < children.size(); ++i)
    PrintRecursive(children[i], depth + 1, max_depth, os);
}

inline size_t PBTree::GetRootCount() const {
  size_t count = 0;
  for (auto* node : all_nodes_)
    if (IsRoot(node)) count++;
  return count;
}

} // end namespace Choreo

#endif // __CHOREO_PARALLEL_BY_TREE_HPP__
