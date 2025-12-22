#include "codegen.hpp"

using namespace Choreo;

std::once_flag CodeGenInfo::init_flag;
std::unique_ptr<CodeGenInfo> CodeGenInfo::instance;
int TMADesc::index = 0;

Choreo::CodeGenInfo& CodeGenInfo::Get() {
  std::call_once(init_flag,
                 []() { instance = std::make_unique<CodeGenInfo>(); });
  return *instance;
}

PBTree::PBTree() = default;

PBTree::~PBTree() { Clear(); }

bool PBTree::AddChild(AST::ParallelBy* parent, AST::ParallelBy* child) {
  // Validate input parameters
  if (!parent || !child) { return false; }

  // Prevent adding a node as its own child
  if (parent == child) { return false; }

  // Check if child already has a parent
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

const std::vector<AST::ParallelBy*>
PBTree::GetChildren(AST::ParallelBy* node) const {
  if (!ValidateNode(node)) { return {}; }

  return children_map_.at(node);
}

AST::ParallelBy* PBTree::GetParent(AST::ParallelBy* node) const {
  if (!ValidateNode(node)) { return nullptr; }

  auto it = parent_map_.find(node);
  return (it != parent_map_.end()) ? it->second : nullptr;
}

const std::vector<AST::ParallelBy*>
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

void PBTree::Clear() {
  children_map_.clear();
  parent_map_.clear();
  all_nodes_.clear();
}

bool PBTree::IsRoot(AST::ParallelBy* node) const {
  if (!ValidateNode(node)) { return false; }

  return parent_map_.find(node) == parent_map_.end();
}

bool PBTree::IsEmpty() const { return all_nodes_.empty(); }

size_t PBTree::GetSize() const { return all_nodes_.size(); }

size_t PBTree::GetDepth(AST::ParallelBy* node) const {
  if (!ValidateNode(node)) { return 0; }

  size_t depth = 0;
  AST::ParallelBy* current = node;

  while (current && !IsRoot(current)) {
    current = GetParent(current);
    depth++;
  }

  return depth;
}

std::vector<AST::ParallelBy*> PBTree::GetAllNodes() const {
  return std::vector<AST::ParallelBy*>(all_nodes_.begin(), all_nodes_.end());
}

std::vector<AST::ParallelBy*>
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

bool PBTree::IsAncestor(AST::ParallelBy* ancestor,
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

bool PBTree::ValidateNode(AST::ParallelBy* node) const {
  return node != nullptr && all_nodes_.find(node) != all_nodes_.end();
}

void PBTree::RemoveFromMaps(AST::ParallelBy* node) {
  if (!node) return;

  // Remove from children map
  auto it = children_map_.find(node);
  if (it != children_map_.end()) { children_map_.erase(it); }

  // Remove from parent map
  parent_map_.erase(node);

  // Remove from all_nodes set
  all_nodes_.erase(node);
}
