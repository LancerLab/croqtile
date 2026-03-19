---
name: code-review-excellence
description: |
  Provides comprehensive code review guidance for React 19, Vue 3, Rust, TypeScript, Java, Python, and C/C++.
  Helps catch bugs, improve code quality, and give constructive feedback.
  Use when: reviewing pull requests, conducting PR reviews, code review, reviewing code changes,
  establishing review standards, mentoring developers, architecture reviews, security audits,
  checking code quality, finding bugs, giving feedback on code.
allowed-tools:
  - Read
  - Grep
  - Glob
  - Bash
  - WebFetch
---

# Code Review Excellence

Transform code reviews from gatekeeping to knowledge sharing through constructive feedback, systematic analysis, and collaborative improvement.

## When to Use This Skill

- Reviewing pull requests and code changes
- Conducting architecture reviews
- Mentoring developers through reviews
- Improving team collaboration
- Maintaining code quality standards

## Core Principles

### The Review Mindset

**Goals of Code Review:**
- Catch bugs and edge cases
- Ensure code maintainability
- Share knowledge across team
- Enforce coding standards
- Improve design and architecture

### Effective Feedback

**Good Feedback is:**
- Specific and actionable
- Educational, not judgmental
- Focused on the code, not the person
- Prioritized (critical vs nice-to-have)

**Severity Labels:**
- `[blocking]` - Must fix before merge
- `[important]` - Should fix, discuss if disagree
- `[nit]` - Nice to have, not blocking
- `[suggestion]` - Alternative approach to consider
- `[praise]` - Good work, keep it up!

## Review Techniques

### Checklist Method
Use checklists for consistent reviews.

### Question Approach
Instead of stating problems, ask questions:
- "What happens if `items` is an empty array?"
- "How should this behave if the API call fails?"

### Suggest, Don't Command
Use collaborative language:
- "Suggestion: async/await might make this more readable."
- "This logic appears in 3 places. Would it make sense to extract it?"

## C++ Code Review Checklist

### Safety and Lifetime
- [ ] Ownership is explicit (RAII, unique_ptr by default)
- [ ] No dangling references or views
- [ ] Rule of 0/3/5 followed for resource-owning types
- [ ] No raw new/delete in business logic
- [ ] Destructors are noexcept and do not throw

### API and Design
- [ ] const-correctness is applied consistently
- [ ] Constructors are explicit where needed
- [ ] override/final used for virtual functions
- [ ] No object slicing (pass by ref or pointer)

### Concurrency
- [ ] Shared data is protected (mutex or atomics)
- [ ] Locking order is consistent
- [ ] No blocking while holding locks

### Performance
- [ ] Unnecessary allocations avoided (reserve, move)
- [ ] Copies avoided in hot paths
- [ ] Algorithmic complexity is reasonable

### Tooling and Tests
- [ ] Builds clean with warnings enabled
- [ ] Sanitizers run on critical code paths
- [ ] Static analysis (clang-tidy) results are addressed

## Architecture Review Guide

For significant changes, assess:
- SOLID principles compliance
- Coupling and cohesion
- Anti-patterns detection
- Design pattern appropriateness

## Performance Review Guide

For performance-critical code, check:
- Algorithm complexity (O notation)
- Memory allocation patterns
- Cache locality
- Hot path optimization opportunities
