---
name: cpp-pro
description: |
  Use when building high-performance C++ systems requiring modern C++20/23 features,
  template metaprogramming, or zero-overhead abstractions for systems programming,
  embedded systems, or performance-critical applications.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
---

# C++ Pro

Senior C++ developer specializing in modern C++20/23, systems programming, and zero-overhead abstractions.

## When to Use This Skill

- Systems programming and embedded development
- Creating custom allocators and memory pools
- Developing concurrent and parallel algorithms
- Optimizing memory-critical systems
- Implementing template metaprogramming solutions
- Building high-performance C++ applications

## Modern C++ Checklist

### C++ Core Guidelines Compliance
- RAII pattern enforcement
- Smart pointer usage (unique_ptr by default, shared_ptr for shared ownership)
- Range-based for loops
- std::optional for optional values
- std::variant for tagged unions

### RAII and Ownership
```cpp
// Good: RAII with unique_ptr
auto make_foo() {
    auto foo = std::make_unique<Foo>();
    if (!foo->init()) {
        return {};
    }
    return foo;
}

// Avoid: manual new/delete with early returns
Foo* make_foo() {
    Foo* foo = new Foo();
    if (!foo->Init()) {
        delete foo;
        return nullptr;
    }
    return foo;
}
```

### Rule of 0/3/5
- Prefer Rule of 0 (RAII types with default operations)
- If owning a resource, define all 5 operations
- Use default/delete for special members

### Move Semantics
```cpp
// Good: move semantics for ownership transfer
class Widget {
    std::vector<int> data_;
public:
    Widget(Widget&& other) noexcept : data_(std::move(other.data_)) {}
    Widget& operator=(Widget&& other) noexcept {
        data_ = std::move(other.data_);
        return *this;
    }
};
```

### Const-Correctness
```cpp
class User {
    std::string name_;
public:
    const std::string& name() const { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }
};
```

### Virtual Functions
```cpp
struct Base {
    virtual void run() = 0;
    virtual ~Base() = default;
};

struct Worker final : Base {
    void run() override {}
};
```

## Template Metaprogramming

### Concepts (C++20)
```cpp
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<Numeric T>
T add(T a, T b) { return a + b; }
```

### constexpr and consteval
```cpp
constexpr auto square(int x) { return x * x; }
consteval int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }
```

## Memory Management

### Smart Pointers
- `std::unique_ptr<T>` - Single ownership
- `std::shared_ptr<T>` - Shared ownership (use sparingly)
- `std::weak_ptr<T>` - Breaking cycles

### Custom Allocators
```cpp
template<typename T>
class PoolAllocator {
    // Pre-allocate memory pool
    // No individual allocations
    // Perfect for fixed-size objects
};
```

## Concurrency

### Atomics for Simple Cases
```cpp
std::atomic<int> counter{0};
void increment() { counter.fetch_add(1, std::memory_order_relaxed); }
```

### Mutex and Locks
```cpp
std::mutex mu;
std::vector<int> data;

void safe_add(int v) {
    std::lock_guard<std::mutex> lock(mu);
    data.push_back(v);
}
```

### Thread-Safe Singletons
```cpp
class Singleton {
    static std::atomic<Singleton*> instance_;
    static std::mutex mutex_;
public:
    static Singleton* get_instance() {
        // Double-checked locking pattern
    }
};
```

## Performance Optimization

### Cache-Friendly Patterns
- Structure of Arrays (SoA) vs Array of Structures (AoS)
- Data alignment for SIMD
- Prefetching for predictable access

### Reserve for Vectors
```cpp
std::vector<int> build(int n) {
    std::vector<int> out;
    out.reserve(n);  // Avoid reallocations
    for (int i = 0; i < n; ++i) {
        out.push_back(i);
    }
    return out;
}
```

## Build System (CMake)

### Modern CMake
```cmake
cmake_minimum_required(VERSION 3.20)
project(myproject)

add_library(mylib STATIC)
target_sources(mylib PUBLIC
    FILE_SET headers BASE_DIRS include
)
target_compile_features(mylib PUBLIC cxx_std_20)
```

### Compiler Flags
```cmake
add_compile_options(
    -Wall -Wextra -Werror
    -Wconversion -Wshadow
)
```

## Static Analysis

### clang-tidy Checks
```bash
clang-tidy src/*.cpp -- -std=c++20
```

### Sanitizers
```bash
# AddressSanitizer
clang++ -fsanitize=address -fno-omit-frame-pointer -g ...

# UndefinedBehaviorSanitizer
clang++ -fsanitize=undefined ...

# ThreadSanitizer
clang++ -fsanitize=thread ...
```

## Error Handling

### Exception Safety
- Use RAII for cleanup
- Destructors should be noexcept
- Consider std::expected (C++23) for expected errors

### noexcept Specifications
```cpp
void do_work() noexcept {
    // Will not throw
}
```

## Best Practices Summary

1. **Default to RAII and smart pointers**
2. **Use const everywhere**
3. **Apply the Rule of 0/3/5**
4. **Leverage modern C++ features (auto, range-for, constexpr)**
5. **Write for performance from the start**
6. **Enable all warnings and treat them as errors**
7. **Run static analysis (clang-tidy)**
8. **Test with sanitizers**
