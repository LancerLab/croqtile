# Modern C++ Guidelines

Croqtile targets **C++17**. Patterns below are appropriate for that baseline. Features that require **C++20 or later** are noted as informational only; do not use them in this codebase until the standard is raised.

### C++ Core Guidelines alignment

- RAII for resource ownership.
- Smart pointers: `std::unique_ptr` by default; `std::shared_ptr` for shared ownership (use sparingly).
- Range-based `for` loops where they improve clarity.
- `std::optional` for optional values.
- `std::variant` for tagged unions.

### RAII and ownership

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

### Rule of 0 / 3 / 5

- Prefer Rule of 0 (composite types whose members manage resources).
- If you own a raw resource, define all five special members appropriately.
- Use `= default` / `= delete` to express intent for special members.

### Move semantics

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

### Const-correctness

```cpp
class User {
    std::string name_;
public:
    const std::string& name() const { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }
};
```

### Virtual functions

```cpp
struct Base {
    virtual void run() = 0;
    virtual ~Base() = default;
};

struct Worker final : Base {
    void run() override {}
};
```

### Template metaprogramming

C++20+ (informational): concepts constrain templates cleanly:

```cpp
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<Numeric T>
T add(T a, T b) { return a + b; }
```

`constexpr` (C++17):

```cpp
constexpr auto square(int x) { return x * x; }
```

C++20+ (informational): `consteval` for compile-time-only functions:

```cpp
consteval int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }
```

### Memory management

Smart pointers:

- `std::unique_ptr<T>` for single ownership.
- `std::shared_ptr<T>` for shared ownership (avoid deep graphs of shared ownership).
- `std::weak_ptr<T>` to break cycles.

Custom allocators:

```cpp
template<typename T>
class PoolAllocator {
    // Pre-allocate memory pool
    // No individual allocations
    // Perfect for fixed-size objects
};
```

### Concurrency

Atomics for simple cases:

```cpp
std::atomic<int> counter{0};
void increment() { counter.fetch_add(1, std::memory_order_relaxed); }
```

Mutex and locks:

```cpp
std::mutex mu;
std::vector<int> data;

void safe_add(int v) {
    std::lock_guard<std::mutex> lock(mu);
    data.push_back(v);
}
```

Thread-safe singleton (illustrative pattern from classic guidance; prefer Meyers singleton in new code when applicable):

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

### Exception safety and noexcept

- Prefer RAII so cleanup runs on all exits.
- Mark functions `noexcept` when they must not throw.
- `noexcept` on destructors is typical for well-behaved types.

```cpp
void do_work() noexcept {
    // Will not throw
}
```

### Best practices summary

1. Default to RAII and smart pointers.
2. Use `const` consistently.
3. Apply Rule of 0 / 3 / 5 where resources are owned.
4. Prefer C++17 features (`auto`, range-for, `constexpr`) that match the project standard.
5. Enable strong warnings and treat them as errors in CI and local workflows when configured.
6. Run static analysis (`clang-tidy`) where integrated.
7. Validate critical paths with sanitizers.
