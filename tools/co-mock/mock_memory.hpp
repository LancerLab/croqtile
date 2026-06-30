#ifndef CHOREO_MOCK_MEMORY_HPP_
#define CHOREO_MOCK_MEMORY_HPP_

#include "types.hpp"
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace Choreo {
namespace Mock {

struct Allocation {
  std::vector<uint8_t> data;
  BaseType elem_type;
  std::vector<size_t> shape;
  Storage storage;

  size_t ElemSize() const;
  size_t TotalElements() const;
  size_t TotalBytes() const { return data.size(); }

  void* RawPtr() { return data.data(); }
  const void* RawPtr() const { return data.data(); }
};

struct FutureInfo {
  std::shared_future<void> handle;
  std::string src_name;
  std::string dst_name;
  size_t bytes = 0;

  bool IsReady() const {
    return handle.valid() &&
           handle.wait_for(std::chrono::seconds(0)) ==
               std::future_status::ready;
  }
};

struct Value {
  enum Kind { Scalar, Pointer, Future, String };
  Kind kind = Scalar;

  BaseType base_type = BaseType::S32;

  union ScalarData {
    int64_t i64;
    uint64_t u64;
    double f64;
    float f32;
    bool b;
    ScalarData() : i64(0) {}
  } scalar;

  std::shared_ptr<Allocation> alloc;
  size_t offset = 0;
  std::shared_ptr<FutureInfo> future_info;
  std::string str_val;

  Value() = default;

  static Value MakeInt(int64_t v);
  static Value MakeUInt(uint64_t v);
  static Value MakeFloat(float v);
  static Value MakeDouble(double v);
  static Value MakeBool(bool v);
  static Value MakePointer(std::shared_ptr<Allocation> a, BaseType bt,
                           size_t off = 0);
  static Value MakeFuture(std::shared_future<void> f,
                          const std::string& src = "",
                          const std::string& dst = "", size_t bytes = 0);
  static Value MakeString(const std::string& s);

  int64_t AsInt() const;
  uint64_t AsUInt() const;
  double AsDouble() const;
  float AsFloat() const;
  bool AsBool() const;

  std::string ToString() const;

  void WriteToAlloc(size_t byte_offset, const Value& v) const;
  Value ReadFromAlloc(size_t byte_offset, BaseType ty) const;
};

struct ThreadInfo {
  std::string pv_name;
  int64_t index;
};

class MockMemory {
  std::vector<std::unordered_map<std::string, Value>> scopes;
  std::vector<ThreadInfo> thread_stack_;
  std::unique_ptr<std::mutex> alloc_mutex_;

public:
  MockMemory();

  void EnterScope();
  void LeaveScope();

  void Define(const std::string& name, Value v);
  Value& Lookup(const std::string& name);
  const Value& Lookup(const std::string& name) const;
  bool Exists(const std::string& name) const;

  std::shared_ptr<Allocation> Allocate(BaseType elem_type,
                                       const std::vector<size_t>& shape,
                                       Storage storage);

  // Fork a child memory for parallel threads: inherits all current scopes
  // (snapshot) but shares Allocation objects via shared_ptr.
  MockMemory Fork() const;

  void PushThread(const std::string& pv_name, int64_t index);
  void PopThread();
  const std::vector<ThreadInfo>& ThreadStack() const { return thread_stack_; }

  const std::unordered_map<std::string, Value>& CurrentScope() const {
    return scopes.back();
  }

  const std::vector<std::unordered_map<std::string, Value>>& AllScopes() const {
    return scopes;
  }
};

} // namespace Mock
} // namespace Choreo

#endif // CHOREO_MOCK_MEMORY_HPP_
