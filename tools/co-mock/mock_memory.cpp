#include "mock_memory.hpp"
#include <cassert>
#include <numeric>
#include <sstream>

namespace Choreo {
namespace Mock {

size_t Allocation::ElemSize() const {
  switch (elem_type) {
  case BaseType::S8:
  case BaseType::U8:
  case BaseType::BOOL: return 1;
  case BaseType::S16:
  case BaseType::U16:
  case BaseType::F16:
  case BaseType::BF16: return 2;
  case BaseType::S32:
  case BaseType::U32:
  case BaseType::F32: return 4;
  case BaseType::S64:
  case BaseType::U64:
  case BaseType::F64: return 8;
  default: return 4;
  }
}

size_t Allocation::TotalElements() const {
  if (shape.empty()) return 1;
  return std::accumulate(shape.begin(), shape.end(), size_t(1),
                         std::multiplies<>());
}

// Value factory methods

Value Value::MakeInt(int64_t v) {
  Value val;
  val.kind = Scalar;
  val.base_type = BaseType::S32;
  val.scalar.i64 = v;
  return val;
}

Value Value::MakeUInt(uint64_t v) {
  Value val;
  val.kind = Scalar;
  val.base_type = BaseType::U32;
  val.scalar.u64 = v;
  return val;
}

Value Value::MakeFloat(float v) {
  Value val;
  val.kind = Scalar;
  val.base_type = BaseType::F32;
  val.scalar.f32 = v;
  return val;
}

Value Value::MakeDouble(double v) {
  Value val;
  val.kind = Scalar;
  val.base_type = BaseType::F64;
  val.scalar.f64 = v;
  return val;
}

Value Value::MakeBool(bool v) {
  Value val;
  val.kind = Scalar;
  val.base_type = BaseType::BOOL;
  val.scalar.b = v;
  return val;
}

Value Value::MakePointer(std::shared_ptr<Allocation> a, BaseType bt,
                         size_t off) {
  Value val;
  val.kind = Pointer;
  val.base_type = bt;
  val.alloc = std::move(a);
  val.offset = off;
  return val;
}

int64_t Value::AsInt() const {
  switch (base_type) {
  case BaseType::S8: return (int8_t)scalar.i64;
  case BaseType::S16: return (int16_t)scalar.i64;
  case BaseType::S32: return (int32_t)scalar.i64;
  case BaseType::S64: return scalar.i64;
  case BaseType::U8: return (uint8_t)scalar.u64;
  case BaseType::U16: return (uint16_t)scalar.u64;
  case BaseType::U32: return (uint32_t)scalar.u64;
  case BaseType::U64: return (int64_t)scalar.u64;
  case BaseType::BOOL: return scalar.b ? 1 : 0;
  case BaseType::F32: return (int64_t)scalar.f32;
  case BaseType::F64: return (int64_t)scalar.f64;
  default: return scalar.i64;
  }
}

uint64_t Value::AsUInt() const {
  switch (base_type) {
  case BaseType::U8: return (uint8_t)scalar.u64;
  case BaseType::U16: return (uint16_t)scalar.u64;
  case BaseType::U32: return (uint32_t)scalar.u64;
  case BaseType::U64: return scalar.u64;
  case BaseType::S32: return (uint64_t)scalar.i64;
  case BaseType::S64: return (uint64_t)scalar.i64;
  default: return scalar.u64;
  }
}

double Value::AsDouble() const {
  switch (base_type) {
  case BaseType::F64: return scalar.f64;
  case BaseType::F32: return (double)scalar.f32;
  case BaseType::S32:
  case BaseType::S64: return (double)scalar.i64;
  case BaseType::U32:
  case BaseType::U64: return (double)scalar.u64;
  default: return scalar.f64;
  }
}

float Value::AsFloat() const {
  switch (base_type) {
  case BaseType::F32: return scalar.f32;
  case BaseType::F64: return (float)scalar.f64;
  case BaseType::S32:
  case BaseType::S64: return (float)scalar.i64;
  case BaseType::U32:
  case BaseType::U64: return (float)scalar.u64;
  default: return scalar.f32;
  }
}

bool Value::AsBool() const {
  switch (base_type) {
  case BaseType::BOOL: return scalar.b;
  case BaseType::S32:
  case BaseType::S64: return scalar.i64 != 0;
  case BaseType::U32:
  case BaseType::U64: return scalar.u64 != 0;
  case BaseType::F32: return scalar.f32 != 0.0f;
  case BaseType::F64: return scalar.f64 != 0.0;
  default: return scalar.i64 != 0;
  }
}

std::string Value::ToString() const {
  std::ostringstream oss;
  if (kind == Pointer) {
    oss << "<" << STR(alloc->storage) << " " << STR(base_type) << "[";
    for (size_t i = 0; i < alloc->shape.size(); ++i) {
      if (i) oss << ", ";
      oss << alloc->shape[i];
    }
    oss << "] @ " << alloc->RawPtr() << "+" << offset << ">";
    return oss.str();
  }
  switch (base_type) {
  case BaseType::S8:
  case BaseType::S16:
  case BaseType::S32:
  case BaseType::S64: oss << scalar.i64; break;
  case BaseType::U8:
  case BaseType::U16:
  case BaseType::U32:
  case BaseType::U64: oss << scalar.u64; break;
  case BaseType::F32: oss << scalar.f32; break;
  case BaseType::F64: oss << scalar.f64; break;
  case BaseType::BOOL: oss << (scalar.b ? "true" : "false"); break;
  default: oss << scalar.i64; break;
  }
  return oss.str();
}

void Value::WriteToAlloc(size_t byte_offset, const Value& v) const {
  assert(alloc && "WriteToAlloc on non-pointer value");
  assert(byte_offset + alloc->ElemSize() <= alloc->TotalBytes());
  uint8_t* dst = alloc->data.data() + byte_offset;
  switch (v.base_type) {
  case BaseType::S8: {
    int8_t tmp = (int8_t)v.scalar.i64;
    std::memcpy(dst, &tmp, 1);
    break;
  }
  case BaseType::U8: {
    uint8_t tmp = (uint8_t)v.scalar.u64;
    std::memcpy(dst, &tmp, 1);
    break;
  }
  case BaseType::S16: {
    int16_t tmp = (int16_t)v.scalar.i64;
    std::memcpy(dst, &tmp, 2);
    break;
  }
  case BaseType::U16: {
    uint16_t tmp = (uint16_t)v.scalar.u64;
    std::memcpy(dst, &tmp, 2);
    break;
  }
  case BaseType::S32: {
    int32_t tmp = (int32_t)v.scalar.i64;
    std::memcpy(dst, &tmp, 4);
    break;
  }
  case BaseType::U32: {
    uint32_t tmp = (uint32_t)v.scalar.u64;
    std::memcpy(dst, &tmp, 4);
    break;
  }
  case BaseType::S64: {
    std::memcpy(dst, &v.scalar.i64, 8);
    break;
  }
  case BaseType::U64: {
    std::memcpy(dst, &v.scalar.u64, 8);
    break;
  }
  case BaseType::F32: {
    std::memcpy(dst, &v.scalar.f32, 4);
    break;
  }
  case BaseType::F64: {
    std::memcpy(dst, &v.scalar.f64, 8);
    break;
  }
  case BaseType::BOOL: {
    uint8_t tmp = v.scalar.b ? 1 : 0;
    std::memcpy(dst, &tmp, 1);
    break;
  }
  default: {
    int32_t tmp = (int32_t)v.scalar.i64;
    std::memcpy(dst, &tmp, 4);
    break;
  }
  }
}

Value Value::ReadFromAlloc(size_t byte_offset, BaseType ty) const {
  assert(alloc && "ReadFromAlloc on non-pointer value");
  const uint8_t* src = alloc->data.data() + byte_offset;
  Value result;
  result.kind = Scalar;
  result.base_type = ty;
  switch (ty) {
  case BaseType::S8: {
    int8_t tmp;
    std::memcpy(&tmp, src, 1);
    result.scalar.i64 = tmp;
    break;
  }
  case BaseType::U8: {
    uint8_t tmp;
    std::memcpy(&tmp, src, 1);
    result.scalar.u64 = tmp;
    break;
  }
  case BaseType::S16: {
    int16_t tmp;
    std::memcpy(&tmp, src, 2);
    result.scalar.i64 = tmp;
    break;
  }
  case BaseType::U16: {
    uint16_t tmp;
    std::memcpy(&tmp, src, 2);
    result.scalar.u64 = tmp;
    break;
  }
  case BaseType::S32: {
    int32_t tmp;
    std::memcpy(&tmp, src, 4);
    result.scalar.i64 = tmp;
    break;
  }
  case BaseType::U32: {
    uint32_t tmp;
    std::memcpy(&tmp, src, 4);
    result.scalar.u64 = tmp;
    break;
  }
  case BaseType::S64: {
    std::memcpy(&result.scalar.i64, src, 8);
    break;
  }
  case BaseType::U64: {
    std::memcpy(&result.scalar.u64, src, 8);
    break;
  }
  case BaseType::F32: {
    std::memcpy(&result.scalar.f32, src, 4);
    break;
  }
  case BaseType::F64: {
    std::memcpy(&result.scalar.f64, src, 8);
    break;
  }
  case BaseType::BOOL: {
    uint8_t tmp;
    std::memcpy(&tmp, src, 1);
    result.scalar.b = tmp != 0;
    break;
  }
  default: {
    int32_t tmp;
    std::memcpy(&tmp, src, 4);
    result.scalar.i64 = tmp;
    break;
  }
  }
  return result;
}

// MockMemory implementation

MockMemory::MockMemory()
    : alloc_mutex_(std::make_unique<std::mutex>()) {
  scopes.emplace_back();
}

void MockMemory::EnterScope() { scopes.emplace_back(); }

void MockMemory::LeaveScope() {
  assert(scopes.size() > 1 && "Cannot leave global scope");
  scopes.pop_back();
}

void MockMemory::Define(const std::string& name, Value v) {
  scopes.back()[name] = std::move(v);
}

Value& MockMemory::Lookup(const std::string& name) {
  for (auto it = scopes.rbegin(); it != scopes.rend(); ++it)
    if (auto found = it->find(name); found != it->end()) return found->second;

  assert(false && "MockMemory::Lookup: variable not found");
  static Value dummy;
  return dummy;
}

const Value& MockMemory::Lookup(const std::string& name) const {
  for (auto it = scopes.rbegin(); it != scopes.rend(); ++it)
    if (auto found = it->find(name); found != it->end()) return found->second;

  assert(false && "MockMemory::Lookup: variable not found");
  static Value dummy;
  return dummy;
}

bool MockMemory::Exists(const std::string& name) const {
  for (auto it = scopes.rbegin(); it != scopes.rend(); ++it)
    if (it->count(name)) return true;
  return false;
}

std::shared_ptr<Allocation>
MockMemory::Allocate(BaseType elem_type, const std::vector<size_t>& shape,
                     Storage storage) {
  std::lock_guard<std::mutex> lock(*alloc_mutex_);
  auto alloc = std::make_shared<Allocation>();
  alloc->elem_type = elem_type;
  alloc->shape = shape;
  alloc->storage = storage;

  size_t total = 1;
  for (auto s : shape) total *= s;
  alloc->data.resize(total * alloc->ElemSize(), 0);
  return alloc;
}

MockMemory MockMemory::Fork() const {
  MockMemory child;
  child.scopes = scopes;
  child.thread_stack_ = thread_stack_;
  return child;
}

void MockMemory::PushThread(const std::string& pv_name, int64_t index) {
  thread_stack_.push_back({pv_name, index});
}

void MockMemory::PopThread() {
  if (!thread_stack_.empty()) thread_stack_.pop_back();
}

} // namespace Mock
} // namespace Choreo
