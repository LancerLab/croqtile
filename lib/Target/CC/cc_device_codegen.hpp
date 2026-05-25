#ifndef __CHOREO_CC_DEVICE_CODEGEN_HPP__
#define __CHOREO_CC_DEVICE_CODEGEN_HPP__

#include "codegen.hpp"
#include <string>
#include <vector>

namespace Choreo {

struct CCDeviceCodeGen : public DeviceCodeGen {
  std::string DeviceName() const override { return "cpu"; }
  std::string TargetName() const override { return "cc"; }
  bool IsHostDevice() const override { return true; }

  void EmitHostIncludes(std::ostream& os) const override {
    os << "#include <future>\n";
    os << "#include <thread>\n";
  }

  void EmitHostPreamble(std::ostream& os) const override {
    os << "static choreo::thread_pool __choreo_cpu_pool("
       << "std::thread::hardware_concurrency());\n\n";
  }

  void EmitSync(std::ostream& os, const std::string& indent) override {
    for (auto& f : pending_futures)
      os << indent << f << ".get();\n";
    pending_futures.clear();
  }

  std::string CompileCommand(const std::string& src,
                             const std::string& obj) const override {
    return "${CXX:-g++} -std=c++17 -O2 -pthread -c " + src + " -o " + obj;
  }

  std::string LinkFlags() const override { return "-lpthread"; }

  void AddPendingFuture(const std::string& name) override {
    pending_futures.push_back(name);
  }

  void Reset() override {
    pending_futures.clear();
  }

private:
  std::vector<std::string> pending_futures;
};

} // namespace Choreo

#endif // __CHOREO_CC_DEVICE_CODEGEN_HPP__
