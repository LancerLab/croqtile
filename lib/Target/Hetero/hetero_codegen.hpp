#ifndef __CHOREO_HETERO_CODEGEN_HPP__
#define __CHOREO_HETERO_CODEGEN_HPP__

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "cc_codegen.hpp"
#include "codegen.hpp"

namespace Choreo {

namespace Hetero {

struct HeteroCodeGen : public CC::CCCodeGen {
public:
  HeteroCodeGen() {
    cpp_name =
        "__choreo_hetero_" + OptionRegistry::GetInstance().GetInputName();
    cmp_dir = CreateUniquePath();
  }

  bool BeforeVisitImpl(AST::Node&) override;
  bool InMidVisitImpl(AST::Node&) override;
  bool AfterVisitImpl(AST::Node&) override;

  bool Visit(AST::ParallelBy&) override;
  bool Visit(AST::Synchronize&) override;

  bool Visit(AST::WithIn&) override;
  bool Visit(AST::ForeachBlock&) override;
  bool Visit(AST::IfElseBlock&) override;
  bool Visit(AST::WhileBlock&) override;
  bool Visit(AST::Assignment&) override;
  bool Visit(AST::DMA&) override;
  bool Visit(AST::Break&) override;
  bool Visit(AST::Continue&) override;
  bool Visit(AST::Call&) override;
  bool Visit(AST::NamedVariableDecl&) override;
  bool Visit(AST::CppSourceCode& n) override;
  bool Visit(AST::Return&) override;
  bool Visit(AST::Wait&) override;
  bool Visit(AST::MMA&) override;
  bool Visit(AST::Trigger&) override;
  bool Visit(AST::Rotate&) override;

private:
  std::string active_device_target;

  // DeviceCodeGen delegates for host includes, sync, link flags, etc.
  std::map<std::string, std::unique_ptr<DeviceCodeGen>> device_codegens;

  DeviceCodeGen* ActiveDeviceCodeGen() {
    auto it = device_codegens.find(active_device_target);
    return (it != device_codegens.end()) ? it->second.get() : nullptr;
  }

  AST::ChoreoFunction* current_func_node_ = nullptr;
  bool in_offload_device_block_ = false;
  int offload_func_counter_ = 0;
  std::map<std::string, int> device_pb_counters_;

  struct OffloadFuncInfo {
    std::string co_func_name;
    std::string parent_fname;
    AST::ptr<AST::ChoreoFunction> offload_func;
    std::string device_source;
    std::string source_ext;
    std::string host_fwd_decl;
    std::string device_name;
    std::string target_name;
    struct BufferInfo {
      std::string host_name;
      std::string device_name;
      std::string base_type;
      std::string size_expr;
      bool is_output = false;
    };
    std::vector<BufferInfo> buffers;
    std::vector<std::string> scalar_params;
    std::vector<std::string> scalar_types;
  };
  std::vector<OffloadFuncInfo> offload_functions_;
  OffloadFuncInfo cur_offload_func_;

  int host_future_counter_ = 0;
  std::vector<std::string> pending_host_futures_;

  void EmitPreamble() override;
  void EmitScript(std::ostream& out, const std::string& exe_fn = "") override;
  bool CompileWithScript(const std::string& action) override;

  void EmitHeteroSource();

  bool CompileOffloadToSource(OffloadFuncInfo& fi);
  void BeginOffloadFunction(AST::ParallelBy& n, DeviceCodeGen& dcg);
  void EndOffloadFunction();
  void EmitOffloadHostCall(const OffloadFuncInfo& fi);

  void ResetFunctionStates() override {
    CCCodeGen::ResetFunctionStates();
    active_device_target = "";
    current_func_node_ = nullptr;
    host_future_counter_ = 0;
    pending_host_futures_.clear();
    in_offload_device_block_ = false;
    cur_offload_func_ = {};
    device_pb_counters_.clear();
  }
};

} // namespace Hetero

} // namespace Choreo

#endif // __CHOREO_HETERO_CODEGEN_HPP__
