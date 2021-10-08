// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STVM_EXECUTION_PROVIDER_H
#define STVM_EXECUTION_PROVIDER_H

#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/providers/stvm/stvm_execution_provider_info.h"
#include "core/platform/ort_mutex.h"

#include "stvm_common.h"

namespace onnxruntime {

namespace stvm_env_vars {
   static const std::string kDumpSubgraphs = "ORT_STVM_DUMP_SUBGRAPHS";
}  // namespace stvm_env_vars

// Information to construct kernel function state.
struct StvmFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  // tvm::runtime::Module* module = nullptr;
  OrtMutex* stvm_mu_ptr = nullptr;
};

class STVMCompiler;

// Logical device representation.
class StvmExecutionProvider : public IExecutionProvider {
  friend STVMCompiler;
 public:
  explicit StvmExecutionProvider(const StvmExecutionProviderInfo& info);
  virtual ~StvmExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

 private:
  bool GPUTargetCheck() const;
  void ProcessInfo();
  void ProcessCPUTarget();
  void ProcessGPUTarget();
  void PrintInfo() const;
 private:
  bool dump_subgraphs_ = false;
  OrtMutex stvm_mu_;
  AllocatorPtr allocator_;
  StvmExecutionProviderInfo info_;
  std::unordered_map<std::string, std::shared_ptr<tvm::runtime::Module>> modules_;
};

}  // namespace onnxruntime

#endif  // STVM_EXECUTION_PROVIDER_H
