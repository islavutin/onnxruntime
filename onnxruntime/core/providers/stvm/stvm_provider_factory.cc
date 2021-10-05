// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/stvm/stvm_provider_factory.h"
#include <atomic>
#include "stvm_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct StvmProviderFactory : IExecutionProviderFactory {
  StvmProviderFactory(const StvmExecutionProviderInfo& info) : info_{info} {}
  ~StvmProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return onnxruntime::make_unique<StvmExecutionProvider>(info_);
 }

 private:
    StvmExecutionProviderInfo info_;
};


std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Stvm(const StvmExecutionProviderInfo& info) {
    return std::make_shared<onnxruntime::StvmProviderFactory>(info);
}
}  // namespace onnxruntime

// TODO(vvchernov): check API, may be need extension
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Stvm, _In_ OrtSessionOptions* options, _In_ const char* backend_type) {
  StvmExecutionProviderInfo info;
  info.target = std::string{backend_type};

  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Stvm(info));
  return nullptr;
}
