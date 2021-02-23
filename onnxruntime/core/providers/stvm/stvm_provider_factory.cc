// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/stvm/stvm_provider_factory.h"
#include <atomic>
#include "stvm_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {

struct StvmProviderFactory : IExecutionProviderFactory {
  StvmProviderFactory(std::string&& type) : backend_type_(std::move(type)) {}
  ~StvmProviderFactory() = default;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    StvmExecutionProviderInfo info{backend_type_};
    return onnxruntime::make_unique<StvmExecutionProvider>(info);
 }

 private:
    const std::string backend_type_;
};


std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Stvm(const char* backend_type) {
    return std::make_shared<onnxruntime::StvmProviderFactory>(std::string{backend_type});
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Stvm, _In_ OrtSessionOptions* options, _In_ const char* backend_type) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Stvm(backend_type));
  return nullptr;
}

