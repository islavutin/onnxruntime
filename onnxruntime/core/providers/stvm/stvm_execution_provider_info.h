// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STVM_EXECUTION_PROVIDER_INFO_H
#define STVM_EXECUTION_PROVIDER_INFO_H

#include "core/framework/provider_options.h"

namespace onnxruntime {

// Information needed to construct an TVM execution provider.
struct StvmExecutionProviderInfo {
  std::string target{"llvm"};
  std::string target_host{"llvm"};
  uint opt_level{3};

  static StvmExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
};

}  // namespace onnxruntime

#endif  // STVM_EXECUTION_PROVIDER_INFO_H
