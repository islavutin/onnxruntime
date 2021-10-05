// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/stvm/stvm_execution_provider_info.h"

#include "core/common/common.h"
#include "core/framework/provider_options_utils.h"

namespace onnxruntime {
namespace stvm {
namespace provider_option_names {
constexpr const char* kTarget = "target";
constexpr const char* kTargetHost = "target_host";
constexpr const char* kOptLevel = "opt_level";
}  // namespace provider_option_names
}  // namespace stvm

StvmExecutionProviderInfo StvmExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  StvmExecutionProviderInfo info{};

  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddAssignmentToReference(stvm::provider_option_names::kTarget, info.target)
          .AddAssignmentToReference(stvm::provider_option_names::kTargetHost, info.target_host)
          .AddAssignmentToReference(stvm::provider_option_names::kOptLevel, info.opt_level)
          .Parse(options));

  return info;
}

}  // namespace onnxruntime
