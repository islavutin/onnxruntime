// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "stvm_allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "xpu_data_transfer.h"


namespace onnxruntime {

DLContext STVMGPUAllocator::get_context() {
    DLContext ctx;
    switch (device_id_) {
    case VULKAN:
      ctx = {kDLVulkan, 0};
      break;
    default:
        ORT_NOT_IMPLEMENTED("STVMGPUAllocator");
        break;
    }
    return ctx;
}
    
void* STVMGPUAllocator::Alloc(size_t size) {

  void* p = nullptr;
  if (size > 0) {
    DLContext ctx = get_context();
    DLDataType dl_type{kDLInt, 8, 1};
    TVMDeviceAllocDataSpace(ctx, size, STVM_ALLOC_ALIGN, dl_type, (void**)&p);
  }
  return p;
}

void STVMGPUAllocator::Free(void* p) {
    DLContext ctx = get_context();
    TVMDeviceFreeDataSpace(ctx, p);
}

}  // namespace onnxruntime
