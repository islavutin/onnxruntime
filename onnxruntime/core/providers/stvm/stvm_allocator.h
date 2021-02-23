// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STVM_ALLOCATOR
#define STVM_ALLOCATOR

#include "core/framework/allocator.h"
#include "stvm_common.h"

namespace onnxruntime {

#define STVM_ALLOC_ALIGN 128

typedef enum {
    VULKAN = 0,
} STVM_DEVICE_ID;
 
class STVMGPUAllocator : public IDeviceAllocator {
 public:
  STVMGPUAllocator(int device_id, const char* name)
    : IDeviceAllocator(
        OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator,
                      OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device_id),
                      device_id, OrtMemTypeDefault)) { device_id_ = device_id;}

  virtual void* Alloc(size_t size) override;
  virtual void Free(void* p) override;
  DLContext get_context();
 private:
  int device_id_;
};

}  // namespace onnxruntime
#endif // STVM_ALLOCATOR
