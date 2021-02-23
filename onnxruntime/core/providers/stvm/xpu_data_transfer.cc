// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xpu_data_transfer.h"

namespace onnxruntime {
GPUDataTransfer::GPUDataTransfer() {
}

GPUDataTransfer::~GPUDataTransfer() {
}

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
    return (src_device.Type() == OrtDevice::GPU || dst_device.Type() == OrtDevice::GPU);
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  const OrtDevice& src_device = src.Location().device;
  const OrtDevice& dst_device = dst.Location().device;

  if ((src_device.Type() == OrtDevice::CPU) && (dst_device.Type() == OrtDevice::CPU)) {
      memcpy(dst_data, src_data, bytes);
  } else {
    DLContext src_context = get_context(src_device);
    DLContext dst_context = get_context(dst_device);
    DLDataType dl_type{kDLInt, 8, 1};
    TVMDeviceCopyDataFromTo(src_data, 0, dst_data, 0, bytes, src_context, dst_context, dl_type, nullptr);
  }
  return Status::OK();
}

DLContext GPUDataTransfer::get_context(const OrtDevice& device) const
{
  DLContext context;
  switch (device.Type()) {
  case OrtDevice::CPU:
      context = {kDLCPU, 0};
      break;
  case OrtDevice::GPU:
      context = {kDLVulkan, 0};
      break;
  default:
      ORT_NOT_IMPLEMENTED("GPUDataTransfer get_context");
      break;
  }
  return context;
}
    
}  // namespace onnxruntime
