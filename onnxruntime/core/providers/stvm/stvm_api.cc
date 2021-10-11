// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "stvm_api.h"

#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>

namespace stvm {

    tvm::runtime::Module TVMCompile(const std::string& onnx_txt, const std::string& target, const std::string& target_host, int opt_level, const std::vector<std::vector<int64_t>>& input_shapes)
{
  tvm::Array<tvm::Array<tvm::Integer>> shapes;
  for (size_t i = 0; i < input_shapes.size(); i++)
  {
    tvm::Array<tvm::Integer> shape;
    for (auto& dim : input_shapes[i])
    {
      shape.push_back(tvm::Integer(dim));
    }
    shapes.push_back(shape);
  }

  const tvm::PackedFunc* compile = tvm::runtime::Registry::Get("tvm_onnx_import_and_compile");
  tvm::runtime::Module mod = (*compile)(TVMByteArray{onnx_txt.data(), onnx_txt.size()}, target, target_host, opt_level, shapes);
  return mod;
}

void TVMExtractOutputShapes(tvm::runtime::Module& mod, size_t num_outputs, std::vector<std::vector<int64_t>>& output_shapes)
{
  tvm::PackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < num_outputs; i++)
  {
    tvm::runtime::NDArray output_array = get_output(i);
    const auto& shape = output_array.Shape();
    std::vector<int64_t> oshape;
    for (const auto& dim : shape)
    {
      oshape.push_back(dim);
    }
    output_shapes.push_back(oshape);
  }
}

void TVMRun(tvm::runtime::Module& mod, std::vector<DLTensor>& inputs, std::vector<DLTensor>& outputs, [[maybe_unused]] tvm::runtime::TVMRetValue *ret)
{
  // TODO(vvchernov): set_input_zero_copy is more preferable but it does not satisfy alignment conditions.
  //tvm::PackedFunc set_input = mod.GetFunction("set_input_zero_copy", false);
  tvm::PackedFunc set_input = mod.GetFunction("set_input", false);
  for (size_t i = 0; i < inputs.size(); i++)
  {
    set_input(i, &inputs[i]);
  }

  const tvm::PackedFunc* run = tvm::runtime::Registry::Get("tvm_run_with_benchmark");
  (*run)(mod);

  tvm::PackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < outputs.size(); i++)
  {
    get_output(i, &outputs[i]);
  }
}

}  // namespace stvm
