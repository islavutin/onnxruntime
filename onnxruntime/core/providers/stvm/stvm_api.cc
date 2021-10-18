// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>

#include "stvm_api.h"

#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>

namespace stvm {

tvm::runtime::Module TVMCompile(const std::string& onnx_txt,
                                const std::string& target,
                                const std::string& target_host,
                                int opt_level,
                                int opset,
                                bool freeze_params,
                                const std::vector<std::vector<int64_t>>& input_shapes)
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
  tvm::runtime::Module mod = (*compile)(TVMByteArray{onnx_txt.data(), onnx_txt.size()}, target, target_host, opt_level, opset, freeze_params, shapes);
  return mod;
}

void TVMSetInputs(tvm::runtime::Module& mod,
                  std::vector<DLTensor>& inputs)
{
  tvm::PackedFunc set_input = mod.GetFunction("set_input", false);
  for (size_t i = 0; i < inputs.size(); i++)
  {
    set_input(i, &inputs[i]);
  }
}

void TVMRun(tvm::runtime::Module& mod,
            std::vector<DLTensor>& inputs,
            std::vector<DLTensor>& outputs,
            [[maybe_unused]] tvm::runtime::TVMRetValue *ret)
{
  // TODO(vvchernov): set_input_zero_copy is more preferable but it does not satisfy alignment conditions.
  //tvm::PackedFunc set_input = mod.GetFunction("set_input_zero_copy", false);

  auto start = std::chrono::system_clock::now();
  TVMSetInputs(mod, inputs);

  const tvm::PackedFunc* run = tvm::runtime::Registry::Get("tvm_run");
  auto end = std::chrono::system_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "preprocess tvm_run: " << dur << " us" << std::endl;

  start = std::chrono::system_clock::now();
  (*run)(mod);
  end = std::chrono::system_clock::now();
  dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "tvm inference run: " << float(dur)/1000 << " ms" << std::endl;

  start = std::chrono::system_clock::now();
  tvm::PackedFunc get_output = mod.GetFunction("get_output", false);
  for (size_t i = 0; i < outputs.size(); i++)
  {
    get_output(i, &outputs[i]);
  }
  end = std::chrono::system_clock::now();
  dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "postprocess tvm run: " << dur << " us" << std::endl;
}

}  // namespace stvm
