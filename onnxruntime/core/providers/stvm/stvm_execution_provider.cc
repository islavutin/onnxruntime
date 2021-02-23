// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/framework/allocatormgr.h"
#include "core/platform/env.h"
#include "core/common/status.h"
#include "onnx/shape_inference/implementation.h"
#include "core/graph/model.h"
#include "stvm_execution_provider.h"
#include "xpu_data_transfer.h"
#include "stvm_allocator.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

static DLDataType GetDataType(ONNXTensorElementDataType type) {
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    return {kDLFloat, 64, 1};
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return {kDLFloat, 32, 1};
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      return {kDLInt, 64, 1};
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      return {kDLInt, 32, 1};
  } else
    ORT_THROW("not implement.");
}

static DLContext GetDLContext(const OrtDevice& device) {
  DLContext context;
  switch (device.Type()) {
  case OrtDevice::CPU:
      context = {kDLCPU, 0};
      break;
  case OrtDevice::GPU:
      context = {kDLVulkan, 0};
      break;
  default:
      ORT_NOT_IMPLEMENTED("Unsupported device");
      break;
  }
  return context;
}
    
struct STVMFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  tvm::runtime::Module* module = nullptr;
  std::function<tvm::runtime::Module*(std::string func_name, const std::vector<std::vector<int64_t>>& input_shapes)> compiler = nullptr;
};    

StvmExecutionProvider::StvmExecutionProvider(const StvmExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kStvmExecutionProvider} {
    backend_type_ = info.backend_type;
    if (backend_type_.find("vulkan") != std::string::npos) {
        DeviceAllocatorRegistrationInfo default_memory_info(
        {OrtMemTypeDefault, [](int id) { return onnxruntime::make_unique<STVMGPUAllocator>(VULKAN, "STVM"); }, std::numeric_limits<size_t>::max()});
        allocator_ = CreateAllocator(default_memory_info, VULKAN, false);
        InsertAllocator(allocator_);
    }

    // Get environment variables
    const Env& env_instance = Env::Default();
    
    const std::string dump_subgraphs_env = env_instance.GetEnvironmentVar(stvm_env_vars::kDumpSubgraphs);
    if (!dump_subgraphs_env.empty()) {
        dump_subgraphs_ = (std::stoi(dump_subgraphs_env) == 0 ? false : true);
    }

}

StvmExecutionProvider::~StvmExecutionProvider() {}

AllocatorPtr StvmExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeDefault) {
    return allocator_;
  } else {
    return IExecutionProvider::GetAllocator(id, mem_type);
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
StvmExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                         const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  if (graph_viewer.IsSubgraph()) {
      return result;
  }
#if 1
  // Construct modelproto from graph
  onnxruntime::Model model(graph_viewer.Name(), true, ModelMetaData(), PathString{},  IOnnxRuntimeOpSchemaRegistryList(), graph_viewer.DomainToVersionMap(), std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
  onnxruntime::Graph& graph_build = model.MainGraph();

  for (const auto& node : graph_viewer.Nodes()) {
      std::vector<onnxruntime::NodeArg*> inputs, outputs;
      for (auto input : node.InputDefs()) {
          auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
          inputs.push_back(&n_input);
      }
      for (auto output : node.OutputDefs()) {
          auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
          outputs.push_back(&n_output);
      }
      graph_build.AddNode(node.Name(), node.OpType(), node.Description(), inputs, outputs, &node.GetAttributes(), node.Domain());
  }

  const auto& init_tensors = graph_viewer.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
      graph_build.AddInitializedTensor(*(tensor.second));
  }
  ORT_ENFORCE(graph_build.Resolve().IsOK());

  ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto status = graph_build.Resolve();
  std::string onnx_string_buffer;
  model_proto.SerializeToString(&onnx_string_buffer);
#endif  

  std::unordered_set<std::string> required_initializers;
  const std::vector<NodeIndex>& sorted_nodes = graph_viewer.GetNodesInTopologicalOrder();
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
  for (auto& node_idx : sorted_nodes) {
      graph_viewer.GetNode(node_idx)->ForEachDef([&required_initializers, &init_tensors](const onnxruntime::NodeArg& node_arg, bool is_input) {
              if(is_input && init_tensors.count(node_arg.Name())) {
                  required_initializers.insert(node_arg.Name());
              }}, true);
  }
      
  auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = "TVMStandalone";
  meta_def->domain = "StandaloneTest";
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
      
  for (auto& nodeArgPtr : graph_viewer.GetInputs()) {
       inputs.push_back(nodeArgPtr->Name());
   }

  for (auto& name : required_initializers) {
      inputs.push_back(name);
  }

  for (auto& nodeArgPtr : graph_viewer.GetOutputs()) {
      outputs.push_back(nodeArgPtr->Name());
  }
  meta_def->inputs = inputs;
  meta_def->outputs = outputs;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  sub_graph->SetMetaDef(std::move(meta_def));
  sub_graph->nodes = sorted_nodes;
  result.push_back(
                   onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  return result;
}

common::Status StvmExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& nodes,
                                              std::vector<NodeComputeInfo>& node_compute_funcs) {

  for (auto* fused_node : nodes) {
    auto func_body = fused_node->GetFunctionBody();
    if (!func_body)
        return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
    const onnxruntime::Graph& node_graph = func_body->Body();
    onnxruntime::Model model(node_graph.Name(), true, ModelMetaData(), PathString(),
                             IOnnxRuntimeOpSchemaRegistryList(), node_graph. DomainToVersionMap(),
                             std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
    ONNX_NAMESPACE::ModelProto model_proto = model.ToProto();

    model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    *(model_proto.mutable_graph()) = node_graph.ToGraphProto();
    auto opset = model_proto.add_opset_import();
    opset->set_domain(kOnnxDomain);
    opset->set_version(node_graph.DomainToVersionMap().at(kOnnxDomain));

    std::string string_buf;
    model_proto.SerializeToString(&string_buf);

    std::fstream dump("/tmp/" + fused_node->Name() + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
    model_proto.SerializeToOstream(&dump);

    const std::string func_name = fused_node->Name();

    auto compiler = [this, model_proto, string_buf](std::string func_name, const std::vector<std::vector<int64_t>>& input_shapes) -> tvm::runtime::Module* {
        if (modules_.count(func_name)) {
            return modules_[func_name].get();
        }

        tvm::runtime::Module mod_f = TVMCompile(string_buf, backend_type_, "llvm", 3, input_shapes);
        auto module_ptr = std::make_shared<tvm::runtime::Module>();
        *module_ptr = mod_f;
        modules_[func_name] = module_ptr;
        return modules_[func_name].get();
    };
    
    NodeComputeInfo compute_info;

    compute_info.create_state_func = [compiler](ComputeContext* context, FunctionState* state) {
        auto* p = new STVMFuncState();
        *p = {context->allocate_func, context->release_func, context->allocator_handle, nullptr, compiler};
        *state = p;
        return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
        if (state)
            delete static_cast<STVMFuncState*>(state);
    };

    compute_info.compute_func = [func_name](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
        Ort::CustomOpApi ort{*api};
        STVMFuncState* tvm_state = reinterpret_cast<STVMFuncState*>(state);
        std::vector<std::vector<int64_t>> input_shapes;
        std::vector<std::vector<int64_t>> output_shapes;
        size_t num_inputs = ort.KernelContext_GetInputCount(context);
        std::vector<DLTensor> dl_tensors_inputs(num_inputs);

        for (auto i = 0u; i < num_inputs; i++) {
            const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
            ORT_ENFORCE(input_tensor->IsTensor());
            const Tensor& tensor = input_tensor->Get<onnxruntime::Tensor>();
            const OrtDevice& device = tensor.Location().device;
            auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
            auto tensor_type = ort.GetTensorElementType(tensor_info);
            input_shapes.emplace_back(ort.GetTensorShape(tensor_info));
            ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

            dl_tensors_inputs[i].ctx = GetDLContext(device);
            dl_tensors_inputs[i].dtype = GetDataType(tensor_type);
            dl_tensors_inputs[i].strides = nullptr;
            dl_tensors_inputs[i].byte_offset = 0;
            dl_tensors_inputs[i].data = const_cast<double*>(ort.GetTensorData<double>(input_tensor));
            dl_tensors_inputs[i].ndim = input_shapes.back().size();
            dl_tensors_inputs[i].shape = input_shapes.back().data();
        }

        size_t num_outputs = ort.KernelContext_GetOutputCount(context);
        std::vector<DLTensor> dl_tensors_outputs(num_outputs);

        for (auto i = 0u; i < num_outputs; i++) {
            //setup output tensor property
            //todo: type should be set by framework.
            output_shapes.push_back(input_shapes[i]);
            OrtValue* output_tensor = ort.KernelContext_GetOutput(context, i, output_shapes[i].data(), output_shapes[i].size());
            ORT_ENFORCE(output_tensor->IsTensor());
            const Tensor& tensor = output_tensor->Get<onnxruntime::Tensor>();
            const OrtDevice& device = tensor.Location().device;
            auto tensor_info = ort.GetTensorTypeAndShape(output_tensor);
            auto tensor_type = ort.GetTensorElementType(tensor_info);
            ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
            
            dl_tensors_outputs[i].ctx = GetDLContext(device);
            dl_tensors_outputs[i].dtype = GetDataType(tensor_type);
            dl_tensors_outputs[i].strides = nullptr;
            dl_tensors_outputs[i].byte_offset = 0;
            dl_tensors_outputs[i].data = ort.GetTensorMutableData<double>(output_tensor);
            dl_tensors_outputs[i].ndim = output_shapes.back().size();
            dl_tensors_outputs[i].shape = output_shapes.back().data();
        }

        tvm::runtime::Module* mod = tvm_state->compiler(func_name, input_shapes);
        tvm::runtime::TVMRetValue rvalue;
        TVMRun(*mod, dl_tensors_inputs, dl_tensors_outputs, &rvalue);
        return Status::OK();
    };
    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

std::unique_ptr<onnxruntime::IDataTransfer> StvmExecutionProvider::GetDataTransfer() const {
    if (backend_type_.find("vulkan") != std::string::npos) {
        return onnxruntime::make_unique<onnxruntime::GPUDataTransfer>();
    } else if (backend_type_.find("llvm") != std::string::npos) {
        return onnxruntime::make_unique<onnxruntime::CPUDataTransfer>();
    } else {
        ORT_NOT_IMPLEMENTED("STVM GetDataTransfer");
    } 
}
    
}  // namespace onnxruntime

