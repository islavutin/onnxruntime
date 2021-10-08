// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>

#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/platform/env.h"
#include "core/graph/model.h"
#include "core/common/cpuid_info.h"

#include "stvm_execution_provider.h"
#include "xpu_data_transfer.h"
#include "stvm_allocator.h"
#include "stvm_utils.h"
#include "stvm_api.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {

// Information to construct kernel function state.
struct STVMFuncState {
  AllocateFunc allocate_func = nullptr;
  DestroyFunc release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  tvm::runtime::Module* module = nullptr;
  std::function<tvm::runtime::Module*(std::string func_name, const std::vector<std::vector<int64_t>>& input_shapes)> compiler = nullptr;
};

class STVMCompiler {
  public:
    STVMCompiler() = delete;
    ~STVMCompiler() = default;

    STVMCompiler(StvmExecutionProvider* ep,
                 const std::string& string_buf) :
      ep_(ep),
      buffer_(string_buf) {}

    tvm::runtime::Module* operator()(std::string func_name, const std::vector<std::vector<int64_t>>& input_shapes) {
      if (ep_->modules_.count(func_name)) {
        return ep_->modules_[func_name].get();
      }

      tvm::runtime::Module mod_f = stvm::TVMCompile(buffer_, ep_->info_.target, ep_->info_.target_host, ep_->info_.opt_level, input_shapes);
      auto module_ptr = std::make_shared<tvm::runtime::Module>();
      *module_ptr = mod_f;
      ep_->modules_[func_name] = module_ptr;
      return ep_->modules_[func_name].get();
    }

  private:
    StvmExecutionProvider* ep_;
    std::string buffer_;
};

class STVMComputer {
  public:
    STVMComputer() = delete;
    ~STVMComputer() = default;

    STVMComputer(const std::string& name) :
      name_(name) {}

    common::Status operator()(FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};
      STVMFuncState* tvm_state = reinterpret_cast<STVMFuncState*>(state);
      std::vector<std::vector<int64_t>> input_shapes;
      size_t num_inputs = ort.KernelContext_GetInputCount(context);
      std::vector<DLTensor> dl_tensors_inputs;

      for (auto i = 0u; i < num_inputs; i++) {
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
        ORT_ENFORCE(input_tensor->IsTensor());
        const Tensor& tensor = input_tensor->Get<onnxruntime::Tensor>();
        const OrtDevice& device = tensor.Location().device;
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        auto tensor_type = ort.GetTensorElementType(tensor_info);
        std::vector<int64_t> input_shape = ort.GetTensorShape(tensor_info);
        int64_t* shape = new int64_t[input_shape.size()];
        for(size_t i = 0; i < input_shape.size(); i++) {
          shape[i] = input_shape[i];
        }
        input_shapes.push_back(input_shape);
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

        DLTensor t;
        t.device = GetDLDevice(device);
        t.dtype = GetDataType(tensor_type);
        t.strides = nullptr;
        t.byte_offset = 0;
        t.data = const_cast<void*>(ort.GetTensorData<void>(input_tensor));
        t.ndim = input_shape.size();
        t.shape = shape;
        dl_tensors_inputs.push_back(t);
      }

      size_t num_outputs = ort.KernelContext_GetOutputCount(context);
      std::vector<DLTensor> dl_tensors_outputs;

      std::vector<std::vector<int64_t>> output_shapes;
      tvm::runtime::Module* mod = tvm_state->compiler(name_, input_shapes);
      stvm::TVMExtractOutputShapes(*mod, num_outputs, output_shapes);

      for (auto i = 0u; i < num_outputs; i++) {
        //setup output tensor property
        OrtValue* output_tensor = ort.KernelContext_GetOutput(context, i, output_shapes[i].data(), output_shapes[i].size());
        ORT_ENFORCE(output_tensor->IsTensor());
        const Tensor& tensor = output_tensor->Get<onnxruntime::Tensor>();
        const OrtDevice& device = tensor.Location().device;
        auto tensor_info = ort.GetTensorTypeAndShape(output_tensor);
        auto tensor_type = ort.GetTensorElementType(tensor_info);
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

        DLTensor t;
        t.device = GetDLDevice(device);
        t.dtype = GetDataType(tensor_type);
        t.strides = nullptr;
        t.byte_offset = 0;
        t.data = ort.GetTensorMutableData<void>(output_tensor);
        t.ndim = output_shapes[i].size();
        t.shape = output_shapes[i].data();
        dl_tensors_outputs.push_back(t);
      }
      tvm::runtime::TVMRetValue rvalue;
      stvm::TVMRun(*mod, dl_tensors_inputs, dl_tensors_outputs, &rvalue);
      return Status::OK();
    }

  private:
    std::string name_;
};

StvmExecutionProvider::StvmExecutionProvider(const StvmExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kStvmExecutionProvider},
      info_{info} {
  ProcessInfo();

  AllocatorCreationInfo default_memory_info = {[](int) {
                                                 return onnxruntime::make_unique<STVMAllocator>();
                                               },
                                               0, false};
  allocator_ = CreateAllocator(default_memory_info);
  InsertAllocator(allocator_);

  // Get environment variables
  const Env& env_instance = Env::Default();

  const std::string dump_subgraphs_env = env_instance.GetEnvironmentVar(stvm_env_vars::kDumpSubgraphs);
  if (!dump_subgraphs_env.empty()) {
    dump_subgraphs_ = (std::stoi(dump_subgraphs_env) == 0 ? false : true);
  }
}

StvmExecutionProvider::~StvmExecutionProvider() {}

AllocatorPtr StvmExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  return allocator_;
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
  onnxruntime::Model model(graph_viewer.Name(), true, ModelMetaData(), PathString{}, IOnnxRuntimeOpSchemaRegistryList(), graph_viewer.DomainToVersionMap(), std::vector<ONNX_NAMESPACE::FunctionProto>(), *GetLogger());
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
              } }, true);
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
                             IOnnxRuntimeOpSchemaRegistryList(), node_graph.DomainToVersionMap(),
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

    STVMCompiler compiler = STVMCompiler(this, string_buf);

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

    compute_info.compute_func = STVMComputer(func_name);

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

std::unique_ptr<onnxruntime::IDataTransfer> StvmExecutionProvider::GetDataTransfer() const {
  LOG(INFO) << "transferring data";
  if (GPUTargetCheck()) {
    return onnxruntime::make_unique<onnxruntime::GPUDataTransfer>();
  } else if (info_.target.find("llvm") != std::string::npos) {
    return onnxruntime::make_unique<onnxruntime::CPUDataTransfer>();
  } else {
    ORT_NOT_IMPLEMENTED("STVM GetDataTransfer is not implemented for target ", info_.target);
  }
}

bool StvmExecutionProvider::GPUTargetCheck() const {
  //TODO(vvchernov): target or target host?
  bool check = (
    info_.target.find("cuda") != std::string::npos ||
    info_.target.find("opencl") != std::string::npos ||
    info_.target.find("metal") != std::string::npos ||
    info_.target.find("vulkan") != std::string::npos
  );
  return check;
}

void StvmExecutionProvider::ProcessInfo() {
  if(info_.target == cpu_target_str ||
     info_.target == llvm_target_str) {
    ProcessCPUTarget();
  } else if(info_.target == gpu_target_str) {
    ProcessGPUTarget();
  } else if(info_.target.empty()) {
    ORT_NOT_IMPLEMENTED("target option is empty!");
  } else {
    // TODO(vvchernov): extend mechanism of auto-definition of target
    // target is gotten from option set up by client
  }

  if((info_.target_host == cpu_target_str ||
      info_.target_host == llvm_target_str) &&
      info_.target_host != info_.target) {
    info_.target_host = info_.target;
  } else if (info_.target_host.empty()) {
    info_.target_host = info_.target;
  } else {
    // TODO(vvchernov): extend mechanism of auto-definition of target host
    // target host is gotten from option set up by client
  }

  if(info_.opt_level < 1) {
    info_.opt_level = default_opt_level;
  }

  PrintInfo();
}

void StvmExecutionProvider::ProcessCPUTarget() {
  const auto& cpu_id_info = CPUIDInfo::GetCPUIDInfo();
  // auto detect from CPU ID
  if (cpu_id_info.HasAVX512Skylake()) {
    info_.target = stvm_cpu_targets::LLVM_TARGET_SKYLAKE_AVX512;
  } else if (cpu_id_info.HasAVX512f()) {
    info_.target = stvm_cpu_targets::LLVM_TARGET_AVX512;
  } else if (cpu_id_info.HasAVX2()) {
    info_.target = stvm_cpu_targets::LLVM_TARGET_AVX2;
  } else if (cpu_id_info.HasAVX()) {
    info_.target = stvm_cpu_targets::LLVM_TARGET_AVX;
  } else  {
    // TODO(vvchernov): extend mechanism of auto-definition of cpu target
    info_.target = llvm_target_str;
  }
}

void StvmExecutionProvider::ProcessGPUTarget() {
  ORT_NOT_IMPLEMENTED("GPU target auto-defenition is not implemented now!");
}

void StvmExecutionProvider::PrintInfo() const {
  std::cout << "STVM ep options:" << std::endl;
  std::cout << "target: " << info_.target << std::endl;
  std::cout << "target_host: " << info_.target_host << std::endl;
  std::cout << "opt level: " << info_.opt_level << std::endl;
  std::cout << "tuning file path: " << info_.tuning_file_path << std::endl;
}

}  // namespace onnxruntime
