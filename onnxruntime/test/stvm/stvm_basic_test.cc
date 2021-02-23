// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include <iostream>
#include <dlfcn.h>
#include <string>

#include <dlpack/dlpack.h>

#include "core/graph/onnx_protobuf.h"
#include "core/graph/model.h"

#include <fstream>
#include <tvm/runtime/module.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/driver/jit_interface.h>
#include "Python.h"

namespace {

std::wstring wide_str(const std::string& str)
{
  std::wostringstream wstm ;
  const std::ctype<wchar_t>& ctfacet = std::use_facet<std::ctype<wchar_t>>(wstm.getloc());
  for(size_t i = 0; i < str.size(); ++i)
  {
    wstm << ctfacet.widen(str[i]);
  }
  return wstm.str();
}

}


namespace onnxruntime {

static DLDataType GetDataType(ONNXTensorElementDataType type) {
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    return {kDLFloat, 64, 1};
  } else
    ORT_THROW("not implement.");
}

namespace test {

struct TVMFuncState {
  AllocateFunc test_allocate_func = nullptr;
  DestroyFunc test_release_func = nullptr;
  AllocatorHandle allocator = nullptr;
  tvm::runtime::Module* module = nullptr;
  std::function<tvm::runtime::Module*(std::string func_name, const std::vector<std::vector<int64_t>>& input_shapes)> compiler = nullptr;
};

class TVMExecutionProviderX : public CPUExecutionProvider {
 public:
  explicit TVMExecutionProviderX(const CPUExecutionProviderInfo& info) : CPUExecutionProvider(info) {
    Py_Initialize();
    auto pythonpath = std::getenv("PYTHONPATH");
    if (pythonpath)
    {
      PySys_SetPath(wide_str(pythonpath).c_str());
    }
     py_name_ = PyUnicode_FromString("tvm.relay");
     py_module_ = PyImport_Import(py_name_);
     PyRun_SimpleString("print(0)\n");
     // PyRun_SimpleString("import tvm.relay");
  }
  ~TVMExecutionProviderX() {
    Py_DECREF(py_name_);
    Py_DECREF(py_module_);
    Py_Finalize();
  }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override {
      std::vector<std::unique_ptr<ComputeCapability>> result;
      if (graph_viewer.IsSubgraph()) {
          return result;
      }

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
      
      for (auto& nodeArgPtr : graph_viewer.GetInputs())
       {
           inputs.push_back(nodeArgPtr->Name());
       }

      for (auto& name : required_initializers)
      {
          inputs.push_back(name);
      }

      for (auto& nodeArgPtr : graph_viewer.GetOutputs())
      {
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

  common::Status Compile(const std::vector<onnxruntime::Node*>& nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override {
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
            if (modules_.count(func_name))
            {
              return modules_[func_name].get();
            }

            void * handle = dlopen("/usr/local/lib/libtvm.so", RTLD_LAZY);
            if (!handle)
              std::cerr << "Cannot load library: " << dlerror() << '\n';
            dlerror();

            typedef tvm::runtime::Module compile_entry_t(const std::string&, const std::string&, const std::string&, int, const std::vector<std::vector<int64_t>>&);
            compile_entry_t* compile_entry = (compile_entry_t*) dlsym(handle, "TVMCompile");

            const char* dlsym_error = dlerror();
            if (dlsym_error) {
              std::cerr << "Cannot load symbol create: " << dlsym_error << '\n';
            }

            tvm::runtime::Module mod_f = compile_entry(string_buf, "llvm", "llvm", 3, input_shapes);

            auto module_ptr = std::make_shared<tvm::runtime::Module>();
            *module_ptr = mod_f;
            modules_[func_name] = module_ptr;
            dlclose(handle);
            return modules_[func_name].get();
          };

          NodeComputeInfo compute_info;

          compute_info.create_state_func = [compiler](ComputeContext* context, FunctionState* state) {
              auto* p = new TVMFuncState();
              *p = {context->allocate_func, context->release_func, context->allocator_handle, nullptr, compiler};
              *state = p;
              return 0;
          };

          compute_info.release_state_func = [](FunctionState state) {
              if (state)
                  delete static_cast<TVMFuncState*>(state);
          };

          compute_info.compute_func = [func_name](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
              Ort::CustomOpApi ort{*api};
              TVMFuncState* tvm_state = reinterpret_cast<TVMFuncState*>(state);
              std::vector<std::vector<int64_t>> input_shapes;
              std::vector<std::vector<int64_t>> output_shapes;

              DLContext cpu_context = {kDLCPU, 0};
              size_t num_inputs = ort.KernelContext_GetInputCount(context);
              std::vector<DLTensor> dl_tensors_inputs(num_inputs);

              for (auto i = 0u; i < num_inputs; i++) {
                  const OrtValue* input_tensor = ort.KernelContext_GetInput(context, i);
                  auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
                  auto tensor_type = ort.GetTensorElementType(tensor_info);
                  input_shapes.emplace_back(ort.GetTensorShape(tensor_info));
                  ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
                  dl_tensors_inputs[i].ctx = cpu_context;
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

                  auto tensor_info = ort.GetTensorTypeAndShape(output_tensor);
                  auto tensor_type = ort.GetTensorElementType(tensor_info);
                  ort.ReleaseTensorTypeAndShapeInfo(tensor_info);

                  dl_tensors_outputs[i].ctx = cpu_context;
                  dl_tensors_outputs[i].dtype = GetDataType(tensor_type);
                  dl_tensors_outputs[i].strides = nullptr;
                  dl_tensors_outputs[i].byte_offset = 0;
                  dl_tensors_outputs[i].data = ort.GetTensorMutableData<double>(output_tensor);
                  dl_tensors_outputs[i].ndim = output_shapes.back().size();
                  dl_tensors_outputs[i].shape = output_shapes.back().data();
              }

              tvm::runtime::Module* mod = tvm_state->compiler(func_name, input_shapes);

              void * run_handle = dlopen("/usr/local/lib/libtvm.so", RTLD_LAZY);
              if (!run_handle)
                  std::cerr << "Cannot load library: " << dlerror() << '\n';

              typedef void run_entry_t(tvm::runtime::Module&, std::vector<DLTensor>&, std::vector<DLTensor>&, tvm::runtime::TVMRetValue*);

              run_entry_t* run_entry = (run_entry_t*) dlsym(run_handle, "TVMRun");
              tvm::runtime::TVMRetValue rvalue;
              run_entry(*mod, dl_tensors_inputs, dl_tensors_outputs, &rvalue);

              return Status::OK();
          };
          node_compute_funcs.push_back(compute_info);
      }
      return Status::OK();
  }
 private:
    PyObject *py_name_;
    PyObject *py_module_;
    std::unordered_map<std::string, std::shared_ptr<tvm::runtime::Module>> modules_;
};

    
static void RunSession(InferenceSession& session_object,
                       RunOptions& run_options,
                       std::vector<int64_t>& dims_x,
                       std::vector<double>& values_x,
                       std::vector<int64_t>& dims_y,
                       std::vector<double>& values_y) {
  // prepare inputs
  OrtValue ml_value;
  CreateMLValue<double>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_x, values_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X1", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y4");
  std::vector<OrtValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  EXPECT_TRUE(st.IsOK());
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(dims_y);
  EXPECT_EQ(expected_shape, rtensor.Shape());
  const std::vector<double> found(rtensor.template Data<double>(), rtensor.template Data<double>() + expected_shape.Size());
  ASSERT_EQ(found.size(), values_y.size());
  for (size_t i = 0; i < found.size(); i++)
    ASSERT_EQ(found[i], values_y[i]);
}

static const std::string MODEL_URI = "testdata/fuse_mul_1.onnx";

TEST(TVMTest, Standalone_tvm) {
    SessionOptions so;
    so.session_logid = "InferenceSessionTests.NoTimeout";
    InferenceSession session_object{so, GetEnvironment()};
    CPUExecutionProviderInfo info;
    auto tvm_xp = onnxruntime::make_unique<TVMExecutionProviderX>(info);
    EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(tvm_xp)).IsOK());
    EXPECT_TRUE(session_object.Load(MODEL_URI).IsOK());
    EXPECT_TRUE(session_object.Initialize().IsOK());

    RunOptions run_options;
    run_options.run_tag = "one session/one tag";

    // prepare inputs
    std::vector<int64_t> dims_x = {
        6,
    };
    std::vector<double> values_x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // prepare expected inputs and outputs
    std::vector<int64_t> expected_dims_y = {
        6,
    };
    // now the expected value should be Mul's result.
    std::vector<double> expected_values_y = {1.0, 32.0, 243.0, 1024.0, 3125.0, 7776.0};
    // Now run
    RunSession(session_object, run_options, dims_x, values_x, expected_dims_y, expected_values_y);
}
    
}  // namespace test

}  // namespace onnxruntime


