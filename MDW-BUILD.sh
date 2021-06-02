#!/bin/bash

./build.sh \
  --config Release \
  --skip_tests \
  --parallel \
  --cudnn_home /usr/include/x86_64-linux-gnu/ \
  --cuda_home /usr/local/cuda \
  --use_tensorrt \
  --tensorrt_home /opt/TensorRT-7.2.3.4/ && \
./build.sh \
  --config Release \
  --skip_tests \
  --parallel \
  --cudnn_home /usr/include/x86_64-linux-gnu/ \
  --cuda_home /usr/local/cuda \
  --use_tensorrt \
  --tensorrt_home /opt/TensorRT-7.2.3.4/ \
  --build_shared_lib && \
./build.sh \
  --config Release \
  --skip_tests \
  --parallel \
  --cudnn_home /usr/include/x86_64-linux-gnu/ \
  --cuda_home /usr/local/cuda \
  --use_tensorrt \
  --tensorrt_home /opt/TensorRT-7.2.3.4/ \
  --build_wheel

