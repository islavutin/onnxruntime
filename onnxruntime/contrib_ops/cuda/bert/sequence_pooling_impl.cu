// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "sequence_pooling_impl.h"
#include <cuda_fp16.h>
#include <stdio.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename InputIt, typename OutputIt>
__device__ inline void PrefixSumLinear(const InputIt* first, const InputIt* last, OutputIt* d_first) {
  if (first == last) {
    return;
  }
  OutputIt sum = *first;
  *d_first = sum;
  while (++first != last) {
    sum += *first;
    *(++d_first) = sum;
  }
}

template <typename T>
__global__ void SequencePoolingKernel(const T* input, const int64_t* sentence_lengthes, T* output, T* masks) {
  // blockDim.x -> num_sequences
  // gridDim.x -> hidden_size
  // gridDim.y -> batch_size
  const int hidden_size = gridDim.x;
  const int num_sequences = blockDim.x;

  __shared__ int sentence_lengthes_prefixsum[512]; // num_sequences is small and suppose to be less than 512

  const int offset = blockIdx.y * num_sequences;

  if (threadIdx.x == 0) {
    PrefixSumLinear(sentence_lengthes + offset, sentence_lengthes + offset + num_sequences, sentence_lengthes_prefixsum);
  }

  if (blockIdx.x == 0) {
    *(masks + offset + threadIdx.x) = 1;
  }

  __syncthreads();

  const int sequence_length_for_split = sentence_lengthes_prefixsum[blockDim.x - 1];
  const int past_sequence_length = (threadIdx.x == 0) ? 0 : sentence_lengthes_prefixsum[threadIdx.x - 1];

  const int input_offset = blockIdx.y * hidden_size * sequence_length_for_split + hidden_size * past_sequence_length + blockIdx.x;
  const int output_offset = blockIdx.y * hidden_size * num_sequences + hidden_size * threadIdx.x + blockIdx.x;

  T local_max;
  const int sequence_length = sentence_lengthes_prefixsum[threadIdx.x] - past_sequence_length;
  for (int i = 0; i < sequence_length; ++i) {
    if (i == 0) {
      local_max = *(input + input_offset);
    } else {
      T value = *(input + input_offset + i * hidden_size);
      local_max = (float)value > (float)local_max ? value : local_max;
    }
  }

  *(output + output_offset) = local_max;
}

template <typename T>
bool SequencePooling(
  cudaStream_t stream,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const T* input,
  const int64_t* sentence_lengthes,
  T* output,
  T* masks) {

  const dim3 grid(hidden_size, batch_size, 1);
  const dim3 block(num_sequences, 1, 1);

  SequencePoolingKernel<T><<<grid, block, 0, stream>>>(input, sentence_lengthes, output, masks);

  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchSequencePoolingKernel(
  void* output,
  void* masks,
  const void* input,
  const void* sentence_lengthes,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const size_t element_size) {
  // use default stream
  const cudaStream_t stream = nullptr;

  if (element_size == 2) {
    return SequencePooling<half>(
      stream,
      batch_size,
      hidden_size,
      num_sequences,
      reinterpret_cast<const half*>(input),
      reinterpret_cast<const int64_t*>(sentence_lengthes),
      reinterpret_cast<half*>(output),
      reinterpret_cast<half*>(masks)
    );
  } else {
    return SequencePooling<float>(
      stream,
      batch_size,
      hidden_size,
      num_sequences,
      reinterpret_cast<const float*>(input),
      reinterpret_cast<const int64_t*>(sentence_lengthes),
      reinterpret_cast<float*>(output),
      reinterpret_cast<float*>(masks)
    );
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
