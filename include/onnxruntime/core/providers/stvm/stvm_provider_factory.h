// Copyright 2020 AMD
// Licensed under the MIT License

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Stvm, _In_ OrtSessionOptions* options, _In_ const char* backend_type);

#ifdef __cplusplus
}
#endif


