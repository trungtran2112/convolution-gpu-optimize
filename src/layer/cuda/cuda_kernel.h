#ifndef SRC_LAYER_CUDA_CUDA_KERNEL_H_
#define SRC_LAYER_CUDA_CUDA_KERNEL_H_
#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"


class cuda_kernel
{
public:
    enum kernel_type {naive = 1, shared = 2, shared_constmem = 3};
    void get_device_info();
    void conv_forward(const float *in, float *out, const float *w,
                                         const int B, const int C_in, const int C_out,
                                         const int H_in, const int W_in, const int K, kernel_type kernel);

};

#endif // SRC_LAYER_CUDA_CUDA_KERNEL_H_