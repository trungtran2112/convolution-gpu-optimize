#include "cuda_kernel.h"
#include<iostream>
using namespace std;

#define TILE_WIDTH 16
__constant__ float constMem[2400]; //K*K*C_in*C_out


// Convolution forward kernel: Naive implementation
__global__ void conv_forward_kernel(const float *in, float *out, const float *weight,
                                    const int channel_in, const int channel_out,
                                    const int height_in, const int width_in, const int kernel_width)
{
    const int height_out = height_in - kernel_width + 1; //24
    const int width_out = width_in - kernel_width + 1; //24

    int height_grid = (height_out - 1) / TILE_WIDTH + 1; //2
    int width_grid = (width_out - 1) / TILE_WIDTH + 1; //2

    int sample_idx = blockIdx.z; //cho biết ảnh thứ mấy trong batch
    int out_feature_idx = blockIdx.x; //cho biết đang xét kernel thứ mấy

    int row = (blockIdx.y / width_grid) * TILE_WIDTH + threadIdx.y; //tính cái dòng hiện tại trong input
    int col = (blockIdx.y % width_grid) * TILE_WIDTH + threadIdx.x; //tính cái cột hiện tại trong input

    float sum = 0;

    if (row < height_out && col < width_out)
    {
      int hw_in = height_in * width_in; //28x28
      int hw_out = height_out * width_out; //24x24

      for (int i = 0; i < channel_in; i++)
      {
          for (int j = 0; j < kernel_width; j++)
          {
              for (int k = 0; k < kernel_width; k++)
              {
                  int x = row + j;
                  int y = col + k;
                  sum += in[sample_idx * channel_in * hw_in + i * hw_in + //sample_idx * channel_in * hw_in tính từ vị trí đầu đến channel khác
                              x * width_in + y] *           //i * hw_in chọn lớp ảnh
                          weight[out_feature_idx * channel_in * kernel_width * kernel_width +
                              i * kernel_width * kernel_width + j * kernel_width + k];
              }
          }
      }
      out[sample_idx * channel_out * hw_out + out_feature_idx * hw_out + row * width_out + col] = sum;
    }

    
}


// Convolution forward kernel: Shared memory implementation
__global__ void conv_forward_kernel_2(const float *X, float *out, const float *W, const int B,
                                      const int C_in, const int C_out,
                                      const int H_in, const int W_in, const int K)
{
  int m, h_base, w_base, h,w; 
  int X_tile_width = TILE_WIDTH + K-1; 
  extern __shared__ float shmem[]; 
  float* X_shared = &shmem[0]; 
  float* W_shared = &shmem[X_tile_width * X_tile_width];

  const int H_out = H_in - K + 1; //24
  const int W_out = W_in - K + 1; //24
  int W_grid = (W_out - 1) / TILE_WIDTH + 1; //2

  m = blockIdx.x; 
  h_base = (blockIdx.y / W_grid) * TILE_WIDTH; // vertical base out data index for the block 
  w_base = (blockIdx.y % W_grid) * TILE_WIDTH; // horizontal base out data index for the block  
  
  int tx = threadIdx.x; 
  int ty = threadIdx.y; 
  h = h_base + ty; 
  w = w_base + tx; 
  int sample_idx = blockIdx.z;
  float acc = 0.; 
  for (int c = 0; c < C_in; c++)
  {
    //load W vào shared memory
    if (( ty < K) && ( tx < K)) 
    {
      W_shared[ty * K + tx]= W[m * C_in * K * K + c * K * K + ty * K + tx];
    }
    // else
    // {
    //   W_shared[ty * K + tx] = 0;
    // }
    __syncthreads(); 

 
    //load từng block từ X sang shared_memory
    for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) 
    { 
      for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) 
      {
        if(i < H_in && j < W_in)
        {
          X_shared[(i - h_base)*X_tile_width + (j - w_base)] = X[sample_idx * C_in * H_in * W_in + W_in * H_in * c + i * W_in + j]; 
        }
        else
        {
          X_shared[(i - h_base)*X_tile_width + (j - w_base)] = 0;
        }
      }
    } 
    __syncthreads(); 

    //Tính tích chập trên 1 kernel
    for (int p=0; p<K; p++) 
    {
      for (int q=0; q<K; q++) {
        if (((ty+p)<X_tile_width) && ((tx+q)<X_tile_width)) {
          acc += X_shared[(ty+p)*X_tile_width+(tx+q)]*W_shared[p*K+q];
        }
      }
    }
    __syncthreads(); 
  }

  if( sample_idx < B && m<C_out && h<H_out && w<W_out) {
   out[sample_idx * C_out * H_out * W_out + m  * H_out * W_out + h * W_out + w] = acc;
  }
}




// Convolution forward kernel: Shared memory and constant memory implementation
__global__ void conv_forward_kernel_3(const float *X, float *out, const float *W, const int B,
                                      const int C_in, const int C_out,
                                      const int H_in, const int W_in, const int K)
{
  /*
    Shared memory: (TILE_WIDTH + K -1)*(TILE_WIDTH + K -1) (Block)
    Constant memory: K*K (Kernel)
  */
  int m, h_base, w_base, h,w; 
  int X_tile_width = TILE_WIDTH + K-1; 
  extern __shared__ float shmem[]; 
  float* X_shared = &shmem[0]; 


  const int H_out = H_in - K + 1; //24
  const int W_out = W_in - K + 1; //24
  int W_grid = (W_out - 1) / TILE_WIDTH + 1; //2

  m = blockIdx.x; 
  h_base = (blockIdx.y / W_grid) * TILE_WIDTH; // vertical base out data index for the block 
  w_base = (blockIdx.y % W_grid) * TILE_WIDTH; // horizontal base out data index for the block  
  
  int tx = threadIdx.x; 
  int ty = threadIdx.y; 
  h = h_base + ty; 
  w = w_base + tx; 
  int sample_idx = blockIdx.z;
  float acc = 0.; 
  float weight = 0.;

  for (int c = 0; c < C_in; c++)
  {
    //load từng block từ X sang shared_memory
    for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) 
    { 
      for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) 
      {
        if(sample_idx < B && i < H_in && j < W_in)
        {
          X_shared[(i - h_base)*X_tile_width + (j - w_base)] = X[sample_idx * C_in * H_in * W_in + W_in * H_in * c + i * W_in + j]; 
        }
        else
        {
          X_shared[(i - h_base)*X_tile_width + (j - w_base)] = 0.0;
        }
      }
    } 
    __syncthreads(); 

    //Tính tích chập trên 1 kernel
    for (int p=0; p<K; p++) 
    {
      for (int q=0; q<K; q++) {
        if (((ty+p)<X_tile_width) && ((tx+q)<X_tile_width)) {
          weight = constMem[m*C_in*K*K + c*K*K + p*K + q];
          acc += X_shared[(ty+p)*X_tile_width+(tx+q)]*weight;
        }
      }
    }
    __syncthreads(); 
  }

  if( sample_idx < B && m<C_out && h<H_out && w<W_out) {
   out[sample_idx * C_out * H_out * W_out + m  * H_out * W_out + h * W_out + w] = acc;
  }
}




__host__ void cuda_kernel::conv_forward(const float *in, float *out, const float *w,
                                         const int B, const int C_in, const int C_out,
                                         const int H_in, const int W_in, const int K, cuda_kernel::kernel_type kernel)
{
  float *d_in;
  float *d_out;
  float *d_w;

  const int H_out = H_in - K + 1;
  const int W_out = W_in - K + 1;

  int inputArrayLength = B*C_in*H_in*W_in;
  int outputArrayLength = B*C_out*H_out*W_out;
  int kernelArrayLength = C_out*C_in*K*K;
  cudaMalloc((void**) &d_in, inputArrayLength*sizeof(float));
  cudaMalloc((void**) &d_out, outputArrayLength*sizeof(float));

  cudaMemcpy(d_in, in, inputArrayLength*sizeof(float), cudaMemcpyHostToDevice);

  if(kernel == cuda_kernel::shared || kernel == cuda_kernel::naive){
    cudaMalloc((void**) &d_w, kernelArrayLength*sizeof(float));
    cudaMemcpy(d_w, w, kernelArrayLength*sizeof(float), cudaMemcpyHostToDevice);
  }
  else{
    cudaMemcpyToSymbol(constMem, w, kernelArrayLength*sizeof(float));
  }

  int grid = ((H_out - 1) / TILE_WIDTH + 1) * ((W_out - 1) / TILE_WIDTH + 1);
  dim3 dimGrid(C_out, grid , B);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  if(kernel == cuda_kernel::naive) // Naive implementation
  {
    conv_forward_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, d_w, C_in, C_out, H_in, W_in, K);
  }
  else if(kernel == cuda_kernel::shared) // Shared memory implementation
  {
    size_t s_mem = ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K*K)  * sizeof(float);
    conv_forward_kernel_2<<<dimGrid, dimBlock, s_mem>>>(d_in, d_out, d_w, B, C_in, C_out, H_in, W_in, K);
  }
  else if(kernel == cuda_kernel::shared_constmem) // Shared + Const memory implementation
  {
    size_t s_mem = (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1)  * sizeof(float);
    conv_forward_kernel_3<<<dimGrid, dimBlock, s_mem>>>(d_in, d_out, d_w, B, C_in, C_out, H_in, W_in, K);
  }
  else
  {
    cout << "Invalid kernel type" << endl;
  }

  CHECK(cudaMemcpy(out, d_out, outputArrayLength * sizeof(float), cudaMemcpyDeviceToHost));

  if(kernel == cuda_kernel::shared || kernel == cuda_kernel::naive){
    CHECK(cudaFree(d_w));
  }
  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
}



