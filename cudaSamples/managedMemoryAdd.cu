/*
Simple example on using the Unified Memory
https://devblogs.nvidia.com/parallelforall/

To compile
nvcc managedMemoryAdd.cu -o managedMemoryAdd

To profile
nvprof ./managedMemoryAdd
*/
#include <iostream>
#include <math.h>

// Simple kernel to add elements

__global__ void addSingleThread(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

__global__ void addMoreThreads(int n, float *x, float *y)
{
  // Let the kernel calculate which part of the input signal to play with
  int index = threadIdx.x;
  int stride = blockDim.x;

  // Just did this to keep the syntax similar to the previous example
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
__global__ void addGridThreads(int n, float *x, float *y)
{
  // Let the kernel calculate which part of the input signal to play with, but
  // now also include the grid information
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  // N will be 1 million (1048576)
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU

  // On this case a grid made of one block, where this block has 1 thread
  addSingleThread<<<1, 1>>>(N, x, y);

  // Now we have a grid of one block and this block has 256 threads
  addMoreThreads<<<1, 256>>>(N, x, y);

  // Now we calculate the grid dimensions
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  addGridThreads<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
