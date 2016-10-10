/*
Compiling on Linux
g++ -I /usr/local/cuda-7.5/include/ -L /usr/local/cuda-7.5/lib64 -o
matrix_big_mul_CL matrix_big_mul_CL.c -lOpenCL -lm
// Test on Juno
g++ -I /home/root/work/Mali_OpenCL_SDK_v1.1.0/include -L
/home/root/work/Mali_OpenCL_SDK_v1.1.0/lib -o matrix_big_mul_CL
matrix_big_mul_CL.c -lOpenCL
*/
#include <alloca.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

#define N_THREADS 8
#define N_ITERATIONS 10

int num_rows_A = 2000;
int num_rows_B = 2000;
int num_rows_C = 2000;
int num_cols_A = 2000;
int num_cols_B = 600;
int num_cols_C = 600;

// I'm forcing a malloc because I want to add the malloc time on the game
float *A = (float *)malloc(sizeof(float) * num_rows_A * num_cols_A);
float *B = (float *)malloc(sizeof(float) * num_rows_B * num_cols_B);
float *C = (float *)malloc(sizeof(float) * num_rows_C * num_cols_C);
float *C_ref = (float *)malloc(sizeof(float) * num_rows_C * num_cols_C);

void matrix_2d_mul_float(float *A, float *B, float *C, int num_rows_A,
                         int num_cols_A, int num_cols_B) {
  float sum = 0;
  int num_rows_C = num_rows_A;
  int num_cols_C = num_cols_B;
// Iterate on each row of A
#pragma omp parallel for schedule(dynamic, 1) collapse(2)
  for (int i = 0; i < num_rows_A; i++) {
    // Iterate on each collumn of B
    for (int k = 0; k < num_cols_B; k++) {
      sum = 0;
      // Do the "multiply add between" row of A and collumn of B
      for (int j = 0; j < num_cols_A; j++) {
        // A[i][j] == A[i*num_cols_A+j]
        // B[j][k] == B[j*num_cols_B+k]
        // sum += A[i][j]*B[j][k];
        sum += A[i * num_cols_A + j] * B[j * num_cols_B + k];
      }
      // C[i][k] == C[i*num_cols_C+k]
      C[i * num_cols_C + k] = sum;
    }
  }
}

void fillRand(float *vec, int minValue, int maxValue, int sizeVec) {
  srand(time(NULL));
  for (int idx = 0; idx < sizeVec; idx++) {
    vec[idx] = rand() % maxValue + minValue;
  }
}

int main() {
  // Define pointers to device memory
  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;

  printf("Multiplying 2 matrices A[%d,%d] * B[%d,%d]\n", num_rows_A, num_cols_A,
         num_rows_B, num_cols_B);

  // Get size in bytes for our vectors
  int numBytesA = sizeof(float) * num_rows_A * num_cols_A;
  int numBytesB = sizeof(float) * num_rows_B * num_cols_B;
  int numBytesC = sizeof(float) * num_rows_C * num_cols_C;
  printf("Size in bytes A: %d\n", numBytesA);
  printf("Size in bytes B: %d\n", numBytesB);
  printf("Size in bytes C: %d\n", numBytesC);

  // Fill arrays
  fillRand(A, 1, 10, num_rows_A * num_cols_A);
  fillRand(B, 1, 10, num_rows_B * num_cols_B);
  memset(C, 0, (num_rows_C * num_cols_C) * sizeof(float));

  // OpenCL part
  cl_platform_id cpPlatform;     // OpenCL platform
  cl_device_id device_id;        // device ID
  cl_context context;            // context
  cl_command_queue queue;        // command queue
  cl_program program;            // program
  cl_kernel kernel;              // kernel
  cl_int err;                    // error
  cl_ulong time_start, time_end; // Time
  double total_time;

  // Load the source code containing the kernel
  FILE *fp;
  char fileName[] = "./matrix_mul_naive.cl";
  char *source_str;
  size_t source_size;
  fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  // Read our kernel to memory (source_str)
  source_str = (char *)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  printf("Initializing OpenCL device...\n");
  // Bind to platform
  err = clGetPlatformIDs(1, &cpPlatform, NULL);
  // Get ID for the device
  err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  // Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  // Create a command queue
  // queue = clCreateCommandQueue(context, device_id, 0, &err);
  // Enable queue profile
  queue =
      clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
  if (!queue) {
    printf("Failed to create command queue!\n");
    return -1;
  }

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                      (const size_t *)&source_size, &err);

  // Build the program (with our kernel)
  printf("Compiling OpenCL kernel...\n");
  const char options[] = "-cl-std=CL1.2";
  err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
  if (err != CL_SUCCESS) {
    cl_build_status status;
    char *programLog;
    size_t logSize;
    printf("Kernel compilation error\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS,
                          sizeof(cl_build_status), &status, NULL);
    // check build log
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &logSize);
    programLog = (char *)calloc(logSize + 1, sizeof(char));
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logSize + 1,
                          programLog, NULL);
    programLog[logSize] = '\0';
    printf("Build failed; error=%d, status=%d, programLog:nn%s\n", err, status,
           programLog);
    free(programLog);
    return -1;
  }

  // Create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program, "matrix_2d_mul_float_gpu", &err);

  // Create the input and output arrays in device memory for our calculation
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, numBytesA, NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, numBytesB, NULL, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numBytesC, NULL, NULL);

  // Configure the global size and local size
  size_t localSize[2], globalSize[2];
  // number of work-items that make up a work-group
  localSize[0] = N_THREADS;
  localSize[1] = N_THREADS;

  globalSize[0] = ceil(num_rows_A / (float)N_THREADS) * N_THREADS;
  globalSize[1] = ceil(num_cols_B / (float)N_THREADS) * N_THREADS;

  printf("Global size[%d, %d]\n", (unsigned int)globalSize[0],
         (unsigned int)globalSize[1]);

  // Call sequential function
  for (int idxLoop = 0; idxLoop < N_ITERATIONS; idxLoop++) {
    // Copy matrices A and B to GPU
    clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, numBytesA, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, numBytesB, B, 0, NULL, NULL);

    // Launch the kernel
    // matrix_2d_mul_float(A,B,C,num_rows_A,num_cols_A,num_cols_B);
    // matrix_2d_mul_float_gpu<<<dimGrid,
    // dimBlock>>>(device_A_mat,device_B_mat,device_C_mat,num_rows_A,num_cols_A,num_cols_B);
    // cudaError_t err = cudaThreadSynchronize();

    // Set the arguments to our compute kernel matrix_2d_mul_float_gpu(__global
    // float *A, __global float *B,  __global float* C, int num_rows_A, int
    // num_cols_A, int num_cols_B)
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&num_rows_A);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&num_cols_A);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&num_cols_B);

    if (err != CL_SUCCESS) {
      printf("Error: Failed to set kernel arguments! %d\n", err);
      return -1;
    }

    // Enqueues a command to execute a kernel on a device. (2 dimensions)
    cl_event kernEvent;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize,
                                 0, NULL, &kernEvent);
    if (err != CL_SUCCESS) {
      printf("Failed to launch kernel! %d\n", err);
      return -1;
    }
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Get the result from the GPU to the CPU
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, numBytesC, C, 0, NULL, NULL);

    clGetEventProfilingInfo(kernEvent, CL_PROFILING_COMMAND_START,
                            sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernEvent, CL_PROFILING_COMMAND_END,
                            sizeof(time_end), &time_end, NULL);
    total_time += time_end - time_start;

    printf("Matrix multiplication done %d\n", idxLoop);
  }
  total_time = total_time / N_ITERATIONS;
  printf("\nKernel Execution time = %0.3f ms\n", (total_time / 1000000.0));

  // Calculate one iteration with the reference function
  printf("Calculating reference\n");
  matrix_2d_mul_float(A, B, C_ref, num_rows_A, num_cols_A, num_cols_B);
  printf("Comparing with reference\n");
  float sumDiff = 0;
  for (int i = 0; i < (num_rows_C * num_cols_C); i++) {
    float diff = C_ref[i] - C[i];
    if (diff > 0.01f) {
      printf("Error at Pos[%d]: Values = (REF)%f -- (GPU)%f\n", i, C_ref[i],
             C[i]);
      sumDiff += diff;
    }
  }
  printf("Difference = %f\n", sumDiff);

  // Free memory
  free(A);
  free(B);
  free(C);
  // Release memories from GPU
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);

  clReleaseContext(context);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);

  return 0;
}
