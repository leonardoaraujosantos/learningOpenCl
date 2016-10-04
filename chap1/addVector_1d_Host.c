/*
Compiling on Linux
g++ -I /usr/local/cuda-7.5/include/ -L /usr/local/cuda-7.5/lib64 -o
addVector_1d_Host addVector_1d_Host.c -lOpenCL -lm
*/
#include <alloca.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef APPLE
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

int main() {
  // Length of vectors
  unsigned int n = 13;

  // Host input vectors
  float *h_a;
  float *h_b;
  // Host output vector
  float *h_c;

  // Device input buffers
  cl_mem d_a;
  cl_mem d_b;
  // Device output buffer
  cl_mem d_c;

  // Size, in bytes, of each vector
  size_t sizeBytesVectors = n * sizeof(float);
  printf("Each vector has %d bytes with %d elements of type float\n",
         (int)sizeBytesVectors, n);

  // Allocate memory for each vector on host
  h_a = (float *)malloc(sizeBytesVectors);
  h_b = (float *)malloc(sizeBytesVectors);
  h_c = (float *)malloc(sizeBytesVectors);

  // Initialize vectors on host
  for (int i = 0; i < n; i++) {
    h_a[i] = (float)i;
    h_b[i] = (float)i + 1;
    h_c[i] = 0;
  }

  size_t globalSize, localSize;
  cl_int err;
  // Number of work items in each local work group
  localSize = 3;
  // Number of total work items
  globalSize = ceil(n / (float)localSize) * localSize;
  printf("LocalSize=%d globalSize=%d\n", (int)localSize, (int)globalSize);

  // OpenCL part
  cl_platform_id cpPlatform; // OpenCL platform
  cl_device_id device_id;    // device ID
  cl_context context;        // context
  cl_command_queue queue;    // command queue
  cl_program program;        // program
  cl_kernel kernel;          // kernel

  // Load the source code containing the kernel
  FILE *fp;
  char fileName[] = "./addVec.cl";
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

  // Bind to platform
  err = clGetPlatformIDs(1, &cpPlatform, NULL);
  // Get ID for the device
  err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  // Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  // Create a command queue
  queue = clCreateCommandQueue(context, device_id, 0, &err);

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                                      (const size_t *)&source_size, &err);
  // Build the program (with our kernel)
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
  kernel = clCreateKernel(program, "addVec", &err);

  // Create the input and output arrays in device memory for our calculation
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeBytesVectors, NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeBytesVectors, NULL, NULL);
  d_c =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeBytesVectors, NULL, NULL);

  // Copy host buffers to GPU memory
  err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, sizeBytesVectors, h_a, 0,
                             NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, sizeBytesVectors, h_b, 0,
                              NULL, NULL);

  // Set the arguments to our compute kernel addVec(a,b,c,n)
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
  err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

  // Execute the kernel over the entire range of the data set
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                               0, NULL, NULL);

  // Wait for the command queue to get serviced before reading back results
  clFinish(queue);

  // Copy from GPU memory to host memory
  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, sizeBytesVectors, h_c, 0, NULL,
                      NULL);

  // Display the result
  for (int i = 0; i < n; i++)
    printf("Z[%d]=%3.2f\n", i, h_c[i]);

  // release OpenCL resources
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // release host memory
  free(h_a);
  free(h_b);
  free(h_c);
}
