/*
Compiling on Linux
gcc -I /usr/local/cuda-7.5/include/ -L /usr/local/cuda-7.5/lib64 -o hello hello_opencl.c -lOpenCL
*/
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
// Linux
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

int main() {
  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem memobj = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  // String with result of our kernel
  char resultString[MEM_SIZE];

  // Load the source code containing the kernel
  FILE *fp;
  char fileName[] = "./hello_kernel.cl";
  char *source_str;
  size_t source_size;

  fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }

  // Read our kernel to memory (source_str)
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  // Get Platform and Device Info
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  // Create a context
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

  // Create the command Queue
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

  /* Create Memory Buffer */
  memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,MEM_SIZE * sizeof(char), NULL, &ret);

  // Create a program (library of kernels)
  program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
  (const size_t *)&source_size, &ret);

  // Compile the kernel
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  // Create the kernel
  kernel = clCreateKernel(program, "hello", &ret);

  // Prepare the parameters
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);

  // Execute OpenCL Kernel
  ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);

  // Get result from device memory buffer
  ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
    MEM_SIZE * sizeof(char),resultString, 0, NULL, NULL);

    // Display result
    puts(resultString);

    // Finalization
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(source_str);

    return 0;
  }
