// Kernels never return they are always void. Also notice that our Parameters
// are pointers to the global memory or scalar parameters.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void addVec( __global float *a,
                      __global float *b,
                      __global float *z,
                      uint32_t sizeVec)
{
  // Get work-item id for work_group dimension 0
  size_t idx = get_global_id(0);

  // Avoid working out of vector boundaries
  if (idx >= sizeVec) {
    printf("Invalid index\n");
    return;
  }

  // Each valid work item will compute this...
  z[idx] = a[idx] + b[idx];
  printf("a[%d](%f) + b[%d](%f) = %f\n",idx,a[idx],idx,b[idx],z[idx]);
}
