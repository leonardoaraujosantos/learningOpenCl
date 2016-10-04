// Kernels never return they are always void. Also notice that our Parameters
// are pointers to the global memory or scalar parameters.
__kernel void addVec( __global float *a,
                      __global float *b,
                      __global float *z,
                      unsigned int sizeVec)
{
  // Get work-item id for work_group dimension 0
  int idx = get_global_id(0);

  // Avoid working out of vector boundaries
  if (idx >= sizeVec) {
    printf("Invalid index %d\n",idx);
    return;
  }

  // Each valid work item will compute this...
  z[idx] = a[idx] + b[idx];
  printf("a[%d](%f) + b[%d](%f) = %f\n",idx,a[idx],idx,b[idx],z[idx]);
}
