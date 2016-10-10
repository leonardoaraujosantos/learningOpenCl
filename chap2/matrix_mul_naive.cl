__kernel void matrix_2d_mul_float_gpu(__global float *A, __global float *B,  __global float* C, int num_rows_A, int num_cols_A, int num_cols_B)
{
   int i = get_global_id(0);
   int k = get_global_id(1);
   if (i > num_rows_A || k > num_cols_B){
     //printf("Invalid index [%d,%d]\n",i,k);
     return;
   }

   // Sum is on the register(local to each thread)
  float sum = 0;

   // This iterate a lot on the global memory 2*j times
  for (int j=0; j<num_cols_A; j++){
    // A[i][j] == A[i*num_cols_A+j]
    // B[j][k] == B[j*num_cols_B+k]
    //sum += A[i][j]*B[j][k];
    sum += A[i*num_cols_A+j]*B[j*num_cols_B+k];
  }

  // And now one more time
  //printf("Sum %f\n",sum);
  C[i*num_cols_B+k]=sum;
}
