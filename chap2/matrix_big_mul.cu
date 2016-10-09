/*
  Now we make the matrix much bigger
  g++ -pg seq_matrix_big_mul.c -o seq_matrix_big_mul
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define N_THREADS 32

int num_rows_A = 2000; int num_rows_B = 2000; int num_rows_C = 2000;
int num_cols_A = 2000; int num_cols_B = 600; int num_cols_C = 600;
//int num_rows_A = 64; int num_rows_B = 64; int num_rows_C = 64;
//int num_cols_A = 64; int num_cols_B = 64; int num_cols_C = 64;

// I'm forcing a malloc because I want to add the malloc time on the game
float *A = (float*) malloc(sizeof(float) * num_rows_A * num_cols_A);
float *B = (float*) malloc(sizeof(float) * num_rows_B * num_cols_B);
float *C = (float*) malloc(sizeof(float) * num_rows_C * num_cols_C);
float *C_ref = (float*) malloc(sizeof(float) * num_rows_C * num_cols_C);

__global__ void matrix_2d_mul_float_gpu(float *A, float *B, float *C, int num_rows_A, int num_cols_A, int num_cols_B) {
  // Same code for all 2d kernel
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > num_rows_A || k > num_cols_B) return;

  float sum = 0;

  for (int j=0; j<num_cols_A; j++){
    // A[i][j] == A[i*num_cols_A+j]
    // B[j][k] == B[j*num_cols_B+k]
    //sum += A[i][j]*B[j][k];
    sum += A[i*num_cols_A+j]*B[j*num_cols_B+k];
  }

  C[i*num_cols_B+k]=sum;
}

void matrix_2d_mul_float(float *A, float *B, float *C, int num_rows_A, int num_cols_A, int num_cols_B) {
  float sum = 0;
  int num_rows_C = num_rows_A;
  int num_cols_C = num_cols_B;
  // Iterate on each row of A
  #pragma omp parallel for schedule(dynamic,1) collapse(2)
  for(int i=0; i<num_rows_A; i++) {
    // Iterate on each collumn of B
    for (int k=0; k<num_cols_B; k++) {
      sum = 0;
      // Do the "multiply add between" row of A and collumn of B
      for (int j=0; j<num_cols_A; j++){
        // A[i][j] == A[i*num_cols_A+j]
        // B[j][k] == B[j*num_cols_B+k]
        //sum += A[i][j]*B[j][k];
        sum += A[i*num_cols_A+j]*B[j*num_cols_B+k];
      }
      // C[i][k] == C[i*num_cols_C+k]
      C[i*num_cols_C+k]=sum;
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
  // Get size in bytes for our vectors
  int numBytesA = sizeof(float) * num_rows_A * num_cols_A;
  int numBytesB = sizeof(float) * num_rows_B * num_cols_B;
  printf("Size in bytes A: %d\n",numBytesA);
  printf("Size in bytes B: %d\n",numBytesB);

  // Fill arrays
  fillRand(A, 1, 100, num_rows_A * num_cols_A);
  fillRand(B, 1, 100, num_rows_B * num_cols_B);
  memset(C, 0, (num_rows_C*num_cols_C)*sizeof(float));

  // Allocate memory on GPU
  float *device_A_mat; float *device_B_mat; float *device_C_mat;
  cudaMalloc((char**)&device_A_mat,numBytesA);
  cudaMalloc((char**)&device_B_mat,numBytesB);
  cudaMalloc((char**)&device_C_mat,numBytesB);

  // Calculate kernel grid and blocks
  dim3 dimBlock(N_THREADS, N_THREADS);
  dim3 dimGrid((num_cols_B + dimBlock.x - 1) / dimBlock.x, (num_rows_A + dimBlock.y - 1) / dimBlock.y);
  // Call sequential function
  //ProfilerStart("nameOfProfile.log");
  for (int idxLoop=0; idxLoop < 10; idxLoop++) {
    // Copy matrices A and B to GPU
    cudaMemcpy(device_A_mat,A,numBytesA,cudaMemcpyHostToDevice);
    cudaMemcpy(device_B_mat,B,numBytesB,cudaMemcpyHostToDevice);

    // Launch the kernel
    //matrix_2d_mul_float(A,B,C,num_rows_A,num_cols_A,num_cols_B);
    matrix_2d_mul_float_gpu<<<dimGrid, dimBlock>>>(device_A_mat,device_B_mat,device_C_mat,num_rows_A,num_cols_A,num_cols_B);
    cudaError_t err = cudaThreadSynchronize();
    //printf("Run kernel: %s\n", cudaGetErrorString(err));

    // Get the result from the GPU to the CPU
    cudaMemcpy(C,device_C_mat,numBytesB,cudaMemcpyDeviceToHost);
    printf("Matrix multiplication done %d\n",idxLoop);
  }

  // Calculate one iteration with the reference function
  /*printf("Calculating reference\n");
  matrix_2d_mul_float(A,B,C_ref,num_rows_A,num_cols_A,num_cols_B);
  printf("Comparing with reference\n");
  float sumDiff = 0;
  for (int i = 0; i < (num_rows_C*num_cols_C); i++) {
	  float diff = C_ref[i] - C[i];
	  if (diff > 0.01f) {
		  printf("Values = %f -- %f\n",C_ref[i], C[i]);
		  sumDiff += diff;
	  }
  }
  printf("Difference = %f\n",sumDiff);*/

  // Free memory
  free(A);free(B);free(C);
  // Release memories from GPU
  cudaFree(device_A_mat);
  cudaFree(device_B_mat);
  cudaFree(device_C_mat);
  return 0;
}
