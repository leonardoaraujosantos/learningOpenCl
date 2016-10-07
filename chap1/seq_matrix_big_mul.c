/*
  Now we make the matrix much bigger
  g++ -pg seq_matrix_big_mul.c -o seq_matrix_big_mul
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

int num_rows_A = 2000; int num_rows_B = 2000; int num_rows_C = 2000;
int num_cols_A = 2000; int num_cols_B = 600; int num_cols_C = 600;

// I'm forcing a malloc because I want to add the malloc time on the game
float *A = (float*) malloc(sizeof(float) * num_rows_A * num_cols_A);
float *B = (float*) malloc(sizeof(float) * num_rows_B * num_cols_B);
float *C = (float*) malloc(sizeof(float) * num_rows_C * num_cols_C);

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

  // Call sequential function
  //ProfilerStart("nameOfProfile.log");
  for (int idxLoop=0; idxLoop < 10; idxLoop++) {
    // Populate matricex on heap

    matrix_2d_mul_float(A,B,C,num_rows_A,num_cols_A,num_cols_B);
    printf("Matrix multiplication done %d\n",idxLoop);
  }
  // Free memory
  free(A);free(B);free(C);
  return 0;
}
