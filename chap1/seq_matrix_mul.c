/*
First sequential problem matrix multiplication
We're going to multiply 2 matrix A(3x3),B(3x2), the expected result is a
matrix C (3x2)

Compiling
g++ -pg seq_matrix_mul.c -o seq_matrix_mul

Check for leaks
valgrind --leak-check=full --show-leak-kinds=all ./seq_matrix_mul

Profile (Instruction)
valgrind --tool=callgrind ./seq_matrix_mul
kcachegrind callgrind.out.26375

Profile (Time old way, 1ms precision)
./seq_matrix_mul
gprof ./seq_matrix_mul gmon.out > timeProf.txt

Profile (Time using Google Performance Tools)
Install https://github.com/gperftools/gperftools
g++ -DWITHGPERFTOOLS -lprofiler -g seq_matrix_mul.c -o seq_matrix_mul
./seq_matrix_mul
pprof --callgrind ./seq_matrix_mul profile.log > profile.callgrind
*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//#include <gperftools/profiler.h>

int num_rows_A = 3; int num_rows_B = 3; int num_rows_C = 3;
int num_cols_A = 3; int num_cols_B = 2; int num_cols_C = 2;

// Create 2d array references
float A_ref[3][3] = {
  {1, 2, 3} ,
  {4, 5, 6} ,
  {1, 3, 2}
};
float B_ref[3][2] = {
  {10, 11} ,
  {7, 5} ,
  {2, 4}
};
float C_ref[3][2] = {
  {30, 33} ,
  {87, 93} ,
  {35, 24}
};

// I'm forcing a malloc because I want to add the malloc time on the game
float *A = (float*) malloc(sizeof(float) * num_rows_A * num_cols_A);
float *B = (float*) malloc(sizeof(float) * num_rows_B * num_cols_B);
float *C = (float*) malloc(sizeof(float) * num_rows_C * num_cols_C);

void matrix_2d_mul_float(float *A, float *B, float *C, int num_rows_A, int num_cols_A, int num_cols_B) {
  float sum = 0;
  int num_rows_C = num_rows_A;
  int num_cols_C = num_cols_B;
  // Iterate on each row of A
  //#pragma omp parallel for
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

void displayVec1d(float *vec, int size, char *desc) {
  // Trick to point to a 2d matrix
  printf("\nVector %s size: %d {",desc,size);
  for (int idx = 0; idx < size; idx++) {
    printf(" %3.2f,",vec[idx]);
  }
  printf("}\n");
}

void displayMatrix2d(float *mat, int num_rows, int num_cols) {
  // Trick to point to a 2d matrix
  printf("\nMatrix %dx%d\n",num_rows,num_cols);
  for (int idx_row = 0; idx_row < num_rows_C; idx_row++) {
    printf("|");
    for (int idx_col = 0; idx_col < num_cols_C; idx_col++) {
      // mat[row][col] == mat[idx_row*num_cols+idx_col]
      printf(" %3.2f ",mat[idx_row*num_cols+idx_col]);
    }
    printf("|");
    printf("\n");
  }
  printf("\n");
}

int main() {
  // Get size in bytes for our vectors
  int numBytesA = sizeof(float) * num_rows_A * num_cols_A;
  int numBytesB = sizeof(float) * num_rows_B * num_cols_B;
  printf("Size in bytes A: %d\n",numBytesA);
  printf("Size in bytes B: %d\n",numBytesB);

  memcpy(A,A_ref,numBytesA);
  displayVec1d(A,num_rows_A * num_cols_A,(char*)"A");

  // Print reference
  printf("Reference result for C\n");
  displayMatrix2d((float*)C_ref,num_rows_C,num_cols_C);

  // Call sequential function
  //ProfilerStart("nameOfProfile.log");  
  for (int idxLoop=1; idxLoop < 1000000; idxLoop++) {
    // Populate matricex on heap
    memcpy(A,A_ref,numBytesA);
    memcpy(B,B_ref,numBytesB);

    matrix_2d_mul_float(A,B,C,num_rows_A,num_cols_A,num_cols_B);
  }
  //ProfilerStop();

  // Print result
  printf("Calculated result for C\n");
  displayMatrix2d(C,num_rows_C,num_cols_C);

  // Free memory
  free(A);free(B);free(C);

  return 0;
}
