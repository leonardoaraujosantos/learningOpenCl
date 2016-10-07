/*
Compile and profile
nvcc checkTransferMatMul.cu -o checkTransferMatMul

If you have problems with gcc version try
sudo ln -s /usr/bin/gcc-4.9 /usr/local/cuda-7.5/bin/gcc

Profile console
nvprof ./checkTransferMatMul
*/
#include <stdio.h>

int main(int argc, char *argv[])
{
    unsigned int BytesToSend=60;
    unsigned int BytesToReceive=24;

    if (argc > 2) {
      BytesToSend =  atoi(argv[1]);
      BytesToReceive =  atoi(argv[2]);
    }

    printf("Checking GPU transfer...\n");
    printf("Sending %d bytes\n",BytesToSend);
    printf("Sending %d bytes\n",BytesToReceive);

    // Alocate memory on CPU
    char *hostArray= (char*)malloc(BytesToSend);
    char *deviceArray;

    // Allocate memory on GPU
    cudaMalloc((char**)&deviceArray,BytesToSend);
    memset(hostArray,0,BytesToSend);

    // Transfer hostArray from CPU to GPU
    cudaMemcpy(deviceArray,hostArray,BytesToSend,cudaMemcpyHostToDevice);
    // Get hostArray from GPU to CPU
    cudaMemcpy(hostArray,deviceArray,BytesToReceive,cudaMemcpyDeviceToHost);

    // Release memory from GPU
    cudaFree(deviceArray);
}
