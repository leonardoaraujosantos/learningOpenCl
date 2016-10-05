/*
Compile and profile
nvcc checkTransfer.cu -o checkTransfer

If you have problems with gcc version try
sudo ln -s /usr/bin/gcc-4.9 /usr/local/cuda-7.5/bin/gcc

Profile console
nvprof ./checkTransfer
*/
int main()
{
    //const unsigned int X=1; //1 Bytes (2us/1us)
    //const unsigned int X=10; //10 Bytes (2us/1us)
    //const unsigned int X=100; //100 Bytes (2us/1us)
    //const unsigned int X=1000; //1k Bytes (2us/1us)
    //const unsigned int X=10000; //10k Bytes (2.7us/2us)
    //const unsigned int X=100000; //100k Bytes (10us/10us)
    //const unsigned int X=1000000; //1 Megabyte (80us/79us)
    //const unsigned int X=10000000; //10 Megabyte (1000us/900us)
    //const unsigned int X=100000000; //100 Megabyte (10000us/10000us)
    const unsigned int X=1000000000; //1000 Megabyte (106000us/103000us)
    //const unsigned int X=256000000; //256 Megabyte (27000us/26000us)
    //const unsigned int X=120*120*3; // 120x120 RGB image (43200 bytes) (7us/6us)
    const unsigned int bytes = X*sizeof(char);
    // Alocate memory on CPU
    char *hostArray= (char*)malloc(bytes);
    char *deviceArray;

    // Allocate memory on GPU
    cudaMalloc((char**)&deviceArray,bytes);
    memset(hostArray,0,bytes);

    // Transfer hostArray from CPU to GPU
    cudaMemcpy(deviceArray,hostArray,bytes,cudaMemcpyHostToDevice);
    // Get hostArray from GPU to CPU
    cudaMemcpy(hostArray,deviceArray,bytes,cudaMemcpyDeviceToHost);

    // Release memory from GPU
    cudaFree(deviceArray);
}
