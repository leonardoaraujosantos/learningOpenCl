// nvcc profilerExample.cu -o profileExample
int main()
{
    const unsigned int X=1048576; //1 Megabyte
    const unsigned int bytes = X*sizeof(int);
    int *hostArray= (int*)malloc(bytes);
    int *deviceArray;
    cudaMalloc((int**)&deviceArray,bytes);
    memset(hostArray,0,bytes);
    cudaMemcpy(deviceArray,hostArray,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(hostArray,deviceArray,bytes,cudaMemcpyDeviceToHost);

    cudaFree(deviceArray);

}
