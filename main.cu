#include <iostream>
#include <cuda_runtime.h>

int main()
{
    std::cout << "Hello, World!" << std::endl;
    std::cout << "1<<20 = " << (1<<20) << std::endl;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
               device, deviceProp.major, deviceProp.minor);
    }
    return 0;
}
