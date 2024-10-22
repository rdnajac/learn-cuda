#include <iostream>
#include <cuda_runtime.h>

// SAXPY kernel: computes y = a * x + y
__global__ void saxpy(int n, float a, float * __restrict__ x, float * __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main()
{
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float *x, *y;
    // Allocate device memory
    cudaMalloc((void**)&x, size);
    cudaMalloc((void**)&y, size);

    // Allocate host memory
    float *h_x = new float[N];
    float *h_y = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = 0.0f;
    }

    // Copy data from host to device
    cudaMemcpy(x, h_x, size, cudaMemcpyHostToDevice); // Explicit memory transfer
    cudaMemcpy(y, h_y, size, cudaMemcpyHostToDevice); // Explicit memory transfer

    // Launch SAXPY kernel
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, x, y);

    // Copy result back from device to host
    cudaMemcpy(h_y, y, size, cudaMemcpyDeviceToHost); // Explicit memory transfer
    // We don't need to synchronize here because cudaMemcpy is a blocking call.

    // Output some results for verification
    for (int i = 0; i < 10; i++)
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;

    // Free device memory
    cudaFree(x);
    cudaFree(y);
    // Free host memory
    delete[] h_x;
    delete[] h_y;

    return 0;
}
