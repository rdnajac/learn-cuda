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

    float *x, *y;
    // Allocate unified memory accessible from both host and device
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // Initialize the arrays directly
    for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i);   // Host writes to unified memory
        y[i] = 0.0f;                    // Host writes to unified memory
    }

    // Launch SAXPY kernel
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, x, y);

    // Ensure GPU has finished before accessing results
    cudaDeviceSynchronize();    // Implicit synchronization

    // Output some results for verification
    for (int i = 0; i < 10; i++)
        std::cout << "y[" << i << "] = " << y[i] << std::endl; // Access results directly

    // Free unified memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
