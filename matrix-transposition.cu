#include <cuda_runtime.h>
#include <iostream>

const int TILE_DIM = 32;

// Transpose kernel
__global__ void transpose(float *odata, const float *idata, int width)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width && y < width)
        odata[y * width + x] = idata[x * width + y];
}

int main()
{
    const int nx = 1024;
    const int ny = 1024;
    const int mem_size = nx * ny * sizeof(float);

    dim3 dimGrid((nx + TILE_DIM - 1) / TILE_DIM, (ny + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    float *h_idata = new float[nx * ny];
    float *h_tdata = new float[nx * ny];

    // Initialize host data
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            h_idata[j * nx + i] = j * nx + i;

    float *d_idata, *d_tdata;
    cudaMalloc(&d_idata, mem_size);
    cudaMalloc(&d_tdata, mem_size);

    cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

    // Launch transpose kernel
    transpose<<<dimGrid, dimBlock>>>(d_tdata, d_idata, nx);

    cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_idata);
    cudaFree(d_tdata);
    delete[] h_idata;
    delete[] h_tdata;

    return 0;
}
