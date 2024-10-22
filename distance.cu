#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Kernel to compute the distance matrix
__global__ void compute_distance_matrix(const float *d_vectors,
                                        float *d_distance_matrix, int N,
                                        int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index

  if (i < N) {
    for (int j = 0; j < N; ++j) { // Column index
      float sum = 0.0f;
      for (int k = 0; k < M; ++k) { // Calculate distance
        float diff = d_vectors[i * M + k] - d_vectors[j * M + k];
        sum += diff * diff;
      }
      d_distance_matrix[i * N + j] = sqrtf(sum); // Store distance
    }
  }
}

void calculate_distance_matrix(const std::vector<std::vector<float>> &vectors) {
  int N = vectors.size();
  int M = vectors[0].size();

  // Flatten vectors for CUDA
  float *h_vectors = new float[N * M];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      h_vectors[i * M + j] = vectors[i][j];
    }
  }

  float *d_vectors;
  float *d_distance_matrix;
  cudaMalloc(&d_vectors, N * M * sizeof(float));
  cudaMalloc(&d_distance_matrix, N * N * sizeof(float));

  cudaMemcpy(d_vectors, h_vectors, N * M * sizeof(float),
             cudaMemcpyHostToDevice);

  // Launch kernel with N threads
  int blockSize = 256; // Adjust block size as necessary
  int numBlocks = (N + blockSize - 1) / blockSize;
  compute_distance_matrix<<<numBlocks, blockSize>>>(d_vectors,
                                                    d_distance_matrix, N, M);

  // Copy results back to host
  std::vector<std::vector<float>> distance_matrix(N, std::vector<float>(N));
  cudaMemcpy(distance_matrix[0].data(), d_distance_matrix,
             N * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Clean up
  delete[] h_vectors;
  cudaFree(d_vectors);
  cudaFree(d_distance_matrix);

  // Print a portion of the distance matrix
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      std::cout << distance_matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

int main() {
  // Example with N vectors of length M
  const int N = 1000; // Number of vectors (N >> M)
  const int M = 5;    // Length of each vector

  // Create N random vectors of length M
  std::vector<std::vector<float>> vectors(N, std::vector<float>(M));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      vectors[i][j] =
          static_cast<float>(rand() % 100); // Random values between 0 and 99
    }
  }

  // Calculate the distance matrix
  calculate_distance_matrix(vectors);

  return 0;
}
