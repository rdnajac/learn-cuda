# CUDA

CUDA is a general-purpose parallel computing platform and programming
model developed by NVIDIA to leverage the massive parallel processing
power of GPUs.

## Overview

CUDA's scalable programming model uses abstractions like thread group
hierarchies, shared memories, and barrier synchronization to enable
programs to automatically scale across various GPU architectures
by adjusting the number of multiprocessors and memory partitions.

### Algorithms that Perform Well on GPUs

| Algorithm Category               | Examples                             | Description                                                                                        |
| -------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------- |
| Data Parallel Algorithms         | Vector Operations, Matrix Operations | Operations applied to large datasets where the same computation is performed on multiple elements. |
| Image Processing Algorithms      | Convolution, Histogram Equalization  | Pixel-wise operations and filters that benefit from parallel execution.                            |
| Machine Learning & Deep Learning | Neural Networks, k-Means Clustering  | Training and inference for models that process large datasets.                                     |
| Physics Simulations              | Molecular Dynamics, Fluid Dynamics   | Simulations involving complex interactions that can be parallelized.                               |
| Sorting Algorithms               | Bitonic Sort, Radix Sort             | Efficiently parallelized sorting methods for large datasets.                                       |
| Graph Algorithms                 | Breadth-First Search, PageRank       | Algorithms that can leverage parallelism for graph traversal and ranking.                          |
| Finite Element Methods           | Structural Analysis, Heat Transfer   | Solving differential equations using a parallelized approach.                                      |

### Binaries

The CUDA Toolkit provides the following binaries:

- `nvcc`: The CUDA compiler
- `nvprof`: The CUDA profiler
- `cuda-memcheck`: The CUDA memory checker
- `cuda-gdb`: The CUDA debugger
- `nsight-sys`: The CUDA system profiler
- `nsight-compute`: The CUDA kernel profiler

### Key Concepts and Terms

- The **host** is the CPU while the **device** is the GPU.
- grid-stride loops: A loop that iterates over all elements in a grid of threads

Acronyms that you should know:

- SIMD: Single Instruction, Multiple Data
- SIMT: Single Instruction, Multiple Threads (the CUDA execution model)
- AoS: Array of Structures
- SoA: Structure of Arrays (preferred for GPU programming)
- SAXPY: Single-precision $A*X$ Plus $Y$, a classic parallel algorithm

## CUDA C++ Programming

The CUDA programming model provides three key language extensions:

- **CUDA blocks**: A group of threads that execute the same kernel code
- **Shared memory**: A memory space shared by all threads in a block
- **Synchronization barriers**: A mechanism to synchronize threads in a block by
  making them wait until all threads have reached a certain point in the code

### Execution

A typical CUDA program consists of three main steps:

> 1. Copy the input data from host memory to device memory, also known as host-to-device transfer.
> 2. Load the GPU program and execute, caching data on-chip for performance.
> 3. Copy the results from device memory to host memory, also called device-to-host transfer.

A kernel is a function that runs on the GPU. It is defined with the
`__global__` keyword and is called from the host code.

```cuda
/**
 * @brief Adds two vectors `a` and `b` and stores the result in `c`.
 *
 * @param a The first input vector
 * @param b The second input vector
 * @param c The output vector
 * @param N The number of elements in the vectors
 *
 * @details
 * Kernel functions always return `void` and have no return value.
 * The `__global__` keyword specifies that the function runs on the device.
 * To modify the input vectors, they must be passed as pointers.
 */
__global__ void vectorAdd(float *a, float *b, float *c, int N)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)  /* Check if the thread is within the array bounds */
        c[i] = a[i] + b[i];
}

int main()
{
    // ...
    // Kernel invocation with 1 block and $N$ threads
    VecAdd<<<1, N>>>(d_a, d_b, d_c);
    // ...
}
```

> [!NOTE]
> The `<<<...>>>` _execution configuration_ syntax is used to specify the
> number of blocks and threads to launch the kernel with. Also note the
> use of the `blockIdx` and `threadIdx` variables to calculate the
> thread's index within the grid and the convention of prefixing variables
> that reside on the device with `d_` (and `h_` for host variables).

#### Breakdown of the Launch Syntax

`add<<<1, 256>>>(N, x, y);`

1. **Kernel Name**: `add`

   - This is the name of the CUDA kernel function being called. It must be defined with the `__global__` qualifier.

2. **Execution Configuration**: `<<<1, 256>>>`

   - This part specifies the grid and block dimensions for launching the kernel:
     - **Grid Size**: `1`
       - This means there is one block in the grid. In this case, it will launch only one block of threads.
     - **Block Size**: `256`
       - This specifies that each block will contain 256 threads.

3. **Kernel Arguments**: `(N, x, y)`
   - These are the parameters passed to the kernel. They can be any type supported in CUDA, including scalars, pointers, etc. In this case, `N` is likely an integer representing the size of the operation, while `x` and `y` are pointers to arrays in device memory.

#### What Happens if You Don’t Need All 256 Threads?

1. **Unused Threads**:

   - If the kernel only requires a certain number of threads (let's say `M` threads, where `M < 256`), the remaining threads in the block (256 - M) will simply be idle. They will still be allocated, but they will not perform any operations or contribute to the computation.

2. **Resource Wastage**:

   - Each thread in a block consumes resources (like registers and shared memory). If many threads are idle, you may not be utilizing the GPU's resources efficiently, leading to wasted computational capacity.

3. **Occupancy**:

   - The occupancy of a kernel (the ratio of active warps to the maximum number of warps supported on the multiprocessor) might not be optimal if a large number of threads remain unused. This could lead to underutilization of the GPU.

4. **Kernel Execution**:

   - The kernel will still execute without error, but the performance may be less than optimal. If your computation only requires `M` threads, it's better to launch a grid and block configuration that aligns with the actual workload, such as using fewer threads or blocks.

### Another Example

```cuda
// Kernel - Adding two matrices MatA and MatB
__global__ void MatAdd(float MatA[N][N], float MatB[N][N], float MatC[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        MatC[i][j] = MatA[i][j] + MatB[i][j];
}

int main()
{
    //...
    // Matrix addition kernel launch from host code
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x -1) / threadsPerBlock.x, (N+threadsPerBlock.y -1) / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(MatA, MatB, MatC);
    //...
}
```

In this example, we:

1. Define Threads per Block (`dim3 threadsPerBlock(16, 16);`)

   - 16 in both x and y dimensions (total of 256 threads).

2. Calculate Number of Blocks (`dim3 numBlocks(...);`)

   - Uses N (number of elements) to determine blocks in both dimensions.
   - The formula ensures that any leftover elements are accounted for.

3. Launch the Kernel (`MatAdd<<<numBlocks, threadsPerBlock>>>(MatA, MatB, MatC);`)

   - Launches the `MatAdd` kernel with the defined number of blocks and threads.
   - `MatA`, `MatB`, and `MatC` are the matrices being processed.

> [!IMPORTANT]
> It is common to see the number of threads per block calcualted as
> above (e.g., `(N + threadsPerBlock.x - 1) / threadsPerBlock.x`).
> This is a common convention to ensure that the grid is large enough
> to cover all elements in the matrix and handles any remainders.

### Thread Hierarchy

The `threadIdx` is a 3-component vector that holds the thread's index
within the block.

![Thread hierarchy](./assets/cuda_indexing.png)

> [!NOTE]
> During execution there is a finer grouping of threads into _warps_.

### Memory Hierarchy

The following memories are exposed by the GPU architecture:

- **Registers**: These are private to each thread, which means that registers assigned to a thread are not visible to other threads. The compiler makes decisions about register utilization.
- **L1/Shared memory (SMEM)**: Every SM has a fast, on-chip scratchpad memory that can be used as L1 cache and shared memory. All threads in a CUDA block can share shared memory, and all CUDA blocks running on a given SM can share the physical memory resource provided by the SM.
- **Read-only memory (ROM)**:Each SM has an instruction cache, constant memory, texture memory and RO cache, which is read-only to kernel code.
- **L2 cache**: The L2 cache is shared across all SMs, so every thread in every CUDA block can access this memory.
- **Global memory (DRAM)**:This is the framebuffer size of the GPU and DRAM sitting in the GPU.

Some things to remember about memory in CUDA:

- Each thread has private local memory.
- Each thread block has shared memory that is:
  - visible to all threads of the block and
  - with the same lifetime as the block.
- Thread blocks in a thread block cluster can perform read, write, and atomic operations on each other’s shared memory.
- All threads have access to the same global memory.

### Read-Only Memory Spaces

The constant and texture memory spaces are read-only memory spaces.

The global, constant, and texture memory spaces are persistent across
kernel launches by the same application.

## Global Memory Coalescing

The device _coalesces_ global memory loads and stores issued by threads
of a warp into as few transactions as possible to minimize DRAM bandwidth

## Thread Synchronization

Imagine a race condition where two threads are trying to
read and write to the same memory location...

CUDA provides a simple barrier synchronization primitive,
`__syncthreads()` that ensures that all threads in a block have
reached the same point in the code before any are allowed to proceed.

In other words, a thread's execution can only proceed past this
barrier when all threads in the block have reached it.

> [!CAUTION]
> Calling `__syncthreads()` in a conditional block (_divergent code_)
> can lead to a deadlock. Remember the four requirements for a deadlock:
>
> 1. Mutual exclusion
> 2. Hold and wait
> 3. No preemption
> 4. Circular wait

## SIMT Architecture

The NVIDIA GPU architecture features a scalable array of multithreaded Streaming
Multiprocessors (SMs) that efficiently execute CUDA kernels. Each multiprocessor
handles hundreds of threads using the SIMT (Single Instruction, Multiple Thread)
architecture, allowing for simultaneous execution of warps—groups of 32 threads.
Warps can diverge during execution, but with the introduction of Independent
Thread Scheduling in the Volta architecture, threads can now execute
independently, enhancing flexibility and performance. The hardware also
maintains execution contexts on-chip, enabling efficient context switching and
execution of multiple warps within the available register and shared memory limits.

The total number of warps in a block is as follows:

$$
\text{ceil} \left( \frac{T}{W_{size}}, 1 \right)
$$

where,

- $T$ is the total number of threads in the block
- $W_{size}$ is the warp size (usually 32)
- ceil is the ceiling function (equal to x rounded up to the nearest integer)

## Appendix

## CUDA C++ Data Types

| Data Type   | Description                                                    | `sizeof` |
| ----------- | -------------------------------------------------------------- | -------- |
| `blockIdx`  | 3-component vector holding the block's index within the grid   | 12 bytes |
| `blockDim`  | 3-component vector holding the block's dimensions              | 12 bytes |
| `threadIdx` | 3-component vector holding the thread's index within the block | 12 bytes |
| `gridDim`   | 3-component vector holding the grid's dimensions               | 12 bytes |
| `warpSize`  | The number of threads in a warp (usually 32)                   | 4 bytes  |

> [!NOTE]
> The 3-component vectors are of type `dim3` (typedef'd as `uint3`), a struct
> with three `unsigned int` fields representing the x, y, and z dimensions.
> The `warpSize` is a constant defined in a header and is just an `int`.

### CUDA C++ Keywords

| Storage Class     | Description                                        | Use Case                                                                   |
| ----------------- | -------------------------------------------------- | -------------------------------------------------------------------------- |
| `__device__`      | Declares a function that runs on the device        | Used for functions that need to be called from kernel code.                |
| `__global__`      | Declares a kernel function that runs on the device | Used for launching a kernel from the host to the device.                   |
| `__host__`        | Declares a function that runs on the host          | Used for functions that are executed on the CPU.                           |
| `__noinline__`    | Prevents the compiler from inlining a function     | Useful for debugging or when inlining would be detrimental to performance. |
| `__forceinline__` | Forces the compiler to inline a function           | Used for small functions where inlining can reduce call overhead.          |
| `__shared__`      | Declares a variable in shared memory               | Used to enable fast data sharing between threads in the same block.        |
| `__constant__`    | Declares a variable in constant memory             | Used for read-only data that can be accessed by all threads and is cached. |
| `__managed__`     | Declares a variable in managed memory              | Automatically handles memory allocation between host and device.           |
| `__restrict__`    | Hints that a pointer is not aliased                | Optimizes memory access when the compiler can assume no aliasing occurs.   |

## Further Reading

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Additional Resources

Blog:

- [Intro](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [Device Properties](https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/)
- [Global Memory](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [Shared Memory](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [Efficient Matrix Transpose](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

TODO:

- [Performance metrics](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)
- [Optimize data transfers](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
- [Overlap data transfers](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)

Important!!!

- [Unified Memory](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)
