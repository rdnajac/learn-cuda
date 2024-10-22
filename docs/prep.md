# Interview Prep

- **Thread**: The smallest unit of execution in CUDA, with its own local memory and registers. Threads can be scheduled independently.
- **Block**: A group of threads that can cooperate through shared memory and can be scheduled independently on a multiprocessor. A block can contain up to 1024 threads, depending on the GPU architecture.

- **Threads**: Threads within the same block can communicate using shared memory, allowing for efficient data sharing.
- **Blocks**: Threads in different blocks cannot communicate directly. Instead, they must use global memory for data exchange, which incurs higher latency.

## What is a Warp?

A warp consists of 32 threads that execute instructions in lockstep
on NVIDIA GPUs. All threads in a warp execute the same instruction
simultaneously but can operate on different data. This execution model
enhances performance by maximizing GPU resource utilization.

## How Many Warps Can Run Simultaneously Inside a Multiprocessor?

The number of warps that can run simultaneously within a multiprocessor (SM)
varies by GPU architecture. Typically, a single multiprocessor can support
up to 64 active warps, enabling efficient management of memory latency
and improved throughput.

## Ontology of the CUDA Framework

The following table offers a non-exact description of the ontology of the CUDA framework:

| Memory (Hardware)      | Memory (Code or Variable Scoping)                                        | Computation (Hardware)          | Computation (Code Syntax)                                    | Computation (Code Semantics)                                |
| ---------------------- | ------------------------------------------------------------------------ | ------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| RAM                    | Non-CUDA variables                                                       | Host                            | Program                                                      | One routine call                                            |
| VRAM, GPU L2 cache     | Global, constant, texture                                                | Device                          | Grid                                                         | Simultaneous call of the same subroutine on many processors |
| GPU L1 cache           | Local, shared                                                            | SM ("Streaming Multiprocessor") | Block                                                        | Individual subroutine call                                  |
| Warp = 32 threads      |                                                                          |                                 | SIMD instructions                                            |                                                             |
| GPU L0 cache, register | Thread (aka. "SP", "Streaming Processor", "CUDA core"; deprecated names) |                                 | Analogous to individual scalar ops within a vector operation |

## Types of memory in a GPU

| Storage Class  | Memory Type     | Location | Description                                                                                                           |
| -------------- | --------------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `__global__`   | Global Memory   | L2       | Memory accessible by all threads across all blocks; ideal for large datasets and communication between blocks.        |
| `__device__`   | Global Memory   | L2       | Memory accessible by all threads across all blocks; similar to `__global__`, but typically used for device functions. |
| `__shared__`   | Shared Memory   | L1       | Memory shared among threads within the same block; allows for faster data sharing and synchronization.                |
| `__constant__` | Constant Memory | L1       | Read-only memory that is cached and optimized for faster access; useful for storing constants used by kernels.        |
| `__local__`    | Local Memory    | L0       | Private memory specific to a thread; used for storing local variables and provides fast access.                       |

## Coalesced vs. Uncoalesced Memory Access

| Type        | Definition                                                                                                                                 | Benefits/Drawbacks                                                                                                                                                | Example                                                                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Coalesced   | Occurs when multiple memory accesses can be combined into a single operation or when contiguous memory locations are accessed efficiently. | Benefits: <ul><li>Reduces the number of memory transactions, improving performance.</li><li>Maximizes cache utilization, leading to fewer cache misses.</li></ul> | In GPU programming, when threads access elements of an array in a contiguous manner, these accesses can be coalesced into a single memory transaction. |
| Uncoalesced | Occurs when memory accesses are scattered or non-contiguous, leading to multiple separate memory transactions.                             | Drawbacks: <ul><li>Increases latency due to more memory transactions.</li><li>Poor cache utilization, resulting in more cache misses.</li></ul>                   | Threads accessing every other element of an array in a non-contiguous manner would result in uncoalesced memory access.                                |

## How a Cache Works

A cache is a smaller, faster type of volatile memory located closer to the
processor than the main memory (RAM). It stores copies of frequently accessed
data and instructions to reduce the time it takes to access this information.
When a CPU or GPU needs to read data, it first checks the cache. If the data is
found (a cache hit), it can be accessed much faster than retrieving it from the
main memory. If the data is not found (a cache miss), it must be fetched from
the slower main memory, which can lead to increased latency.

## Difference Between Shared Memory and Registers

Shared memory is a type of on-chip memory that is accessible by all threads
within a block. It allows threads to collaborate and share data efficiently,
making it ideal for data that needs to be accessed by multiple threads. However,
shared memory has a limited size and can lead to contention if many threads
attempt to access it simultaneously.

Registers, on the other hand, are the fastest type of memory available to a
thread. Each thread has its own set of registers, which store local variables
and temporary data. Registers provide very low-latency access but are limited in
quantity, meaning that only a small amount of data can be stored. Unlike shared
memory, registers cannot be accessed by other threads.

## Understanding Kernel Occupancy in CUDA

Kernel occupancy is a critical metric in CUDA programming that indicates how
well the resources of a GPU's Streaming Multiprocessor (SM) are utilized during
kernel execution. It is defined as the ratio of active warps to the maximum
number of warps that can be supported by an SM, typically expressed as a
percentage. Understanding and optimizing occupancy can significantly impact the
performance of CUDA applications.

### Importance of Occupancy

- **Resource Utilization**: Higher occupancy allows for better utilization of
  GPU resources, enabling more warps to execute concurrently. This helps hide
  memory latency, as multiple warps can be ready to execute while waiting for
  memory operations to complete.
- **Performance Indicator**: While high occupancy is generally desirable, it
  does not guarantee optimal performance. Other factors, such as memory
  bandwidth and computation efficiency, must also be considered.

### Factors Influencing Occupancy

1. **Registers**: Each thread in a kernel requires registers for its execution.
   Excessive register usage can reduce the number of active threads that fit into
   the available resources, thereby lowering occupancy.

2. **Shared Memory**: If a kernel consumes significant shared memory, this can
   limit the number of warps that can reside on an SM simultaneously, impacting
   overall occupancy.

3. **Thread Block Size**: The number of threads per block affects occupancy.
   Finding the right balance is crucial; too few threads may lead to
   underutilization, while too many can exceed resource limits.

### Performance Considerations

- **Latency Hiding**: High occupancy can help in masking latency associated with
  memory accesses, which is essential for maintaining high throughput.
- **Balance with Other Factors**: Achieving optimal occupancy should be balanced
  with memory bandwidth and computational efficiency. A kernel with lower
  occupancy might still perform well if it effectively utilizes memory and
  performs significant computation.
