# Understanding CUDA Streams

## What are CUDA Streams?

1. **Definition**: A CUDA stream is a sequence of operations (kernels and memory
   transfers) that are executed in the order they are issued. Each stream
   operates independently, allowing for overlapping execution of operations.

2. **Default Stream**: When you do not explicitly create a stream, all
   operations are assigned to the default stream. In this case, operations are
   executed sequentially, which can limit performance.

## Key Features

1. **Concurrency**: Streams allow multiple operations to run concurrently. For
   example, while one kernel is executing, another can be launched, and memory
   transfers can occur simultaneously.

2. **Synchronization**: Within a stream, operations are executed in the order
   they are issued. However, operations in different streams can execute
   concurrently, which means you may need to manage synchronization between streams
   explicitly if necessary.

3. **Non-blocking Operations**: Memory transfers can be initiated in a
   non-blocking manner. This means that the CPU can continue executing other
   code while the data transfer is taking place, further enhancing performance.

## Benefits of Using Streams

1. **Improved Throughput**: By overlapping computation and data transfer,
   streams can help maximize the utilization of the GPU, reducing idle time and
   improving throughput.

2. **Better Resource Management**: Streams allow for more fine-grained control
   over how resources are used, making it easier to optimize performance for
   specific workloads.

3. **Enhanced Performance**: Applications that utilize streams effectively can
   see significant performance improvements, especially in scenarios involving
   multiple kernels and large data transfers.

```cuda
cudaStream_t stream; cudaStreamCreate(&stream);

// Launch a kernel in the stream myKernel<<<gridSize, blockSize, 0,
stream>>>(...);

// Perform memory copy in the stream cudaMemcpyAsync(dst, src, size,
cudaMemcpyHostToDevice, stream);

// Synchronize the stream to ensure all operations are complete
cudaStreamSynchronize(stream);

// Cleanup cudaStreamDestroy(stream);
```
