# SAXPY: A Simple Unified Memory Example

This example demonstrates how to use Unified Memory in CUDA to simplify memory
management. Unified Memory allows you to access data on both the CPU and GPU
without needing to manually copy data between the two. This example uses Unified
Memory to simplify the memory management of the `saxpy` program.

## Separate Memory Management

```console
==89976== Profiling application: ./saxpy
==89976== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   60.07%  708.23us         2  354.11us  353.76us  354.47us  [CUDA memcpy HtoD]
                   35.75%  421.51us         1  421.51us  421.51us  421.51us  [CUDA memcpy DtoH]
                    4.18%  49.248us         1  49.248us  49.248us  49.248us  saxpy(int, float, float*, float*)
      API calls:   98.77%  188.17ms         2  94.086ms  67.345us  188.10ms  cudaMalloc
                    0.80%  1.5209ms         3  506.95us  392.96us  697.27us  cudaMemcpy
                    0.20%  380.60us         2  190.30us  130.42us  250.18us  cudaFree
                    0.19%  366.35us       202  1.8130us     187ns  79.881us  cuDeviceGetAttribute
                    0.02%  30.643us         1  30.643us  30.643us  30.643us  cudaLaunchKernel
                    0.01%  19.042us         2  9.5210us  5.3750us  13.667us  cuDeviceGetName
                    0.01%  14.077us         2  7.0380us  2.3700us  11.707us  cuDeviceGetPCIBusId
                    0.00%  8.3950us         2  4.1970us     380ns  8.0150us  cuDeviceTotalMem
                    0.00%  1.5700us         3     523ns     237ns     923ns  cuDeviceGetCount
                    0.00%  1.5370us         4     384ns     190ns     905ns  cuDeviceGet
                    0.00%     679ns         2     339ns     263ns     416ns  cuDeviceGetUuid
```

## Unified Memory Management

```console

==90019== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  1.7528ms         1  1.7528ms  1.7528ms  1.7528ms  saxpy(int, float, float*, float*)
      API calls:   98.69%  193.95ms         2  96.977ms  62.545us  193.89ms  cudaMallocManaged
                    0.89%  1.7537ms         1  1.7537ms  1.7537ms  1.7537ms  cudaDeviceSynchronize
                    0.22%  438.24us       202  2.1690us     253ns  97.714us  cuDeviceGetAttribute
                    0.15%  292.06us         2  146.03us  145.75us  146.31us  cudaFree
                    0.02%  39.026us         1  39.026us  39.026us  39.026us  cudaLaunchKernel
                    0.01%  24.949us         2  12.474us  7.0190us  17.930us  cuDeviceGetName
                    0.01%  15.636us         2  7.8180us  3.4090us  12.227us  cuDeviceGetPCIBusId
                    0.01%  10.879us         2  5.4390us     603ns  10.276us  cuDeviceTotalMem
                    0.00%  2.8620us         3     954ns     461ns  1.8410us  cuDeviceGetCount
                    0.00%  2.2810us         4     570ns     285ns  1.3260us  cuDeviceGet
                    0.00%     888ns         2     444ns     358ns     530ns  cuDeviceGetUuid

==90019== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 1080 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     116  70.620KB  4.0000KB  0.9922MB  8.000000MB  747.1660us  Host To Device
       2  32.000KB  4.0000KB  60.000KB  64.00000KB  5.888000us  Device To Host
      13         -         -         -           -  1.836068ms  Gpu page fault groups
Total CPU Page faults: 25
```
