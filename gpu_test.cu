#include <cuda_runtime.h>
#include <iostream>

__global__ void testKernel()
{
    printf("Hello from the RTX 3050! Block %d, Thread %d\n",
           blockIdx.x, threadIdx.x);
}

int main()
{
    int deviceCount = 0;

    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        std::cout << "No CUDA GPU detected.\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "CUDA GPU detected:\n";
    std::cout << "Name: " << prop.name << "\n";
    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: "
              << prop.totalGlobalMem / (1024 * 1024)
              << " MB\n";

    testKernel<<<1, 8>>>();

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cout << "CUDA kernel failed: "
                  << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "CUDA kernel executed successfully.\n";

    return 0;
}