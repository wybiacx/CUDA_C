
// system includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


template <int BLOCK_SIZE>
__global__
void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = by * BLOCK_SIZE * wA;

    int aEnd = aBegin + wA - 1;

    int aStep = BLOCK_SIZE;

    int bBegin = bx * BLOCK_SIZE;

    int bStep = wB * BLOCK_SIZE;
    // C[bx * BLOCK_SIZE + tx][by * BLOCK_SIZE + ty] = Csub;
    // C[(by * BLOCK_SIZE + ty) * wC + bx * BLOCK_SIZE + tx] = Csub;
    // C[(by * BLOCK_SIZE + ty) * wB + bx * BLOCK_SIZE + tx] = Csub;
    float Csub = 0;

    for(int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + ty * wA + tx];
        Bs[ty][tx] = B[b + ty * wB + tx];

        __syncthreads();

#pragma unroll

        for(int k = 0; k < BLOCK_SIZE; k++)
            Csub += As[ty][k] * Bs[k][tx];
        
        __syncthreads();
    }

    int pos = (by * BLOCK_SIZE + ty) * wB + bx * BLOCK_SIZE + tx;
    C[pos] = Csub;
}

void ConstantInit(float *data, int size, float val) {
    for(int i = 0; i < size; i++)
        data[i] = val;
}


int MatrixMultiply(int argc, char **argv,int block_size, 
    const dim3 &dimsA, const dim3 &dimsB) {
    
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));

    cudaStream_t stream;

    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);


    float *d_A, *d_B, *d_C;
    // x, y, z
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C;
    
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    if (h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(
        cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream)
    );

    checkCudaErrors(
        cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream)
    );

    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup
    if (block_size == 16) {
        MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }else {
        MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    printf("warmup done.\n");

    checkCudaErrors(cudaStreamSynchronize(stream));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, stream));

    int nIter = 300;

    for(int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }else {
            MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }
    
    // Record the start event
    checkCudaErrors(cudaEventRecord(stop, stream));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) * 
                                static_cast<double>(dimsA.y) * 
                                static_cast<double>(dimsB.x);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);

    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
        " WorkgroupSize= %u threads/block\n",
        gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);
    
    checkCudaErrors(
        cudaMemcpyAsync(h_C ,d_C ,mem_size_C , cudaMemcpyDeviceToHost, stream)
    );
    checkCudaErrors(cudaStreamSynchronize(stream));
    
    printf("Checking computed result for correctness: ");
    bool correct = true;

    double eps = 1.e-6;

    for(int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    if (correct)
        return EXIT_SUCCESS;
    else return EXIT_FAILURE;

}

int main(int argc, char **argv) {

    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?")) {
            printf("Usage -device=n (n >= 0 for deviceID)\n");
            printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
            printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
            printf("  Note: Outer matrix dimensions of A & B matrices" \
                " must be equal.\n");

            exit(EXIT_SUCCESS);
        }

    int dev = findCudaDevice(argc, (const char **)argv);

    int block_size = 32;

    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
    }

    if (dimsA.x != dimsB.y) {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
    dimsB.x, dimsB.y);

    checkCudaErrors(cudaProfilerStart());
    int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
    checkCudaErrors(cudaProfilerStop());
    
    exit(matrix_result);
}



