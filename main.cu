#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>
#include<cmath>
#include<cuda_fp16.h>
#include<mma.h>
#include<algorithm>

#define N 4096
#define ITERATIONS 10
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8
#define TILE_K 32
#define TILE_M 64
#define COLS_PER_THREAD 4
#define TILE_N (DIM_THREAD_BLOCK_X * COLS_PER_THREAD)
#define ROWS_PER_THREAD (TILE_M / DIM_THREAD_BLOCK_Y)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WMMA_WARPS_M 4
#define WMMA_WARPS_N 2
using namespace std;

inline void cudaCheck(cudaError_t err, const char* file, int line){
  if(err != cudaSuccess){
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

// Shared-memory tiled SGEMM using double buffering and register blocking.
__global__ void sgemm_optimized(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int n, float alpha, float beta) {
  __shared__ float As[2][TILE_M][TILE_K];
  __shared__ float Bs[2][TILE_K][TILE_N];

  const int threadCol = threadIdx.x;
  const int threadRow = threadIdx.y;
  const int blockRow = blockIdx.y * TILE_M;
  const int blockCol = blockIdx.x * TILE_N;
  const int globalColBase = blockCol + threadCol * COLS_PER_THREAD;

  float accum[ROWS_PER_THREAD][COLS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < ROWS_PER_THREAD; ++i) {
    #pragma unroll
    for (int j = 0; j < COLS_PER_THREAD; ++j) {
      accum[i][j] = 0.f;
    }
  }

  const int threadsPerBlock = blockDim.x * blockDim.y;
  const int linearThreadId = threadRow * blockDim.x + threadCol;

  const int tileCount = (n + TILE_K - 1) / TILE_K;
  int stage = 0;

  for (int loadIdx = linearThreadId; loadIdx < TILE_M * TILE_K; loadIdx += threadsPerBlock) {
    int row = loadIdx / TILE_K;
    int col = loadIdx % TILE_K;
    int globalRow = blockRow + row;
    int globalColA = col;
    if (globalRow < n && globalColA < n) {
      As[stage][row][col] = A[globalRow * n + globalColA];
    } else {
      As[stage][row][col] = 0.f;
    }
  }
  for (int loadIdx = linearThreadId; loadIdx < TILE_K * TILE_N; loadIdx += threadsPerBlock) {
    int row = loadIdx / TILE_N;
    int col = loadIdx % TILE_N;
    int globalRowB = row;
    int globalColB = blockCol + col;
    if (globalRowB < n && globalColB < n) {
      Bs[stage][row][col] = B[globalRowB * n + globalColB];
    } else {
      Bs[stage][row][col] = 0.f;
    }
  }
  __syncthreads();

  for (int tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
    int readStage = stage;
    int nextTile = tileIdx + 1;
    int writeStage = stage ^ 1;

    if (nextTile < tileCount) {
      int nextBaseK = nextTile * TILE_K;
      for (int loadIdx = linearThreadId; loadIdx < TILE_M * TILE_K; loadIdx += threadsPerBlock) {
        int row = loadIdx / TILE_K;
        int col = loadIdx % TILE_K;
        int globalRow = blockRow + row;
        int globalColA = nextBaseK + col;
        if (globalRow < n && globalColA < n) {
          As[writeStage][row][col] = A[globalRow * n + globalColA];
        } else {
          As[writeStage][row][col] = 0.f;
        }
      }
      for (int loadIdx = linearThreadId; loadIdx < TILE_K * TILE_N; loadIdx += threadsPerBlock) {
        int row = loadIdx / TILE_N;
        int col = loadIdx % TILE_N;
        int globalRowB = nextBaseK + row;
        int globalColB = blockCol + col;
        if (globalRowB < n && globalColB < n) {
          Bs[writeStage][row][col] = B[globalRowB * n + globalColB];
        } else {
          Bs[writeStage][row][col] = 0.f;
        }
      }
    }

    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      float bVals[COLS_PER_THREAD];
      #pragma unroll
      for (int j = 0; j < COLS_PER_THREAD; ++j) {
        int col = threadCol * COLS_PER_THREAD + j;
        bVals[j] = Bs[readStage][k][col];
      }
      #pragma unroll
      for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        int row = threadRow + i * blockDim.y;
        float aVal = As[readStage][row][k];
        #pragma unroll
        for (int j = 0; j < COLS_PER_THREAD; ++j) {
          accum[i][j] += aVal * bVals[j];
        }
      }
    }

    if (nextTile < tileCount) {
      __syncthreads();
      stage = writeStage;
    }
  }

  #pragma unroll
  for (int i = 0; i < ROWS_PER_THREAD; ++i) {
    int globalRow = blockRow + threadRow + i * blockDim.y;
    if (globalRow < n) {
      #pragma unroll
      for (int j = 0; j < COLS_PER_THREAD; ++j) {
        int globalCol = globalColBase + j;
        if (globalCol < n) {
          int idx = globalRow * n + globalCol;
          float cVal = C[idx];
          C[idx] = beta * cVal + alpha * accum[i][j];
        }
      }
    }
  }
}

// WMMA-based SGEMM using Tensor Cores (FP16 inputs, FP32 accumulation).
__global__ void sgemm_wmma(const half* __restrict__ A,
                           const half* __restrict__ B,
                           float* __restrict__ C,
                           int n, float alpha, float beta) {
  using namespace nvcuda;

  const int warpId = threadIdx.y;
  const int warpRow = warpId / WMMA_WARPS_N;
  const int warpCol = warpId % WMMA_WARPS_N;

  const int tileRow = (blockIdx.y * WMMA_WARPS_M + warpRow) * WMMA_M;
  const int tileCol = (blockIdx.x * WMMA_WARPS_N + warpCol) * WMMA_N;

  if (tileRow + WMMA_M > n || tileCol + WMMA_N > n) {
    return;
  }

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
  wmma::fill_fragment(cFrag, 0.0f);

  for (int k = 0; k < n; k += WMMA_K) {
    if (k + WMMA_K > n) {
      break;
    }
    const half* aTile = A + tileRow * n + k;
    const half* bTile = B + k * n + tileCol;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;

    wmma::load_matrix_sync(aFrag, aTile, n);
    wmma::load_matrix_sync(bFrag, bTile, n);
    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
  }

  float* cTile = C + tileRow * n + tileCol;
  if (beta != 0.0f) {
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cMem;
    wmma::load_matrix_sync(cMem, cTile, n, wmma::mem_row_major);
    #pragma unroll
    for (int i = 0; i < cFrag.num_elements; ++i) {
      cFrag.x[i] = alpha * cFrag.x[i] + beta * cMem.x[i];
    }
  } else {
    #pragma unroll
    for (int i = 0; i < cFrag.num_elements; ++i) {
      cFrag.x[i] = alpha * cFrag.x[i];
    }
  }

  wmma::store_matrix_sync(cTile, cFrag, n, wmma::mem_row_major);
}

void compare(float* res1, float* res2, int n, float tolerance=0.0005f){
  int fail=0;
  for(int i=0; i<n; i++){
    float a,b;
    if(res1[i]<0)
      a=res1[i]*(-1);
    else 
      a=res1[i];
    if(res2[i]<0)
      b=res2[i]*(-1);
    else 
      b=res2[i];
    if((a<0.01)&&(b<0.01)){
      continue;
    }
    if(i<10)
      printf("i=%d %lf %lf\n",i,a,b);
    float diff=(a-b)/(a+0.000001);
    if(diff<0)
      diff=diff*(-1);
    if(diff>tolerance)
      fail++;
  }
  printf("Number of errors: %d\n", fail);
}

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

int main(){
  float alpha=0.5f, beta=0.3f;
  float *A=new float[N*N];
  float *B=new float[N*N];
  float *C_init=new float[N*N];
  float *C_cpu=new float[N*N];
  float *C_smem_result=new float[N*N];
  float *C_wmma_result=new float[N*N];
  half *A_half_host=new half[N*N];
  half *B_half_host=new half[N*N];

  srand(0);
  for(int i=0; i<N*N; ++i){
    float aVal=static_cast<float>(rand())/static_cast<float>(RAND_MAX);
    float bVal=static_cast<float>(rand())/static_cast<float>(RAND_MAX);
    float cVal=static_cast<float>(rand())/static_cast<float>(RAND_MAX);
    A[i]=aVal;
    B[i]=bVal;
    C_init[i]=cVal;
    C_cpu[i]=cVal;
    C_smem_result[i]=0.f;
    C_wmma_result[i]=0.f;
    A_half_host[i]=__float2half(aVal);
    B_half_host[i]=__float2half(bVal);
  }

  for(int j=0; j<N; j++){
    for(int i=0; i<N; i++){
      float tmp = beta*C_cpu[i*N+j];
      for(int k=0; k<N; k++){
        tmp += alpha*A[i*N+k]*B[k*N+j];
      }
      C_cpu[i*N+j]=tmp;
    }
  }

  float *A_gpu;
  float *B_gpu;
  float *C_gpu;
  half *A_gpu_half;
  half *B_gpu_half;
  size_t matrixBytesF = sizeof(float)*N*N;
  size_t matrixBytesH = sizeof(half)*N*N;

  CUDA_CHECK(cudaMalloc((void **)&A_gpu, matrixBytesF));
  CUDA_CHECK(cudaMalloc((void **)&B_gpu, matrixBytesF));
  CUDA_CHECK(cudaMalloc((void **)&C_gpu, matrixBytesF));
  CUDA_CHECK(cudaMalloc((void **)&A_gpu_half, matrixBytesH));
  CUDA_CHECK(cudaMalloc((void **)&B_gpu_half, matrixBytesH));

  CUDA_CHECK(cudaMemcpy(A_gpu, A, matrixBytesF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_gpu, B, matrixBytesF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(C_gpu, C_init, matrixBytesF, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(A_gpu_half, A_half_host, matrixBytesH, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_gpu_half, B_half_host, matrixBytesH, cudaMemcpyHostToDevice));

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((N + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);

  sgemm_optimized<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, alpha, beta);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(C_smem_result, C_gpu, matrixBytesF, cudaMemcpyDeviceToHost));
  compare(C_cpu, C_smem_result, N*N);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaMemcpy(C_gpu, C_init, matrixBytesF, cudaMemcpyHostToDevice));
  for(int warm=0; warm<3; ++warm){
    sgemm_optimized<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, alpha, beta);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){
    sgemm_optimized<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, alpha, beta);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsedMs = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
  double smemTime = (elapsedMs / 1000.0) / ITERATIONS;
  double flops = 2.0*N*N*(double)N;
  double smemGflopsPerSecond = flops/1.0e9/smemTime;
  double smemMatrixBytes = 3.0 * N * N * sizeof(float);
  double smemGB = smemMatrixBytes/1.0e9;
  double smemGBpS = smemGB/smemTime;
  printf("Shared-memory kernel: GFLOPS/s=%lf GB/s=%lf time(s)=%lf\n",
         smemGflopsPerSecond, smemGBpS, smemTime);

  bool tensorCoresAvailable = false;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    tensorCoresAvailable = prop.major >= 7;
  }

  if (tensorCoresAvailable) {
    CUDA_CHECK(cudaMemcpy(C_gpu, C_init, matrixBytesF, cudaMemcpyHostToDevice));

    dim3 wmmaBlock(32, WMMA_WARPS_M * WMMA_WARPS_N);
    dim3 wmmaGrid((N + (WMMA_N * WMMA_WARPS_N) - 1) / (WMMA_N * WMMA_WARPS_N),
                  (N + (WMMA_M * WMMA_WARPS_M) - 1) / (WMMA_M * WMMA_WARPS_M));

    sgemm_wmma<<<wmmaGrid, wmmaBlock>>>(A_gpu_half, B_gpu_half, C_gpu, N, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C_wmma_result, C_gpu, matrixBytesF, cudaMemcpyDeviceToHost));
    compare(C_cpu, C_wmma_result, N*N, 0.05f);

    CUDA_CHECK(cudaMemcpy(C_gpu, C_init, matrixBytesF, cudaMemcpyHostToDevice));
    for(int warm=0; warm<3; ++warm){
      sgemm_wmma<<<wmmaGrid, wmmaBlock>>>(A_gpu_half, B_gpu_half, C_gpu, N, alpha, beta);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){
      sgemm_wmma<<<wmmaGrid, wmmaBlock>>>(A_gpu_half, B_gpu_half, C_gpu, N, alpha, beta);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    double wmmaTime = (elapsedMs / 1000.0) / ITERATIONS;
    double wmmaGflopsPerSecond = flops/1.0e9/wmmaTime;
    double wmmaMatrixBytes = (2.0 * N * N * sizeof(half) + N * N * sizeof(float));
    double wmmaGB = wmmaMatrixBytes/1.0e9;
    double wmmaGBpS = wmmaGB/wmmaTime;
    printf("Tensor Core kernel: GFLOPS/s=%lf GB/s=%lf time(s)=%lf\n",
           wmmaGflopsPerSecond, wmmaGBpS, wmmaTime);
  } else {
    printf("Tensor Cores unavailable on this device; skipping WMMA path.\n");
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(A_gpu));
  CUDA_CHECK(cudaFree(B_gpu));
  CUDA_CHECK(cudaFree(C_gpu));
  CUDA_CHECK(cudaFree(A_gpu_half));
  CUDA_CHECK(cudaFree(B_gpu_half));
  delete[] A;
  delete[] B;
  delete[] C_init;
  delete[] C_cpu;
  delete[] C_smem_result;
  delete[] C_wmma_result;
  delete[] A_half_host;
  delete[] B_half_host;
  return 0;
}
