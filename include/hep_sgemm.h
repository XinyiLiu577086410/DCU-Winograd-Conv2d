#include <hip/hip_runtime.h>
#include <cstdint>

template <typename inoutT,
          typename calcT,
          int  DIM_M,
          int  DIM_N,
          int  BLK_M,
          int  BLK_N,
          int  BLK_K,
          int  DIM_M_A,
          int  DIM_N_A,
          int  DIM_M_B,
          int  DIM_N_B>
static __global__ __launch_bounds__(DIM_M * DIM_N) void
gemm_batched_general_kernel(int32_t    M,
                            int32_t    N,
                            int32_t    K,
                            inoutT      alpha,
                            inoutT *    dA_input,
                            int32_t    lda,
                            inoutT *    dB_input,
                            int32_t    ldb,
                            inoutT      beta,
                            inoutT *    dC_input,
                            int32_t    ldc,
                            int32_t    batch_count)
{
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_M * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % DIM_M_A; // thread's m position for loading A
    int thyA = idt / DIM_M_A; // thread's n position for loading A
    int thxB = idt % DIM_M_B; // thread's m position for loading B
    int thyB = idt / DIM_M_B; // thread's n position for loading B

    __shared__ calcT sA[BLK_K][BLK_M]; // shared memory for A
    __shared__ calcT sB[BLK_N][BLK_K]; // shared memory for B
    calcT            rC[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for C

    int a_i_offset = thxA + BLK_M * blx;
    int a_j_offset = thyA;
    int b_i_offset = thxB;
    int b_j_offset = thyB + BLK_N * bly;

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_M / DIM_M; ++m)
            rC[n][m] = (calcT)0.0;

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        for(int n = 0; n < BLK_K; n += DIM_N_A)
        {
            for(int m = 0; m < BLK_M; m += DIM_M_A)
            {
                int i = m + a_i_offset;
                int j = n + kk + a_j_offset;
                if(i < M && j < K)
                {
                    sA[n + thyA][m + thxA] = (calcT)dA_input[i * size_t(lda) + j];
                }
                else
                {
                    sA[n + thyA][m + thxA] = (calcT)0.0;
                }
            }
        }

        for(int n = 0; n < BLK_N; n += DIM_N_B)
        {
            for(int m = 0; m < BLK_K; m += DIM_M_B)
            {
                int i = m + kk + b_i_offset;
                int j = n + b_j_offset;
                if(i < K && j < N)
                {
                    sB[n + thyB][m + thxB] = (calcT)dB_input[i + j * size_t(ldb)];
                }
                else
                {
                    sB[n + thyB][m + thxB] = (calcT)0;
                }
            }
        }

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k)
            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                    rC[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

        __syncthreads();
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n)
    {
        for(int m = 0; m < BLK_M / DIM_M; ++m)
        {
            int coord_dCm = blx * BLK_M + m * DIM_M + thx;
            int coord_dCn = bly * BLK_N + n * DIM_N + thy;
            if(coord_dCn < N && coord_dCm < M)
            {
                dC_input[coord_dCn * size_t(ldc) + coord_dCm] = (inoutT)alpha * rC[n][m];
            }
        }
    }
}

// large index support is not needed for lda, ldb, ldc as this kernel is only intended for small m, n, k
// templated alpha, beta, restricted m, n, k
template <typename inoutT,
          typename calcT,
          int  DIM_M,
          int  DIM_N,
          int  BLK_M,
          int  BLK_N,
          int  BLK_K,
          int  DIM_M_A,
          int  DIM_N_A,
          int  DIM_M_B,
          int  DIM_N_B>
static __global__ __launch_bounds__(DIM_M * DIM_N) void
gemm_batched_kernel(int32_t    M,
                    int32_t    N,
                    int32_t    K,
                    inoutT *    dA_input,
                    int32_t    lda,
                    inoutT *    dB_input,
                    int32_t    ldb,
                    inoutT *    dC_input,
                    int32_t    ldc,
                    int32_t    batch_count)
{
    int thx  = threadIdx.x; // thread's m position in C
    int thy  = threadIdx.y; // thread's n position in C
    int idt  = DIM_M * thy + thx; // thread's number
    int blx  = blockIdx.x; // block's m position
    int bly  = blockIdx.y; // block's n position
    int blz  = blockIdx.z; // block's matrix in the batch
    int thxA = idt % DIM_M_A; // thread's m position for loading A
    int thyA = idt / DIM_M_A; // thread's n position for loading A
    int thxB = idt % DIM_M_B; // thread's m position for loading B
    int thyB = idt / DIM_M_B; // thread's n position for loading B

    __shared__ calcT sA[BLK_K][BLK_M]; // shared memory for A
    __shared__ calcT sB[BLK_N][BLK_K]; // shared memory for B
    calcT            rC[BLK_N / DIM_N][BLK_M / DIM_M]; // registers for C

    size_t coord_A, coord_B;

    coord_A = (blx * BLK_M + thxA) * size_t(lda) + thyA;
    coord_B = (bly * BLK_N + thyB) * size_t(ldb) + thxB;

    for(int n = 0; n < BLK_N / DIM_N; ++n)
        for(int m = 0; m < BLK_M / DIM_M; ++m)
            rC[n][m] = (calcT)0.0;

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        for(int n = 0; n < BLK_K; n += DIM_N_A)
            for(int m = 0; m < BLK_M; m += DIM_M_A)
                sA[n + thyA][m + thxA] = (calcT)dA_input[coord_A + (n + m * size_t(lda))];

        for(int n = 0; n < BLK_N; n += DIM_N_B)
            for(int m = 0; m < BLK_K; m += DIM_M_B)
                sB[n + thyB][m + thxB] = (calcT)dB_input[coord_B + (n * size_t(ldb) + m)];

        __syncthreads();

        for(int k = 0; k < BLK_K; ++k)
            for(int n = 0; n < BLK_N / DIM_N; ++n)
                for(int m = 0; m < BLK_M / DIM_M; ++m)
                    rC[n][m] += sA[k][m * DIM_M + thx] * sB[n * DIM_N + thy][k];

        __syncthreads();

        coord_A += BLK_K;
        coord_B += BLK_K;
    }

    for(int n = 0; n < BLK_N / DIM_N; ++n)
    {
        for(int m = 0; m < BLK_M / DIM_M; ++m)
        {
            int coord_dCm = blx * BLK_M + m * DIM_M + thx;
            int coord_dCn = bly * BLK_N + n * DIM_N + thy;

            dC_input[coord_dCn * size_t(ldc) + coord_dCm] = (inoutT)rC[n][m];
        }
    }
}

template <typename inoutT, typename calcT>
static void hep_sgemm(int32_t       m,
                      int32_t       n,
                      int32_t       k,
                      inoutT        alpha,
                      void *      	dA_,
                      int32_t       lda,
                      void *        dB_,
                      int32_t       ldb,
                      inoutT        beta,
                      void *        dC_,
                      int32_t       ldc,
                      int32_t       batch_count,
                      hipStream_t   stream)
{
	auto dA = reinterpret_cast<inoutT*>(dA_);
	auto dB = reinterpret_cast<inoutT*>(dB_);
	auto dC = reinterpret_cast<inoutT*>(dC_);
    // if((m % 64 == 0) && (n % 64 == 0) && (k % 4 == 0))
    // {
    //     //m is mult of 64, n is mult of 64, k is mult of 4
    //     const int dim_m = 16;
    //     const int dim_n = 16;
    //     const int blk_m = 64;
    //     const int blk_n = 64;
    //     const int blk_k = 4;
    //     dim3      dimBlock(dim_m, dim_n, 1);
    //     dim3      dimGrid(m / blk_m, n / blk_n, batch_count);
    //     // assert(alpha == 1.0 && beta == 0.0)
    //     gemm_batched_kernel<inoutT, calcT, dim_m, dim_n, blk _m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, dA, lda, dB, ldb, dC, ldc, batch_count);
    // }
    // else 
    if((m % 32 == 0) && (n % 32 == 0) && (k % 8 == 0))
    {
        // m is mult of 32, n is mult of 32, k is mult of 8
        const int dim_m = 16;
        const int dim_n = 16;
        const int blk_m = 32;
        const int blk_n = 32;
        const int blk_k = 8;
        dim3      dimBlock(dim_m, dim_n, 1);
        dim3      dimGrid(m / blk_m, n / blk_n, batch_count);
        // assert(alpha == 1.0 && beta == 0.0)
        gemm_batched_kernel<inoutT, calcT, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, dA, lda, dB, ldb, dC, ldc, batch_count);
    }
    else
    {
        const int dim_m = 16;
        const int dim_n = 16;
        const int blk_m = 32;
        const int blk_n = 32;
        const int blk_k = 8;
        dim3      dimBlock(dim_m, dim_n, 1);
        dim3      dimGrid(((m - 1) / blk_m) + 1, ((n - 1) / blk_n) + 1, batch_count);
        // assert(beta == 0)
        // general m, n, k, alpha; beta == 0
        gemm_batched_general_kernel<inoutT, calcT, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc, batch_count);
    }
}