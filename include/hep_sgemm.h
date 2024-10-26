#include <hip/hip_runtime.h>
#include <cstdint>
#define HEP_WARP_SIZE 64

typedef _Float16 fp16x8 __attribute__((ext_vector_type(8)));
typedef _Float16 fp16x4 __attribute__((ext_vector_type(4)));
typedef float fp32x4 __attribute__((ext_vector_type(4)));
typedef _Float16 fp16;

union RegisterUnion
{
  fp16x8 vector8;
  struct
  {
    fp16x4 vector_front;
    fp16x4 vector_rear;
  };
};

template <int  BLK_M,
          int  BLK_N,
          int  BLK_K>
static __global__ __launch_bounds__(HEP_WARP_SIZE) void
gemm_batched_general_kernel_tensorcore_32x32x16_fp16fp32
                   (int32_t    M,
                    int32_t    N,
                    int32_t    K,
                    _Float16*  dA_input,
                    int32_t    lda,
                    _Float16*  dB_input,
                    int32_t    ldb,
                    _Float16*  dC_input,
                    int32_t    ldc)
{
    int thx  = threadIdx.x;
    int blx  = blockIdx.x;  // block's m position
    int bly  = blockIdx.y;  // block's n position
    int blz  = blockIdx.z;  // block's matrix in the batch 
    dA_input += size_t(blz) * M * K;
    dB_input += size_t(blz) * N * K;
    dC_input += size_t(blz) * N * M;


    fp32x4 C_reg_acc[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        RegisterUnion fragAB, fragAB2;
        
        size_t read_dim_mn = thx % BLK_K;
        size_t read_dim_k  = thx / BLK_K * 4;

        size_t offset_A = size_t(kk + read_dim_k) * size_t(lda) + size_t(blx * BLK_M + read_dim_mn);
        size_t offset_B = size_t(kk + read_dim_k) * size_t(ldb) + size_t(bly * BLK_N + read_dim_mn);
        
        int flag[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
        for(int i = 0; i < 4; i++)
        {
            flag[0][i] = (kk + read_dim_k + i < K && blx * BLK_M + read_dim_mn < M);
            flag[1][i] = (kk + read_dim_k + i < K && bly * BLK_N + read_dim_mn < N);
            flag[2][i] = (kk + read_dim_k + i < K && blx * BLK_M + read_dim_mn + 16 < M);
            flag[3][i] = (kk + read_dim_k + i < K && bly * BLK_N + read_dim_mn + 16 < N);
        }
        fragAB.vector_front = { flag[0][0] ? dA_input[offset_A + 0 * size_t(lda)] : (_Float16)0.0,
                                flag[0][1] ? dA_input[offset_A + 1 * size_t(lda)] : (_Float16)0.0,
                                flag[0][2] ? dA_input[offset_A + 2 * size_t(lda)] : (_Float16)0.0,
                                flag[0][3] ? dA_input[offset_A + 3 * size_t(lda)] : (_Float16)0.0 };
        fragAB.vector_rear  = { flag[1][0] ? dB_input[offset_B + 0 * size_t(ldb)] : (_Float16)0.0,
                                flag[1][1] ? dB_input[offset_B + 1 * size_t(ldb)] : (_Float16)0.0,
                                flag[1][2] ? dB_input[offset_B + 2 * size_t(ldb)] : (_Float16)0.0,
                                flag[1][3] ? dB_input[offset_B + 3 * size_t(ldb)] : (_Float16)0.0 };
        offset_A += 16;
        offset_B += 16;

        fragAB2.vector_front = { flag[2][0] ? dA_input[offset_A + 0 * size_t(lda)] : (_Float16)0.0,
                                 flag[2][1] ? dA_input[offset_A + 1 * size_t(lda)] : (_Float16)0.0,
                                 flag[2][2] ? dA_input[offset_A + 2 * size_t(lda)] : (_Float16)0.0,
                                 flag[2][3] ? dA_input[offset_A + 3 * size_t(lda)] : (_Float16)0.0 };
        fragAB2.vector_rear  = { flag[3][0] ? dB_input[offset_B + 0 * size_t(ldb)] : (_Float16)0.0,
                                 flag[3][1] ? dB_input[offset_B + 1 * size_t(ldb)] : (_Float16)0.0,
                                 flag[3][2] ? dB_input[offset_B + 2 * size_t(ldb)] : (_Float16)0.0,
                                 flag[3][3] ? dB_input[offset_B + 3 * size_t(ldb)] : (_Float16)0.0 };

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");

        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[0]), "+v"(fragAB.vector_front), "+v"(fragAB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[1]), "+v"(fragAB.vector_front), "+v"(fragAB2.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[2]), "+v"(fragAB2.vector_front), "+v"(fragAB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[3]), "+v"(fragAB2.vector_front), "+v"(fragAB2.vector_rear));

    }
    __syncthreads();

    size_t output_row = thx % 16;
    size_t output_col = thx / 16;

    size_t offset_C_row = size_t(bly * BLK_N + output_col);
    size_t offset_C_col = size_t(blx * BLK_M + output_row);
    size_t offset_C = size_t(offset_C_row) * size_t(ldc) + size_t(offset_C_col);

    size_t offset_C0 = size_t(offset_C_row +  0) * size_t(ldc) + size_t(offset_C_col +  0);
    size_t offset_C1 = size_t(offset_C_row + 16) * size_t(ldc) + size_t(offset_C_col +  0);
    size_t offset_C2 = size_t(offset_C_row +  0) * size_t(ldc) + size_t(offset_C_col + 16);
    size_t offset_C3 = size_t(offset_C_row + 16) * size_t(ldc) + size_t(offset_C_col + 16);

    int flagC[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    flagC[0] = (offset_C_row +  0 + 0 < N && offset_C_col +  0 < M);
    flagC[1] = (offset_C_row +  4 + 0 < N && offset_C_col +  0 < M);
    flagC[2] = (offset_C_row +  8 + 0 < N && offset_C_col +  0 < M);
    flagC[3] = (offset_C_row + 12 + 0 < N && offset_C_col +  0 < M);

    flagC[4] = (offset_C_row +  0 + 16 < N && offset_C_col +  0 < M);
    flagC[5] = (offset_C_row +  4 + 16 < N && offset_C_col +  0 < M);
    flagC[6] = (offset_C_row +  8 + 16 < N && offset_C_col +  0 < M);
    flagC[7] = (offset_C_row + 12 + 16 < N && offset_C_col +  0 < M);

    flagC[8] = (offset_C_row +  0 + 0 < N && offset_C_col + 16 < M);
    flagC[9] = (offset_C_row +  4 + 0 < N && offset_C_col + 16 < M);
    flagC[10] = (offset_C_row +  8 + 0 < N && offset_C_col + 16 < M);
    flagC[11] = (offset_C_row + 12 + 0 < N && offset_C_col + 16 < M);

    flagC[12] = (offset_C_row +  0 + 16 < N && offset_C_col + 16 < M);
    flagC[13] = (offset_C_row +  4 + 16 < N && offset_C_col + 16 < M);
    flagC[14] = (offset_C_row +  8 + 16 < N && offset_C_col + 16 < M);
    flagC[15] = (offset_C_row + 12 + 16 < N && offset_C_col + 16 < M);


    if(flagC[0]) dC_input[offset_C0 +  0 * size_t(ldc)] = (_Float16)C_reg_acc[0].x;
    if(flagC[1]) dC_input[offset_C0 +  4 * size_t(ldc)] = (_Float16)C_reg_acc[0].y;
    if(flagC[2]) dC_input[offset_C0 +  8 * size_t(ldc)] = (_Float16)C_reg_acc[0].z;
    if(flagC[3]) dC_input[offset_C0 + 12 * size_t(ldc)] = (_Float16)C_reg_acc[0].w;

    if(flagC[4]) dC_input[offset_C1 +  0 * size_t(ldc)] = (_Float16)C_reg_acc[1].x;
    if(flagC[5]) dC_input[offset_C1 +  4 * size_t(ldc)] = (_Float16)C_reg_acc[1].y;
    if(flagC[6]) dC_input[offset_C1 +  8 * size_t(ldc)] = (_Float16)C_reg_acc[1].z;
    if(flagC[7]) dC_input[offset_C1 + 12 * size_t(ldc)] = (_Float16)C_reg_acc[1].w;

    if(flagC[8]) dC_input[offset_C2 +  0 * size_t(ldc)] = (_Float16)C_reg_acc[2].x;
    if(flagC[9]) dC_input[offset_C2 +  4 * size_t(ldc)] = (_Float16)C_reg_acc[2].y;
    if(flagC[10]) dC_input[offset_C2 +  8 * size_t(ldc)] = (_Float16)C_reg_acc[2].z;
    if(flagC[11]) dC_input[offset_C2 + 12 * size_t(ldc)] = (_Float16)C_reg_acc[2].w;

    if(flagC[12]) dC_input[offset_C3 +  0 * size_t(ldc)] = (_Float16)C_reg_acc[3].x;
    if(flagC[13]) dC_input[offset_C3 +  4 * size_t(ldc)] = (_Float16)C_reg_acc[3].y;
    if(flagC[14]) dC_input[offset_C3 +  8 * size_t(ldc)] = (_Float16)C_reg_acc[3].z;
    if(flagC[15]) dC_input[offset_C3 + 12 * size_t(ldc)] = (_Float16)C_reg_acc[3].w;
}


template <int  BLK_M,
          int  BLK_N,
          int  BLK_K>
static __global__ __launch_bounds__(HEP_WARP_SIZE) void
gemm_batched_kernel_tensorcore_32x32x16_fp16fp32
                   (int32_t    M,
                    int32_t    N,
                    int32_t    K,
                    _Float16*  dA_input,
                    int32_t    lda,
                    _Float16*  dB_input,
                    int32_t    ldb,
                    _Float16*  dC_input,
                    int32_t    ldc)
{
    static_assert(BLK_M == 32 && BLK_N == 32 && BLK_K == 16, "incorrect block shape");
    int thx  = threadIdx.x;
    int blx  = blockIdx.x;  // block's m position
    int bly  = blockIdx.y;  // block's n position
    int blz  = blockIdx.z;  // block's matrix in the batch
    dA_input += size_t(blz) * M * K;
    dB_input += size_t(blz) * N * K;
    dC_input += size_t(blz) * N * M;

    fp32x4 C_reg_acc[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        RegisterUnion fragAB, fragAB2;
        
        size_t read_dim_mn = thx % BLK_K;
        size_t read_dim_k  = thx / BLK_K * 4;

        size_t offset_A = size_t(kk + read_dim_k) * size_t(lda) + size_t(blx * BLK_M + read_dim_mn);
        size_t offset_B = size_t(kk + read_dim_k) * size_t(ldb) + size_t(bly * BLK_N + read_dim_mn);

        fragAB.vector_front = { dA_input[offset_A + 0 * size_t(lda)],
                                dA_input[offset_A + 1 * size_t(lda)],
                                dA_input[offset_A + 2 * size_t(lda)],
                                dA_input[offset_A + 3 * size_t(lda)] };
        fragAB.vector_rear  = { dB_input[offset_B + 0 * size_t(ldb)],
                                dB_input[offset_B + 1 * size_t(ldb)],
                                dB_input[offset_B + 2 * size_t(ldb)],
                                dB_input[offset_B + 3 * size_t(ldb)] };
        offset_A += 16;
        offset_B += 16;
        fragAB2.vector_front = { dA_input[offset_A + 0 * size_t(lda)],
                                 dA_input[offset_A + 1 * size_t(lda)],
                                 dA_input[offset_A + 2 * size_t(lda)],
                                 dA_input[offset_A + 3 * size_t(lda)] };
        fragAB2.vector_rear  = { dB_input[offset_B + 0 * size_t(ldb)],
                                 dB_input[offset_B + 1 * size_t(ldb)],
                                 dB_input[offset_B + 2 * size_t(ldb)],
                                 dB_input[offset_B + 3 * size_t(ldb)] };

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");

        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[0]), "+v"(fragAB.vector_front), "+v"(fragAB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[1]), "+v"(fragAB.vector_front), "+v"(fragAB2.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[2]), "+v"(fragAB2.vector_front), "+v"(fragAB.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[3]), "+v"(fragAB2.vector_front), "+v"(fragAB2.vector_rear));

    }
    __syncthreads();

    size_t output_row = thx % 16;
    size_t output_col = thx / 16;
    size_t offset_C = size_t(bly * BLK_N + output_col) * size_t(ldc) + size_t(blx * BLK_M + output_row);

    size_t offset_C0 = size_t(bly * BLK_N + output_col +  0) * size_t(ldc) + size_t(blx * BLK_M + output_row +  0);
    size_t offset_C1 = size_t(bly * BLK_N + output_col + 16) * size_t(ldc) + size_t(blx * BLK_M + output_row +  0);
    size_t offset_C2 = size_t(bly * BLK_N + output_col +  0) * size_t(ldc) + size_t(blx * BLK_M + output_row + 16);
    size_t offset_C3 = size_t(bly * BLK_N + output_col + 16) * size_t(ldc) + size_t(blx * BLK_M + output_row + 16);

    dC_input[offset_C0 +  0 * size_t(ldc)] = (_Float16)C_reg_acc[0].x;
    dC_input[offset_C0 +  4 * size_t(ldc)] = (_Float16)C_reg_acc[0].y;
    dC_input[offset_C0 +  8 * size_t(ldc)] = (_Float16)C_reg_acc[0].z;
    dC_input[offset_C0 + 12 * size_t(ldc)] = (_Float16)C_reg_acc[0].w;

    dC_input[offset_C1 +  0 * size_t(ldc)] = (_Float16)C_reg_acc[1].x;
    dC_input[offset_C1 +  4 * size_t(ldc)] = (_Float16)C_reg_acc[1].y;
    dC_input[offset_C1 +  8 * size_t(ldc)] = (_Float16)C_reg_acc[1].z;
    dC_input[offset_C1 + 12 * size_t(ldc)] = (_Float16)C_reg_acc[1].w;

    dC_input[offset_C2 +  0 * size_t(ldc)] = (_Float16)C_reg_acc[2].x;
    dC_input[offset_C2 +  4 * size_t(ldc)] = (_Float16)C_reg_acc[2].y;
    dC_input[offset_C2 +  8 * size_t(ldc)] = (_Float16)C_reg_acc[2].z;
    dC_input[offset_C2 + 12 * size_t(ldc)] = (_Float16)C_reg_acc[2].w;

    dC_input[offset_C3 +  0 * size_t(ldc)] = (_Float16)C_reg_acc[3].x;
    dC_input[offset_C3 +  4 * size_t(ldc)] = (_Float16)C_reg_acc[3].y;
    dC_input[offset_C3 +  8 * size_t(ldc)] = (_Float16)C_reg_acc[3].z;
    dC_input[offset_C3 + 12 * size_t(ldc)] = (_Float16)C_reg_acc[3].w;
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
    if((m % 32 == 0) && (n % 32 == 0) && (k % 16 == 0))
    {
        const int blk_m = 32;
        const int blk_n = 32;
        const int blk_k = 16;
        dim3      dimBlock(HEP_WARP_SIZE);
        dim3      dimGrid(m / blk_m, n / blk_n, batch_count);
        gemm_batched_kernel_tensorcore_32x32x16_fp16fp32<blk_m, blk_n, blk_k><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, dA, lda, dB, ldb, dC, ldc);
    }
    else
    {
        const int blk_m = 32;
        const int blk_n = 32;
        const int blk_k = 16;
        dim3      dimBlock(HEP_WARP_SIZE);
        dim3      dimGrid((m - 1) / blk_m + 1, (n - 1) / blk_n + 1, batch_count);
        gemm_batched_general_kernel_tensorcore_32x32x16_fp16fp32<blk_m, blk_n, blk_k><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, dA, lda, dB, ldb, dC, ldc);
    }
}