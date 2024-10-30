#include <hip/hip_runtime.h>
#include <cstdint>
#include <iostream>
#include "common.h"

template <int  BLK_M,
          int  BLK_N,
          int  BLK_K>
static __global__ __launch_bounds__(HEP_WARP_SIZE) void
gemm_batched_general_kernel_tensorcore_32x32x16_fp16fp32_Akxm_Bnxk_Cnxm
                   (int32_t    M,
                    int32_t    N,
                    int32_t    K,
                    fp16*  dA_input,
                    int32_t    lda,
                    fp16*  dB_input,
                    int32_t    ldb,
                    fp16*  dC_input,
                    int32_t    ldc)
{
    static_assert(BLK_M == 32 && BLK_N == 32 && BLK_K == 16, "incorrect block shape");
    const int thx  = threadIdx.x;
    const int blx  = blockIdx.x;  // block's m position
    const int bly  = blockIdx.y;  // block's n position
    const int blz  = blockIdx.z;  // block's matrix in the batch 
    dA_input += size_t(blz) * M * K;
    dB_input += size_t(blz) * N * K;
    dC_input += size_t(blz) * N * M;
    __shared__ struct {
        fp16 B[BLK_K][BLK_N / 2 + 1][2];
    } lds;

    fp32x4 C_reg_acc[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        RegisterUnion frag_AB_0, frag_AB_1;
        
        size_t read_dim_k  = thx % BLK_K;
        size_t read_dim_mn = thx / BLK_K;
        size_t offset_B = size_t(kk + read_dim_k) + size_t(bly * BLK_N + read_dim_mn) * size_t(ldb);
        int flag_if_read_B[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        for(int i = 0; i < 4; i++)
        {
            flag_if_read_B[0][i] = (kk + read_dim_k < K && bly * BLK_N + read_dim_mn + i * 4 < N);
            flag_if_read_B[1][i] = (kk + read_dim_k < K && bly * BLK_N + read_dim_mn + i * 4 + 16 < N);
        }
        lds.B[read_dim_k][(read_dim_mn +  0) % (BLK_N / 2)][(read_dim_mn +  0) / (BLK_N / 2)] = flag_if_read_B[0][0] ? dB_input[offset_B +  0 * size_t(ldb)] : (fp16)0.0;
        lds.B[read_dim_k][(read_dim_mn +  4) % (BLK_N / 2)][(read_dim_mn +  4) / (BLK_N / 2)] = flag_if_read_B[0][1] ? dB_input[offset_B +  4 * size_t(ldb)] : (fp16)0.0;
        lds.B[read_dim_k][(read_dim_mn +  8) % (BLK_N / 2)][(read_dim_mn +  8) / (BLK_N / 2)] = flag_if_read_B[0][2] ? dB_input[offset_B +  8 * size_t(ldb)] : (fp16)0.0;
        lds.B[read_dim_k][(read_dim_mn + 12) % (BLK_N / 2)][(read_dim_mn + 12) / (BLK_N / 2)] = flag_if_read_B[0][3] ? dB_input[offset_B + 12 * size_t(ldb)] : (fp16)0.0;
        lds.B[read_dim_k][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)] = flag_if_read_B[1][0] ? dB_input[offset_B + 16 * size_t(ldb)] : (fp16)0.0;
        lds.B[read_dim_k][(read_dim_mn + 20) % (BLK_N / 2)][(read_dim_mn + 20) / (BLK_N / 2)] = flag_if_read_B[1][1] ? dB_input[offset_B + 20 * size_t(ldb)] : (fp16)0.0;
        lds.B[read_dim_k][(read_dim_mn + 24) % (BLK_N / 2)][(read_dim_mn + 24) / (BLK_N / 2)] = flag_if_read_B[1][2] ? dB_input[offset_B + 24 * size_t(ldb)] : (fp16)0.0;
        lds.B[read_dim_k][(read_dim_mn + 28) % (BLK_N / 2)][(read_dim_mn + 28) / (BLK_N / 2)] = flag_if_read_B[1][3] ? dB_input[offset_B + 28 * size_t(ldb)] : (fp16)0.0;



        read_dim_mn = thx % BLK_K;  // shall be (BLK_MN / 2) not BLK_K
        read_dim_k  = thx / BLK_K * 4;
        size_t offset_A = size_t(kk + read_dim_k) * size_t(lda) + size_t(blx * BLK_M + read_dim_mn);
        int flag_if_read_A[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        for(int i = 0; i < 4; i++)
        {
            flag_if_read_A[0][i] = (kk + read_dim_k + i < K && blx * BLK_M + read_dim_mn < M);
            flag_if_read_A[1][i] = (kk + read_dim_k + i < K && blx * BLK_M + read_dim_mn + 16 < M);
        }
        frag_AB_0.vector_front =  { flag_if_read_A[0][0] ? dA_input[offset_A + 0 * size_t(lda)] : (fp16)0.0,
                                    flag_if_read_A[0][1] ? dA_input[offset_A + 1 * size_t(lda)] : (fp16)0.0,
                                    flag_if_read_A[0][2] ? dA_input[offset_A + 2 * size_t(lda)] : (fp16)0.0,
                                    flag_if_read_A[0][3] ? dA_input[offset_A + 3 * size_t(lda)] : (fp16)0.0 };
        offset_A += 16;
        frag_AB_1.vector_front =  { flag_if_read_A[1][0] ? dA_input[offset_A + 0 * size_t(lda)] : (fp16)0.0,
                                    flag_if_read_A[1][1] ? dA_input[offset_A + 1 * size_t(lda)] : (fp16)0.0,
                                    flag_if_read_A[1][2] ? dA_input[offset_A + 2 * size_t(lda)] : (fp16)0.0,
                                    flag_if_read_A[1][3] ? dA_input[offset_A + 3 * size_t(lda)] : (fp16)0.0 };

        frag_AB_0.vector_rear = { lds.B[read_dim_k + 0][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)], 
                                  lds.B[read_dim_k + 1][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)],
                                  lds.B[read_dim_k + 2][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)], 
                                  lds.B[read_dim_k + 3][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)] };
        frag_AB_1.vector_rear = { lds.B[read_dim_k + 0][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)],
                                  lds.B[read_dim_k + 1][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)], 
                                  lds.B[read_dim_k + 2][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)], 
                                  lds.B[read_dim_k + 3][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)] };

        asm volatile("s_waitcnt lgkmcnt(0)\n\t");

        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[0]), "+v"(frag_AB_0.vector_front), "+v"(frag_AB_0.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[1]), "+v"(frag_AB_0.vector_front), "+v"(frag_AB_1.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[2]), "+v"(frag_AB_1.vector_front), "+v"(frag_AB_0.vector_rear));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[3]), "+v"(frag_AB_1.vector_front), "+v"(frag_AB_1.vector_rear));

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


    if(flagC[0]) dC_input[offset_C0 +  0 * size_t(ldc)] = (fp16)C_reg_acc[0].x;
    if(flagC[1]) dC_input[offset_C0 +  4 * size_t(ldc)] = (fp16)C_reg_acc[0].y;
    if(flagC[2]) dC_input[offset_C0 +  8 * size_t(ldc)] = (fp16)C_reg_acc[0].z;
    if(flagC[3]) dC_input[offset_C0 + 12 * size_t(ldc)] = (fp16)C_reg_acc[0].w;

    if(flagC[4]) dC_input[offset_C1 +  0 * size_t(ldc)] = (fp16)C_reg_acc[1].x;
    if(flagC[5]) dC_input[offset_C1 +  4 * size_t(ldc)] = (fp16)C_reg_acc[1].y;
    if(flagC[6]) dC_input[offset_C1 +  8 * size_t(ldc)] = (fp16)C_reg_acc[1].z;
    if(flagC[7]) dC_input[offset_C1 + 12 * size_t(ldc)] = (fp16)C_reg_acc[1].w;

    if(flagC[8]) dC_input[offset_C2 +  0 * size_t(ldc)] = (fp16)C_reg_acc[2].x;
    if(flagC[9]) dC_input[offset_C2 +  4 * size_t(ldc)] = (fp16)C_reg_acc[2].y;
    if(flagC[10]) dC_input[offset_C2 +  8 * size_t(ldc)] = (fp16)C_reg_acc[2].z;
    if(flagC[11]) dC_input[offset_C2 + 12 * size_t(ldc)] = (fp16)C_reg_acc[2].w;

    if(flagC[12]) dC_input[offset_C3 +  0 * size_t(ldc)] = (fp16)C_reg_acc[3].x;
    if(flagC[13]) dC_input[offset_C3 +  4 * size_t(ldc)] = (fp16)C_reg_acc[3].y;
    if(flagC[14]) dC_input[offset_C3 +  8 * size_t(ldc)] = (fp16)C_reg_acc[3].z;
    if(flagC[15]) dC_input[offset_C3 + 12 * size_t(ldc)] = (fp16)C_reg_acc[3].w;
}


template <int  BLK_M,
          int  BLK_N,
          int  BLK_K>
static __global__ __launch_bounds__(HEP_WARP_SIZE) void
gemm_batched_kernel_tensorcore_32x32x16_fp16fp32_Akxm_Bnxk_Cnxm
                   (int32_t    M,
                    int32_t    N,
                    int32_t    K,
                    fp16*  dA_input,
                    int32_t    lda,
                    fp16*  dB_input,
                    int32_t    ldb,
                    fp16*  dC_input,
                    int32_t    ldc)
{
    static_assert(BLK_M == 32 && BLK_N == 32 && BLK_K == 16, "incorrect block shape");
    const int thx  = threadIdx.x;
    const int blx  = blockIdx.x;  // block's m position
    const int bly  = blockIdx.y;  // block's n position
    const int blz  = blockIdx.z;  // block's matrix in the batch
    dA_input += size_t(blz) * M * K;
    dB_input += size_t(blz) * N * K;
    dC_input += size_t(blz) * N * M;
    __shared__ struct {
        fp16 B[BLK_K][BLK_N / 2 + 1][2];
    } lds;

    fp32x4 C_reg_acc[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        RegisterUnion fragAB, fragAB2;

        size_t read_dim_k  = thx % BLK_K;
        size_t read_dim_mn = thx / BLK_K;
        size_t offset_B = size_t(kk + read_dim_k) + size_t(bly * BLK_N + read_dim_mn) * size_t(ldb);
        lds.B[read_dim_k][(read_dim_mn +  0) % (BLK_N / 2)][(read_dim_mn +  0) / (BLK_N / 2)] = dB_input[offset_B +  0 * size_t(ldb)];
        lds.B[read_dim_k][(read_dim_mn +  4) % (BLK_N / 2)][(read_dim_mn +  4) / (BLK_N / 2)] = dB_input[offset_B +  4 * size_t(ldb)];
        lds.B[read_dim_k][(read_dim_mn +  8) % (BLK_N / 2)][(read_dim_mn +  8) / (BLK_N / 2)] = dB_input[offset_B +  8 * size_t(ldb)];
        lds.B[read_dim_k][(read_dim_mn + 12) % (BLK_N / 2)][(read_dim_mn + 12) / (BLK_N / 2)] = dB_input[offset_B + 12 * size_t(ldb)];
        lds.B[read_dim_k][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)] = dB_input[offset_B + 16 * size_t(ldb)];
        lds.B[read_dim_k][(read_dim_mn + 20) % (BLK_N / 2)][(read_dim_mn + 20) / (BLK_N / 2)] = dB_input[offset_B + 20 * size_t(ldb)];
        lds.B[read_dim_k][(read_dim_mn + 24) % (BLK_N / 2)][(read_dim_mn + 24) / (BLK_N / 2)] = dB_input[offset_B + 24 * size_t(ldb)];
        lds.B[read_dim_k][(read_dim_mn + 28) % (BLK_N / 2)][(read_dim_mn + 28) / (BLK_N / 2)] = dB_input[offset_B + 28 * size_t(ldb)];
        


        read_dim_mn = thx % BLK_K;   // shall be (BLK_MN / 2) not BLK_K
        read_dim_k  = thx / BLK_K * 4;
        size_t offset_A = size_t(kk + read_dim_k) * size_t(lda) + size_t(blx * BLK_M + read_dim_mn);
        fragAB.vector_front = { dA_input[offset_A + 0 * size_t(lda)],
                                dA_input[offset_A + 1 * size_t(lda)],
                                dA_input[offset_A + 2 * size_t(lda)],
                                dA_input[offset_A + 3 * size_t(lda)] };
        offset_A += 16;
        fragAB2.vector_front = { dA_input[offset_A + 0 * size_t(lda)],
                                 dA_input[offset_A + 1 * size_t(lda)],
                                 dA_input[offset_A + 2 * size_t(lda)],
                                 dA_input[offset_A + 3 * size_t(lda)] };

        fragAB.vector_rear  = { lds.B[read_dim_k + 0][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)], 
                                lds.B[read_dim_k + 1][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)],
                                lds.B[read_dim_k + 2][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)], 
                                lds.B[read_dim_k + 3][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)] };
        fragAB2.vector_rear = { lds.B[read_dim_k + 0][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)],
                                lds.B[read_dim_k + 1][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)], 
                                lds.B[read_dim_k + 2][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)], 
                                lds.B[read_dim_k + 3][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)] };

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

    dC_input[offset_C0 +  0 * size_t(ldc)] = (fp16)C_reg_acc[0].x;
    dC_input[offset_C0 +  4 * size_t(ldc)] = (fp16)C_reg_acc[0].y;
    dC_input[offset_C0 +  8 * size_t(ldc)] = (fp16)C_reg_acc[0].z;
    dC_input[offset_C0 + 12 * size_t(ldc)] = (fp16)C_reg_acc[0].w;

    dC_input[offset_C1 +  0 * size_t(ldc)] = (fp16)C_reg_acc[1].x;
    dC_input[offset_C1 +  4 * size_t(ldc)] = (fp16)C_reg_acc[1].y;
    dC_input[offset_C1 +  8 * size_t(ldc)] = (fp16)C_reg_acc[1].z;
    dC_input[offset_C1 + 12 * size_t(ldc)] = (fp16)C_reg_acc[1].w;

    dC_input[offset_C2 +  0 * size_t(ldc)] = (fp16)C_reg_acc[2].x;
    dC_input[offset_C2 +  4 * size_t(ldc)] = (fp16)C_reg_acc[2].y;
    dC_input[offset_C2 +  8 * size_t(ldc)] = (fp16)C_reg_acc[2].z;
    dC_input[offset_C2 + 12 * size_t(ldc)] = (fp16)C_reg_acc[2].w;

    dC_input[offset_C3 +  0 * size_t(ldc)] = (fp16)C_reg_acc[3].x;
    dC_input[offset_C3 +  4 * size_t(ldc)] = (fp16)C_reg_acc[3].y;
    dC_input[offset_C3 +  8 * size_t(ldc)] = (fp16)C_reg_acc[3].z;
    dC_input[offset_C3 + 12 * size_t(ldc)] = (fp16)C_reg_acc[3].w;
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
        gemm_batched_kernel_tensorcore_32x32x16_fp16fp32_Akxm_Bnxk_Cnxm<blk_m, blk_n, blk_k><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, dA, lda, dB, ldb, dC, ldc);
    }
    else
    {
        const int blk_m = 32;
        const int blk_n = 32;
        const int blk_k = 16;
        dim3      dimBlock(HEP_WARP_SIZE);
        dim3      dimGrid((m - 1) / blk_m + 1, (n - 1) / blk_n + 1, batch_count);
        gemm_batched_general_kernel_tensorcore_32x32x16_fp16fp32_Akxm_Bnxk_Cnxm<blk_m, blk_n, blk_k><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, dA, lda, dB, ldb, dC, ldc);
    }
}