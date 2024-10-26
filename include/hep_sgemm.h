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
    // __shared__ struct {
    //     fp16 A[BLK_K][BLK_M / 2 + 1][2]; // shared memory for A, B
    //     fp16 B[BLK_K][BLK_N / 2 + 1][2];
    // } lds;

    fp32x4 C_reg_acc[4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        // size_t read_dim_k  = thx % BLK_K;
        // size_t read_dim_mn = thx / BLK_K;
        // size_t offset_A = size_t(kk + read_dim_k) + size_t(blx * BLK_M + read_dim_mn) * size_t(lda);
        // size_t offset_B = size_t(kk + read_dim_k) + size_t(bly * BLK_N + read_dim_mn) * size_t(ldb);
        
        // lds.A[read_dim_k][(read_dim_mn +  0) % (BLK_M / 2)][(read_dim_mn +  0) / (BLK_M / 2)] = dA_input[offset_A +  0 * size_t(lda)];
        // lds.A[read_dim_k][(read_dim_mn +  4) % (BLK_M / 2)][(read_dim_mn +  4) / (BLK_M / 2)] = dA_input[offset_A +  4 * size_t(lda)];
        // lds.A[read_dim_k][(read_dim_mn +  8) % (BLK_M / 2)][(read_dim_mn +  8) / (BLK_M / 2)] = dA_input[offset_A +  8 * size_t(lda)];
        // lds.A[read_dim_k][(read_dim_mn + 12) % (BLK_M / 2)][(read_dim_mn + 12) / (BLK_M / 2)] = dA_input[offset_A + 12 * size_t(lda)];
        // lds.A[read_dim_k][(read_dim_mn + 16) % (BLK_M / 2)][(read_dim_mn + 16) / (BLK_M / 2)] = dA_input[offset_A + 16 * size_t(lda)];
        // lds.A[read_dim_k][(read_dim_mn + 20) % (BLK_M / 2)][(read_dim_mn + 20) / (BLK_M / 2)] = dA_input[offset_A + 20 * size_t(lda)];
        // lds.A[read_dim_k][(read_dim_mn + 24) % (BLK_M / 2)][(read_dim_mn + 24) / (BLK_M / 2)] = dA_input[offset_A + 24 * size_t(lda)];
        // lds.A[read_dim_k][(read_dim_mn + 28) % (BLK_M / 2)][(read_dim_mn + 28) / (BLK_M / 2)] = dA_input[offset_A + 28 * size_t(lda)];

        // lds.B[read_dim_k][(read_dim_mn +  0) % (BLK_N / 2)][(read_dim_mn +  0) / (BLK_N / 2)] = dB_input[offset_B +  0 * size_t(ldb)];
        // lds.B[read_dim_k][(read_dim_mn +  4) % (BLK_N / 2)][(read_dim_mn +  4) / (BLK_N / 2)] = dB_input[offset_B +  4 * size_t(ldb)];
        // lds.B[read_dim_k][(read_dim_mn +  8) % (BLK_N / 2)][(read_dim_mn +  8) / (BLK_N / 2)] = dB_input[offset_B +  8 * size_t(ldb)];
        // lds.B[read_dim_k][(read_dim_mn + 12) % (BLK_N / 2)][(read_dim_mn + 12) / (BLK_N / 2)] = dB_input[offset_B + 12 * size_t(ldb)];
        // lds.B[read_dim_k][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)] = dB_input[offset_B + 16 * size_t(ldb)];
        // lds.B[read_dim_k][(read_dim_mn + 20) % (BLK_N / 2)][(read_dim_mn + 20) / (BLK_N / 2)] = dB_input[offset_B + 20 * size_t(ldb)];
        // lds.B[read_dim_k][(read_dim_mn + 24) % (BLK_N / 2)][(read_dim_mn + 24) / (BLK_N / 2)] = dB_input[offset_B + 24 * size_t(ldb)];
        // lds.B[read_dim_k][(read_dim_mn + 28) % (BLK_N / 2)][(read_dim_mn + 28) / (BLK_N / 2)] = dB_input[offset_B + 28 * size_t(ldb)];

        RegisterUnion fragAB, fragAB2;
        
        size_t read_dim_mn = thx % BLK_K;
        size_t read_dim_k  = thx / BLK_K * 4;
        // size_t read_dim_k  = thx % BLK_K;
        // size_t read_dim_mn = thx / BLK_K;
        size_t offset_A = size_t(kk + read_dim_k) * size_t(lda) + size_t(blx * BLK_M + read_dim_mn);
        size_t offset_B = size_t(kk + read_dim_k) * size_t(ldb) + size_t(bly * BLK_N + read_dim_mn);

        // fragAB.vector_front  = {lds.A[read_dim_k + 0][read_dim_mn % (BLK_M / 2)][read_dim_mn / (BLK_M / 2)], 
        //                         lds.A[read_dim_k + 1][read_dim_mn % (BLK_M / 2)][read_dim_mn / (BLK_M / 2)], 
        //                         lds.A[read_dim_k + 2][read_dim_mn % (BLK_M / 2)][read_dim_mn / (BLK_M / 2)], 
        //                         lds.A[read_dim_k + 3][read_dim_mn % (BLK_M / 2)][read_dim_mn / (BLK_M / 2)]};
        // fragAB.vector_rear   = {lds.B[read_dim_k + 0][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)], 
        //                         lds.B[read_dim_k + 1][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)],
        //                         lds.B[read_dim_k + 2][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)], 
        //                         lds.B[read_dim_k + 3][read_dim_mn % (BLK_N / 2)][read_dim_mn / (BLK_N / 2)]};
        // fragAB2.vector_front = {lds.A[read_dim_k + 0][(read_dim_mn + 16) % (BLK_M / 2)][(read_dim_mn + 16) / (BLK_M / 2)], 
        //                         lds.A[read_dim_k + 1][(read_dim_mn + 16) % (BLK_M / 2)][(read_dim_mn + 16) / (BLK_M / 2)], 
        //                         lds.A[read_dim_k + 2][(read_dim_mn + 16) % (BLK_M / 2)][(read_dim_mn + 16) / (BLK_M / 2)],
        //                         lds.A[read_dim_k + 3][(read_dim_mn + 16) % (BLK_M / 2)][(read_dim_mn + 16) / (BLK_M / 2)]};
        // fragAB2.vector_rear  = {lds.B[read_dim_k + 0][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)],
        //                         lds.B[read_dim_k + 1][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)], 
        //                         lds.B[read_dim_k + 2][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)], 
        //                         lds.B[read_dim_k + 3][(read_dim_mn + 16) % (BLK_N / 2)][(read_dim_mn + 16) / (BLK_N / 2)]};
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


template <int  BLK_M,
          int  BLK_N,
          int  BLK_K>
static __global__ __launch_bounds__(HEP_WARP_SIZE) void
gemm_batched_kernel_tensorcore_16x16x16_fp16fp32
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
    static_assert(BLK_M == 16 && BLK_N == 16 && BLK_K == 16, "incorrect block shape");
    int thx  = threadIdx.x;
    int blx  = blockIdx.x;  // block's m position
    int bly  = blockIdx.y;  // block's n position
    __shared__ struct {
        fp16 AB[BLK_M][2][BLK_K]; // shared memory for A, B
    } lds;

    fp32x4 C_reg_acc = {0, 0, 0, 0};

    int kk = 0;
    for(; kk < K; kk += BLK_K)
    {
        size_t read_dim_k  = thx % BLK_K;
        size_t read_dim_mn = thx / BLK_K;
        size_t offset_A = size_t(kk + read_dim_k) + size_t(blx * BLK_M + read_dim_mn) * size_t(lda);
        size_t offset_B = size_t(kk + read_dim_k) + size_t(bly * BLK_N + read_dim_mn) * size_t(ldb);

        lds.AB[read_dim_k][0][read_dim_mn +  0] = dA_input[offset_A +  0 * size_t(lda)];
        lds.AB[read_dim_k][0][read_dim_mn +  4] = dA_input[offset_A +  4 * size_t(lda)];
        lds.AB[read_dim_k][0][read_dim_mn +  8] = dA_input[offset_A +  8 * size_t(lda)];
        lds.AB[read_dim_k][0][read_dim_mn + 12] = dA_input[offset_A + 12 * size_t(lda)];

        lds.AB[read_dim_k][1][read_dim_mn +  0] = dB_input[offset_B +  0 * size_t(ldb)];
        lds.AB[read_dim_k][1][read_dim_mn +  4] = dB_input[offset_B +  4 * size_t(ldb)];
        lds.AB[read_dim_k][1][read_dim_mn +  8] = dB_input[offset_B +  8 * size_t(ldb)];
        lds.AB[read_dim_k][1][read_dim_mn + 12] = dB_input[offset_B + 12 * size_t(ldb)];
        asm volatile("s_waitcnt lgkmcnt(0)\n\t");

        RegisterUnion fragAB;
        fp32x4 C_reg_res = {0, 0, 0, 0};
        int lds_read_offset = (thx * 8) * sizeof(_Float16);

        read_dim_mn = thx % BLK_M;
        read_dim_k  = thx / BLK_M * 4;
        fragAB.vector_front = {lds.AB[read_dim_k + 0][0][read_dim_mn], lds.AB[read_dim_k + 1][0][read_dim_mn], lds.AB[read_dim_k + 2][0][read_dim_mn], lds.AB[read_dim_k + 3][0][read_dim_mn]};
        fragAB.vector_rear  = {lds.AB[read_dim_k + 0][1][read_dim_mn], lds.AB[read_dim_k + 1][1][read_dim_mn], lds.AB[read_dim_k + 2][1][read_dim_mn], lds.AB[read_dim_k + 3][1][read_dim_mn]};

        __syncthreads();
        // asm volatile("ds_read_m32x16_b16 %0, %1 offset:0\n\t": "+v"(fragAB.vector8), "+v"(lds_read_offset));
        asm volatile("s_waitcnt lgkmcnt(0)\n\t");
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc), "+v"(fragAB.vector_front), "+v"(fragAB.vector_rear));

        // C_reg_acc += C_reg_res;
    }
    __syncthreads();

    size_t output_row = thx % 16;
    size_t output_col = thx / 16;
    size_t offset_C = size_t(bly * BLK_N + output_col) * size_t(ldc) + size_t(blx * BLK_M + output_row);

    dC_input[offset_C +  0 * size_t(ldc)] = (_Float16)C_reg_acc.x;
    dC_input[offset_C +  4 * size_t(ldc)] = (_Float16)C_reg_acc.y;
    dC_input[offset_C +  8 * size_t(ldc)] = (_Float16)C_reg_acc.z;
    dC_input[offset_C + 12 * size_t(ldc)] = (_Float16)C_reg_acc.w;
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
    // else if((m % 16 == 0) && (n % 16 == 0) && (k % 16 == 0))
    // {
        // std::cout << "this kernel not implemented!!!" << std::endl;
        // std::exit(-1);
        // m is mult of 16, n is mult of 16, k is mult of 16
        // const int blk_m = 16;
        // const int blk_n = 16;
        // const int blk_k = 16;
        // dim3      dimBlock(HEP_WARP_SIZE);
        // dim3      dimGrid(m / blk_m, n / blk_n, 1);
        // gemm_batched_kernel_tensorcore_16x16x16_fp16fp32<blk_m, blk_n, blk_k><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, dA, lda, dB, ldb, dC, ldc);
    // }
    // else
    // {   
        // std::cout << "General kernel not implemented!!!" << std::endl;
        // std::exit(-1);
        // const int dim_m = 16;
        // const int dim_n = 16;
        // const int blk_m = 32;
        // const int blk_n = 32;
        // const int blk_k = 8;
        // dim3      dimBlock(dim_m, dim_n, 1);
        // dim3      dimGrid(((m - 1) / blk_m) + 1, ((n - 1) / blk_n) + 1, batch_count_unused);
        // gemm_batched_general_kernel<inoutT, calcT, dim_m, dim_n, blk_m, blk_n, blk_k, blk_m, blk_k, blk_k, blk_n><<<dimGrid, dimBlock, 0, stream>>>(m, n, k, alpha, dA, lda, dB, ldb, beta, dC, ldc, batch_count_unused);
    // }
}