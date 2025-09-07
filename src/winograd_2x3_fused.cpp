#define FLT_H      3L
#define FLT_W      3L
#define TILE_IN_H  4L
#define TILE_IN_W  4L
#define TILE_OUT_H 2L
#define TILE_OUT_W 2L
#include "common.h"
#include "error.h"

template <int  BLK_M,
          int  BLK_N,
          int  BLK_K>
__global__ void __launch_bounds__(32*16*2)
winograd_2x3_kernel(_Float16* filter_d,
                    FltShape  fs,
                    _Float16* image_d,
                    ImgShape  is,
                    uint64_t  padding_h,
                    uint64_t  padding_w,
                    _Float16* out_d,
                    OutShape  os,
                    TileShape ts)
{
  /*! Dimensions
      M => P / Tiles
      N => K / Output channels
      K => C / Input channels
  */
  __shared__ union {
    struct {
      fp16 V[TILE_IN_H][TILE_IN_W][BLK_K][BLK_M];
      fp16 Ut[TILE_IN_H][TILE_IN_W][BLK_N][BLK_K];
    };
    struct {
      fp16 Y[TILE_IN_H][TILE_IN_W][BLK_M][BLK_N];
    };
  } lds;
  
  const int thx = threadIdx.x;
  const int thy = threadIdx.y;
  const int tid = thx + thy * blockDim.x;
  const int blx = blockIdx.x;
  const int bly = blockIdx.y;
  const int TCU_SIZE = 16; 
  static_assert(BLK_M == 32 && BLK_N == 32 && BLK_K == 16, "incorrect block shape");
  
  typedef fp16 (*image_tensor_t) [is.ic][is.h][is.w];
  auto image = (image_tensor_t) image_d;
  typedef fp16 (*filter_tensor_t) [fs.ic][fs.h][fs.w];
  auto filter = (filter_tensor_t) filter_d;
  typedef fp16 (*output_tensor_t) [os.oc][os.h][os.w];
  auto output = (output_tensor_t) out_d;
  fp32x4 C_reg_acc[2][2] = {0};
  int local_oc_idx;
  int local_tile_idx;
  int local_ic_idx;
  int tile_blk = BLK_M * blx;
  int oc_blk   = BLK_N * bly;

  for(int ic_blk = 0; ic_blk < is.ic; ic_blk += BLK_K)
  {
    int local_ic_idx;
    int local_oc_idx;
    fp16 z0, z1, z2, z3, z6;
    
    if(thy == 1) { // must be 1 not 0
      local_ic_idx = thx % BLK_K;
      local_oc_idx = thx / BLK_K;
      // ! read filter slice from global memory into shared memory
      typedef fp16 filter_tile_t __attribute__((ext_vector_type(9)));
      filter_tile_t filter_tile = {0};
      if(oc_blk + local_oc_idx < fs.oc)
        if(ic_blk + local_ic_idx < fs.ic)
          filter_tile = *(filter_tile_t*)&filter[oc_blk + local_oc_idx][ic_blk + local_ic_idx][0][0];
      
      fp16 tmp[TILE_IN_H][TILE_IN_W];
      
      //! filter transform
      for (int w = 0; w < FLT_W; ++w) {
        z6 = filter_tile[w + 0 * FLT_W];
        z0 = ((fp16)( 1.0f )) * z6;
        z1 = ((fp16)( 1.0f / 2.0f )) * z6;
        z2 = ((fp16)( 1.0f / 2.0f )) * z6;

        z6 = filter_tile[w + 1 * FLT_W];
        z1 += ((fp16)( 1.0f / 2.0f )) * z6;
        z2 += ((fp16)(-1.0f / 2.0f )) * z6;

        z6 = filter_tile[w + 2 * FLT_W];
        z1 += ((fp16)( 1.0f / 2.0f )) * z6;
        z2 += ((fp16)( 1.0f / 2.0f )) * z6;
        z3 =  ((fp16)( 1.0f )) * z6;

        tmp[0][w] = z0;
        tmp[1][w] = z1;
        tmp[2][w] = z2;
        tmp[3][w] = z3;
      }

      for (int h = 0; h < TILE_IN_H; ++h) {
        z6 = tmp[h][0];
        z0 = ((fp16)( 1.0f )) * z6;
        z1 = ((fp16)( 1.0f / 2.0f )) * z6;
        z2 = ((fp16)( 1.0f / 2.0f )) * z6;

        z6 = tmp[h][1];
        z1 += ((fp16)( 1.0f / 2.0f )) * z6;
        z2 += ((fp16)(-1.0f / 2.0f )) * z6;

        z6 = tmp[h][2];
        z1 += ((fp16)( 1.0f / 2.0f)) * z6;
        z2 += ((fp16)( 1.0f / 2.0f)) * z6;
        z3 = ((fp16)( 1.0f )) * z6;

        lds.Ut[h][0][local_oc_idx][local_ic_idx] = z0;
        lds.Ut[h][1][local_oc_idx][local_ic_idx] = z1;
        lds.Ut[h][2][local_oc_idx][local_ic_idx] = z2;
        lds.Ut[h][3][local_oc_idx][local_ic_idx] = z3;
      }
    }

    if(thy == 0) { // must be 0 not 1
      local_tile_idx = thx % BLK_M;
      local_ic_idx   = thx / BLK_M;
      fp16x4 img_tile[TILE_IN_H];
      //! read image slice from global memory into shared memory
      if(tile_blk + local_tile_idx < ts.numTileTotal && ic_blk + local_ic_idx < is.ic) {
        const TileIndex ti = getTileIndex(tile_blk + local_tile_idx, ts);
        const uint64_t b = ti.b, th = ti.th, tw = ti.tw;
        for(int h = 0; h < TILE_IN_H; ++h) {
          if(th * TILE_OUT_H + h - padding_h < is.h) {
              img_tile[h][0] = (tw * TILE_OUT_W + 0 - padding_w < is.w)
                                ? image[b][ic_blk + local_ic_idx][th * TILE_OUT_H + h - padding_h][tw * TILE_OUT_W + 0 - padding_w] : 0;
              img_tile[h][1] = (tw * TILE_OUT_W + 1 - padding_w < is.w)
                                ? image[b][ic_blk + local_ic_idx][th * TILE_OUT_H + h - padding_h][tw * TILE_OUT_W + 1 - padding_w] : 0;
              img_tile[h][2] = (tw * TILE_OUT_W + 2 - padding_w < is.w)
                                ? image[b][ic_blk + local_ic_idx][th * TILE_OUT_H + h - padding_h][tw * TILE_OUT_W + 2 - padding_w] : 0;
              img_tile[h][3] = (tw * TILE_OUT_W + 3 - padding_w < is.w)
                                ? image[b][ic_blk + local_ic_idx][th * TILE_OUT_H + h - padding_h][tw * TILE_OUT_W + 3 - padding_w] : 0;
          }
        }
      } else {
        for(int h = 0; h < TILE_IN_H; ++h)
          for(int w = 0; w < TILE_IN_W; ++w)
            img_tile[h][w] = 0;
      }
      //！image transform
      for (int w = 0; w < TILE_IN_W; ++w) {
        z6 = img_tile[0][w];
        z0 = ((fp16) 1.0f) * z6;

        z6 = img_tile[1][w];
        z1 = ((fp16) 1.0f) * z6;
        z2 = ((fp16)-1.0f) * z6;
        z3 = ((fp16) 1.0f) * z6;

        z6 = img_tile[2][w];
        z0 += ((fp16)-1.0f) * z6;
        z1 += ((fp16) 1.0f) * z6;
        z2 += ((fp16) 1.0f) * z6;

        z6 = img_tile[3][w];
        z3 += ((fp16)-1.0f) * z6;

        lds.V[0][w][local_ic_idx][local_tile_idx] = z0;
        lds.V[1][w][local_ic_idx][local_tile_idx] = z1;
        lds.V[2][w][local_ic_idx][local_tile_idx] = z2;
        lds.V[3][w][local_ic_idx][local_tile_idx] = z3;
      }
      for (int h = 0; h < TILE_IN_H; ++h) {
        z6 = lds.V[h][0][local_ic_idx][local_tile_idx];
        z0 = ((fp16) 1.0f) * z6;

        z6 = lds.V[h][1][local_ic_idx][local_tile_idx];
        z1 = ((fp16) 1.0f) * z6;
        z2 = ((fp16)-1.0f) * z6;
        z3 = ((fp16) 1.0f) * z6;

        z6 = lds.V[h][2][local_ic_idx][local_tile_idx];
        z0 += ((fp16)-1.0f) * z6;
        z1 += ((fp16) 1.0f) * z6;
        z2 += ((fp16) 1.0f) * z6;

        z6 = lds.V[h][3][local_ic_idx][local_tile_idx];
        z3 += ((fp16)-1.0f) * z6;

        lds.V[h][0][local_ic_idx][local_tile_idx] = z0;
        lds.V[h][1][local_ic_idx][local_tile_idx] = z1;
        lds.V[h][2][local_ic_idx][local_tile_idx] = z2;
        lds.V[h][3][local_ic_idx][local_tile_idx] = z3;
      }
    }


    __syncthreads();

    
#ifdef HYGON_DCU_MATRIX_CORE
    asm volatile("s_waitcnt lgkmcnt(0)\n\t");
    const int elem_idx = tid / HEP_WARP_SIZE;
    const int h = elem_idx / TILE_IN_W;
    const int w = elem_idx % TILE_IN_W;
    //! do batched gemm
    fp16x4 frag_A[2], frag_B[2];
    const size_t read_dim_k  = (tid % HEP_WARP_SIZE) / TCU_SIZE * 4;
    const size_t read_dim_mn = (tid % HEP_WARP_SIZE) % TCU_SIZE;
    frag_A[0] = { lds.V[h][w][read_dim_k + 0][read_dim_mn +  0], 
                  lds.V[h][w][read_dim_k + 1][read_dim_mn +  0], 
                  lds.V[h][w][read_dim_k + 2][read_dim_mn +  0], 
                  lds.V[h][w][read_dim_k + 3][read_dim_mn +  0] };
    frag_A[1] = { lds.V[h][w][read_dim_k + 0][read_dim_mn + 16], 
                  lds.V[h][w][read_dim_k + 1][read_dim_mn + 16], 
                  lds.V[h][w][read_dim_k + 2][read_dim_mn + 16], 
                  lds.V[h][w][read_dim_k + 3][read_dim_mn + 16] };

    frag_B[0] = { lds.Ut[h][w][read_dim_mn +  0][read_dim_k + 0], 
                  lds.Ut[h][w][read_dim_mn +  0][read_dim_k + 1], 
                  lds.Ut[h][w][read_dim_mn +  0][read_dim_k + 2], 
                  lds.Ut[h][w][read_dim_mn +  0][read_dim_k + 3] };
    frag_B[1] = { lds.Ut[h][w][read_dim_mn + 16][read_dim_k + 0], 
                  lds.Ut[h][w][read_dim_mn + 16][read_dim_k + 1], 
                  lds.Ut[h][w][read_dim_mn + 16][read_dim_k + 2], 
                  lds.Ut[h][w][read_dim_mn + 16][read_dim_k + 3] };
    asm volatile("s_waitcnt lgkmcnt(0)\n\t");
    #ifndef IGNORE_NOP
    	#define NOP_48_CYCLES() \
			    asm volatile("s_nop 8\n\t"); \
                            asm volatile("s_nop 8\n\t"); \
                            asm volatile("s_nop 8\n\t"); \
                            asm volatile("s_nop 8\n\t"); \
                            asm volatile("s_nop 8\n\t"); \
                            asm volatile("s_nop 8\n\t");
    #else
    	#define NOP_48_CYCLES() 
    #endif
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[0][0]), "+v"(frag_A[0]), "+v"(frag_B[0]));
    NOP_48_CYCLES();
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[0][1]), "+v"(frag_A[0]), "+v"(frag_B[1]));
    NOP_48_CYCLES();
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[1][0]), "+v"(frag_A[1]), "+v"(frag_B[0]));
    NOP_48_CYCLES();
    asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0\n\t":"+v"(C_reg_acc[1][1]), "+v"(frag_A[1]), "+v"(frag_B[1]));
    NOP_48_CYCLES();
#else 
    const int elem_idx = tid / HEP_WARP_SIZE;
    const int h = elem_idx / TILE_IN_W;
    const int w = elem_idx % TILE_IN_W;
    size_t write_dim_m = (tid % HEP_WARP_SIZE) % BLK_K;
    size_t write_dim_n = (tid % HEP_WARP_SIZE) / BLK_K;

    #pragma unroll(BLK_K)
    for (int k = 0; k < BLK_K; k++) {
      float v = lds.V[h][w][k][write_dim_m +  0];
      #pragma unroll(4)
      for (int t = 0; t < 4; ++t) {
        C_reg_acc[0][0][t] += v * (float)lds.Ut[h][w][write_dim_n +  0 + t * 4][k];
      }
    }
    #pragma unroll(BLK_K)
    for (int k = 0; k < BLK_K; k++) {
      float v = lds.V[h][w][k][write_dim_m +  0];
      #pragma unroll(4)
      for (int t = 0; t < 4; ++t) {
        C_reg_acc[0][1][t] += v * (float)lds.Ut[h][w][write_dim_n + 16 + t * 4][k];
      }
    }
    #pragma unroll(BLK_K)
    for (int k = 0; k < BLK_K; k++) {
      float v = lds.V[h][w][k][write_dim_m + 16];
      #pragma unroll(4)
      for (int t = 0; t < 4; ++t) {
        C_reg_acc[1][0][t] += v * (float)lds.Ut[h][w][write_dim_n +  0 + t * 4][k];
      }
    }
    #pragma unroll(BLK_K)
    for (int k = 0; k < BLK_K; k++) {
      float v = lds.V[h][w][k][write_dim_m + 16];
      #pragma unroll(4)
      for (int t = 0; t < 4; ++t) {
        C_reg_acc[1][1][t] += v * (float)lds.Ut[h][w][write_dim_n + 16 + t * 4][k];
      }
    }
#endif
  } //! end for ic_blk = 0 to K by BLK_K


  __syncthreads();

  //! Store the result matrices of batched gemm into lds.
  const int elem_idx = tid / HEP_WARP_SIZE;
  const int h = elem_idx / TILE_IN_W;
  const int w = elem_idx % TILE_IN_W;
  const size_t write_local_tile = (tid % HEP_WARP_SIZE) % TCU_SIZE;
  const size_t write_local_oc   = (tid % HEP_WARP_SIZE) / TCU_SIZE;
  lds.Y[h][w][write_local_oc +  0 +  0][write_local_tile +  0] = (_Float16)C_reg_acc[0][0].x;
  lds.Y[h][w][write_local_oc +  4 +  0][write_local_tile +  0] = (_Float16)C_reg_acc[0][0].y;
  lds.Y[h][w][write_local_oc +  8 +  0][write_local_tile +  0] = (_Float16)C_reg_acc[0][0].z;
  lds.Y[h][w][write_local_oc + 12 +  0][write_local_tile +  0] = (_Float16)C_reg_acc[0][0].w;

  lds.Y[h][w][write_local_oc +  0 + 16][write_local_tile +  0] = (_Float16)C_reg_acc[0][1].x;
  lds.Y[h][w][write_local_oc +  4 + 16][write_local_tile +  0] = (_Float16)C_reg_acc[0][1].y;
  lds.Y[h][w][write_local_oc +  8 + 16][write_local_tile +  0] = (_Float16)C_reg_acc[0][1].z;
  lds.Y[h][w][write_local_oc + 12 + 16][write_local_tile +  0] = (_Float16)C_reg_acc[0][1].w;

  lds.Y[h][w][write_local_oc +  0 +  0][write_local_tile + 16] = (_Float16)C_reg_acc[1][0].x;
  lds.Y[h][w][write_local_oc +  4 +  0][write_local_tile + 16] = (_Float16)C_reg_acc[1][0].y;
  lds.Y[h][w][write_local_oc +  8 +  0][write_local_tile + 16] = (_Float16)C_reg_acc[1][0].z;
  lds.Y[h][w][write_local_oc + 12 +  0][write_local_tile + 16] = (_Float16)C_reg_acc[1][0].w;

  lds.Y[h][w][write_local_oc +  0 + 16][write_local_tile + 16] = (_Float16)C_reg_acc[1][1].x;
  lds.Y[h][w][write_local_oc +  4 + 16][write_local_tile + 16] = (_Float16)C_reg_acc[1][1].y;
  lds.Y[h][w][write_local_oc +  8 + 16][write_local_tile + 16] = (_Float16)C_reg_acc[1][1].z;
  lds.Y[h][w][write_local_oc + 12 + 16][write_local_tile + 16] = (_Float16)C_reg_acc[1][1].w;

  __syncthreads();
  

  local_oc_idx   = tid / BLK_M;
  local_tile_idx = tid % BLK_M;

  //! do output transform 
  fp16 z0, z1, z4;
  for (int w = 0; w < TILE_IN_W; ++w) {
    z4 = lds.Y[0][w][local_oc_idx][local_tile_idx];
    z0 =  z4;

    z4 = lds.Y[1][w][local_oc_idx][local_tile_idx];
    z0 += z4;
    z1 =  z4;
    
    z4 = lds.Y[2][w][local_oc_idx][local_tile_idx];
    z0 +=  z4;
    z1 += -z4;

    z4 = lds.Y[3][w][local_oc_idx][local_tile_idx];
    z1 += -z4;


    lds.Y[0][w][local_oc_idx][local_tile_idx] = z0;
    lds.Y[1][w][local_oc_idx][local_tile_idx] = z1;
  }

  for (int h = 0; h < TILE_OUT_H; ++h) {
    z4 = lds.Y[h][0][local_oc_idx][local_tile_idx];
    z0 =  z4;

    z4 = lds.Y[h][1][local_oc_idx][local_tile_idx];
    z0 += z4;
    z1 =  z4;

    z4 = lds.Y[h][2][local_oc_idx][local_tile_idx];
    z0 +=  z4;
    z1 += -z4;

    z4 = lds.Y[h][3][local_oc_idx][local_tile_idx];
    z1 += -z4;

    int oc = oc_blk + local_oc_idx;
    int tile = tile_blk + local_tile_idx;
    if(oc < os.oc && tile < ts.numTileTotal) {
      TileIndex ti = getTileIndex(tile, ts);
      int n = ti.b, tw = ti.tw, th = ti.th;
      if(th * TILE_OUT_H + h < os.h && tw * TILE_OUT_W + 0 < os.w)
        output[n][oc][th * TILE_OUT_H + h][tw * TILE_OUT_W + 0] = (_Float16) z0;
      if(th * TILE_OUT_H + h < os.h && tw * TILE_OUT_W + 1 < os.w)
        output[n][oc][th * TILE_OUT_H + h][tw * TILE_OUT_W + 1] = (_Float16) z1;
    }
  }
}


void winograd_2x3_fused(const void* param_ptr) {
    // std::cout << "winograd_2x3_fused" << std::endl;
    const mykernelParamType* param = (const mykernelParamType*)param_ptr;
    _Float16* filter_d = param->pweight;
    _Float16* image_d  = param->pin;
    _Float16* out_d    = param->pout;
    uint64_t padding_h = param->p;
    uint64_t padding_w = param->q;

    ImgShape  is = {param->n, param->c, param->h, param->w};
    FltShape  fs = {param->k, param->c, param->r, param->s};
    OutShape  os = getOutShape(is, fs, padding_h, padding_w);
    TileShape ts = getTileShape(os);

    const int blk_m = 32;
    const int blk_n = 32;
    const int blk_k = 16;
    dim3 gridDim(DIV_UP(ts.numTileTotal, blk_m), DIV_UP(fs.oc, blk_n), 1), blockDim(blk_m * blk_k, 2);
    winograd_2x3_kernel<blk_m, blk_n, blk_k><<<gridDim, blockDim, 0, hipStreamDefault>>>(filter_d, fs, image_d, is, padding_h, padding_w, out_d, os, ts);   
    HIP_CHECK_KERNEL("Kernel panic!!!");
}

