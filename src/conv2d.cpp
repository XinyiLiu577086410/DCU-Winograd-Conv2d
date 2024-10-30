#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "conv2d.h"
#include "common.h"
#include "error.h"
#include "hep_sgemm.h"

/*选手自定义的kernel入参结构体*/
typedef struct mykernelParamType
{
    _Float16*   pin;                            //输入数据地址
    _Float16*   pweight;                        //权值数据地址
    _Float16*   pout;                           //输出数据地址

    _Float16*   U_d;
    _Float16*   V_d;
    _Float16*   M_d;
    _Float16*   Y_d;

    unsigned int      n;                              //batch szie            
    unsigned int      c;                              //channel number        
    unsigned int      h;                              //数据高                
    unsigned int      w;                              //数据宽                
    unsigned int      k;                              //卷积核数量            
    unsigned int      r;                              //卷积核高              
    unsigned int      s;                              //卷积核宽              
    unsigned int      u;                              //卷积在高方向上的步长  
    unsigned int      v;                              //卷积在宽方向上的步长  
    unsigned int      p;                              //卷积在高方向上的补边  
    unsigned int      q;                              //卷积在宽方向上的补边  
    unsigned int      Oh;                             //卷积在高方向上输出大小    
    unsigned int      Ow;                             //卷积在宽方向上输出大小
    unsigned int      revs0;                          //预留                          
    unsigned int      revs1;                          //预留
    unsigned int      revs2;                          //预留
    unsigned int      revs3;                          //预留
    unsigned int      revs4;                          //预留
    unsigned int      revs5;                          //预留
    unsigned int      revs6;                          //预留
    unsigned int      revs7;                          //预留
}mykernelParamType;                          

template <typename inoutT, 
          typename calcT,
          size_t   work_group_size>
__global__ void input_transform_collapsed_ic_x_tile
                            (fp16*     __restrict__ image, 
                             ImgShape               is,  
                             void*     __restrict__ V_, 
                             VShape                 vs, 
                             int                    simdDimSize, 
                             TileShape              ts, 
                             uint64_t               padding_h, 
                             uint64_t               padding_w)
{
  auto V = reinterpret_cast<inoutT*>(V_);
  __shared__ calcT tmp[work_group_size][TILE_IN_H][TILE_IN_W];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int itx = threadIdx.x;
  calcT z0, z1, z2, z3, z4, z5, z6;

  const uint64_t ic   = idx / vs.numTileTotal;
  const uint64_t tile = idx % vs.numTileTotal;
  const TileIndex ti = getTileIndex(tile, ts);
  const uint64_t  b  = ti.b, th = ti.th, tw = ti.tw;
  typedef fp16 (*image_tensor_t) [is.ic][is.h][is.w];
  image_tensor_t image_tensor = (image_tensor_t)image;

  for (int w = 0; w < TILE_IN_W; ++w) {

    z0 = z1 = z2 = z3 = z4 = z5 = (calcT)0.0;
    if(th * TILE_OUT_H + 0 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
      z6 = (calcT)image_tensor[b][ic][th * TILE_OUT_H + 0 - padding_h][tw * TILE_OUT_W + w - padding_w];
      z0 = ((calcT)4.0f) * z6;
    }

    if(th * TILE_OUT_H + 1 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
      z6 = (calcT)image_tensor[b][ic][th * TILE_OUT_H + 1 - padding_h][tw * TILE_OUT_W + w - padding_w];
      z1 = ((calcT)-4.0f) * z6;
      z2 = ((calcT) 4.0f) * z6;
      z3 = ((calcT)-2.0f) * z6;
      z4 = ((calcT) 2.0f) * z6;
      z5 = ((calcT) 4.0f) * z6;
    }

    if(th * TILE_OUT_H + 2 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
      z6 =  (calcT)image_tensor[b][ic][th * TILE_OUT_H + 2 - padding_h][tw * TILE_OUT_W + w - padding_w];
      z0 += ((calcT)-5.0f) * z6;
      z1 += ((calcT)-4.0f) * z6;
      z2 += ((calcT)-4.0f) * z6;
      z3 += -z6;
      z4 += -z6;
    }

    if(th * TILE_OUT_H + 3 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
      z6 =  (calcT)image_tensor[b][ic][th * TILE_OUT_H + 3 - padding_h][tw * TILE_OUT_W + w - padding_w];
      z1 +=  z6;
      z2 += -z6;
      z3 += ((calcT) 2.0f) * z6;
      z4 += ((calcT)-2.0f) * z6;
      z5 += ((calcT)-5.0f) * z6;
    }

    if(th * TILE_OUT_H + 4 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
      z6 =  (calcT)image_tensor[b][ic][th * TILE_OUT_H + 4 - padding_h][tw * TILE_OUT_W + w - padding_w];
      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;
    }

    if(th * TILE_OUT_H + 5 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
      z6 =  (calcT)image_tensor[b][ic][th * TILE_OUT_H + 5 - padding_h][tw * TILE_OUT_W + w - padding_w];
      z5 += z6;
    }

    tmp[itx][0][w] = z0;
    tmp[itx][1][w] = z1;
    tmp[itx][2][w] = z2;
    tmp[itx][3][w] = z3;
    tmp[itx][4][w] = z4;
    tmp[itx][5][w] = z5;

  }

  for (int h = 0; h < TILE_IN_H; ++h) {
    z6 = tmp[itx][h][0];

    z0 = ((calcT)4.0f) * z6;

    z6 = tmp[itx][h][1];

    z1 = ((calcT)-4.0f) * z6;
    z2 = ((calcT) 4.0f) * z6;
    z3 = ((calcT)-2.0f) * z6;
    z4 = ((calcT) 2.0f) * z6;
    z5 = ((calcT) 4.0f) * z6;

    z6 = tmp[itx][h][2];

    z0 += ((calcT)-5.0f) * z6;
    z1 += ((calcT)-4.0f) * z6;
    z2 += ((calcT)-4.0f) * z6;
    z3 += -z6;
    z4 += -z6;

    z6 = tmp[itx][h][3];

    z1 +=  z6;
    z2 += -z6;
    z3 += ((calcT) 2.0f) * z6;
    z4 += ((calcT)-2.0f) * z6;
    z5 += ((calcT)-5.0f) * z6;

    z6 = tmp[itx][h][4];

    z0 += z6;
    z1 += z6;
    z2 += z6;
    z3 += z6;
    z4 += z6;

    z6 = tmp[itx][h][5];

    z5 += z6;

    V[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
    V[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
    V[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
    V[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
    V[h * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx] = z4;
    V[h * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx] = z5;
  }

}

template <typename inoutT, 
          typename calcT , 
          int      IC_BLK,
          int      OC_BLK>
__global__ __launch_bounds__(OC_BLK * IC_BLK)  
void filter_transform_2_dims_transpose( fp16*     __restrict__ filter_, 
                                        void*     __restrict__ U_,
                                        UShape                 us ) 
{
  static_assert(OC_BLK == IC_BLK);
  auto filter = reinterpret_cast<inoutT(*)[us.ic][FLT_H][FLT_W]>(filter_);
  auto U = reinterpret_cast<inoutT(*)[TILE_IN_W][us.ic][us.oc]>(U_);
  __shared__ calcT tmp[IC_BLK][OC_BLK][TILE_IN_H][TILE_IN_W];
  const int oc_local = threadIdx.y;  // thread's index on OC
  const int ic_local = threadIdx.x;  // thread's index on IC
  const int oc_blk   = blockIdx.y;   // workgroup's index on OC
  const int ic_blk   = blockIdx.x;   // workgroup's index on IC
  calcT z0, z1, z2, z3, z4, z5, z6;

  for(int w = 0; w < FLT_W; ++w)
    for(int h = 0; h < FLT_H; ++h)
      if(oc_blk * OC_BLK + oc_local < us.oc && ic_blk * IC_BLK + ic_local < us.ic)
        tmp[ic_local][oc_local][h][w] = filter[oc_blk * OC_BLK + oc_local][ic_blk * IC_BLK + ic_local][h][w];
  
  __syncthreads();
  
  for (int w = 0; w < FLT_HW; ++w) {
    z6 = tmp[ic_local][oc_local][0][w];

    z0 = ((calcT)( 1.0f / 4.0f )) * z6;
    z1 = ((calcT)(-1.0f / 6.0f )) * z6;
    z2 = ((calcT)(-1.0f / 6.0f )) * z6;
    z3 = ((calcT)( 1.0f / 24.0f)) * z6;
    z4 = ((calcT)( 1.0f / 24.0f)) * z6;

    z6 = tmp[ic_local][oc_local][1][w];

    z1 += ((calcT)(-1.0f / 6.0f )) * z6;
    z2 += ((calcT)( 1.0f / 6.0f )) * z6;
    z3 += ((calcT)( 1.0f / 12.0f)) * z6;
    z4 += ((calcT)(-1.0f / 12.0f)) * z6;

    z6 = tmp[ic_local][oc_local][2][w];

    z1 += ((calcT)(-1.0f / 6.0f)) * z6;
    z2 += ((calcT)(-1.0f / 6.0f)) * z6;
    z3 += ((calcT)( 1.0f / 6.0f)) * z6;
    z4 += ((calcT)( 1.0f / 6.0f)) * z6;
    z5 = z6;

    tmp[ic_local][oc_local][0][w] = z0;
    tmp[ic_local][oc_local][1][w] = z1;
    tmp[ic_local][oc_local][2][w] = z2;
    tmp[ic_local][oc_local][3][w] = z3;
    tmp[ic_local][oc_local][4][w] = z4;
    tmp[ic_local][oc_local][5][w] = z5;
  }

  for (int h = 0; h < TILE_IN_H; ++h) {
    z6 = tmp[ic_local][oc_local][h][0];

    z0 = ((calcT)( 1.0f / 4.0f )) * z6;
    z1 = ((calcT)(-1.0f / 6.0f )) * z6;
    z2 = ((calcT)(-1.0f / 6.0f )) * z6;
    z3 = ((calcT)( 1.0f / 24.0f)) * z6;
    z4 = ((calcT)( 1.0f / 24.0f)) * z6;

    z6 = tmp[ic_local][oc_local][h][1];

    z1 += ((calcT)(-1.0f / 6.0f )) * z6;
    z2 += ((calcT)( 1.0f / 6.0f )) * z6;
    z3 += ((calcT)( 1.0f / 12.0f)) * z6;
    z4 += ((calcT)(-1.0f / 12.0f)) * z6;

    z6 = tmp[ic_local][oc_local][h][2];

    z1 += ((calcT)(-1.0f / 6.0f)) * z6;
    z2 += ((calcT)(-1.0f / 6.0f)) * z6;
    z3 += ((calcT)( 1.0f / 6.0f)) * z6;
    z4 += ((calcT)( 1.0f / 6.0f)) * z6;
    z5 = z6;

    if(oc_blk * OC_BLK + oc_local < us.oc && ic_blk * IC_BLK + ic_local < us.ic) {
      U[h][0][ic_blk * IC_BLK + ic_local][oc_blk * OC_BLK + oc_local] = z0;
      U[h][1][ic_blk * IC_BLK + ic_local][oc_blk * OC_BLK + oc_local] = z1;
      U[h][2][ic_blk * IC_BLK + ic_local][oc_blk * OC_BLK + oc_local] = z2;
      U[h][3][ic_blk * IC_BLK + ic_local][oc_blk * OC_BLK + oc_local] = z3;
      U[h][4][ic_blk * IC_BLK + ic_local][oc_blk * OC_BLK + oc_local] = z4;
      U[h][5][ic_blk * IC_BLK + ic_local][oc_blk * OC_BLK + oc_local] = z5;
    }
  }

  __syncthreads();

}

template <typename inoutT, 
          typename calcT, 
          int work_group_size>
__global__ void filter_transform_no_transpose(fp16*     __restrict__ filter, 
                                              void*     __restrict__ U_,
                                              UShape                 us, 
                                              int                    simdDimSize) 
{
  auto U = reinterpret_cast<inoutT*>(U_);
  __shared__ calcT tmp[work_group_size][TILE_IN_H][TILE_IN_W] ;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= simdDimSize) 
    return;
  
  int itx = threadIdx.x;
  calcT z0, z1, z2, z3, z4, z5, z6;
  for (int i = 0; i < FLT_HW; ++i) {
    z6 = filter[idx * FLT_H * FLT_W + 0 * FLT_W  + i];

    z0 = ((calcT)( 1.0f / 4.0f )) * z6;
    z1 = ((calcT)(-1.0f / 6.0f )) * z6;
    z2 = ((calcT)(-1.0f / 6.0f )) * z6;
    z3 = ((calcT)( 1.0f / 24.0f)) * z6;
    z4 = ((calcT)( 1.0f / 24.0f)) * z6;

    z6 = filter[idx * FLT_H * FLT_W + 1 * FLT_W  + i];

    z1 += ((calcT)(-1.0f / 6.0f )) * z6;
    z2 += ((calcT)( 1.0f / 6.0f )) * z6;
    z3 += ((calcT)( 1.0f / 12.0f)) * z6;
    z4 += ((calcT)(-1.0f / 12.0f)) * z6;

    z6 = filter[idx * FLT_H * FLT_W + 2 * FLT_W  + i];

    z1 += ((calcT)(-1.0f / 6.0f)) * z6;
    z2 += ((calcT)(-1.0f / 6.0f)) * z6;
    z3 += ((calcT)( 1.0f / 6.0f)) * z6;
    z4 += ((calcT)( 1.0f / 6.0f)) * z6;
    z5 = z6;

    tmp[itx][0][i] = z0;
    tmp[itx][1][i] = z1;
    tmp[itx][2][i] = z2;
    tmp[itx][3][i] = z3;
    tmp[itx][4][i] = z4;
    tmp[itx][5][i] = z5;
  }

  for (int i = 0; i < TILE_IN_H; ++i) {
    z6 = tmp[itx][i][0];

    z0 = ((calcT)( 1.0f / 4.0f )) * z6;
    z1 = ((calcT)(-1.0f / 6.0f )) * z6;
    z2 = ((calcT)(-1.0f / 6.0f )) * z6;
    z3 = ((calcT)( 1.0f / 24.0f)) * z6;
    z4 = ((calcT)( 1.0f / 24.0f)) * z6;

    z6 = tmp[itx][i][1];

    z1 += ((calcT)(-1.0f / 6.0f )) * z6;
    z2 += ((calcT)( 1.0f / 6.0f )) * z6;
    z3 += ((calcT)( 1.0f / 12.0f)) * z6;
    z4 += ((calcT)(-1.0f / 12.0f)) * z6;

    z6 = tmp[itx][i][2];

    z1 += ((calcT)(-1.0f / 6.0f)) * z6;
    z2 += ((calcT)(-1.0f / 6.0f)) * z6;
    z3 += ((calcT)( 1.0f / 6.0f)) * z6;
    z4 += ((calcT)( 1.0f / 6.0f)) * z6;
    z5 = z6;

    U[i * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
    U[i * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
    U[i * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
    U[i * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
    U[i * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx] = z4;
    U[i * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx] = z5;
  }
}


template <typename inoutT, 
          typename calcT,
          int work_group_size>
__global__ void output_transform(void* __restrict__    M_, 
                                 int                   simdDimSize,
                                 fp16*    __restrict__ out, 
                                 OutShape              os,  
                                 TileShape             ts) 
{
  auto M = reinterpret_cast<inoutT*>(M_);
  __shared__ calcT tmp[work_group_size][TILE_OUT_H][TILE_IN_W];
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  int itx = threadIdx.x;

  calcT z0, z1, z2, z3, z4;
  for (int w = 0; w < TILE_IN_W; ++w) {
    z4 = M[0 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

    z0 = z4;

    z4 = M[1 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

    z0 = z0 + z4;
    z1 = z4;
    z2 = z4;
    z3 = z4;
    
    z4 = M[2 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

    z0 +=  z4;
    z1 += -z4;
    z2 +=  z4;
    z3 += -z4;

    z4 = M[3 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

    z0 += z4;
    z1 += ((calcT)2.0f) * z4;
    z2 += ((calcT)4.0f) * z4;
    z3 += ((calcT)8.0f) * z4;

    z4 = M[4 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

    z0 += z4;
    z1 += ((calcT)-2.0f) * z4;
    z2 += ((calcT) 4.0f) * z4;
    z3 += ((calcT)-8.0f) * z4;

    z4 = M[5 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

    z3 += z4;

    tmp[itx][0][w] = z0;
    tmp[itx][1][w] = z1;
    tmp[itx][2][w] = z2;
    tmp[itx][3][w] = z3;
  }

  for (int h = 0; h < TILE_OUT_HW; ++h) {
    z4 = tmp[itx][h][0];

    z0 = z4;


    z4 = tmp[itx][h][1];

    z0 += z4;
    z1 = z4;
    z2 = z4;
    z3 = z4;

    z4 = tmp[itx][h][2];
    
    z0 += z4;
    z1 += -z4;
    z2 += z4;
    z3 += -z4;

    z4 = tmp[itx][h][3];

    z0 += z4;
    z1 += ((calcT)2.0f) * z4;
    z2 += ((calcT)4.0f) * z4;
    z3 += ((calcT)8.0f) * z4;


    z4 = tmp[itx][h][4];


    z0 += z4;
    z1 += ((calcT)-2.0f) * z4;
    z2 += ((calcT) 4.0f) * z4;
    z3 += ((calcT)-8.0f) * z4;


    z4 = tmp[itx][h][5];

    z3 += z4;


    int k = idx / ts.numTileTotal;
    int b = idx % ts.numTileTotal;
    TileIndex ti = getTileIndex(b, ts);
    int n = ti.b, tw = ti.tw, th = ti.th;

    if(th * 4 + h < os.h && tw * 4 + 0 < os.w)
      out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * 4 + h) * os.w + (tw * 4 + 0)] = (fp16) z0;
    if(th * 4 + h < os.h && tw * 4 + 1 < os.w)
      out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * 4 + h) * os.w + (tw * 4 + 1)] = (fp16) z1;
    if(th * 4 + h < os.h && tw * 4 + 2 < os.w)
      out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * 4 + h) * os.w + (tw * 4 + 2)] = (fp16) z2;
    if(th * 4 + h < os.h && tw * 4 + 3 < os.w)
      out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * 4 + h) * os.w + (tw * 4 + 3)] = (fp16) z3;
  }
}


extern "C" void winconv_4x3(const void* param_ptr) {
    const mykernelParamType* param = (const mykernelParamType*)param_ptr;
    fp16* filter_d = param->pweight;
    fp16* image_d  = param->pin;
    fp16* out_d    = param->pout;
    void* U_d = reinterpret_cast<void*> (param->U_d);
    void* V_d = reinterpret_cast<void*> (param->V_d);
    void* M_d = reinterpret_cast<void*> (param->M_d);
    void* Y_d = reinterpret_cast<void*> (param->Y_d);
    uint64_t padding_h = param->p;
    uint64_t padding_w = param->q;

    ImgShape  is = {param->n, param->c, param->h, param->w};
    FltShape  fs = {param->k, param->c, param->r, param->s};
    OutShape  os = getOutShape(is, fs, padding_h, padding_w);
    TileShape ts = getTileShape(os);
    UShape    us = getUShape(fs);
    VShape    vs = getVShape(is, ts);
  
    const size_t work_group_size = 64;
    input_transform_collapsed_ic_x_tile <fp16, fp16, work_group_size> <<< vs.ic * vs.numTileTotal / work_group_size, work_group_size >>> (image_d, is, V_d, vs, vs.ic * vs.numTileTotal, ts, padding_h, padding_w);
    HIP_CHECK_KERNEL("Kernel panic!!!");
    const int blk_oc = 16;
    const int blk_ic = 16;
    dim3 block_dim(blk_ic, blk_oc);
    dim3 grid_dim(DIV_UP(us.ic , blk_ic), DIV_UP(us.oc, blk_oc));
    filter_transform_no_transpose <fp16, fp16, work_group_size> <<<DIV_UP(us.ic * us.oc, work_group_size), work_group_size>>> (filter_d, U_d, us, us.ic * us.oc);
    HIP_CHECK_KERNEL("Kernel panic!!!");    
    const float alpha = 1.0, beta = 0.0;
    hep_sgemm<fp16, float>( vs.numTileTotal, us.oc, us.ic,
                                alpha,
                                (void*)(V_d),
                                vs.numTileTotal,  // if you change V's layout, you need to change this
                                (void*)(U_d),
                                us.ic,            // if you change U's layout, you need to change this
                                beta,
                                (void*)(M_d),
                                vs.numTileTotal,  // if you change M's layout, you need to change this
                                TILE_IN_H * TILE_IN_W,
                                hipStreamDefault );

    output_transform <fp16, fp16, work_group_size> <<< us.oc * vs.numTileTotal / work_group_size, work_group_size >>>(M_d, us.oc * vs.numTileTotal, out_d, os, ts);
    HIP_CHECK_KERNEL("Kernel panic!!!");    

}

/*选手需要返回自定义kernel入参结构体的size*/
int getParamsize(__in__ problem_t* problem, __out__ int* paramSize)
{
    *paramSize = sizeof(mykernelParamType);

    return 0;
}

/*选手需要返回自己优化的kernel的grid信息与kernel函数的指针*/
int getkernelInfo(__in__ problem_t* problem, __out__  kernelInfo_t* kernelInfo, __in_out__ void* param)
{
    mykernelParamType* pArgs = (mykernelParamType*)param;

    unsigned int n = problem->n;
    unsigned int c = problem->c;
    unsigned int h = problem->h;
    unsigned int w = problem->w;
    unsigned int k = problem->k;
    unsigned int r = problem->r;
    unsigned int s = problem->s;
    unsigned int u = problem->u;
    unsigned int v = problem->v;
    unsigned int p = problem->p;
    unsigned int q = problem->q;


    unsigned int outh = (h - r + 2*p)/u + 1;
    unsigned int outw = (w - s + 2*q)/v + 1;

    kernelInfo->blockx   = (outh*outw + 15)/16;                    //blockx  number
    kernelInfo->blocky   = (k+15)/16;                    //blocky  number
    kernelInfo->blockz   = n;                    //blockz  number
    kernelInfo->threadx  = 16;                   //threadx number per block
    kernelInfo->thready  = 16;                   //thready number per block
    kernelInfo->threadz  = 1;                   //threadz number per block
    kernelInfo->dynmicLdsSize = 0;
    kernelInfo->kernelPtr= (void*)winconv_4x3;                 //kernel ptr

    pArgs->pin = problem->in;
    pArgs->pweight = problem->weight;
    pArgs->pout = problem->out;
    pArgs->n = n;                              //batch szie              default value 1
    pArgs->c = c;                              //channel number          default value 32
    pArgs->h = h;                              //数据高                  default value 32
    pArgs->w = w;                              //数据宽                  default value 32
    pArgs->k = k;                              //卷积核数量              default value 32
    pArgs->r = r;                              //卷积核高                default value 1
    pArgs->s = s;                              //卷积核宽                default value 1
    pArgs->u = u;                              //卷积在高方向上的步长     default value 1
    pArgs->v = v;                              //卷积在宽方向上的步长     default value 1
    pArgs->p = p;                              //卷积在高方向上的补边     default value 0
    pArgs->q = q;                              //卷积在宽方向上的补边     default value 0
    pArgs->Oh = outh;
    pArgs->Ow = outw;

    ImgShape  is = {n, c, h, w};
    FltShape  fs = {k, c, FLT_H, FLT_W};
    OutShape  os = {n, k, outh, outw};
    TileShape ts = getTileShape(os);
    UShape    us = getUShape(fs);
    VShape    vs = getVShape(is, ts);
    unsigned int image_size = n * c * h * w;
    unsigned int filter_size = k * c * r * s;
    unsigned int out_size = n * k * outh * outw;

    unsigned int U_size = TILE_IN_H * TILE_IN_W * k * c;
    unsigned int V_size = TILE_IN_H * TILE_IN_W * vs.numTileTotal * c;
    unsigned int M_size = TILE_IN_H  * TILE_IN_W  * k * vs.numTileTotal;
#ifdef PRINT_TENSOR_SIZE
    std::cout << "n = " << n << ", c = " << c << ", h = " << h << ", w = " << w << ", k = " << k << ", r = " << r << ", s = " << s << ", u = " << u << ", v = " << v << ", p = " << p << ", q = " << q << std::endl;
    std::cout << "image_size: " << image_size / 1024.0 << " KiB" << std::endl;
    std::cout << "filter_size: " << filter_size / 1024.0 << " KiB" << std::endl;
    std::cout << "out_size: " << out_size / 1024.0 << " KiB" << std::endl;
    std::cout << "U_size: " << U_size / 1024.0 << " KiB" << std::endl;
    std::cout << "V_size: " << V_size / 1024.0 << " KiB" << std::endl;
    std::cout << "M_size: " << M_size / 1024.0 << " KiB" << std::endl;
    std::exit(0);
#endif
    unsigned int malloc_size = sizeof(fp16) * (
          U_size
        + V_size
        + M_size
    );
    
    hipMalloc(&pArgs->U_d, malloc_size);
    
    pArgs->V_d = pArgs->U_d + U_size;
    pArgs->M_d = pArgs->V_d + V_size;
    pArgs->Y_d = pArgs->M_d + M_size;
    return 0;
}

extern "C" void free_extra_vram(const void* param_ptr) {
    mykernelParamType* pArgs = (mykernelParamType*)param_ptr;
    hipFree(pArgs->U_d);
}
