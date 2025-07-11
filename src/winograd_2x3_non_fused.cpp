#define FLT_H      3L
#define FLT_W      3L
#define TILE_IN_H  4L
#define TILE_IN_W  4L
#define TILE_OUT_H 2L
#define TILE_OUT_W 2L

#include "common.h"
#include "error.h"
#include "hep_sgemm.h"
                      

template <typename inoutT, 
          typename calcT,
          size_t   work_group_size>
__global__ static void input_transform_collapsed_wino2x3
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

  if (idx >= simdDimSize) 
    return;

  int itx = threadIdx.x;
  calcT z0, z1, z2, z3, z6;

  const uint64_t ic   = idx / vs.numTileTotal;
  const uint64_t tile = idx % vs.numTileTotal;
  const TileIndex ti = getTileIndex(tile, ts);
  const uint64_t  b  = ti.b, th = ti.th, tw = ti.tw;
  typedef fp16 (*image_tensor_t) [is.ic][is.h][is.w];
  image_tensor_t image_tensor = (image_tensor_t)image;

  for (int w = 0; w < TILE_IN_W; ++w) {

    z0 = z1 = z2 = z3 = (calcT)0.0;
    if(th * TILE_OUT_H + 0 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
      z6 = (calcT)image_tensor[b][ic][th * TILE_OUT_H + 0 - padding_h][tw * TILE_OUT_W + w - padding_w];
      z0 = ((calcT) 1.0f) * z6;
    }

    if(th * TILE_OUT_H + 1 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
      z6 = (calcT)image_tensor[b][ic][th * TILE_OUT_H + 1 - padding_h][tw * TILE_OUT_W + w - padding_w];
      z1 = ((calcT) 1.0f) * z6;
      z2 = ((calcT)-1.0f) * z6;
      z3 = ((calcT) 1.0f) * z6;
    }

    if(th * TILE_OUT_H + 2 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
      z6 =  (calcT)image_tensor[b][ic][th * TILE_OUT_H + 2 - padding_h][tw * TILE_OUT_W + w - padding_w];
      z0 += ((calcT)-1.0f) * z6;
      z1 += ((calcT) 1.0f) * z6;
      z2 += ((calcT) 1.0f) * z6;
    }

    if(th * TILE_OUT_H + 3 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
      z6 =  (calcT)image_tensor[b][ic][th * TILE_OUT_H + 3 - padding_h][tw * TILE_OUT_W + w - padding_w];
      z3 += ((calcT)-1.0f) * z6;
    }

    tmp[itx][0][w] = z0;
    tmp[itx][1][w] = z1;
    tmp[itx][2][w] = z2;
    tmp[itx][3][w] = z3;
  }

  for (int h = 0; h < TILE_IN_H; ++h) {
    z6 = tmp[itx][h][0];
    z0 = ((calcT) 1.0f) * z6;

    z6 = tmp[itx][h][1];
    z1 = ((calcT) 1.0f) * z6;
    z2 = ((calcT)-1.0f) * z6;
    z3 = ((calcT) 1.0f) * z6;

    z6 = tmp[itx][h][2];
    z0 += ((calcT)-1.0f) * z6;
    z1 += ((calcT) 1.0f) * z6;
    z2 += ((calcT) 1.0f) * z6;

    z6 = tmp[itx][h][3];
    z3 += ((calcT)-1.0f) * z6;

    V[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
    V[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
    V[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
    V[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
  }

}


template <typename inoutT, 
          typename calcT, 
          int work_group_size>
__global__ static void filter_transform_wino2x3
                                        ( fp16*     __restrict__ filter, 
                                          void*     __restrict__ U_,
                                          UShape                 us, 
                                          int                    simdDimSize ) 
{
  auto U = reinterpret_cast<inoutT*>(U_);
  __shared__ calcT tmp[work_group_size][TILE_IN_H][TILE_IN_W] ;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx >= simdDimSize) 
    return;
  
  int itx = threadIdx.x;
  calcT z0, z1, z2, z3, z6;
  for (int i = 0; i < FLT_W; ++i) {
    z6 = filter[idx * FLT_H * FLT_W + 0 * FLT_W  + i];
    z0 = ((calcT)( 1.0f )) * z6;
    z1 = ((calcT)( 1.0f / 2.0f )) * z6;
    z2 = ((calcT)( 1.0f / 2.0f )) * z6;

    z6 = filter[idx * FLT_H * FLT_W + 1 * FLT_W  + i];
    z1 += ((calcT)( 1.0f / 2.0f )) * z6;
    z2 += ((calcT)(-1.0f / 2.0f )) * z6;

    z6 = filter[idx * FLT_H * FLT_W + 2 * FLT_W  + i];
    z1 += ((calcT)( 1.0f / 2.0f )) * z6;
    z2 += ((calcT)( 1.0f / 2.0f )) * z6;
    z3 =  ((calcT)( 1.0f )) * z6;

    tmp[itx][0][i] = z0;
    tmp[itx][1][i] = z1;
    tmp[itx][2][i] = z2;
    tmp[itx][3][i] = z3;
  }

  for (int i = 0; i < TILE_IN_H; ++i) {
    z6 = tmp[itx][i][0];
    z0 = ((calcT)( 1.0f )) * z6;
    z1 = ((calcT)( 1.0f / 2.0f )) * z6;
    z2 = ((calcT)( 1.0f / 2.0f )) * z6;
    
    z6 = tmp[itx][i][1];
    z1 += ((calcT)( 1.0f / 2.0f )) * z6;
    z2 += ((calcT)(-1.0f / 2.0f )) * z6;

    z6 = tmp[itx][i][2];
    z1 += ((calcT)( 1.0f / 2.0f )) * z6;
    z2 += ((calcT)( 1.0f / 2.0f )) * z6;
    z3 =  ((calcT)( 1.0f )) * z6;

    U[i * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
    U[i * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
    U[i * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
    U[i * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
  }
}


template <typename inoutT, 
          typename calcT,
          int work_group_size>
__global__ static void output_transform_wino2x3
                            (void*     __restrict__ M_, 
                              int                   simdDimSize,
                              fp16*    __restrict__ out, 
                              OutShape              os,  
                              TileShape             ts) 
{
  auto M = reinterpret_cast<inoutT*>(M_);
  __shared__ calcT tmp[work_group_size][TILE_OUT_H][TILE_IN_W];
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  if (idx >= simdDimSize) 
    return;

  int itx = threadIdx.x;

  calcT z0, z1,  z4;
  for (int w = 0; w < TILE_IN_W; ++w) {
    z4 = M[0 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];
    z0 =  z4;

    z4 = M[1 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];
    z0 += z4;
    z1 =  z4;
    
    z4 = M[2 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];
    z0 +=  z4;
    z1 += -z4;

    z4 = M[3 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];
    z1 += -z4;

    tmp[itx][0][w] = z0;
    tmp[itx][1][w] = z1;
  }

  for (int h = 0; h < TILE_OUT_H; ++h) {
    z4 = tmp[itx][h][0];
    z0 =  z4;

    z4 = tmp[itx][h][1];
    z0 += z4;
    z1 =  z4;

    z4 = tmp[itx][h][2];
    z0 +=  z4;
    z1 += -z4;

    z4 = tmp[itx][h][3];
    z1 += -z4;

    int k = idx / ts.numTileTotal;
    int b = idx % ts.numTileTotal;
    TileIndex ti = getTileIndex(b, ts);
    int n = ti.b, tw = ti.tw, th = ti.th;

    if(th * TILE_OUT_H + h < os.h && tw * TILE_OUT_W + 0 < os.w)
      out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * TILE_OUT_H + h) * os.w + (tw * TILE_OUT_W + 0)] = (fp16) z0;
    if(th * TILE_OUT_H + h < os.h && tw * TILE_OUT_W + 1 < os.w)
      out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * TILE_OUT_H + h) * os.w + (tw * TILE_OUT_W + 1)] = (fp16) z1;
  }
}

template <typename inoutT, 
          typename calcT, 
          int work_group_size>
__global__ static void input_transform_filter_transform_winograd2x3
                                   (fp16*     __restrict__ filter, 
                                    void*     __restrict__ U_,
                                    UShape                 us, 
                                    int                    simdDimSize1,
                                    fp16*     __restrict__ image, 
                                    ImgShape               is,  
                                    void*     __restrict__ V_, 
                                    VShape                 vs, 
                                    int                    simdDimSize2, 
                                    TileShape              ts, 
                                    uint64_t               padding_h, 
                                    uint64_t               padding_w, 
                                    size_t                 div)
{
  int blx = blockIdx.x;
  if(blx < div) 
  {
    auto U = reinterpret_cast<inoutT*>(U_);
    __shared__ calcT tmp[work_group_size][TILE_IN_H][TILE_IN_W] ;
    int idx = blx * blockDim.x + threadIdx.x;
    
    if (idx >= simdDimSize1) 
      return;
    
    int itx = threadIdx.x;
    calcT z0, z1, z2, z3, z6;
    for (int i = 0; i < FLT_W; ++i) {
      z6 = filter[idx * FLT_H * FLT_W + 0 * FLT_W  + i];
      z0 = ((calcT)( 1.0f )) * z6;
      z1 = ((calcT)( 1.0f / 2.0f )) * z6;
      z2 = ((calcT)( 1.0f / 2.0f )) * z6;

      z6 = filter[idx * FLT_H * FLT_W + 1 * FLT_W  + i];
      z1 += ((calcT)( 1.0f / 2.0f )) * z6;
      z2 += ((calcT)(-1.0f / 2.0f )) * z6;

      z6 = filter[idx * FLT_H * FLT_W + 2 * FLT_W  + i];
      z1 += ((calcT)( 1.0f / 2.0f )) * z6;
      z2 += ((calcT)( 1.0f / 2.0f )) * z6;
      z3 =  ((calcT)( 1.0f )) * z6;

      tmp[itx][0][i] = z0;
      tmp[itx][1][i] = z1;
      tmp[itx][2][i] = z2;
      tmp[itx][3][i] = z3;
    }

    for (int i = 0; i < TILE_IN_H; ++i) {
      z6 = tmp[itx][i][0];
      z0 = ((calcT)( 1.0f )) * z6;
      z1 = ((calcT)( 1.0f / 2.0f )) * z6;
      z2 = ((calcT)( 1.0f / 2.0f )) * z6;
      
      z6 = tmp[itx][i][1];
      z1 += ((calcT)( 1.0f / 2.0f )) * z6;
      z2 += ((calcT)(-1.0f / 2.0f )) * z6;

      z6 = tmp[itx][i][2];
      z1 += ((calcT)( 1.0f / 2.0f )) * z6;
      z2 += ((calcT)( 1.0f / 2.0f )) * z6;
      z3 =  ((calcT)( 1.0f )) * z6;

      U[i * TILE_IN_W * simdDimSize1 + 0 * simdDimSize1 + idx] = z0;
      U[i * TILE_IN_W * simdDimSize1 + 1 * simdDimSize1 + idx] = z1;
      U[i * TILE_IN_W * simdDimSize1 + 2 * simdDimSize1 + idx] = z2;
      U[i * TILE_IN_W * simdDimSize1 + 3 * simdDimSize1 + idx] = z3;
    }
  }
  else 
  {
    blx -= div;
    auto V = reinterpret_cast<inoutT*>(V_);
    __shared__ calcT tmp[work_group_size][TILE_IN_H][TILE_IN_W];
    int idx = blx * blockDim.x + threadIdx.x;
    if (idx >= simdDimSize2) 
      return;

    int itx = threadIdx.x;
    calcT z0, z1, z2, z3, z6;

    const uint64_t ic   = idx / vs.numTileTotal;
    const uint64_t tile = idx % vs.numTileTotal;
    const TileIndex ti = getTileIndex(tile, ts);
    const uint64_t  b  = ti.b, th = ti.th, tw = ti.tw;
    typedef fp16 (*image_tensor_t) [is.ic][is.h][is.w];
    image_tensor_t image_tensor = (image_tensor_t)image;

    for (int w = 0; w < TILE_IN_W; ++w) {

      z0 = z1 = z2 = z3 = (calcT)0.0;
      if(th * TILE_OUT_H + 0 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
        z6 = (calcT)image_tensor[b][ic][th * TILE_OUT_H + 0 - padding_h][tw * TILE_OUT_W + w - padding_w];
        z0 = ((calcT) 1.0f) * z6;
      }

      if(th * TILE_OUT_H + 1 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
        z6 = (calcT)image_tensor[b][ic][th * TILE_OUT_H + 1 - padding_h][tw * TILE_OUT_W + w - padding_w];
        z1 = ((calcT) 1.0f) * z6;
        z2 = ((calcT)-1.0f) * z6;
        z3 = ((calcT) 1.0f) * z6;
      }

      if(th * TILE_OUT_H + 2 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
        z6 =  (calcT)image_tensor[b][ic][th * TILE_OUT_H + 2 - padding_h][tw * TILE_OUT_W + w - padding_w];
        z0 += ((calcT)-1.0f) * z6;
        z1 += ((calcT) 1.0f) * z6;
        z2 += ((calcT) 1.0f) * z6;
      }

      if(th * TILE_OUT_H + 3 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
        z6 =  (calcT)image_tensor[b][ic][th * TILE_OUT_H + 3 - padding_h][tw * TILE_OUT_W + w - padding_w];
        z3 += ((calcT)-1.0f) * z6;
      }

      tmp[itx][0][w] = z0;
      tmp[itx][1][w] = z1;
      tmp[itx][2][w] = z2;
      tmp[itx][3][w] = z3;
    }

    for (int h = 0; h < TILE_IN_H; ++h) {
      z6 = tmp[itx][h][0];
      z0 = ((calcT) 1.0f) * z6;

      z6 = tmp[itx][h][1];
      z1 = ((calcT) 1.0f) * z6;
      z2 = ((calcT)-1.0f) * z6;
      z3 = ((calcT) 1.0f) * z6;

      z6 = tmp[itx][h][2];
      z0 += ((calcT)-1.0f) * z6;
      z1 += ((calcT) 1.0f) * z6;
      z2 += ((calcT) 1.0f) * z6;

      z6 = tmp[itx][h][3];
      z3 += ((calcT)-1.0f) * z6;

      V[h * TILE_IN_W * simdDimSize2 + 0 * simdDimSize2 + idx] = z0;
      V[h * TILE_IN_W * simdDimSize2 + 1 * simdDimSize2 + idx] = z1;
      V[h * TILE_IN_W * simdDimSize2 + 2 * simdDimSize2 + idx] = z2;
      V[h * TILE_IN_W * simdDimSize2 + 3 * simdDimSize2 + idx] = z3;
    }
  }
 
}


void winograd_2x3_none_fused(const void* param_ptr) {
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
  
    const size_t work_group_size = HEP_WARP_SIZE;
    input_transform_filter_transform_winograd2x3 <fp16, fp16, work_group_size> <<< DIV_UP(us.ic * us.oc, work_group_size) + DIV_UP(vs.ic * vs.numTileTotal, work_group_size), work_group_size >>> (filter_d, U_d, us, us.ic * us.oc, image_d, is, V_d, vs, vs.ic * vs.numTileTotal, ts, padding_h, padding_w, DIV_UP(us.ic * us.oc, work_group_size));   
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

    output_transform_wino2x3 <fp16, fp16, work_group_size> <<< DIV_UP(us.oc * vs.numTileTotal, work_group_size), work_group_size >>>(M_d, us.oc * vs.numTileTotal, out_d, os, ts);
    HIP_CHECK_KERNEL("Kernel panic!!!");    

}
