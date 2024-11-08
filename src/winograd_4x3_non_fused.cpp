#define FLT_H      3L
#define FLT_W      3L
#define TILE_IN_H  6L
#define TILE_IN_W  6L
#define TILE_OUT_H 4L
#define TILE_OUT_W 4L
#include "common.h"
#include "error.h"
#include "hep_sgemm.h"

template <typename inoutT, 
          typename calcT,
          size_t   work_group_size>
__global__ static void input_transform_collapsed_wino4x3
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
          typename calcT,
          size_t   work_group_size,
          size_t   H,
          size_t   W,
          uint64_t padding_h, 
          uint64_t padding_w>
__global__ static void input_transform_every_img_wino4x3
                            (fp16*     __restrict__ image, 
                             ImgShape               is,  
                             void*     __restrict__ V_, 
                             VShape                 vs)
{
  const int vec_len = 8;
  static_assert(padding_h == 1 && padding_w == 1, "padding_h and padding_w must be 1");
  static_assert(!((W % vec_len)), "W must be multiple of 8");
  static_assert(!((H * W) % work_group_size), "H * W must be multiple of work_group_size");
  static_assert(!(H % TILE_OUT_H) && !(W % TILE_OUT_W), "H and W must be multiple of TILE_OUT_H and TILE_OUT_W");
  
  __shared__ calcT tmp[H + 2 * padding_h][W + 2 * padding_w];
  int thx = threadIdx.x;
  int blx = blockIdx.x;
  int bly = blockIdx.y;
  calcT z0, z1, z2, z3, z4, z5, z6;

  const uint64_t ic = bly;
  const uint64_t b  = blx;
  auto V            = reinterpret_cast<inoutT(*)[TILE_IN_W][vs.ic][vs.numTileTotal]>(V_);
  auto image_tensor = reinterpret_cast<inoutT(*)[is.ic][H][W]>(image);
  fp16 frag[TILE_IN_H][TILE_IN_W];

  const int lds_tile_h          = H / TILE_OUT_H;
  const int lds_tile_w          = W / TILE_OUT_W;
  const int lds_tile_total      = lds_tile_h * lds_tile_w;
  static_assert(!(lds_tile_total % work_group_size), "lds_tile_total must be multiple of work_group_size");
  const int tile_num_per_thread = lds_tile_total / work_group_size;
  const int tile_idx_base       = thx * tile_num_per_thread;
  const TileShape lds_ts = {1, lds_tile_total, lds_tile_total, lds_tile_h, lds_tile_h};


  // Initialize padding in shared memory
  for (int i = thx; i < H + 2 * padding_h; i += work_group_size) {
    tmp[i][0] = 0;
    tmp[i][W + 2 * padding_w - 1] = 0;
  }
  for (int i = thx; i < W + 2 * padding_w; i += work_group_size) {
    tmp[0][i] = 0;
    tmp[H + 2 * padding_h - 1][i] = 0;
  }

  __shared__ fp16 result[TILE_IN_H][TILE_IN_W][lds_tile_total];

  const int block_size = H * W / work_group_size;
  const int block_base = block_size * thx;

  for(int w = 0; w < block_size / vec_len; w++) {
    const int block_offset = block_base + w * vec_len;
    const int w_read = block_offset % W;
    const int h_read = block_offset / W;
    *reinterpret_cast<fp16x8*>(&tmp[h_read + padding_h][w_read + padding_w])
                      = *reinterpret_cast<fp16x8*>(&image_tensor[b][ic][h_read][w_read]);
  }

  __syncthreads();

  for(int tidx = tile_idx_base; tidx < tile_idx_base + tile_num_per_thread; ++tidx) {
    const TileIndex lds_ti = getTileIndex(tidx, lds_ts);
    const int h1 = lds_ti.th * TILE_OUT_H;
    const int w1 = lds_ti.tw * TILE_OUT_W;
    for (int w = 0; w < TILE_IN_W; ++w) {
      z0 = z1 = z2 = z3 = z4 = z5 = (calcT)0.0;
      z6 = tmp[h1][w + w1];
      z0 = ((calcT)4.0f) * z6;

      z6 = tmp[h1 + 1][w + w1];
      z1 = ((calcT)-4.0f) * z6;
      z2 = ((calcT) 4.0f) * z6;
      z3 = ((calcT)-2.0f) * z6;
      z4 = ((calcT) 2.0f) * z6;
      z5 = ((calcT) 4.0f) * z6;

      z6 = tmp[h1 + 2][w + w1];
      z0 += ((calcT)-5.0f) * z6;
      z1 += ((calcT)-4.0f) * z6;
      z2 += ((calcT)-4.0f) * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = tmp[h1 + 3][w + w1];
      z1 +=  z6;
      z2 += -z6;
      z3 += ((calcT) 2.0f) * z6;
      z4 += ((calcT)-2.0f) * z6;
      z5 += ((calcT)-5.0f) * z6;

      z6 = tmp[h1 + 4][w + w1];
      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = tmp[h1 + 5][w + w1];
      z5 += z6;

      frag[0][w] = z0;
      frag[1][w] = z1;
      frag[2][w] = z2;
      frag[3][w] = z3;
      frag[4][w] = z4;
      frag[5][w] = z5;
    }

    for (int h = 0; h < TILE_IN_H; ++h) {
      z6 = frag[h][0];

      z0 = ((calcT)4.0f) * z6;

      z6 = frag[h][1];

      z1 = ((calcT)-4.0f) * z6;
      z2 = ((calcT) 4.0f) * z6;
      z3 = ((calcT)-2.0f) * z6;
      z4 = ((calcT) 2.0f) * z6;
      z5 = ((calcT) 4.0f) * z6;

      z6 = frag[h][2];

      z0 += ((calcT)-5.0f) * z6;
      z1 += ((calcT)-4.0f) * z6;
      z2 += ((calcT)-4.0f) * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = frag[h][3];

      z1 +=  z6;
      z2 += -z6;
      z3 += ((calcT) 2.0f) * z6;
      z4 += ((calcT)-2.0f) * z6;
      z5 += ((calcT)-5.0f) * z6;

      z6 = frag[h][4];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = frag[h][5];

      z5 += z6;

      result[h][0][tidx] = z0;
      result[h][1][tidx] = z1;
      result[h][2][tidx] = z2;
      result[h][3][tidx] = z3;
      result[h][4][tidx] = z4;
      result[h][5][tidx] = z5;
    }
  }
  
  // write back to global memory
  for(int h = 0; h < TILE_IN_H; ++h) {
    for(int w = 0; w < TILE_IN_W; ++w) {
      for(int tidx = tile_idx_base; tidx < tile_idx_base + tile_num_per_thread; tidx++) {
        const int global_tile = b * lds_tile_total + tidx;
        V[h][w][ic][global_tile] = result[h][w][tidx];
      }
    }
  }
}

template <typename inoutT, 
          typename calcT, 
          int work_group_size>
__global__ static void filter_transform_wino4x3
                    (fp16*    __restrict__ filter, 
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
  for (int i = 0; i < FLT_W; ++i) {
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
__global__ static void output_transform_wino4x3
                                (void* __restrict__    M_, 
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

  for (int h = 0; h < TILE_OUT_H; ++h) {
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

    if(th * TILE_OUT_H + h < os.h && tw * TILE_OUT_W + 0 < os.w)
      out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * TILE_OUT_H + h) * os.w + (tw * TILE_OUT_W + 0)] = (fp16) z0;
    if(th * TILE_OUT_H + h < os.h && tw * TILE_OUT_W + 1 < os.w)
      out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * TILE_OUT_H + h) * os.w + (tw * TILE_OUT_W + 1)] = (fp16) z1;
    if(th * TILE_OUT_H + h < os.h && tw * TILE_OUT_W + 2 < os.w)
      out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * TILE_OUT_H + h) * os.w + (tw * TILE_OUT_W + 2)] = (fp16) z2;
    if(th * TILE_OUT_H + h < os.h && tw * TILE_OUT_W + 3 < os.w)
      out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * TILE_OUT_H + h) * os.w + (tw * TILE_OUT_W + 3)] = (fp16) z3;
  }
}

static void input_transform_wino4x3
      ( fp16*     __restrict__  image_d, 
        ImgShape                is, 
        fp16*     __restrict__  V_d, 
        VShape                  vs, 
        int                     simdDimSize, 
        TileShape               ts, 
        uint64_t                padding_h, 
        uint64_t                padding_w ) 
{
    const int work_group_size = 64;
    dim3 block_dim(work_group_size);
    dim3 grid_dim(is.numImg, vs.ic);
    if(padding_h == 1 && padding_w == 1) {
      if(is.h == 64 && is.w == 64)
        input_transform_every_img_wino4x3 <fp16, fp16, work_group_size, 64, 64, 1, 1> <<< grid_dim, block_dim >>> (image_d, is, V_d, vs);
      else if(is.h == 64 && is.w == 32)
        input_transform_every_img_wino4x3 <fp16, fp16, work_group_size, 64, 32, 1, 1> <<< grid_dim, block_dim >>> (image_d, is, V_d, vs);
      else if(is.h == 64 && is.w == 16)
        input_transform_every_img_wino4x3 <fp16, fp16, work_group_size, 64, 16, 1, 1> <<< grid_dim, block_dim >>> (image_d, is, V_d, vs);
      else if(is.h == 32 && is.w == 64)
        input_transform_every_img_wino4x3 <fp16, fp16, work_group_size, 32, 64, 1, 1> <<< grid_dim, block_dim >>> (image_d, is, V_d, vs);
      else if(is.h == 32 && is.w == 32)
        input_transform_every_img_wino4x3 <fp16, fp16, work_group_size, 32, 32, 1, 1> <<< grid_dim, block_dim >>> (image_d, is, V_d, vs);
      else if(is.h == 16 && is.w == 64)
        input_transform_every_img_wino4x3 <fp16, fp16, work_group_size, 16, 64, 1, 1> <<< grid_dim, block_dim >>> (image_d, is, V_d, vs);
      else
        input_transform_collapsed_wino4x3 <fp16, fp16, work_group_size> <<< DIV_UP(vs.ic * vs.numTileTotal, work_group_size), work_group_size >>> (image_d, is, V_d, vs, vs.ic * vs.numTileTotal, ts, padding_h, padding_w);
    }
    else
      input_transform_collapsed_wino4x3 <fp16, fp16, work_group_size> <<< DIV_UP(vs.ic * vs.numTileTotal, work_group_size), work_group_size >>> (image_d, is, V_d, vs, vs.ic * vs.numTileTotal, ts, padding_h, padding_w);
}


void winograd_4x3_none_fused(const void* param_ptr) {
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
    input_transform_wino4x3(image_d, is, reinterpret_cast<fp16*>(V_d), vs, vs.ic * vs.numTileTotal, ts, padding_h, padding_w);
    HIP_CHECK_KERNEL("Kernel panic!!!");
    filter_transform_wino4x3 <fp16, fp16, work_group_size> <<<DIV_UP(us.ic * us.oc, work_group_size), work_group_size>>> (filter_d, U_d, us, us.ic * us.oc);
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
    
    const size_t work_group_size_2 = 4 * HEP_WARP_SIZE; 
    output_transform_wino4x3 <fp16, fp16, work_group_size_2> <<< DIV_UP(us.oc * vs.numTileTotal , work_group_size_2), work_group_size_2 >>>(M_d, us.oc * vs.numTileTotal, out_d, os, ts);
    HIP_CHECK_KERNEL("Kernel panic!!!");    

}