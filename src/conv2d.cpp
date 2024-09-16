#include <cinttypes>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "conv2d.h"
#include "common.h"
#include "error.h"
#include <rocblas.h>


/*选手自定义的kernel入参结构体*/
typedef struct mykernelParamType
{
    _Float16*   pin;                            //输入数据地址
    _Float16*   pweight;                        //权值数据地址
    _Float16*   pout;                           //输出数据地址

    float*   U_d;
    float*   V_d;
    float*   M_d;
    float*   Y_d;

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


__global__ void srcTransform(_Float16* __restrict__ image, ImgShape is,  float* __restrict__ V, VShape vs, int simdDimSize, TileShape ts, uint64_t padding_h, uint64_t padding_w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float z0, z1, z2, z3, z4, z5, z6;
  while (idx < simdDimSize) {
    const uint64_t ic = idx % vs.ic;
    const TileIndex ti = getTileIndex(idx / vs.ic, ts);
    const uint64_t b = ti.b, th = ti.th, tw = ti.tw;
    typedef _Float16 (*imageTensor_t) [is.ic][is.h][is.w];
    imageTensor_t imageTensor = (imageTensor_t)image;
    for (int w = 0; w < TILE_IN_W; ++w) {
     
      z0 = z1 = z2 = z3 = z4 = z5 = 0.0f;
      if(th * TILE_OUT_H + 0 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
        z6 = imageTensor[b][ic][th * TILE_OUT_H + 0 - padding_h][tw * TILE_OUT_W + w - padding_w];
        z0 = 4.0f * z6;
      }

      if(th * TILE_OUT_H + 1 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
        z6 = imageTensor[b][ic][th * TILE_OUT_H + 1 - padding_h][tw * TILE_OUT_W + w - padding_w];
        z1 = -4.0f * z6;
        z2 =  4.0f * z6;
        z3 = -2.0f * z6;
        z4 =  2.0f * z6;
        z5 =  4.0f * z6;
      }

      if(th * TILE_OUT_H + 2 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
        z6 = imageTensor[b][ic][th * TILE_OUT_H + 2 - padding_h][tw * TILE_OUT_W + w - padding_w];
        z0 += -5.0f * z6;
        z1 += -4.0f * z6;
        z2 += -4.0f * z6;
        z3 += -z6;
        z4 += -z6;
      }

      if(th * TILE_OUT_H + 3 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
        z6 = imageTensor[b][ic][th * TILE_OUT_H + 3 - padding_h][tw * TILE_OUT_W + w - padding_w];
        z1 +=  z6;
        z2 += -z6;
        z3 +=  2.0f * z6;
        z4 += -2.0f * z6;
        z5 += -5.0f * z6;
      }

      if(th * TILE_OUT_H + 4 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
        z6 = imageTensor[b][ic][th * TILE_OUT_H + 4 - padding_h][tw * TILE_OUT_W + w - padding_w];
        z0 += z6;
        z1 += z6;
        z2 += z6;
        z3 += z6;
        z4 += z6;
      }

      if(th * TILE_OUT_H + 5 - padding_h < is.h && tw * TILE_OUT_W + w - padding_w < is.w) {
        z6 = imageTensor[b][ic][th * TILE_OUT_H + 5 - padding_h][tw * TILE_OUT_W + w - padding_w];
        z5 += z6;
      }

      V[0 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z0;
      V[1 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z1;
      V[2 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z2;
      V[3 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z3;
      V[4 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z4;
      V[5 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z5;
    }

    for (int h = 0; h < TILE_IN_H; ++h) {
      z6 = V[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx];

      z0 = 4.0f * z6;

      z6 = V[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx];

      z1 = -4.0f * z6;
      z2 =  4.0f * z6;
      z3 = -2.0f * z6;
      z4 =  2.0f * z6;
      z5 =  4.0f * z6;

      z6 = V[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = V[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx];

      z1 +=  z6;
      z2 += -z6;
      z3 +=  2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = V[h * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = V[h * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx];

      z5 += z6;

      V[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
      V[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
      V[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
      V[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
      V[h * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx] = z4;
      V[h * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx] = z5;
    }
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void filterTransform(_Float16* __restrict__ filter, float* __restrict__ U, UShape us, int simdDimSize) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float z0, z1, z2, z3, z4, z5, z6;
  while (idx < simdDimSize) {
    for (int i = 0; i < FLT_HW; ++i) {
      z6 = filter[idx * FLT_H * FLT_W + 0 * FLT_W  + i];

      z0 = (1.0f / 4.0f)  * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = filter[idx * FLT_H * FLT_W + 1 * FLT_W  + i];

      z1 += (-1.0f / 6.0f)  * z6;
      z2 += ( 1.0f / 6.0f)  * z6;
      z3 += (1.0f  / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = filter[idx * FLT_H * FLT_W + 2 * FLT_W  + i];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += ( 1.0f / 6.0f) * z6;
      z4 += ( 1.0f / 6.0f) * z6;
      z5 = z6;

      U[0 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z0;
      U[1 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z1;
      U[2 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z2;
      U[3 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z3;
      U[4 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z4;
      U[5 * TILE_IN_W * simdDimSize + i * simdDimSize + idx] = z5;
    }

    for (int i = 0; i < TILE_IN_H; ++i) {
      z6 = U[i * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx];

      z0 = (1.0f / 4.0f)  * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = U[i * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx];

      z1 += (-1.0f / 6.0f)  * z6;
      z2 += ( 1.0f / 6.0f)  * z6;
      z3 += (1.0f  / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = U[i * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += ( 1.0f / 6.0f) * z6;
      z4 += ( 1.0f / 6.0f) * z6;
      z5 = z6;

      U[i * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
      U[i * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
      U[i * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
      U[i * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
      U[i * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx] = z4;
      U[i * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx] = z5;
    }
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void destTransform(float* __restrict__ M, float* __restrict__ Y, int simdDimSize) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x; 

  float z0, z1, z2, z3, z4;
  while (idx < simdDimSize) {
    for (int w = 0; w < TILE_IN_W; ++w) {
      z4 = M[0 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 = z4;

      z4 = M[1 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 = z0 + z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;
      
      z4 = M[2 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = M[3 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      z4 = M[4 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      z4 = M[5 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z3 += z4;

      Y[0 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z0;
      Y[1 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z1;
      Y[2 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z2;
      Y[3 * TILE_IN_W * simdDimSize + w * simdDimSize + idx] = z3;
    }

    for (int h = 0; h < TILE_OUT_HW; ++h) {
      z4 = Y[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx];

      z0 = z4;

      
      z4 = Y[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx];

      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;

      z4 = Y[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx];
      
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      z4 = Y[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx];

      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;


      z4 = Y[h * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx];


      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;


      z4 = Y[h * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx];

      z3 += z4;

      Y[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
      Y[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
      Y[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
      Y[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
    }
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void destStore(float* __restrict__ Y, _Float16* __restrict__ out, OutShape os,  TileShape ts, uint64_t padding_h, uint64_t padding_w) {
  for(int h = 0; h < TILE_OUT_H; ++h)
    for(int w = 0; w < TILE_OUT_W; ++w)
      for(int k = blockIdx.x * blockDim.x + threadIdx.x; 
          k < os.oc; 
          k += blockDim.x * gridDim.x)
        for(int b = blockIdx.y * blockDim.y + threadIdx.y;
            b < ts.numTileTotal; 
            b += blockDim.y * gridDim.y) {
          TileIndex ti = getTileIndex(b, ts);
          int n = ti.b, tw = ti.tw, th = ti.th;
          if(th * 4 + h < os.h && tw * 4 + w < os.w) 
            out[n * os.oc * os.h * os.w + k * os.h * os.w + (th * 4 + h) * os.w + (tw * 4 + w)] 
              = (_Float16) Y[h * TILE_IN_W * os.oc * ts.numTileTotal + w * os.oc * ts.numTileTotal + k * ts.numTileTotal + b];
        }
}


extern "C" void winconv_4x3(const void* param_ptr) {
    
    const mykernelParamType* param = (const mykernelParamType*)param_ptr;
    _Float16* filter_d = param->pweight;
    _Float16* image_d  = param->pin;
    _Float16* out_d    = param->pout;
    float* U_d = param->U_d;
    float* V_d = param->V_d;
    float* M_d = param->M_d;
    float* Y_d = param->Y_d;
    uint64_t padding_h = param->p;
    uint64_t padding_w = param->q;

    ImgShape  is = {param->n, param->c, param->h, param->w};
    FltShape  fs = {param->k, param->c, param->r, param->s};
    OutShape  os = getOutShape(is, fs, padding_h, padding_w);
    TileShape ts = getTileShape(os);
    UShape    us = getUShape(fs);
    VShape    vs = getVShape(is, ts);


    srcTransform<<<100, 256, 0, hipStreamDefault>>>(image_d, is, V_d, vs, vs.ic * vs.numTileTotal, ts, padding_h, padding_w);
    HIP_CHECK_KERNEL("Kernel panic!!!");    
    filterTransform<<<100, 256, 0, hipStreamDefault>>>(filter_d, U_d, us, us.ic * us.oc);
    HIP_CHECK_KERNEL("Kernel panic!!!");    

    rocblas_handle handle;
    rocblas_create_handle(&handle);
    const float alpha = 1.0, beta = 0.0;
    rocblas_operation transa = rocblas_operation_transpose, transb = rocblas_operation_none;
    for(int i = 0; i < TILE_IN_H * TILE_IN_W; ++i) {
        typedef const float (*UTensor_t) [TILE_IN_W][     us.oc     ][us.ic];
        typedef float (*VTensor_t) [TILE_IN_W][vs.numTileTotal][vs.ic];
        typedef float (*MTensor_t) [TILE_IN_W][us.oc][vs.numTileTotal];
        UTensor_t UTensor = (UTensor_t) U_d;
        VTensor_t VTensor = (VTensor_t) V_d;
        MTensor_t MTensor = (MTensor_t) M_d;
        rocblas_sgemm(handle, transa, transb,
                    vs.numTileTotal, us.oc, us.ic,
                    &alpha,
                    (const float*)(VTensor[i/TILE_IN_W][i%TILE_IN_W]),
                    vs.ic, 
                    (const float*)(UTensor[i/TILE_IN_W][i%TILE_IN_W]),
                    us.ic, 
                    &beta, 
                    (float*)(MTensor[i/TILE_IN_W][i%TILE_IN_W]),
                    vs.numTileTotal);
    }
    rocblas_destroy_handle(handle);

    destTransform<<<100, 256, 0, hipStreamDefault>>>(M_d, Y_d, us.oc * vs.numTileTotal);
    HIP_CHECK_KERNEL("Kernel panic!!!");    

    destStore<<<dim3(10, 10), dim3(16, 16), 0, hipStreamDefault>>>(Y_d, out_d, os, ts, padding_h, padding_w);
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

    // printf("n:%d, c:%d, h:%d, w:%d, k:%d, r:%d, s:%d, u:%d, v:%d, p:%d, q:%d\n", n, c, h, w, k, r, s, u, v, p, q);

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
    unsigned int Y_size = TILE_OUT_H * TILE_IN_W  * k * vs.numTileTotal;
    
    unsigned int malloc_size = sizeof(float) * (
          U_size
        + V_size
        + M_size
        + Y_size
    );
    

    HIP_CHECK(hipMalloc(&pArgs->U_d, malloc_size));
    
    pArgs->V_d = pArgs->U_d + U_size;
    pArgs->M_d = pArgs->V_d + V_size;
    pArgs->Y_d = pArgs->M_d + M_size;
    return 0;
}

extern "C" void free_extra_vram(const void* param_ptr) {
    mykernelParamType* pArgs = (mykernelParamType*)param_ptr;
    hipFree(pArgs->U_d);
}
