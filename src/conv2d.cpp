#include <cstdio>
#include <cstdint>
#include "conv2d.h"
#include "common.h"
#include "winograd.h"
/*选手自定义的kernel入参结构体*/

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

  // kernelInfo->blockx   = (outh*outw + 15)/16;  //blockx  number
  // kernelInfo->blocky   = (k+15)/16;            //blocky  number
  // kernelInfo->blockz   = n;                    //blockz  number
  // kernelInfo->threadx  = 16;                   //threadx number per block
  // kernelInfo->thready  = 16;                   //thready number per block
  // kernelInfo->threadz  = 1;                    //threadz number per block
  // kernelInfo->dynmicLdsSize = 0;
  // kernelInfo->kernelPtr= (void*)NULL;                 //kernel ptr

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

  unsigned int malloc_size = sizeof(fp16) * (U_size + V_size + M_size);
  
  hipMalloc(&pArgs->U_d, malloc_size);
  
  pArgs->V_d = pArgs->U_d + U_size;
  pArgs->M_d = pArgs->V_d + V_size;
  pArgs->Y_d = pArgs->M_d + M_size;
  return 0;
}

extern "C" void conv2d_fp16(const void* param_ptr) {
  auto param = reinterpret_cast<const mykernelParamType*>(param_ptr);
  switch(param->kernel) {
    case winograd_select::select_4x3_non_fused:
      winograd_4x3_none_fused(param_ptr);
      break;

    default:
      winograd_4x3_none_fused(param_ptr);
  };
}