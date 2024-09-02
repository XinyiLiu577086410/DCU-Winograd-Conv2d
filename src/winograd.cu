#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cublas_v2.h>
#include "Error.h"
#include "common.h"

__global__ void srcTransform(float* __restrict__ packedImage, float* __restrict__ V, VShape vs, int simdDimSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float z0, z1, z2, z3, z4, z5, z6;
  while (idx < simdDimSize) {
    for (int w = 0; w < TILE_IN_W; ++w) {
      z6 = packedImage[0 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 = 4.0f * z6;

      z6 = packedImage[1 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z1 = -4.0f * z6;
      z2 =  4.0f * z6;
      z3 = -2.0f * z6;
      z4 =  2.0f * z6;
      z5 =  4.0f * z6;

      z6 = packedImage[2 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 += -5.0f * z6;
      z1 += -4.0f * z6;
      z2 += -4.0f * z6;
      z3 += -z6;
      z4 += -z6;

      z6 = packedImage[3 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z1 +=  z6;
      z2 += -z6;
      z3 +=  2.0f * z6;
      z4 += -2.0f * z6;
      z5 += -5.0f * z6;

      z6 = packedImage[4 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z0 += z6;
      z1 += z6;
      z2 += z6;
      z3 += z6;
      z4 += z6;

      z6 = packedImage[5 * TILE_IN_W * simdDimSize + w * simdDimSize + idx];

      z5 += z6;

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

__global__ void filterTransform(float* __restrict__ packedFilter, float* __restrict__ U, UShape us, int simdDimSize) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float z0, z1, z2, z3, z4, z5, z6;
  while (idx < simdDimSize) {
    for (int i = 0; i < FLT_HW; ++i) {
      z6 = packedFilter[0 * FLT_W * simdDimSize + i * simdDimSize + idx];

      z0 = (1.0f / 4.0f)  * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = packedFilter[1 * FLT_W * simdDimSize + i * simdDimSize + idx];

      z1 += (-1.0f / 6.0f)  * z6;
      z2 += ( 1.0f / 6.0f)  * z6;
      z3 += (1.0f  / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = packedFilter[2 * FLT_W * simdDimSize + i * simdDimSize + idx];

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

      // z4 = svld1(pg, &YTensor[h][1][idx]);
      // z4 = YTensor[h][1][idx];
      z4 = Y[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx];

      z0 += z4;
      z1 = z4;
      z2 = z4;
      z3 = z4;
      
      // z4 = svld1(pg, &YTensor[h][2][idx]);
      // z4 = YTensor[h][2][idx];
      z4 = Y[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx];
      
      z0 += z4;
      z1 += -z4;
      z2 += z4;
      z3 += -z4;

      // z4 = svld1(pg, &YTensor[h][3][idx]);
      // z4 = YTensor[h][3][idx];
      z4 = Y[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx];

      z0 += z4;
      z1 += 2.0f * z4;
      z2 += 4.0f * z4;
      z3 += 8.0f * z4;

      // z4 = svld1(pg, &YTensor[h][4][idx]);
      // z4 = YTensor[h][4][idx];
      z4 = Y[h * TILE_IN_W * simdDimSize + 4 * simdDimSize + idx];


      z0 += z4;
      z1 += -2.0f * z4;
      z2 += 4.0f * z4;
      z3 += -8.0f * z4;

      // z4 = svld1(pg, &YTensor[h][5][idx]);
      // z4 = YTensor[h][5][idx];
      z4 = Y[h * TILE_IN_W * simdDimSize + 5 * simdDimSize + idx];

      z3 += z4;

      // svst1_f32(pg, &YTensor[h][0][idx], z0);
      // svst1_f32(pg, &YTensor[h][1][idx], z1);
      // svst1_f32(pg, &YTensor[h][2][idx], z2);
      // svst1_f32(pg, &YTensor[h][3][idx], z3);
      // YTensor[h][0][idx] = z0;
      // YTensor[h][1][idx] = z1;
      // YTensor[h][2][idx] = z2;
      // YTensor[h][3][idx] = z3;
      Y[h * TILE_IN_W * simdDimSize + 0 * simdDimSize + idx] = z0;
      Y[h * TILE_IN_W * simdDimSize + 1 * simdDimSize + idx] = z1;
      Y[h * TILE_IN_W * simdDimSize + 2 * simdDimSize + idx] = z2;
      Y[h * TILE_IN_W * simdDimSize + 3 * simdDimSize + idx] = z3;
    }
    idx += blockDim.x * gridDim.x;
  }
}

__global__ void filterOcIcPack(float* __restrict__ filter, FltShape fs, float* __restrict__ packedFilter) {
  for(int h = 0; h < FLT_HW; ++h)
    for(int w = 0; w < FLT_HW; ++w)
      for(int k = blockIdx.x * blockDim.x + threadIdx.x; 
          k < fs.oc; 
          k += blockDim.x * gridDim.x) {
        for(int c = blockIdx.y * blockDim.y + threadIdx.y;
            c < fs.ic; 
            c += blockDim.y * gridDim.y){ 
          packedFilter[h * fs.w * fs.oc * fs.ic + w * fs.oc * fs.ic + k * fs.ic + c] 
              = filter[k * fs.ic * fs.h * fs.w + c * fs.h * fs.w + h * fs.w + w];
        }
      }
}

__global__ void ImageTileIcPack(float* __restrict__ image, ImgShape is,  float* __restrict__ packedImage,  TileShape ts) {
  for(int tile = blockIdx.x * blockDim.x + threadIdx.x; 
      tile < ts.numTileTotal; 
      tile += blockDim.x * gridDim.x) {
    for(int ic = blockIdx.y * blockDim.y + threadIdx.y;
        ic < is.ic;
        ic += blockDim.y * gridDim.y) {
      for(int h = 0; h < TILE_IN_H; ++h) {
        for(int w = 0; w < TILE_IN_W; ++w) {
          TileIndex ti = getTileIndex(tile, ts);
          int b = ti.b, x = ti.tw, y = ti.th;
          if(y * 4 + h < is.h && x * 4 + w < is.w)
            packedImage[h * TILE_IN_W * ts.numTileTotal * is.ic + w * ts.numTileTotal * is.ic + tile * is.ic + ic] 
              = image[b * is.ic * is.h * is.w + ic * is.h * is.w + (y * 4 + h) * is.w + (x * 4 + w)];
          else
            packedImage[h * TILE_IN_W * ts.numTileTotal * is.ic + w * ts.numTileTotal * is.ic + tile * is.ic + ic] = 0;
        }
      }
    }
  }
}

__global__ void destStore(float* __restrict__ Y, float* __restrict__ out, OutShape os,  TileShape ts) {
  for(int h = 0; h < TILE_OUT_H; ++h)
    for(int w = 0; w < TILE_OUT_W; ++w)
      for(int k = blockIdx.x * blockDim.x + threadIdx.x; 
          k < os.oc; 
          k += blockDim.x * gridDim.x)
        for(int b = blockIdx.y * blockDim.y + threadIdx.y;
            b < ts.numTileTotal; 
            b += blockDim.y * gridDim.y) {
          TileIndex ti = getTileIndex(b, ts);
          int n = ti.b, x = ti.tw, y = ti.th;
          if(y * 4 + h < os.h && x * 4 + w < os.w) 
            out[n * os.oc * os.h * os.w + k * os.h * os.w + (y * 4 + h) * os.w + (x * 4 + w)] 
              = Y[h * TILE_IN_W * os.oc * ts.numTileTotal + w * os.oc * ts.numTileTotal + k * ts.numTileTotal + b];
        }
}

float winconv_2x3(float *__restrict__ image, const int inHeight,
                 const int inWidth, const int numInChannel, float *__restrict__ filter,
                 const int numOutChannel, const int numBatch, float *__restrict__ out,
                 float *__restrict__ U_no_use, float *__restrict__ V_no_use,
                 float *__restrict__ M_no_use) {

  /* new vars of shape */
  ImgShape  is = {numBatch, numInChannel, inHeight, inWidth};
  FltShape  fs = {numOutChannel, numInChannel, FLT_H, FLT_W};
  OutShape  os = getOutShape(is, fs);
  TileShape ts = getTileShape(is, os);
  UShape    us = getUShape(fs);
  VShape    vs = getVShape(is, ts);

  float *image_d, *filter_d;
  HANDLER_ERROR_ERR(cudaMalloc(&image_d, sizeof(float) * is.numImg * is.ic * is.h * is.w));
  HANDLER_ERROR_ERR(cudaMalloc(&filter_d, sizeof(float) * fs.oc * fs.ic * fs.h * fs.w));

  float *packedFilter_d, *packedImage_d;
  HANDLER_ERROR_ERR(cudaMalloc(&packedFilter_d, sizeof(float) * FLT_H * FLT_W * us.oc * us.ic));
  HANDLER_ERROR_ERR(cudaMalloc(&packedImage_d , sizeof(float) * TILE_IN_H * TILE_IN_H * vs.numTileTotal * vs.ic));

  float *U_d, *V_d;
  HANDLER_ERROR_ERR(cudaMalloc(&U_d, sizeof(float) * TILE_IN_H * TILE_IN_W * us.oc * us.ic));
  HANDLER_ERROR_ERR(cudaMalloc(&V_d, sizeof(float) * TILE_IN_H * TILE_IN_W * vs.numTileTotal * vs.ic)); 

  float *M_d;
  HANDLER_ERROR_ERR(cudaMalloc(&M_d, sizeof(float) * TILE_IN_H  * TILE_IN_W  * us.oc * vs.numTileTotal));

  float *Y_d;
  HANDLER_ERROR_ERR(cudaMalloc(&Y_d, sizeof(float) * TILE_OUT_H * TILE_IN_W  * us.oc * vs.numTileTotal));
  
  float *out_d;
  HANDLER_ERROR_ERR(cudaMalloc(&out_d, sizeof(float) * os.numImg * os.oc * os.h * os.w));

  HANDLER_ERROR_ERR(cudaMemcpy(image_d, image, sizeof(float) * is.numImg * is.ic * is.h * is.w, cudaMemcpyHostToDevice));
  HANDLER_ERROR_ERR(cudaMemcpy(filter_d, filter, sizeof(float) * fs.oc * fs.ic * fs.h * fs.w, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  HANDLER_ERROR_ERR(cudaEventCreate(&start));
  HANDLER_ERROR_ERR(cudaEventCreate(&stop));
  HANDLER_ERROR_ERR(cudaEventRecord(start));

  filterOcIcPack<<<dim3(10, 10), dim3(16, 16)>>>(filter_d, fs, packedFilter_d);
  HANDLER_ERROR_MSG("kernel panic!!!");

  ImageTileIcPack<<<dim3(10, 10), dim3(16, 16)>>>(image_d, is, packedImage_d, ts);
  HANDLER_ERROR_MSG("kernel panic!!!");
  
  srcTransform<<<100, 256>>>(packedImage_d, V_d, vs, vs.ic * vs.numTileTotal);
  HANDLER_ERROR_MSG("kernel panic!!!");
  
  filterTransform<<<100, 256>>>(packedFilter_d, U_d, us, us.ic * us.oc);
  HANDLER_ERROR_MSG("kernel panic!!!");
  
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0f, beta = 0.0f;
  for(int i = 0; i < TILE_IN_H * TILE_IN_W; ++i) {
    typedef float (*UTensor_t) [TILE_IN_W][     us.oc     ][us.ic];
    typedef float (*VTensor_t) [TILE_IN_W][vs.numTileTotal][vs.ic];
    typedef float (*MTensor_t) [TILE_IN_W][us.oc][vs.numTileTotal];
    UTensor_t UTensor = (UTensor_t) U_d;
    VTensor_t VTensor = (VTensor_t) V_d;
    MTensor_t MTensor = (MTensor_t) M_d;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                vs.numTileTotal, us.oc, us.ic,
                &alpha,
                (float*)(VTensor[i/TILE_IN_W][i%TILE_IN_W]),
                vs.ic, 
                (float*)(UTensor[i/TILE_IN_W][i%TILE_IN_W]),
                us.ic, 
                &beta, 
                (float*)(MTensor[i/TILE_IN_W][i%TILE_IN_W]),
                vs.numTileTotal);
  }
  cublasDestroy(handle);

  destTransform<<<100, 256>>>(M_d, Y_d, us.oc * vs.numTileTotal);
  HANDLER_ERROR_MSG("kernel panic!!!");

  destStore<<<dim3(10, 10), dim3(16, 16)>>>(Y_d, out_d, os, ts);
  HANDLER_ERROR_MSG("kernel panic!!!");

  HANDLER_ERROR_ERR(cudaEventRecord(stop));
  HANDLER_ERROR_ERR(cudaEventSynchronize(stop));
  float milliseconds;
  HANDLER_ERROR_ERR(cudaEventElapsedTime(&milliseconds, start, stop));

  HANDLER_ERROR_ERR(cudaMemcpy(out, out_d, sizeof(float) * os.numImg * os.oc * os.h * os.w, cudaMemcpyDeviceToHost));

  cudaFree(image_d);
  cudaFree(filter_d);
  cudaFree(packedImage_d);
  cudaFree(packedFilter_d);
  cudaFree(U_d);
  cudaFree(V_d);
  cudaFree(M_d);
  cudaFree(Y_d);
  cudaFree(out_d);

  return milliseconds;
}