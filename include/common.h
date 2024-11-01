#pragma once
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <stdint.h>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#define MIN(A, B) (((A) < (B)) ? (A) : (B))
#define MAX(A, B) (((A) > (B)) ? (A) : (B))

#define FLT_H 3L
#define FLT_W 3L
#define FLT_HW 3L

#define TILE_IN_HW 6L
#define TILE_IN_H 6L
#define TILE_IN_W 6L

#define TILE_OUT_HW 4L
#define TILE_OUT_H 4L
#define TILE_OUT_W 4L


#define ROUND(A, B) ((A) / (B) * (B))
#define ROUND_UP(A, B) (((A) + (B) - 1) / (B) * (B))

#define DIVIDE(A, B) ((A) / (B))
#define DIVIDE_UP(A, B) (((A) + (B) - 1) / (B))
#define DIV(A, B) ((A) / (B))
#define DIV_UP(A, B) (((A) + (B) - 1) / (B))


typedef struct {
  uint64_t b;
  uint64_t th;
  uint64_t tw;
} TileIndex;

// Tensors' shape, pass them when passing tensor as a pointer
typedef struct {
  uint64_t oc;	// number of output channels
  uint64_t ic;   // number of input channels
  uint64_t h;
  uint64_t w;
} FltShape;


typedef struct {
  uint64_t numImg;
  uint64_t ic;   // number of input channels
  uint64_t h;
  uint64_t w;
} ImgShape;

typedef struct {
  uint64_t oc;
  uint64_t ic;   // number of input channels
  uint64_t h;
  uint64_t w;
} UShape;

typedef struct {
  uint64_t numTileTotal;
  uint64_t ic;   // number of input channels
  uint64_t h;
  uint64_t w;
} VShape;

typedef struct {
  uint64_t numImg;
  uint64_t oc;   // number of output channels
  uint64_t h;
  uint64_t w;
} OutShape;

typedef struct {
  uint64_t numImg;
  uint64_t numTilePerImg;
  uint64_t numTileTotal;
  uint64_t h;
  uint64_t w;
} TileShape;

inline OutShape getOutShape(ImgShape is, FltShape fs, uint64_t padding_h, uint64_t padding_w) {
  OutShape os;
  os.numImg = is.numImg;
  os.oc = fs.oc;
  os.h = is.h - fs.h + 1 + 2 * padding_h;
  os.w = is.w - fs.w + 1 + 2 * padding_w;
  return os;
}

inline TileShape getTileShape(OutShape os) {
  TileShape ts;
  ts.h = DIV_UP(os.h, TILE_OUT_H);
  ts.w = DIV_UP(os.w, TILE_OUT_W);
  ts.numImg = os.numImg;
  ts.numTilePerImg = ts.h * ts.w;
  ts.numTileTotal = ts.numTilePerImg * ts.numImg;
  return ts;
}

inline UShape getUShape(FltShape fs) {
  UShape us;
  us.oc = fs.oc;
  us.ic = fs.ic;
  us.h = TILE_IN_W;
  us.w = TILE_IN_W;
  return us;
}

inline VShape getVShape(ImgShape is, TileShape ts) {
  VShape vs;
  vs.numTileTotal = ts.numTileTotal;
  vs.ic = is.ic;
  vs.h = TILE_IN_H;
  vs.w = TILE_IN_W;
  return vs;
}

inline __device__ __host__ TileIndex getTileIndex(uint64_t tileNo, TileShape ts) {
  TileIndex ti;
  ti.b = tileNo / ts.numTilePerImg;
  tileNo = tileNo % ts.numTilePerImg;
  ti.th = tileNo / ts.w;
  ti.tw = tileNo % ts.w;
  return ti;
}

#define HEP_WARP_SIZE 64

typedef _Float16 fp16;
typedef fp16 fp16x8 __attribute__((ext_vector_type(8)));
typedef fp16 fp16x4 __attribute__((ext_vector_type(4)));
typedef float fp32x4 __attribute__((ext_vector_type(4)));


union RegisterUnion
{
  fp16x8 vector8;
  struct
  {
    fp16x4 vector_front;
    fp16x4 vector_rear;
  };
};

enum winograd_select {
  select_4x3_non_fused,
  select_2x3_fused
};

typedef struct mykernelParamType
{
  _Float16*   pin;                            //输入数据地址
  _Float16*   pweight;                        //权值数据地址
  _Float16*   pout;                           //输出数据地址

  _Float16*   U_d = NULL;
  _Float16*   V_d = NULL;
  _Float16*   M_d = NULL;
  _Float16*   Y_d = NULL;

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

  winograd_select   kernel;                     //kernel type

  ~mykernelParamType() { if(U_d) hipFree(this->U_d); }
}mykernelParamType;                          
