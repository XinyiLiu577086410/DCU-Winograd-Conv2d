#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "verfiy.h"
#include "conv2d.h"
#include "error.h"
extern "C" void winconv_4x3(const void* param_ptr) ;
extern "C" void free_param(const void* param_ptr) ;
int main(int argc, char**argv)
{
    int n = atoi(argv[1]);
    int c = atoi(argv[2]);
    int h = atoi(argv[3]);
    int w = atoi(argv[4]);
    int k = atoi(argv[5]);
    int r = atoi(argv[6]);
    int s = atoi(argv[7]);
    int u = atoi(argv[8]);
    int v = atoi(argv[9]);
    int p = atoi(argv[10]);
    int q = atoi(argv[11]);

    int outh = (h - r + 2*p)/u + 1;
    int outw = (w - s + 2*q)/v + 1;


    _Float16 *pIn       = (_Float16*)malloc(n*c*h*w*sizeof(_Float16));
    _Float16 *pWeight   = (_Float16*)malloc(k*c*r*s*sizeof(_Float16));
    _Float16 *pOut      = (_Float16*)malloc(n*k*outh*outw*sizeof(_Float16));
    _Float16 *pOut_host = (_Float16*)malloc(n*k*outh*outw*sizeof(_Float16));

    _Float16 *pIn_device,*pWeight_device,*pOut_device;
    HIP_CHECK(hipMalloc((void**)&pIn_device, n*c*h*w*sizeof(_Float16)));
    HIP_CHECK(hipMalloc((void**)&pWeight_device, k*c*r*s*sizeof(_Float16)));
    HIP_CHECK(hipMalloc((void**)&pOut_device, n*k*outh*outw*sizeof(_Float16)));
    
    // printf("pIn_device size %ld GiB\n", n*c*h*w*sizeof(_Float16) / 1024 / 1024);
    // printf("pWeight_device size %ld GiB\n", k*c*r*s*sizeof(_Float16) / 1024 / 1024);
    // printf("pOut_device size %ld GiB\n", n*k*outh*outw*sizeof(_Float16) / 1024 / 1024);
    // printf("n:%d, c:%d, h:%d, w:%d, k:%d, r:%d, s:%d, u:%d, v:%d, p:%d, q:%d\n", n, c, h, w, k, r, s, u, v, p, q);

    for(int i = 0; i < n*c*h*w; i++)
    {
        pIn[i] = (rand()%255)/255.0;
    }
    
    for(int i = 0; i < k*c*r*s; i++)
    {
        pWeight[i] = (rand()%255)/255.0;
    }
    
    for(int i = 0; i < n*k*outh*outw; i++)
    {
        pOut[i] = 0.0;
        pOut_host[i] = 0.0;
    }
           
    HIP_CHECK(hipMemcpy(pIn_device, pIn, n*c*h*w*sizeof(_Float16),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(pWeight_device,pWeight,k*c*r*s*sizeof(_Float16),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(pOut_device,pOut,n*k*outh*outw*sizeof(_Float16),hipMemcpyHostToDevice));
   
    /********************step 1*****************************/

    problem_t problem;
    int paramSize;
    kernelInfo_t kernelInfo;

    problem.in        = pIn_device;        
    problem.weight    = pWeight_device;
    problem.out       = pOut_device;             
    problem.n         = n;                             
    problem.c         = c;                             
    problem.h         = h;                             
    problem.w         = w;                             
    problem.k         = k;                             
    problem.r         = r;                             
    problem.s         = s;                             
    problem.u         = u;                             
    problem.v         = v;                             
    problem.p         = p;                             
    problem.q         = q;                               

    /********************************** step 2****************************/
    getParamsize(&problem, &paramSize);
    printf("paramsize:%d\n", paramSize);
    void* param = malloc(paramSize);
    
    getkernelInfo(&problem, &kernelInfo, param);

    dim3 groups(kernelInfo.blockx, kernelInfo.blocky, kernelInfo.blockz);
    dim3 threads(kernelInfo.threadx, kernelInfo.thready, kernelInfo.threadz);
    int ldsSize = kernelInfo.dynmicLdsSize;
        
    /*******************************warm up and get result************************************/
    // hipExtLaunchKernel(kernelInfo.kernelPtr,groups,threads,(void**)&param,ldsSize,0,0,0,0);
    winconv_4x3(param);

    HIP_CHECK(hipMemcpy(pOut_host, pOut_device,  n*k*outh*outw*sizeof(_Float16), hipMemcpyDeviceToHost));

    /*******************************cost time test************************************/
    hipEvent_t start,stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start,0);
    float time_elapsed=0.0;
    
    int iternum = 100;
    for(int i=0; i<iternum; i++)
    {
        // hipExtLaunchKernel(kernelInfo.kernelPtr,groups,threads,(void**)&param,ldsSize,0,0,0,0); 
        winconv_4x3(param);
    }
    hipEventRecord(stop,0);

    hipEventSynchronize(stop);
    hipEventElapsedTime(&time_elapsed,start,stop);

    printf("time: %f us\n", time_elapsed*1000/iternum);
    hipEventDestroy(start);
    hipEventDestroy(stop);  
    
    free_param(param);

    free(param);

    printf("===================start verfiy===================\n");
    conv2dcpu(pIn, pWeight, pOut, n, c, h, w, k, r, s, u, v, p, q);

    int error=0;
    for(int i=0;i<n*k*outh*outw;i++)
    {
        float device_out = pOut_host[i];
        //! was 0.01!!!, for _Fp16 debugging purposes, we need to increase the threshold
        if((fabs(pOut_host[i] - pOut[i]))/pOut_host[i] > 0.03|| isnan(device_out) ||isinf(device_out))
        {
            printf("error, postion:%d, gpuvalue:%f, cpuvalue:%f, error: %f\%\n", i, (float)pOut_host[i], (float)pOut[i], ((float)(pOut_host[i] - pOut[i]))/pOut_host[i]*100);
            error++;
            // break;
            if(error>1000)
                break;
        }        
    }
    printf("================finish,error:%d=========================\n",error);

    HIP_CHECK(hipFree(pIn_device));
    HIP_CHECK(hipFree(pWeight_device));
    HIP_CHECK(hipFree(pOut_device));
    
    free(pIn);
    free(pWeight);
    free(pOut);
    free(pOut_host);

    return 0;
}