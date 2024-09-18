#ifndef __ERROR_H__
#define __ERROR_H__
#include <iostream>
#include <hip/hip_runtime.h>
#include <cstdlib>

#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(-1);                                                     \
        }                                                                                   \
    }

#define HIP_CHECK_KERNEL(msg) (hip_error(msg, __FILE__, __LINE__))

static void hip_error(const char* msg, const char* file, int line) {
    hipError_t err = hipGetLastError();
    if (hipSuccess != err) {
        fprintf(stderr, "%s: %s in %s at line %d\n", msg,
                hipGetErrorString(err), file, line);
       std::exit(-1);  
    }
}


#define COPY_TO_HOST_AND_PRINT_ARRAY(array_d, size) \
    {                                               \
        _Float16* array = (_Float16*)malloc(size * sizeof(_Float16));           \
        assert (array != NULL); \
        printf("array = %p, array_d = %p, size = %d\n", array, array_d, size); \
        HIP_CHECK(hipMemcpy(array, array_d, size * sizeof(_Float16), hipMemcpyDeviceToHost)); \
        for(int i = 0; i < size; i++) \
        { \
            printf("%f ", (float)array[i]); \
        } \
        std::cout << std::endl; \
        free(array); \
    }
#endif
