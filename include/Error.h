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


#endif
