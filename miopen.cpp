#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>

#define CHECK_MIOPEN(status) \
    if(status != miopenStatusSuccess) { \
        std::cerr << "MIOpen failure: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

#define CHECK_HIP(status) \
    if(status != hipSuccess) { \
        std::cerr << "HIP failure: " << hipGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

// 计算输出尺寸
inline int calc_output_size(int input_size, int kernel_size, int pad, int stride) {
    return (input_size - kernel_size + 2 * pad) / stride + 1;
}

int main(int argc, char* argv[]) {
    if (argc != 12) {
        std::cerr << "Usage: " << argv[0] << " n c h w k r s u v p q\n";
        return 1;
    }

    // 解析参数
    int n = atoi(argv[1]);   // batch_size
    int c = atoi(argv[2]);   // in_channels
    int h = atoi(argv[3]);   // in_height
    int w = atoi(argv[4]);   // in_width
    int k = atoi(argv[5]);   // out_channels
    int r = atoi(argv[6]);   // kernel_h
    int s = atoi(argv[7]);   // kernel_w
    int u = atoi(argv[8]);   // stride_h
    int v = atoi(argv[9]);   // stride_w
    int p = atoi(argv[10]);  // pad_h
    int q = atoi(argv[11]);  // pad_w

    // 计算输出尺寸
    int out_h = calc_output_size(h, r, p, u);
    int out_w = calc_output_size(w, s, q, v);

    std::cout << "Params: n=" << n << " c=" << c << " h=" << h << " w=" << w
              << " k=" << k << " r=" << r << " s=" << s << " u=" << u
              << " v=" << v << " p=" << p << " q=" << q << "\n";
    std::cout << "Output: " << out_h << " x " << out_w << std::endl;

    // 初始化HIP
    CHECK_HIP(hipSetDevice(0));
    
    // 初始化MIOpen
    miopenHandle_t miopen_handle;
    CHECK_MIOPEN(miopenCreate(&miopen_handle));

    // 创建张量描述符
    miopenTensorDescriptor_t input_desc, output_desc, weight_desc;
    CHECK_MIOPEN(miopenCreateTensorDescriptor(&input_desc));
    CHECK_MIOPEN(miopenCreateTensorDescriptor(&output_desc));
    CHECK_MIOPEN(miopenCreateTensorDescriptor(&weight_desc));

    // 设置张量维度 (NCHW格式)
    int input_dims[4] = {n, c, h, w};
    int output_dims[4] = {n, k, out_h, out_w};
    int weight_dims[4] = {k, c, r, s};

    CHECK_MIOPEN(miopenSet4dTensorDescriptor(input_desc, miopenHalf, 
                  input_dims[0], input_dims[1], input_dims[2], input_dims[3]));
    CHECK_MIOPEN(miopenSet4dTensorDescriptor(output_desc, miopenHalf, 
                  output_dims[0], output_dims[1], output_dims[2], output_dims[3]));
    CHECK_MIOPEN(miopenSet4dTensorDescriptor(weight_desc, miopenHalf, 
                  weight_dims[0], weight_dims[1], weight_dims[2], weight_dims[3]));

    // 创建卷积描述符
    miopenConvolutionDescriptor_t conv_desc;
    CHECK_MIOPEN(miopenCreateConvolutionDescriptor(&conv_desc));
    CHECK_MIOPEN(miopenInitConvolutionDescriptor(conv_desc, 
                  miopenConvolution, p, q, u, v, 1, 1)); // dilation=1

    // 分配GPU内存
    size_t input_size = n * c * h * w * sizeof(short);
    size_t weight_size = k * c * r * s * sizeof(short);
    size_t output_size = n * k * out_h * out_w * sizeof(short);

    void *d_input, *d_weight, *d_output;
    CHECK_HIP(hipMalloc(&d_input, input_size));
    CHECK_HIP(hipMalloc(&d_weight, weight_size));
    CHECK_HIP(hipMalloc(&d_output, output_size));

    // 初始化数据 (实际应用中应使用真实数据)
    CHECK_HIP(hipMemset(d_input, 0, input_size));
    CHECK_HIP(hipMemset(d_weight, 0, weight_size));
    CHECK_HIP(hipMemset(d_output, 0, output_size));

    // 寻找最优算法
    int request_algo_count = 1;
    int returned_algo_count = 0;
    miopenConvAlgoPerf_t perf_results;
    CHECK_MIOPEN(miopenFindConvolutionForwardAlgorithm(
        miopen_handle, input_desc, d_input, weight_desc, d_weight,
        conv_desc, output_desc, d_output, request_algo_count,
        &returned_algo_count, &perf_results, nullptr, 0, false));

    if (returned_algo_count == 0) {
        std::cerr << "No convolution algorithm found!" << std::endl;
        return 1;
    }

    std::cout << "Selected algorithm: " << perf_results.fwd_algo 
              << ", time: " << perf_results.time << " ms" 
              << ", workspace: " << perf_results.memory << " bytes\n";

    // 分配工作空间
    void* d_workspace = nullptr;
    if (perf_results.memory > 0) {
        CHECK_HIP(hipMalloc(&d_workspace, perf_results.memory));
    }

    // 准备卷积操作
    float alpha = 1.0f, beta = 0.0f;
    CHECK_MIOPEN(miopenConvolutionForward(
        miopen_handle, &alpha, input_desc, d_input, weight_desc, d_weight,
        conv_desc, perf_results.fwd_algo, &beta, output_desc, d_output,
        d_workspace, perf_results.memory));

    // 预热
    CHECK_HIP(hipDeviceSynchronize());
    for (int i = 0; i < 10; ++i) {
        CHECK_MIOPEN(miopenConvolutionForward(
            miopen_handle, &alpha, input_desc, d_input, weight_desc, d_weight,
            conv_desc, perf_results.fwd_algo, &beta, output_desc, d_output,
            d_workspace, perf_results.memory));
    }
    CHECK_HIP(hipDeviceSynchronize());

    // 性能测试
    const int runs = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        miopenConvolutionForward(
            miopen_handle, &alpha, input_desc, d_input, weight_desc, d_weight,
            conv_desc, perf_results.fwd_algo, &beta, output_desc, d_output,
            d_workspace, perf_results.memory);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // 计算性能
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    double avg_time = elapsed / runs * 1000 * 1000; //us
    double gflops = (2.0 * n * k * out_h * out_w * c * r * s) / (avg_time * 1e3); // TFLOPs

    std::cout << "Average time: " << avg_time << " us" << std::endl;
    std::cout << "Throughput: " << gflops << " TFLOPs" << std::endl;

    // 清理资源
    if (d_workspace) hipFree(d_workspace);
    hipFree(d_input);
    hipFree(d_weight);
    hipFree(d_output);
    miopenDestroyConvolutionDescriptor(conv_desc);
    miopenDestroyTensorDescriptor(input_desc);
    miopenDestroyTensorDescriptor(weight_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroy(miopen_handle);

    return 0;
}
