# 项目结构
```bash
.
├── conf.txt
│   # 比赛算例的测试脚本
├── final.sh
├── pre.sh
├── profile_final.sh
├── profile_pre.sh
│   # 头文件
├── include
│   ├── common.h
│   ├── conv2d.h
│   ├── error.h
│   ├── hep_sgemm.h
│   ├── verfiy.h
│   └── winograd.h
├── Makefile
├── readme.txt
│   # 源码文件
└── src
    ├── conv2d.cpp                  # 卷积计算入口，负责算子选择
    ├── main.cpp                    # 主函数
    ├── winograd_2x3_fused.cpp      # 融合式Winograd卷积算子实现
    ├── winograd_2x3_non_fused.cpp  # 非融合式Winograd卷积算子实现
    └── winograd_4x3_non_fused.cpp  # 非融合式Winograd卷积算子实现
```