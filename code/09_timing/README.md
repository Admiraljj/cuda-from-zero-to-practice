# 第 9 章：计时与带宽实验

配合 [第 9 章正文](../../09-正确计时与带宽基线.md) 阅读。共有 7 个独立程序与 1 个公共统计头文件。所有时间必须由你自己的环境测量，不能用正文手算数字冒充结果。

## CPU 入门、统计与对照

从教材根目录执行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/09_timing/clock_basics.cpp -o clock_basics
./clock_basics
g++ -std=c++17 -O2 -Wall -Wextra code/09_timing/statistics.cpp -o statistics
./statistics
g++ -std=c++17 -O3 -Wall -Wextra code/09_timing/cpu_vector_add.cpp -o cpu_vector_add
./cpu_vector_add
```

clock_basics 教你识别时间范围；statistics 只处理固定人工数据，没有读取 GPU 时间；cpu_vector_add 是可能被编译器向量化的普通单线程对照，不代表 CPU 最优库性能。

benchmark_stats.h 要与使用它的源文件在同一目录，不需要单独运行。

## CUDA 计时程序

```bash
nvcc -std=c++17 -O3 -lineinfo code/09_timing/submit_vs_complete.cu -o submit_vs_complete
./submit_vs_complete
nvcc -std=c++17 -O3 -lineinfo code/09_timing/event_timing.cu -o event_timing
./event_timing
nvcc -std=c++17 -O3 -lineinfo code/09_timing/bandwidth.cu -o bandwidth
./bandwidth
nvcc -std=c++17 -O3 -lineinfo code/09_timing/transfer_timing.cu -o transfer_timing
./transfer_timing
```

每个源文件都有自己的 main，应分别编译。Windows 把输出命名为 .exe，PowerShell 使用 .\程序名.exe，并先配置相应 CUDA/主机编译环境。

| 程序 | 测量范围 | 验证要求 |
|---|---|---|
| submit_vs_complete | 主机提交循环、提交到完成 | 两个时间不负，完成时刻不早；65536 个值正确 |
| event_timing | 100 次 kernel 的 Event 区间 | 有限正时间，1048576 个结果正确 |
| bandwidth | 多批相同 kernel；有效字节除以时间 | 每批结果正确、完整样本数、最后 PASS |
| transfer_timing | H2D/kernel/D2H Event 与复用缓冲请求 | 单一默认 stream，逐元素结果正确；不是重叠实验 |

这些程序不设固定毫秒或带宽通过阈值。不同 GPU 速度不同；极快的错误结果不算成功。

## 带宽参数

```bash
./bandwidth 1048576 100 256 5
```

依次为 n、repeats、threads、rounds，都是正的十进制整数。上限分别是 67108864、1000、1024、21；这只是教学输入限制，还需满足当前设备启动限制和实际内存条件。

默认三份设备数组共请求 12 MiB。最大 n 需要约 768 MiB 的设备数组存储，主机也有对应数组；实际资源更多，不建议直接从最大规模开始。

参数错误实验：

```bash
./bandwidth 0 100 256 5
./bandwidth 1024 100 256 5x
```

应显示 argument error 并返回 2，且不运行设备基准。正常成功返回 0；API、数值或有效时间检查失败返回 1。实际设备启动限制不满足也会返回 2，但有不同的配置诊断。

## 正确性检查与性能测量分开

可以在有工具的机器上检查：

```bash
compute-sanitizer --tool memcheck ./bandwidth 1024 2 128 2
compute-sanitizer --tool memcheck ./transfer_timing
```

检测工具会影响时间。此时看错误诊断与结果验证，不将工具下的速度用于正常性能排名。

## 可选 CMake

在教材根目录配置 Release，再构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release -R "^(timing_|bandwidth)" --output-on-failure
```

单配置构建器使用配置阶段的 CMAKE_BUILD_TYPE；多配置构建器使用构建/测试的 Release 选择。架构值需要时按自己的设备设置。

目标：timing_clock_basics、timing_statistics、timing_cpu_vector_add、timing_submit_vs_complete、timing_event、bandwidth、timing_transfer。默认测试不设跨设备速度阈值，带宽小规模案例额外验证非整除尾部。

当前编辑环境未完成 C++/CUDA 编译、CPU/GPU 或 Compute Sanitizer 执行。请按本章命令实测并保留环境与原始时间记录。
