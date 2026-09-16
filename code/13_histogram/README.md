# 第 13 章：原子操作与直方图案例

先读[第 13 章正文](../../13-原子操作与直方图.md)，再按下表顺序运行。每个源文件都有自己的 `main`，是独立程序；不要把全部 `.cpp/.cu` 一次交给链接器。

## 学习顺序

|顺序|源码|回答的问题|
|---|---|---|
|1|atomic_interleaving_model.cpp|普通 read-modify-write 怎样丢更新|
|2|atomic_tickets.cu|`atomicAdd` 为什么返回旧值，票号集合怎样验证|
|3|global_histogram.cu|最简单的 256-bin global atomic 版本|
|4|atomic_contention_benchmark.cu|只改变 cycle/hot/skew 时争用如何测量|
|5|shared_histogram.cu|每 block 私有 shared bins 与两个 barrier|
|6|two_phase_histogram.cu|把 G×256 partial 暴露出来并由第二 kernel 合并|
|7|dynamic_bins_histogram.cu|16/256 bins、动态 shared 与第三个 launch 参数|
|8|numeric_limits_model.cpp|32 位回绕和 float 累加顺序差异|
|9|cas_bounded_counter.cu|CAS 返回旧值、失败重试和有上限领取|
|10|atomic_ref_block_tickets.cu|`cuda::atomic_ref`、block scope 与 relaxed order|
|11|warp_aggregated_histogram.cu|按 warp 内相同 bin 聚合并验证 atomic 次数|
|12|histogram_benchmark.cu|同一协议比较 global/shared/two-phase/warp|
|13|sanitizer_faults.cu|默认安全路径与四类显式 Sanitizer 故障|

## 单文件构建

CPU 模型：

```bash
g++ -std=c++17 -O2 code/13_histogram/atomic_interleaving_model.cpp -o atomic_interleaving_model
./atomic_interleaving_model
g++ -std=c++17 -O2 -fno-fast-math code/13_histogram/numeric_limits_model.cpp -o numeric_limits_model
./numeric_limits_model
```

基础 CUDA 阶段：

```bash
nvcc -std=c++17 -O2 -lineinfo code/13_histogram/atomic_tickets.cu -o atomic_tickets
./atomic_tickets
nvcc -std=c++17 -O2 -lineinfo code/13_histogram/global_histogram.cu -o global_histogram
./global_histogram 1031 128 4 cycle
nvcc -std=c++17 -O2 -lineinfo code/13_histogram/shared_histogram.cu -o shared_histogram
./shared_histogram 1031 64 4 cycle
nvcc -std=c++17 -O2 -lineinfo code/13_histogram/two_phase_histogram.cu -o two_phase_histogram
./two_phase_histogram 1031 64 4 cycle
nvcc -std=c++17 -O2 -lineinfo code/13_histogram/dynamic_bins_histogram.cu -o dynamic_bins_histogram
./dynamic_bins_histogram 1031 16 64 4 cycle
nvcc -std=c++17 -O2 -lineinfo code/13_histogram/cas_bounded_counter.cu -o cas_bounded_counter
./cas_bounded_counter 32 10
```

Scoped atomic 至少需要 compute capability 6.0；本书命令以该最低目标演示：

```bash
nvcc -std=c++17 -O2 -lineinfo -arch=sm_60 \
  code/13_histogram/atomic_ref_block_tickets.cu -o atomic_ref_block_tickets
./atomic_ref_block_tickets 4 64
```

warp match 路径至少需要 compute capability 7.0：

```bash
nvcc -std=c++17 -O2 -lineinfo -arch=sm_70 \
  code/13_histogram/warp_aggregated_histogram.cu -o warp_aggregated_histogram
./warp_aggregated_histogram 1031 128
nvcc -std=c++17 -O3 -lineinfo -arch=sm_70 \
  code/13_histogram/histogram_benchmark.cu -o histogram_benchmark
./histogram_benchmark 16777216 15 256 0
```

为自己的服务器构建时，把 `sm_60/sm_70` 换成实际 GPU 架构。程序在设备功能不足时返回 77，表示该实验跳过；它不表示算法通过。

## 参数矩阵

```bash
./global_histogram 0 128 4 cycle
./global_histogram 1 1 1 hot
./global_histogram 1031 128 4 cycle
./global_histogram 1031 128 4 hot
./global_histogram 1031 128 4 skew

./shared_histogram 7 32 2 cycle
./shared_histogram 1031 64 4 hot
./two_phase_histogram 7 2 2 cycle
./two_phase_histogram 1031 64 4 skew

./dynamic_bins_histogram 1031 16 64 4 cycle
./dynamic_bins_histogram 1031 256 64 4 cycle
./dynamic_bins_histogram 1031 32 64 4 cycle   # 应拒绝

./cas_bounded_counter 32 0
./cas_bounded_counter 32 10
./cas_bounded_counter 32 32
./cas_bounded_counter 8 9                       # 应拒绝
```

`cycle/hot/skew` 是固定输入规格，不是随机样本。所有正确性程序都检查完整输出或本节声明的不变量；只看到退出码 0 仍应阅读它具体检查了什么。

## Benchmark

先验证调用次数机制：

```bash
./warp_aggregated_histogram 1031 128
```

再运行不带调用计数探针的统一实验：

```bash
./histogram_benchmark 16777216 21 256 0
./histogram_benchmark 16777216 21 256 320
```

只比较同一段输出中记录了相同 GPU、n、blocks、threads、repeats 和 metric 的结果。Sanitizer、debug 构建和带 atomic 调用计数的程序都不用于性能排名。

## Compute Sanitizer

```bash
nvcc -std=c++17 -O2 -lineinfo code/13_histogram/sanitizer_faults.cu -o sanitizer_faults
./sanitizer_faults safe
compute-sanitizer --tool memcheck  --error-exitcode 99 ./sanitizer_faults safe
compute-sanitizer --tool racecheck --error-exitcode 99 ./sanitizer_faults safe
compute-sanitizer --tool initcheck --initcheck-address-space all --error-exitcode 99 ./sanitizer_faults safe
compute-sanitizer --tool synccheck --error-exitcode 99 ./sanitizer_faults safe
```

确认安全路径无报告后，才分别执行正文中的 `oob/race/uninit/divergent` 故障实验。错误模式不进入 CTest，也不要在共享生产服务器上批量运行。工具时间不是性能结果；无报告也不能替代 CPU reference 和代码审查。

## CMake 与 CTest

示例以 `sm_80` 表示一台 Ampere 服务器；请按实际 GPU 修改：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --config Release --target \
  histogram_atomic_model histogram_numeric_model histogram_tickets histogram \
  histogram_contention histogram_shared histogram_two_phase histogram_dynamic \
  histogram_cas histogram_atomic_ref histogram_warp histogram_benchmark \
  histogram_sanitizer_faults
ctest --test-dir build -C Release -R "^histogram" --output-on-failure
```

当前编辑环境没有 nvcc、C++ 编译器、GPU 或 Compute Sanitizer；以上是待服务器执行的验收入口，不是已完成的实机报告。运行后请保存 GPU/驱动/Toolkit、完整命令、退出码、Sanitizer 报告和 benchmark 原始样本。
