# 第 11 章：并行归约案例

从教材根目录执行下列命令，不要先进入本代码目录。先读[第 11 章正文](../../11-并行归约-从错误到高效.md)，了解每个案例的输入与预期输出。

## 学习顺序

|顺序|源码|任务|
|---|---|---|
|1|lost_update_model.cpp|用串行模拟解释丢失更新，不执行真实数据竞争|
|2|reduction_tree.cpp|打印 shared 归约树的 CPU 数学模型|
|3|block_partials.cu|每个 block 生成部分和，最后在 CPU 合并|
|4|reduction.cu|GPU 多阶段合并；每个阶段下载并检查|
|5|two_per_thread.cu|每线程先加两项，再使用同一棵树|
|6|floating_order.cpp|CPU 浮点结合顺序与舍入实验|
|7|grid_stride_reduce.cu|固定较少 block，线程循环累加局部和|
|8|maximum.cu|最大值归约，使用负无穷单位元|
|9|cub_sum.cu|CUB 临时存储查询与完整调用|
|10|reduction_benchmark.cu|统一比较四种完整求和路径|

## CPU 案例

```bash
g++ -std=c++17 -O2 code/11_reduction/lost_update_model.cpp -o lost_update_model
./lost_update_model
g++ -std=c++17 -O2 code/11_reduction/reduction_tree.cpp -o reduction_tree
./reduction_tree
g++ -std=c++17 -O2 -fno-fast-math code/11_reduction/floating_order.cpp -o floating_order
./floating_order
```

浮点实验要求 binary32 和 round-to-nearest；不满足时返回 77，表示跳过，不是通过。不使用 fast-math 改写结合顺序。

## CUDA 案例

```bash
nvcc -std=c++17 -lineinfo code/11_reduction/block_partials.cu -o block_partials
./block_partials
nvcc -std=c++17 -lineinfo code/11_reduction/reduction.cu -o reduction
./reduction 129 8
nvcc -std=c++17 -lineinfo code/11_reduction/two_per_thread.cu -o two_per_thread
./two_per_thread 129 8
nvcc -std=c++17 -lineinfo code/11_reduction/grid_stride_reduce.cu -o grid_stride_reduce
./grid_stride_reduce 23 4 2
nvcc -std=c++17 -lineinfo code/11_reduction/maximum.cu -o maximum
./maximum 13 8
nvcc -std=c++17 -lineinfo code/11_reduction/cub_sum.cu -o cub_sum
./cub_sum 1003
nvcc -std=c++17 -O3 -lineinfo code/11_reduction/reduction_benchmark.cu -o reduction_benchmark
./reduction_benchmark 1048576 128 128 50 5
```

reduction、two_per_thread、maximum 的参数为 n threads；grid_stride_reduce 为 n threads grid_cap；cub_sum 只接受 n。可全部省略而用默认值，但不能只传一部分参数。n 为 0—1048576；threads 为 2—1024 内的 2 的幂且不得超过设备限制；grid_cap 为 1—256。空输入路径在主机直接返回单位元，不调用 CUDA。

benchmark 的五个参数为 n threads grid_cap repeats rounds：n 至少为 1，repeats 为 1—1000，rounds 为 1—21；其余限制同上。它保留原始输入，只写阶段工作区，测量完整 GPU 归约的 Event 区间；分配、传输、校验和打印不在区间内。input_GBps 是输入处理率，不是 DRAM 流量或计数器数据；CUB 的 stages_per_call=-1 表示没有查询内部 kernel 数量。默认打印五批每调用时间，再按批统计，不伪造性能结果。缺少 CUB 头文件时先核对 Toolkit/CCCL 安装，而不是添加 -lcub。

## 小规模检查

```bash
./reduction 0 8
./reduction 1 8
./reduction 9 8
./reduction 129 8
./two_per_thread 17 8
./grid_stride_reduce 129 8 2
./maximum 1 8
./cub_sum 0
compute-sanitizer --tool memcheck ./reduction 129 8
compute-sanitizer --tool racecheck ./reduction 129 8
compute-sanitizer --tool synccheck ./reduction 129 8
compute-sanitizer --tool memcheck ./reduction_benchmark 65 8 2 2 2
compute-sanitizer --tool racecheck ./reduction_benchmark 65 8 2 2 2
compute-sanitizer --tool synccheck ./reduction_benchmark 65 8 2 2 2
```

不要用 Sanitizer 下的时间比较性能。设备/软件资源变化可能影响计时；极短区间为 0 时程序会报错，需增加 repeats。非法 n、非 2 的幂 block 或不完整参数应返回 2，并在启动前拒绝。

## CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target reduction_lost_update reduction_tree_model reduction_float_order reduction_partials reduction reduction_pairs reduction_stride reduction_maximum reduction_cub_sum reduction_benchmark
ctest --test-dir build -C Release -R "^reduction" --output-on-failure
```

若暂不学习 CUB，在配置阶段添加 -DBUILD_CUB_EXAMPLES=OFF，并从目标列表删去 reduction_cub_sum、reduction_benchmark。这不会禁止其他手写归约案例。公开仓库也支持 -DBUILD_CUDA_EXAMPLES=OFF，仅构建 CPU 教学程序；具体见根目录 README。

当前只完成静态与数学模型核对；这些程序尚未在本编辑环境编译或执行。请保存自己的实际命令、输出、环境版本和检查报告。
