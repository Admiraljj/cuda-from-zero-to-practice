# 第 12 章：Warp 通信、掩码与分支案例

从教材根目录执行下列命令。先读[第 12 章正文](../../12-Warp原语与分支发散.md)，弄清 lane、参与 mask、有效数据和结果有效位置，再修改代码。

## 学习顺序

|顺序|源码|任务|
|---|---|---|
|1|lane_mask_model.cpp|用 CPU 模型理解多维线程线性化、warp/lane、位图和稀疏 rank|
|2|shuffle_routes.cu|逐 lane 检查 direct/down/up/xor 和 width=8 子组|
|3|vote_records.cu|区分参与者、有效数据与 predicate 结果|
|4|warp_reduce_trace.cu|保存并检查完整 32-lane 求和的每轮轨迹|
|5|warp_sum.cu|每 block 一个完整 warp，处理任意长度与尾部单位元|
|6|sparse_selected_sum.cu|证明稀疏 mask 不会自动压紧 lane|
|7|warp_shared_exchange.cu|建立 shared 的写、读和复用顺序|
|8|block_warp_reduce.cu|用 shuffle+shared 完成两级 block 归约|
|9|branch_records.cu|记录分支路径与工作量，不虚构时间结论|
|10|warp_collectives.cu|对照原始 shuffle、Cooperative Groups 和 CUB|
|11|warp_scan_preview.cu|预习四个独立 8-lane inclusive scan|

每个文件都有自己的 `main`，是独立程序，不要把所有源文件一起链接。

## CPU 模型

```bash
g++ -std=c++17 -O2 code/12_warp/lane_mask_model.cpp -o lane_mask_model
./lane_mask_model
```

它只验证编号与位运算，没有模拟 GPU 调度或证明 CUDA 原语正确。

## CUDA 案例

```bash
nvcc -std=c++17 -lineinfo code/12_warp/shuffle_routes.cu -o shuffle_routes
./shuffle_routes
nvcc -std=c++17 -lineinfo code/12_warp/vote_records.cu -o vote_records
./vote_records
nvcc -std=c++17 -lineinfo code/12_warp/warp_reduce_trace.cu -o warp_reduce_trace
./warp_reduce_trace
nvcc -std=c++17 -lineinfo code/12_warp/warp_sum.cu -o warp_sum
./warp_sum 65
nvcc -std=c++17 -lineinfo code/12_warp/sparse_selected_sum.cu -o sparse_selected_sum
./sparse_selected_sum
nvcc -std=c++17 -lineinfo code/12_warp/warp_shared_exchange.cu -o warp_shared_exchange
./warp_shared_exchange
nvcc -std=c++17 -lineinfo code/12_warp/block_warp_reduce.cu -o block_warp_reduce
./block_warp_reduce 1003 128
nvcc -std=c++17 -lineinfo code/12_warp/branch_records.cu -o branch_records
./branch_records
nvcc -std=c++17 -lineinfo code/12_warp/warp_collectives.cu -o warp_collectives
./warp_collectives
nvcc -std=c++17 -lineinfo code/12_warp/warp_scan_preview.cu -o warp_scan_preview
./warp_scan_preview
```

PowerShell 中通常写 `.\warp_sum.exe 65`。`warp_sum` 接受 `n=0..1048576`；`block_warp_reduce` 接受完整的 `n threads` 两个参数，其中 `n=0..1048576`，`threads=32..1024` 且必须是 32 的倍数，并受实际设备限制。其余程序使用正文中为手算固定的规模，不接收参数。

`warp_sum 0` 与 `block_warp_reduce 0 32` 在主机打印空和并返回，不调用 CUDA。参数格式错误返回 2；设备/API/数值错误返回 1；完整成功返回 0。

`warp_collectives.cu` 使用 CUDA Toolkit/CCCL 中的 `<cub/cub.cuh>`。找不到头文件时先核对安装版本和 include 路径；CUB 是头文件模板库，本例不添加一个虚构的 `-lcub`。

## 边界与错误检查

```bash
./warp_sum 0
./warp_sum 1
./warp_sum 31
./warp_sum 32
./warp_sum 33
./warp_sum 65
./block_warp_reduce 31 32
./block_warp_reduce 33 32
./block_warp_reduce 95 96
./block_warp_reduce 97 96
./block_warp_reduce 1003 128
./block_warp_reduce 13 48       # 应拒绝，返回非零
./block_warp_reduce 13 0        # 应拒绝，返回非零
./warp_sum -1                   # 应拒绝，返回非零
```

96 线程合法，是因为它恰有三个完整 warp；本示例不要求 block 宽度为 2 的幂。48 线程是 CUDA 可以启动的普通宽度，但不满足这份教学程序“每个参与 warp 完整”的主动约束。

## Compute Sanitizer

先对正确的小案例检查：

```bash
compute-sanitizer --tool memcheck ./warp_sum 65
compute-sanitizer --tool memcheck ./block_warp_reduce 97 96
compute-sanitizer --tool racecheck ./warp_shared_exchange
compute-sanitizer --tool synccheck ./warp_shared_exchange
compute-sanitizer --tool synccheck ./block_warp_reduce 97 96
```

Sanitizer 下的时间不用于性能比较。无报告不能代替 CPU 数值参考；逐项地址合法但公式写错时，工具可能不会报错。

## CMake 与 CTest

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target warp_lane_mask_model warp_shuffle_routes warp_vote_records warp_reduce_trace warp_sum warp_sparse_selected_sum warp_shared_exchange warp_block_reduce warp_branch_records warp_scan_preview
ctest --test-dir build -C Release -R "^warp_" --output-on-failure
```

默认还会构建依赖 CUB 的 `warp_collectives`。暂时缺少 CUB 头文件时，在配置阶段添加 `-DBUILD_CUB_EXAMPLES=OFF`，并不要指定该目标；其他 warp 案例仍可构建。公开仓库也支持 `-DBUILD_CUDA_EXAMPLES=OFF`，只构建 CPU 教学程序，具体见根 README。

当前编辑环境没有可用的 C++/CUDA 工具链，因此这里只完成静态和数学模型核对，尚未实际编译、运行 GPU 或执行 Sanitizer。请保存你在服务器上的工具链版本、完整命令、退出码与报告。
