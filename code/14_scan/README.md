# 第 14 章代码：Scan 与流压缩

请先读教材第 14 章，再按下表运行。示例按“CPU 语义 → 单 block 树 → 大数组分层 → 实际应用”的顺序排列。

| 文件 | 需要 GPU | 内容 | 默认是否跑危险行为 |
|---|---|---|---|
| `scan_reference.cpp` | 否 | inclusive/exclusive CPU oracle 与交叉不变量 | 否 |
| `single_block_scan.cu` | 是 | 动态 shared memory、补零、Blelloch 单 block exclusive scan | 否 |
| `hierarchical_scan.cu` | 是 | tile sums、递归 scan、uniform add 的任意长度 scan | 否 |
| `stream_compaction.cu` | 是 | flags + CUB exclusive scan + stable scatter | 否 |
| `scan_benchmark.cu` | 是 | 同输入比较手写分层 scan 与 CUB，完整验证后报告样本 | 否 |

## 直接编译

在仓库根目录执行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra \
    code/14_scan/scan_reference.cpp -o scan_reference

nvcc -std=c++17 -O2 -lineinfo \
    code/14_scan/single_block_scan.cu -o single_block_scan

nvcc -std=c++17 -O2 -lineinfo \
    code/14_scan/hierarchical_scan.cu -o hierarchical_scan

nvcc -std=c++17 -O2 -lineinfo \
    code/14_scan/stream_compaction.cu -o stream_compaction

nvcc -std=c++17 -O2 -lineinfo \
    code/14_scan/scan_benchmark.cu -o scan_benchmark
```

`stream_compaction.cu` 使用 CUDA Toolkit 自带的 CUB 头文件。若找不到 `cub/device/device_scan.cuh`，先确认执行的 `nvcc --version` 与 CUDA Toolkit 安装和环境变量一致，不要随便从网络复制一份头文件混用。

## 建议运行顺序

```bash
./scan_reference
./scan_reference 5 -2 0 7

./single_block_scan 13 out
./single_block_scan 13 inplace
./single_block_scan 257 out

./hierarchical_scan 257 out
./hierarchical_scan 65537 inplace
./hierarchical_scan 1000003 out

./stream_compaction 1 none
./stream_compaction 1 all
./stream_compaction 1031 mixed
./stream_compaction 1031 none
./stream_compaction 1031 all

./scan_benchmark 1000003 15 4
```

每个程序成功时都会打印 `PASS`，但含义不同：

- `scan_reference` 检查 inclusive/exclusive 的逐项关系；
- 两个手写 GPU scan 检查每一个前缀和总和；
- 流压缩同时检查 flags、offsets、输出数量、内容与稳定顺序。
- benchmark 在每次预热和正式样本后逐项检查结果，并在每轮检查原始输入未被修改。

不要只搜索 `PASS` 就忽略前面的 CUDA 错误或参数错误。返回码 0 才表示该次运行通过。

## 用 CMake 构建和测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure \
  -R "scan_(reference|single_block|hierarchical|compaction)"
```

若有多个 CUDA Toolkit，可显式给出编译器路径，例如：

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release
```

路径必须按你的服务器实际安装位置填写。

## Compute Sanitizer

所有正常模式都应当可以安全地交给动态检查：

```bash
compute-sanitizer --tool memcheck ./single_block_scan 257 out
compute-sanitizer --tool memcheck ./hierarchical_scan 65537 out
compute-sanitizer --tool racecheck ./single_block_scan 257 inplace
compute-sanitizer --tool synccheck ./single_block_scan 257 out
compute-sanitizer --tool initcheck ./stream_compaction 1031 mixed
compute-sanitizer --tool memcheck ./scan_benchmark 1031 3 1
```

这些工具检查的维度不同：

- `memcheck`：越界、未对齐等内存访问错误；
- `racecheck`：shared memory 数据竞争风险；
- `synccheck`：同步原语使用错误；
- `initcheck`：读取未初始化的 device global memory。

动态检查通过不等于数学结果正确，所以仍必须保留程序自己的 CPU oracle。反过来，一次结果正确也不能证明没有竞争或越界。

当前章节不会提供默认触发未定义行为的程序。若以后加入故障注入示例，它必须通过明确命令行模式启用，绝不能混入默认测试。

## 当前实现边界

- 教学 scan 使用 64 位有符号整数；输入范围经过限制，测试数据不会溢出。
- `single_block_scan` 最多 512 项，并查询设备线程和 shared memory 上限。
- `hierarchical_scan` tile 固定为 256，临时分配以清楚为先，不代表性能最佳。
- `stream_compaction` 最多为每项产生一个输出，因此预留 `n` 项容量。
- benchmark 的 Event 区间只包含 scan 设备工作；分配、复制、哨兵填充、验证和打印均在区间外。
- 当前机器若没有 `nvcc`、GPU 和 Compute Sanitizer，只能做静态与 CPU 模型检查；需要在 CUDA 服务器上完成真实构建和运行。
