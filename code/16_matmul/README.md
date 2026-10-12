# 第 16 章案例：从点积到 GEMM 对照

按本目录顺序学习：shape/下标 → 一线程一输出 → tile 复用 → tiled kernel → 行列主序映射 → 公平 benchmark。主教材见[第 16 章](../../16-矩阵乘法-朴素到Tiled-GEMM.md)。

| 文件 | 任务 | 是否需要 GPU |
|---|---|---|
| matmul_reference.cpp | 手算对应的 CPU 点积，打印每项下标 | 否 |
| tile_reuse_model.cpp | 单 K phase 的加载与复用计数 | 否 |
| cublas_layout_model.cpp | 验证 C^T=B^T A^T 的地址映射 | 否 |
| naive_matmul.cu | 每线程一个 C，全量结果/输入验证 | 是 |
| tiled_matmul.cu | 16×16 shared、三维尾部与阶段同步 | 是 |
| matmul_benchmark.cu | 同一实验比较 naive/tiled，可加入 cuBLAS | 是 |

以下命令从教材仓库根目录执行。Linux 可执行文件用 `./名称` 运行；Windows 构建时可给输出加 `.exe`，在 PowerShell 用 `./名称.exe`。多行续行符请按所用 shell 调整。

## CPU 模型

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/16_matmul/matmul_reference.cpp -o matmul_reference
g++ -std=c++17 -O2 -Wall -Wextra code/16_matmul/tile_reuse_model.cpp -o matmul_tile_reuse
g++ -std=c++17 -O2 -Wall -Wextra code/16_matmul/cublas_layout_model.cpp -o matmul_cublas_layout
./matmul_reference
./matmul_reference 3 5 2
./matmul_tile_reuse 4
./matmul_tile_reuse 8
./matmul_cublas_layout
```

默认 reference 得到 C 的一维数组 `[58,64,139,154]`。复用模型 T=4 得到 naive 128 次输入元素请求、tiled 32 次 global 元素加载；布局模型得到相同 C 数组。三者各打印 PASS。它们不模拟 GPU 调度、cache、屏障或真实性能。

## 独立 CUDA 程序

```bash
nvcc -std=c++17 -O2 -lineinfo code/16_matmul/naive_matmul.cu -o matmul_naive
nvcc -std=c++17 -O2 -lineinfo code/16_matmul/tiled_matmul.cu -o matmul_tiled
./matmul_naive 37 29 23
./matmul_tiled 37 29 23
./matmul_tiled 1 1 1
./matmul_tiled 17 16 16
./matmul_tiled 16 17 16
./matmul_tiled 16 16 17
./matmul_tiled 17 17 17
./matmul_tiled 1 31 1
```

参数顺序固定 M K N，默认 129 83 117，正整数范围 1..4096；每矩阵至多 16 Mi 个元素。GPU 输出逐项与 CPU 比较，A/B 回读检查不变；输出先填 NaN sentinel。预期格式见正文，不含伪造时间。返回 77 表示 tiled 配置不被设备资源上限支持。

## 两方法/三方法 benchmark

```bash
nvcc -std=c++17 -O2 -lineinfo code/16_matmul/matmul_benchmark.cu -o matmul
nvcc -std=c++17 -O2 -lineinfo -DMATMUL_WITH_CUBLAS=1 \
    code/16_matmul/matmul_benchmark.cu -lcublas -o matmul_cublas
./matmul 37 29 23 6 2 exact
./matmul_cublas 37 29 23 6 2 exact
./matmul_cublas 37 29 23 6 2 decimal
./matmul_cublas 512 512 512 6 3 exact
./matmul_cublas 513 509 503 6 3 decimal
```

加入与设备匹配的 `-arch=sm_XX`，记录完整命令。三方法版需要 cuBLAS，使用 `CUBLAS_COMPUTE_32F_PEDANTIC`，需要支持该枚举的 Toolkit（本书 CUDA 12.x/13.x 基线）。两方法版不链接 cuBLAS。

运行参数为 M K N rounds repeats exact|decimal；默认 129 83 117 7 3 exact，维度 1..2048、rounds 1..31、repeats 1..1000。CPU 校验预算 MKN≤268435456，超出则在 GPU 操作前拒绝。

所有方法采用 C=AB、相同 row-major 输入与独立输出。每方法先单次验证、预热三次，测量次序轮换；单 stream 的 Event 包围 repeats 次 GEMM，分配/H2D/D2H/校验在外。按批间中位时间计算 useful GFLOP/s。短任务的 host 提交间隙仍可能进入区间；这是重复提交实验，不等同于单个 kernel 的纯 duration。

exact 模式以可精确表示的二进制分数验证每个输出；decimal 用实际 float 输入的 double oracle、sum_abs 和解释过的误差预算，任何 NaN/Inf 均失败。每轮另检查 A/B 未改变。输出包含资源、版本、误差、原始样本和 min/median/mean/max。

## CMake/CTest

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --target matmul_reference matmul_tile_reuse matmul_cublas_layout matmul_naive matmul_tiled matmul matmul_cublas
ctest --test-dir build -R '^matmul' --output-on-failure
```

把 80 换成已核对的 GPU 架构。没有 cuBLAS 对照需求，可配置 `-DBUILD_CUBLAS_EXAMPLE=OFF`，并从 build target 列表删去 matmul_cublas。返回 77 的资源不满足用例被 CTest 标为 skipped；拒绝参数用例必须失败才算符合预期。

## 内存、同步与性能诊断

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 ./matmul_tiled 37 29 23
compute-sanitizer --tool racecheck --error-exitcode 99 ./matmul_tiled 37 29 23
compute-sanitizer --tool synccheck --error-exitcode 99 ./matmul_tiled 37 29 23
compute-sanitizer --tool memcheck --error-exitcode 99 ./matmul_cublas 37 29 23 3 1 exact
ncu --list-sections
ncu --kernel-name-base function --kernel-name regex:matmul_tiled_kernel \
    --launch-count 1 --set full -o gemm_tiled ./matmul_tiled 513 509 503
```

先保存普通运行的正确性和 Event 数据，再独立采集 `.ncu-rep`。工具采集/重放中的时间不用于替代普通 benchmark。GPU 型号、驱动、Toolkit、cuBLAS、构建选项、shape、精度模式和计时范围必须一起记录。
