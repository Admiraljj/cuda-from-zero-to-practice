# 第 15 章源码：矩阵转置与合并访存

本目录包含两个读者程序：

| 文件 | 作用 |
|---|---|
| `transpose_mapping.cpp` | 纯 CPU 地址模型；打印非方阵每个输入坐标、输入下标、输出坐标和输出下标，并验证唯一所有者与二次转置 |
| `transpose.cu` | copy、naive、tiled-no-padding、tiled-padded 四方法统一正确性与 kernel-only benchmark |

## 先运行 CPU 地址模型

```bash
g++ -std=c++17 -O2 code/15_transpose/transpose_mapping.cpp -o transpose_mapping
./transpose_mapping
./transpose_mapping 5 3
./transpose_mapping 3 5
```

参数是 `width height`，每项允许 1～16。默认输入为 3 列、2 行。程序必须打印：

```text
PASS every output has one owner and transpose twice restores input.
```

## 编译 CUDA 对照实验

Linux：

```bash
nvcc -std=c++17 -O2 -lineinfo code/15_transpose/transpose.cu -o transpose
```

Windows PowerShell：

```powershell
nvcc -std=c++17 -O2 -lineinfo code/15_transpose/transpose.cu -o transpose.exe
```

运行参数依次是 `width height rounds warmups`：

```bash
./transpose 37 29 3 1
./transpose 1003 777 20 5
./transpose 4096 4096 20 5
```

范围：width/height 为 1～16384，总元素不超过 16 Mi；rounds 为 3～100；warmups 为 1～20。四种方法都必须逐项通过自己的 CPU oracle，且原始 device input 必须保持不变。

## CMake 与 CTest

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target transpose_mapping transpose
ctest --test-dir build -C Release -R "transpose" --output-on-failure
```

## Compute Sanitizer

```bash
compute-sanitizer --tool memcheck ./transpose 37 29 3 1
compute-sanitizer --tool synccheck ./transpose 37 29 3 1
```

Sanitizer 插桩后的耗时不能用于四方法性能排名。

## Nsight Compute

先确认本机 section：

```bash
ncu --list-sections
```

再分别采集无 padding 与 padded 的内存表：

```bash
ncu --section MemoryWorkloadAnalysis_Tables \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*transpose_tiled_kernel<0>.*' \
    --launch-count 1 -o transpose-no-padding \
    ./transpose 4096 4096 3 1

ncu --section MemoryWorkloadAnalysis_Tables \
    --kernel-name-base demangled \
    --kernel-name 'regex:.*transpose_tiled_kernel<1>.*' \
    --launch-count 1 -o transpose-padded \
    ./transpose 4096 4096 3 1
```

底层 metric 名和可用 section 可能随工具版本与 GPU 架构变化。以本机 `ncu --list-sections`、`ncu --query-metrics-mode all` 的结果为准。
