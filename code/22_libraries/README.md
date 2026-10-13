# 第 22 章：库 API 案例

先读 [正文](../../22-Thrust-CUB-cuBLAS与库优先原则.md) 的数据契约和手算，再逐个运行。代码使用 Toolkit 附带的匹配 Thrust/CUB，以及 cuBLAS；接口按 CUDA 12.x 常见版本组织，核对 CCCL v2.5.0 与 cuBLAS 12.6.3。本机尚未编译或运行 GPU 案例，以下是读者的执行入口。

从教材根目录执行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/22_libraries/library_contracts.cpp -o library_contracts
./library_contracts
nvcc -std=c++17 -O2 -lineinfo code/22_libraries/thrust_vectors.cu -o thrust_vectors
nvcc -std=c++17 -O2 -lineinfo code/22_libraries/thrust_saxpy.cu -o thrust_saxpy
nvcc -std=c++17 -O2 -lineinfo code/22_libraries/thrust_fancy_reduce.cu -o thrust_fancy_reduce
nvcc -std=c++17 -O2 -lineinfo code/22_libraries/cub_workflow.cu -o cub_workflow
nvcc -std=c++17 -O2 -lineinfo code/22_libraries/cublas_saxpy.cu -lcublas -o cublas_saxpy
nvcc -std=c++17 -O2 -lineinfo code/22_libraries/cublas_gemm.cu -lcublas -o cublas_gemm
nvcc -std=c++17 -O2 -lineinfo code/22_libraries/saxpy_benchmark.cu -lcublas -o saxpy_benchmark
```

| 程序 | 任务 | 默认值 |
|---|---|---|
| library_contracts | 固定 CPU 手算/stride 模型 | 无参数 |
| thrust_vectors | device/host 容器、sum、sort | n=1031 |
| thrust_saxpy | 手写/Thrust 对照，输入不变 | n=1031 |
| thrust_fancy_reduce | transform_reduce 与变换 iterator | n=1031 |
| cub_workflow | Sum/ExclusiveScan/Select，顺序复用 scratch | n=1031,mode=mixed,rounds=3 |
| cublas_saxpy | strided AXPY，host/device scalar | n=1031,incx=2,incy=3 |
| cublas_gemm | padded 非方阵、beta!=0、严格 FP32 | M=3,K=5,N=2,pad=3 |
| saxpy_benchmark | custom/par/nosync/cuBLAS 四方法 | n=1048576,repeats=16,rounds=5 |

小输入与极端分布：

```bash
./thrust_vectors 5
./thrust_saxpy 0
./thrust_saxpy 1
./thrust_saxpy 255
./thrust_saxpy 256
./thrust_saxpy 257
./thrust_saxpy 1031
./thrust_fancy_reduce 1031
./cub_workflow 0 mixed 1
./cub_workflow 1031 mixed 3
./cub_workflow 1031 none 3
./cub_workflow 1031 all 3
./cublas_saxpy 4 2 3
./cublas_saxpy 1031 1 1
./cublas_gemm 3 5 2 3
./cublas_gemm 17 19 23 0
./cublas_gemm 17 19 23 5
./saxpy_benchmark 1031 4 3
```

向量/CUB 空输入为应用层返回空结果，没有运行 GPU 库。GEMM wrapper 只接受正维度；严格计算路径不支持时返回 77，表示跳过而非通过。

```bash
./thrust_saxpy -1
./cub_workflow 10 unknown 1
./cublas_saxpy 10 0 1
./cublas_gemm 0 5 2 3
compute-sanitizer --tool memcheck ./thrust_saxpy 1031
compute-sanitizer --tool memcheck ./cub_workflow 1031 mixed 2
compute-sanitizer --tool initcheck ./cub_workflow 1031 all 2
compute-sanitizer --tool memcheck ./cublas_saxpy 1031 2 3
compute-sanitizer --tool memcheck ./cublas_gemm 17 19 23 5
```

前四条应拒绝并返回非零。Sanitizer 与全量 oracle/guard/输入不变性检查互补；不因库返回 SUCCESS 就删除检查。

可选批量构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target library_contracts thrust_vectors thrust_saxpy thrust_fancy_reduce cub_workflow cublas_saxpy cublas_gemm saxpy_benchmark
ctest --test-dir build -C Release -R "library_contracts|thrust_|cub_workflow_|cublas_saxpy_|cublas_gemm_|saxpy_benchmark_" --output-on-failure
```

需要时指定经设备确认的 CMAKE_CUDA_ARCHITECTURES。BUILD_THRUST_EXAMPLES 控制本章 Thrust 示例；BUILD_CUB_EXAMPLES 控制本书独立 CUB 示例，不移除 Thrust 内部所需 CUB 头文件；BUILD_CUBLAS_EXAMPLE 控制 cuBLAS。四方法 benchmark 仅在 Thrust/cuBLAS 示例同时启用时构建；CPU model 可用首条 g++ 命令独立编译。

先保存无 profiler 的输出，再采集：

```bash
./saxpy_benchmark 1048576 16 5
nsys profile --trace=cuda,cublas --sample=none --cpuctxsw=none -o library_saxpy ./saxpy_benchmark 1048576 16 5
nsys stats library_saxpy.nsys-rep
```

benchmark 包含每 workflow 共同的 D2D 恢复+SAXPY，不是裸 kernel 时间。par/nosync 策略的同步差异单独解释；本机未采集真实性能数字。
