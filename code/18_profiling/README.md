# 第18章：性能分析案例

正文：[Nsight性能分析](../../18-Nsight性能分析与优化方法.md)。performance_model.cpp只演示合成时间算术；profiling_workload.cu提供真实GPU任务与sync/queued对照，复用第17章头文件的严格整数解析。

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/18_profiling/performance_model.cpp -o performance_model
./performance_model 60 3
nvcc -std=c++17 -O2 -lineinfo code/18_profiling/profiling_workload.cu -o profiling_workload
./profiling_workload 1031 32 sync
./profiling_workload 1031 32 queued
```

参数n repeats mode，默认1048576/32/queued。覆盖同一output=2*input+1任务，input不变。n=0跳过GPU。两种时间区间排除分配/复制/校验，预热3次；结果全量验证。

可选NVTX3头文件版本，Linux用：

```bash
nvcc -std=c++17 -O2 -lineinfo -DPROFILE_NVTX=1 code/18_profiling/profiling_workload.cu -ldl -o profiling_workload_nvtx
nsys profile --trace=cuda,nvtx --sample=none -o queued ./profiling_workload_nvtx 1031 32 queued
nsys profile --trace=cuda,nvtx --sample=none -o sync ./profiling_workload_nvtx 1031 32 sync
nsys stats queued.nsys-rep
nsys profile --trace=cuda,nvtx --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop -o focused ./profiling_workload_nvtx 1031 32 queued
ncu --kernel-name-base function --kernel-name regex:affine_kernel --launch-skip 3 --launch-count 1 --set basic -o affine ./profiling_workload 1048576 1 queued
```

无NVTX版仍可用`--trace=cuda`；NVTX3 include目录可用-I指定，参考NVIDIA/NVTX安装说明。Windows不使用-ldl。先查nsys/ncu版本和help；ncu的launch-skip针对匹配kernel，须核对本机筛选日志。report与普通benchmark排名数据分开保存。

```bash
cmake --build build --target performance_model profiling_workload
ctest --test-dir build -R '^(performance_model|profiling_workload)' --output-on-failure
```

可选CMake `-DBUILD_NVTX_EXAMPLE=ON`、必要时`-DNVTX3_INCLUDE_DIR=路径`，构建profiling_workload_nvtx。默认关闭额外头文件依赖。
