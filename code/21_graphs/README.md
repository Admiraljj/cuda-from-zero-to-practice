# 第 21 章案例：CUDA Graphs

先读 [正文](../../21-CUDA-Graphs与重复工作流.md) 的数学与对象生命周期，再从 dag_model、graph_capture 逐步运行。接口基线为 CUDA Toolkit 12.0+；graph_update 使用该基线的三参数更新接口。下面不是已经在本机执行成功的声明。

所有命令从教材根目录执行。CPU 模型不需要 GPU：

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/21_graphs/dag_model.cpp -o dag_model
./dag_model 10
nvcc -std=c++17 -O2 -lineinfo code/21_graphs/graph_capture.cu -o graph_capture
nvcc -std=c++17 -O2 -lineinfo code/21_graphs/capture_failure.cu -o capture_failure
nvcc -std=c++17 -O2 -lineinfo code/21_graphs/graph_update.cu -o graph_update
nvcc -std=c++17 -O2 -lineinfo code/21_graphs/graph_fork_join.cu -o graph_fork_join
nvcc -std=c++17 -O2 -lineinfo code/21_graphs/graph_benchmark.cu -o graph_benchmark
```

| 程序 | 目的 | 参数默认值 |
|---|---|---|
| dag_model | 假想 DAG、提交次数、盈亏平衡 | repeats=10 |
| graph_capture | pinned 完整往返、标量记录与重放 | n=1031,repeats=3 |
| capture_failure | 正常捕获；显式协议故障与恢复 | safe |
| graph_update | 定义/exec、单节点与整图更新、重建 | n=1031 |
| graph_fork_join | 两分支捕获、origin 汇合 | n=1031 |
| graph_benchmark | 普通三阶段/Graph/融合三方法 | n=4096,repeats=1000,rounds=5 |

先用小数据验收，预期均 PASS；时间必须实测：

```bash
./graph_capture 0 1
./graph_capture 1 3
./graph_capture 255 3
./graph_capture 256 3
./graph_capture 257 3
./graph_capture 1031 3
./graph_update 1031
./graph_fork_join 1031
./capture_failure safe
./graph_benchmark 1031 10 3
./graph_capture -1 3
./graph_benchmark 10 0 3
```

最后两条应拒绝并返回非零。只有主动研究捕获错误时才运行：

```bash
./capture_failure invalid-sync
```

这个故障仅违反捕获同步协议，不制造越界或数据竞争；默认 CTest 不运行故障模式。C++/CUDA 程序对确定的二进制精确输入逐项验证；它们不是一般十进制误差比较模板。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target graph_dag_model graph_capture graph_update graph_fork_join capture_failure graph_benchmark
ctest --test-dir build -C Release -R "graph_|capture_failure_safe" --output-on-failure
compute-sanitizer --tool memcheck ./graph_capture 1031 3
compute-sanitizer --tool memcheck ./graph_update 1031
compute-sanitizer --tool memcheck ./graph_fork_join 1031
compute-sanitizer --tool initcheck ./graph_benchmark 1031 10 3
```

CMake 是可选批量入口；若需指定架构，增加经过设备查询确认的 CMAKE_CUDA_ARCHITECTURES。CPU 模型也可按首条 g++ 命令独立编译，不需要先构建所有 GPU 示例。服务器 GPU 实测未在编写主机完成。

性能实验先保存不带 profiler 的输出，再采集；检查本机 nsys 版本/帮助是否支持所列选项：

```bash
./graph_benchmark 4096 1000 5
nsys profile --trace=cuda --cuda-graph-trace=graph --sample=none --cpuctxsw=none -o graph_overview ./graph_benchmark 4096 1000 5
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o graph_nodes ./graph_benchmark 4096 100 3
nsys stats graph_overview.nsys-rep
```

Graph benchmark 不计 H2D/D2H/验证；最小 capture 程序则捕获完整往返。不要比较这两种不同范围的时间。node tracing 会改变开销；正文解释 CPU 提交、完成、Event、建立与上传五个范围。
