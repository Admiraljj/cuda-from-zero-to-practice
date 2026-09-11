# 第 8 章：内存与资源管理实验

先读 [第 8 章正文](../../08-显存-拷贝-统一内存与RAII.md)。这里有 9 个小程序与 2 个头文件，按顺序学习，不需要一次理解所有封装。

## 1. 进入代码目录

从教材根目录执行：

```bash
cd code/08_memory
```

以下命令以 Linux 为例。-o 后面是输出程序名，./名称 才是执行。Windows 可以把输出改成 名称.exe，再用 PowerShell 的 .\名称.exe 运行；要先配置 CUDA Toolkit 与其支持的主机编译器。

## 2. 不需要 GPU 的两个程序

```bash
g++ -std=c++17 -Wall -Wextra byte_fill.cpp -o byte_fill
./byte_fill
g++ -std=c++17 -Wall -Wextra scope_trace.cpp -o scope_trace
./scope_trace
```

byte_fill 比较按字节填充与设置 float 数值，预期 bytes_all_one=1、value_is_one=0，随后 fill 的 value_is_one=1。scope_trace 打印正常离开作用域与异常展开时的构造/析构顺序。完整预期输出在正文。

## 3. CUDA 内存实验

下面每个程序单独编译，不要把所有 .cu 一次链接在一起：它们各自有 main，也可能使用相同的示例函数名。

```bash
nvcc -std=c++17 -lineinfo managed_scale.cu -o managed_scale
./managed_scale
nvcc -std=c++17 -lineinfo memory_capabilities.cu -o memory_capabilities
./memory_capabilities
nvcc -std=c++17 -lineinfo pinned_roundtrip.cu -o pinned_roundtrip
./pinned_roundtrip
nvcc -std=c++17 -lineinfo memory_snapshot.cu -o memory_snapshot
./memory_snapshot
nvcc -std=c++17 -lineinfo managed_prefetch.cu -o managed_prefetch
./managed_prefetch
```

| 程序 | 预期检查 | 条件与边界 |
|---|---|---|
| managed_scale | 10 个值 2、4、…、20 | 需要支持 managed 分配；CPU 读取前等待 GPU |
| memory_capabilities | 当前设备与 4 个能力值，查询成功标记 | 不能把查询成功当成性能或 kernel 验证 |
| pinned_roundtrip | 8 个值 10、11、…、17 | 单独的主机/设备分配；不是异步重叠实验 |
| memory_snapshot | before、allocated、released 快照 | 4 MiB 请求；不要求空闲差值精确等于请求 |
| managed_prefetch | 10 个值 3、6、…、30，或明确跳过 | 以 Toolkit 12+ 为目标；12.x/13.x 条件编译；能力不足返回 77 |

所有成功执行的普通示例返回 0。API 失败或结果不符返回 1。预取程序只有在成功查询后发现 managed 或 concurrent 能力不足时才返回 77，并打印 SKIP 行；不能把任意错误当作跳过。没有 GPU 时记录“未运行”。

## 4. RAII 与移动所有权

保留同目录中的 float_buffer.cuh 与 device_buffer.cuh，不要单独编译头文件。

```bash
nvcc -std=c++17 -lineinfo raii_scale.cu -o raii_scale
./raii_scale
nvcc -std=c++17 -lineinfo move_buffer.cu -o move_buffer
./move_buffer
```

raii_scale 应输出 8 个值 2 到 16。move_buffer 应显示 source_size=0 destination_size=4 same_address=1，取回 9、8、7、6；它没有 kernel，移动构造也没有复制 GPU 数组。

显式选择普通 C++ 异常分支：

```bash
./raii_scale --fail-after-allocation
echo $?
```

分配成功后，应报告 planned failure after allocation 并返回 1。这是预期失败，不是普通 PASS；它不提交 kernel，也不制造 GPU 越界。PowerShell 用 $LASTEXITCODE 查看返回码。

需要真实设备上的清理检查时可运行：

```bash
compute-sanitizer --tool memcheck --leak-check full ./raii_scale
compute-sanitizer --tool memcheck --leak-check full ./raii_scale --fail-after-allocation
compute-sanitizer --tool memcheck --leak-check full ./move_buffer
```

检查工具诊断与泄漏汇总，不仅看应用退出码。第二条应用故意返回非零，不应误认为出现了预期的 GPU 故障。

## 5. 批量构建

可选：在教材根目录使用 CMake，目标分别为 memory_byte_fill、memory_scope_trace、memory_managed_scale、memory_capabilities、memory_pinned_roundtrip、memory_snapshot、memory_managed_prefetch、memory_raii_scale、memory_move_buffer。

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build -R "^memory_" --output-on-failure
```

默认测试仅执行正常路径；预取的 77 在 CTest 中标为跳过，不计为成功执行。需要指定架构时请使用自己查询的值。

这些命令是给学习者的运行入口。当前案例尚未在本书编辑环境完成 C++/CUDA 编译与 GPU 实测；正文输出为预期值，不是性能测量结果。
