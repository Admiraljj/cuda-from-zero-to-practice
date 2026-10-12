# 第20章：异步、事件与有限缓冲流水线

正文：[Stream与流水线](../../20-Stream-Event-Pinned-Memory与流水线.md)。依次运行CPU chunk模型、单stream异步往返、双producer汇合、串行/流水线对照。

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/20_streams/chunk_model.cpp -o chunk_model
./chunk_model 23 8 2
nvcc -std=c++17 -O2 -lineinfo code/20_streams/async_roundtrip.cu -o async_roundtrip
nvcc -std=c++17 -O2 -lineinfo code/20_streams/event_dependency.cu -o event_dependency
nvcc -std=c++17 -O2 -lineinfo code/20_streams/stream_pipeline.cu -o stream_pipeline
./async_roundtrip 1031
./event_dependency 1031
./stream_pipeline 23 8 2 3 pinned
./stream_pipeline 1031 257 3 3 pinned
./stream_pipeline 1 8 3 3 pinned
./stream_pipeline 0 8 2 1 pinned
```

前两个GPU例子只接受n；pipeline参数n chunk streams rounds memory，默认4194304/262144/3/5/pinned。chunk是元素容量，不是bytes；最后count取min。n0跳过分配与启动，非法参数先拒绝。

有限slot包含两host缓冲、一device缓冲、worker stream和done。reuse前等D2H完成并消费旧结果；结束排空。Pool禁用复制，所有GPU工作完成后释放。每轮完整output/不可变input验证，单槽与多槽使用相同数学和内存模式。

```bash
./stream_pipeline 4194304 262144 3 5 pinned
./stream_pipeline 4194304 262144 3 5 pageable
nsys profile --trace=cuda --sample=none -o pipeline ./stream_pipeline 4194304 262144 3 3 pinned
compute-sanitizer --tool memcheck --error-exitcode 99 ./stream_pipeline 1031 257 3 1 pinned
cmake --build build --target chunk_model async_roundtrip event_dependency pipeline
ctest --test-dir build -R '^(chunk_model|async_roundtrip|event_dependency|pipeline)' --output-on-failure
```

CMake目标pipeline与nvcc输出stream_pipeline来自同一源码。host时间含staging/submit/reuse等待/gather，不含分配、原始生成与校验；GPU Event以control fork/join包含全部used slot最新done。logical GB/s是任务完成率，不冒充PCIe计数器。

query可能立即complete或not-ready；两者都合法。PASS说明数值和覆盖符合协议，是否重叠另看Systems时间线。
