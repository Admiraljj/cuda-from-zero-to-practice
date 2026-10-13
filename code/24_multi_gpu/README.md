# 第24章案例：归属、分片与跨卡传输

先读[正文](../../24-多GPU入门.md)。以下是读者验收命令，不是已实机运行的记录。

从教材根目录运行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/24_multi_gpu/partition_model.cpp -o partition_model
./partition_model 7 2
./partition_model 1 8
cmake -S code/24_multi_gpu -B build-book24-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-book24-cpu --config Release
ctest --test-dir build-book24-cpu -C Release --output-on-failure
nvcc -std=c++17 -O2 -lineinfo code/24_multi_gpu/device_inventory.cu -o device_inventory
nvcc -std=c++17 -O2 -lineinfo code/24_multi_gpu/multi_gpu_scale.cu -o multi_gpu_scale
nvcc -std=c++17 -O2 -lineinfo code/24_multi_gpu/transfer_pipeline.cu -o transfer_pipeline
./device_inventory
./multi_gpu_scale 7 1
./multi_gpu_scale 1031 2
./transfer_pipeline 1031 staged
./transfer_pipeline 1031 peer
./transfer_pipeline 1031 auto
```

GPU独立CMake：添加 BOOK24_ENABLE_CUDA=ON 与经核对的 CMAKE_CUDA_ARCHITECTURES，详见正文；两卡不同架构时按第23章设置合法列表。默认CPU-only无需CUDA。全书根工程仅注册对应目标，不重复add_subdirectory本章独立工程。

正常为0，真实失败1，CLI拒绝2，设备/所要求peer能力不可用77。空n没有GPU执行；没有双卡/peer时跳过，不能视为该GPU路径验证通过。auto仅对成功查询的能力缺失选择中转，真实enable/copy错误不隐藏。

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 ./multi_gpu_scale 1031 2
compute-sanitizer --tool memcheck --error-exitcode 99 ./transfer_pipeline 1031 staged
compute-sanitizer --tool memcheck --error-exitcode 99 ./transfer_pipeline 1031 peer
nsys profile --trace=cuda,osrt -o book24_peer ./transfer_pipeline 1031 peer
```

分片程序时间只是一次诊断样本，包含sentinel/输入快照，不是预热后的公平benchmark。传输程序验证3x中间值、6x+1最终值与原输入不变；不把peer能力查询、peer调用完成和拓扑带宽当成同一证据。
