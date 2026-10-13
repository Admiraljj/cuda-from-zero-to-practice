# 第25章Softmax：先运行五阶段，再运行完整工程

完整解释见[第25章](../../25-综合项目-GPU-Softmax.md)。下面是读者验收入口，不是实机结果。CPU-only先用CMake构建三个CPU阶段和参考/验证器测试：

```bash
cmake -S code/25_softmax_walkthrough -B build-book25-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-book25-cpu --config Release
ctest --test-dir build-book25-cpu -C Release --output-on-failure
```

GPU独立工程添加BOOK25_ENABLE_CUDA=ON与核对后的CMAKE_CUDA_ARCHITECTURES；NVIDIA GPU/Toolkit/host compiler要求见第2章，架构列表见第23章。

手动构建最终程序，从教材根目录：

```bash
nvcc -std=c++17 -O3 -lineinfo -Icode/25_softmax_walkthrough/include code/25_softmax_walkthrough/src/main.cpp code/25_softmax_walkthrough/src/softmax_reference.cpp code/25_softmax_walkthrough/src/softmax.cu -o softmax
./softmax 2 3 mixed block 256 1
./softmax 7 31 mixed compare 128 5
./softmax 7 257 negative compare 64 5
./softmax 257 1003 extreme block 256 3
compute-sanitizer --tool memcheck --error-exitcode 99 ./softmax 7 257 mixed block 64 1
compute-sanitizer --tool racecheck --error-exitcode 99 ./softmax 7 257 negative block 128 1
compute-sanitizer --tool synccheck --error-exitcode 99 ./softmax 7 31 mixed compare 128 1
compute-sanitizer --tool initcheck --error-exitcode 99 ./softmax 7 257 extreme block 256 1
```

参数依次rows、cols、input pattern、method、block_threads、samples。正维度有限输入，预算4194304项；block线程为32..512的2次幂且设备支持，warp仅宽度<=32，serial/warp固定布局不由block_threads控制。compare宽行明确排除warp。

0=完成验证，1=真实失败，2=非法CLI，77=已定义不可用/未支持。CPU参考测试不能冒充GPU通过，GPU跳过也不等于数值通过。

最终main预检/预热后轮换方法，每样本10次相同输入的kernel调用，Event与host-complete计时都排除分配/H2D/D2H/验证。所有样本检查元素误差、非负/finite、行和、guard与device input snapshot。不直接填写未经实测的性能排名。
