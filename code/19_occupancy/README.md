# 第19章：资源预算与配置扫描

正文：[Occupancy与资源权衡](../../19-Occupancy-寄存器与资源权衡.md)。resource_model是合成SM粗略预算；resource_sweep用真实设备/函数属性与Occupancy API，并验证每个支持候选的输出。

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/19_occupancy/resource_model.cpp -o resource_model
./resource_model 256 64 32768
./resource_model 256 64 16384
./resource_model 256 32 16384
nvcc -std=c++17 -O2 -lineinfo -Xptxas=-v code/19_occupancy/resource_sweep.cu -o resource_sweep
./resource_sweep 1031 1 1
./resource_sweep 1048576 5 8
```

CPU模型默认预测3blocks/24warps/37.5%，不含硬件分配粒度。GPU参数n rounds repeats，默认1048576/5/8，扫描Items1/4、B64/128/256/512、dynamic shared0/16/32KiB。

输出真实regs/local、预测blocks/SM/occupancy、grid和时间。预测不是achieved；动态shared为教学资源探针，0/非零还改变屏障路径；Items也改变地址布局与每thread工作。正文解释实验混杂，不能仅凭时间唯一归因occupancy。

```bash
cmake --build build --target resource_model resource_sweep
ctest --test-dir build -R '^(resource_model|resource_sweep|resource_reject)' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 ./resource_sweep 1031 1 1
compute-sanitizer --tool synccheck --error-exitcode 99 ./resource_sweep 1031 1 1
```

unsupported候选打印SKIP，全部无候选返回77；n0不启动。先小尾部，再较大输入，保存完整命令、资源与普通计时，profiler采到的配置必须另核对。
