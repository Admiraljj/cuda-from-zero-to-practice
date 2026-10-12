# 第17章：二维滤波案例

正文：[卷积与Stencil](../../17-卷积与Stencil.md)。从仓库根目录执行以下命令；CPU数学和GPU程序共用本目录filter_reference.h。

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/17_convolution/filter_model.cpp -o filter_model
./filter_model gaussian zero corr
./filter_model gaussian clamp corr
./filter_model sobel zero conv
nvcc -std=c++17 -O2 -lineinfo code/17_convolution/convolution.cu -o convolution
./convolution 37 29 gaussian zero 1 corr 5
./convolution 37 29 asymmetric clamp 1 conv 5
./convolution 17 15 box zero 4 corr 3
```

参数顺序width height filter boundary radius operation rounds；范围/拒绝规则在正文和Usage中。程序比较global-filter、constant-filter、shared，只有可分离filter增加horizontal+vertical路径。corr不翻转、conv翻转180°；zero和clamp是不同数学任务。

默认131×97、gaussian/zero/R1/corr；每次输出NaN sentinel、完整double oracle、有限值与误差预算、每轮input不变性。Event包括本方法全部kernel，separable不能只测垂直；输出像素率，不伪造GPU时间。

CPU固定5×5模型：Gaussian中心13，zero角落1.6875，clamp角落2.5；Sobel中心corr8/conv−8；非对称中心corr105/conv103。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --target filter_model conv2d
ctest --test-dir build -R '^(filter_model|conv2d)' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 ./convolution 17 15 gaussian zero 1 corr 1
compute-sanitizer --tool racecheck --error-exitcode 99 ./convolution 17 15 box clamp 4 corr 1
compute-sanitizer --tool synccheck --error-exitcode 99 ./convolution 17 15 gaussian zero 1 corr 1
```

80替换为已核对架构。CMake目标名conv2d，直接nvcc例子输出名convolution，两者来自同一源码。支持不足返回77，实际GPU执行仍需目标机器验收。
