# 第 10 章配套程序

先阅读根目录的第 10 章正文。九份源码各自是一个独立程序，不要把所有源文件链接到同一个可执行文件。

从教材根目录编译；只有两个 CPU 模型可以在没有 CUDA 的机器上编译运行。

## CPU 模型

```bash
g++ -std=c++17 -O2 code/10_shared/tile_model.cpp -o tile_model
g++ -std=c++17 -O2 code/10_shared/bank_model.cpp -o bank_model
./tile_model
./bank_model
```

第一个展示邻域槽位，第二个展示32-bit标量读取的bank地址模型。都不证明GPU同步或硬件事务行为。

## CUDA 分阶段案例

```bash
nvcc -std=c++17 -lineinfo code/10_shared/stencil_naive.cu -o stencil_naive
nvcc -std=c++17 -lineinfo code/10_shared/stencil_static.cu -o stencil_static
nvcc -std=c++17 -lineinfo code/10_shared/stencil_dynamic.cu -o stencil_dynamic
nvcc -std=c++17 -lineinfo code/10_shared/reuse_tile.cu -o reuse_tile
nvcc -std=c++17 -lineinfo code/10_shared/stencil_radius.cu -o stencil_radius
nvcc -std=c++17 -lineinfo code/10_shared/stencil1d.cu -o stencil1d
nvcc -std=c++17 -lineinfo code/10_shared/shared_resources.cu -o shared_resources
./stencil_naive
./stencil_static
./stencil_dynamic
./reuse_tile
./stencil_radius --matrix
./stencil1d 65 4 32
./shared_resources
```

动态半径1版本包含52组配置；可变半径矩阵包含112组，分别含4和16个空案例。资源查询程序不启动kernel。

`stencil_radius [n radius threads]` 默认10 2 4，参数分别为长度、半径、block线程数。允许n=0..1048576，radius=0..64，threads=1..1024，之后检查设备限制。

`stencil1d [n steps threads]` 默认10 3 4，参数分别为长度、平均轮数、block线程数。允许steps=0..32；零填充边界始终固定除以3。逐轮CPU参考与下载检查不是性能基准。

## 正确性与边界

```bash
./stencil_radius 0 2 4
./stencil_radius 1 8 1
./stencil_radius 17 8 4
./stencil1d 0 3 4
./stencil1d 10 0 4
./stencil1d 1 3 4
./stencil1d 1000 5 256
compute-sanitizer --tool memcheck ./stencil_radius --matrix
compute-sanitizer --tool racecheck ./stencil_radius --matrix
compute-sanitizer --tool synccheck ./stencil_radius --matrix
compute-sanitizer --tool racecheck ./reuse_tile
compute-sanitizer --tool synccheck ./reuse_tile
compute-sanitizer --tool memcheck ./stencil1d 65 4 32
compute-sanitizer --tool racecheck ./stencil1d 65 4 32
compute-sanitizer --tool synccheck ./stencil1d 65 4 32
```

参数拒绝测试：`./stencil_radius 17 -1 4`、`./stencil_radius 17 2 0`、`./stencil1d 10 33 4`应返回2。它们在主机拒绝无效参数，不会故意提交错误kernel。默认构建与测试只运行正确示例。

## CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target shared_tile_model shared_bank_model shared_naive shared_static shared_dynamic shared_reuse shared_radius stencil1d shared_resources
ctest --test-dir build -C Release -R "^(shared_|stencil1d)" --output-on-failure
```

本文命令和正文中的输出是学习及验收入口。当前案例尚未完成实际C++/CUDA编译、GPU执行与Compute Sanitizer验收，不能把静态检查或纸笔推导当成运行结果。
