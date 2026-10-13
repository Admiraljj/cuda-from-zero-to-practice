# 第23章：四个完整工程阶段

先读 [正文](../../23-CMake-项目结构-测试与可复现构建.md)。本章程序是正确性/工程案例，不是性能benchmark；本机没有C++/CUDA/CMake工具链，下面是读者执行入口，非实测声明。

所有命令从教材根目录执行，除非明确要求进入某阶段。CPU模型不需GPU：

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/23_project/01_single_cpu/main.cpp -o book23_single
./book23_single
g++ -std=c++17 -O2 -Wall -Wextra code/23_project/02_split_cpu/main.cpp code/23_project/02_split_cpu/transform.cpp -o book23_split
./book23_split
cmake -S code/23_project/02_split_cpu -B build-book23-split -DCMAKE_BUILD_TYPE=Release
cmake --build build-book23-split --config Release
ctest --test-dir build-book23-split -C Release --output-on-failure
cmake -S code/23_project/03_project -B build-book23-cpu -DCMAKE_BUILD_TYPE=Release -DBOOK23_ENABLE_CUDA=OFF -DBOOK23_ENABLE_CUBLAS=OFF
cmake --build build-book23-cpu --config Release
ctest --test-dir build-book23-cpu -C Release --output-on-failure
./build-book23-cpu/book23_app 1031 cpu 7
./build-book23-cpu/book23_app 257 cuda 7
```

最后一条应NOT RUN、exit77；CPU-only的正n GPU tests会skip，不等于GPU通过。CLI非法参数应exit2+Usage：

```bash
./build-book23-cpu/book23_app -1 cpu 7
./build-book23-cpu/book23_app 10 unknown 7
./build-book23-cpu/book23_app 10 cpu -1
```

最低CMake3.24/C++17。以下80仅用于已经核对compute capability8.0且nvcc支持sm_80的设备；否则先替换：

```bash
cmake -S code/23_project/03_project -B build-book23-cuda -DCMAKE_BUILD_TYPE=Release -DBOOK23_ENABLE_CUDA=ON -DBOOK23_ENABLE_CUBLAS=OFF -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-book23-cuda --config Release --verbose
ctest --test-dir build-book23-cuda -C Release --output-on-failure
cmake -S code/23_project/03_project -B build-book23-cublas -DCMAKE_BUILD_TYPE=Release -DBOOK23_ENABLE_CUDA=ON -DBOOK23_ENABLE_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-book23-cublas --config Release
ctest --test-dir build-book23-cublas -C Release --output-on-failure
cmake -S code/23_project/04_device_link -B build-book23-link -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-book23-link --config Release --verbose
ctest --test-dir build-book23-link -C Release --output-on-failure
```

阶段三是同步host wrapper，普通host调用跨文件不自动需要RDC；阶段四才是真正跨.cu的device函数。阶段四手动命令应在其目录执行，仍只用核对过的架构：

```bash
nvcc -std=c++17 -O2 -lineinfo -rdc=true --cudart=shared -arch=sm_80 main.cpp launch.cu device_math.cu -o book23_device_link
./book23_device_link
```

Ninja preset在阶段三目录执行：

```bash
cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset cpu-release
export BOOK23_CUDA_ARCHS=80
cmake --preset cublas-release
cmake --build --preset cublas-release
ctest --preset cublas-release
```

没有Ninja时用前面的默认generator命令。PowerShell设置环境用$env:BOOK23_CUDA_ARCHS="80"。Windows多配置产物常在Release/下，实际路径看generator；不要误运行另一个build tree的旧程序。

```bash
ctest --test-dir build-book23-cpu -C Release -N
ctest --test-dir build-book23-cpu -C Release -L cpu --output-on-failure
ctest --test-dir build-book23-cublas -C Release -L gpu --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 ./build-book23-cuda/book23_app 1031 cuda 7
compute-sanitizer --tool memcheck --error-exitcode 99 ./build-book23-cublas/book23_app 1031 cublas 7
```

共享源码/Presets与[运行记录模板](run_record_template.md)，不共享build/cache、CMakeUserPresets或账号信息。全书根构建默认将阶段三按CPU-only集成；单独学习本章优先使用上述独立工程入口。
