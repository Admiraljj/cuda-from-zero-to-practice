# 附录A：命令速查与使用边界

这里用于查找，不替代首次解释。终端命令不写进.cu文件；输入路径相对当前目录，修改源码后要重新构建。先确认运行的是哪个build目录的程序。

## A.1 环境与设备

```bash
nvidia-smi
nvidia-smi -L
nvidia-smi topo -m
nvcc --version
g++ --version
cmake --version
which nvcc
```

nvidia-smi报告驱动/GPU，不等于已安装对应Toolkit；nvcc --version报告编译工具，不等于运行时GPU可用。-L列设备，topo -m查拓扑；程序本地编号可能被CUDA_VISIBLE_DEVICES重排。详细见[第2章](02-检查GPU服务器与CUDA环境.md)、[第24章](24-多GPU入门.md)。

集群若用module，按管理员给定版本加载；不要照抄私有account/partition。无需在教材或报告公布服务器登录凭据。

## A.2 单文件：构建、运行、退出码

```bash
nvcc -std=c++17 hello.cu -o hello
./hello
echo $?
```

nvcc是编译驱动；-std=c++17选语言标准；hello.cu是输入；-o hello选输出。Linux用./hello，PowerShell常用.\hello.exe；输出名不是固定API。echo $?在Linux shell查看上一程序退出码，PowerShell用$LASTEXITCODE。首次拆解见[第3章](03-第一段CUDA程序与NVCC.md)。

```bash
nvcc -std=c++17 -O3 -lineinfo app.cu -o app
nvcc -std=c++17 -O0 -g -G app.cu -o app_debug
nvcc -std=c++17 -O3 -Xptxas=-v app.cu -o app_resources
```

-lineinfo提供device源码行定位，不打印变量；-g/-G用于host/device调试，不用于真实性能比较；-Xptxas=-v请求资源报告，具体字段见[第19章](19-Occupancy-寄存器与资源权衡.md)。-O选host优化级别，不是“自动检测GPU错误”。

## A.3 架构、库与多文件

```bash
nvcc -std=c++17 -arch=sm_80 app.cu -o app
nvcc -std=c++17 app.cu -lcublas -o app_cublas
g++ -std=c++17 -c main.cpp -o main.o
g++ -std=c++17 -c transform.cpp -o transform.o
g++ main.o transform.o -o app_cpu
```

sm_80仅在GPU/Toolkit支持相符时使用，不是所有NVIDIA卡都照抄80。-lcublas请求链接库，不安装cuBLAS。-c停在对象文件，最后一条链接定义与引用。

真实跨.cu device引用的示例见[第23章](23-CMake-项目结构-测试与可复现构建.md)：-dc/-rdc与device link不是普通host wrapper分文件的默认要求。

## A.4 全书与独立工程CMake

公开教材根目录CPU-only：

```bash
cmake -S . -B build-cpu -DBUILD_CUDA_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
```

-S源目录，-B构建目录；配置不是编译，build不是运行，CTest不是只看构建状态。单配置用CMAKE_BUILD_TYPE，多配置用--config/-C，详见第23章。

GPU构建加入经确认的CMAKE_CUDA_ARCHITECTURES。第23/24/25/26章另有独立工程，CUDA开关分别BOOK23/24/25/26_ENABLE_CUDA，默认关闭；第23章cuBLAS另有BOOK23_ENABLE_CUBLAS。不要用一个工程的变量猜另一个工程的行为。

```bash
ctest --test-dir build-cpu -C Release -N
ctest --test-dir build-cpu -C Release -R book26 --output-on-failure
cmake --build build-cpu --config Release --verbose
```

-N列测试不执行，-R按名称筛选，--verbose显示真实构建命令。passed/failed/skipped分别记录；“0 failed”不等于GPU已跑。

## A.5 Sanitizer与性能工具

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 ./app
compute-sanitizer --tool initcheck --error-exitcode 99 ./app
compute-sanitizer --tool racecheck --error-exitcode 99 ./app
compute-sanitizer --tool synccheck --error-exitcode 99 ./app
nsys profile --trace=cuda,osrt -o report ./app
nsys stats report.nsys-rep
ncu --set basic ./app
```

99使工具检出错误时失败；先memcheck，再按算法检查初始化/竞争/同步。工具无GPU/未安装/路径跳过不写通过。

Systems看整体队列/host/传输，Compute看指定kernel。匹配kernel、预热与launch筛选、报告读取见[第18章](18-Nsight性能分析与优化方法.md)。不要把profile干扰后的时间直接充当未profile正常基准。

## A.6 中间代码与调试：需要时再用

```bash
nvcc -std=c++17 -ptx kernel.cu -o kernel.ptx
nvcc -std=c++17 -cubin -arch=sm_80 kernel.cu -o kernel.cubin
cuobjdump --dump-sass app
nvdisasm kernel.cubin
cuda-gdb ./app_debug
```

PTX是虚拟指令表示，cubin/SASS面向实际架构。含host-only main的整个工程不一定适合作为-ptx输入；按工具文档选择device源。指令数少不等于更快，调试代码也不能代替Sanitizer/数值参考。

## A.7 记录真实命令，不共享缓存

使用[第26章记录模板](code/26_capstone/project_record_template.md)保存commit/平台/构建/测试/范围/未执行项目。CMakePresets共享可复现选择，CMakeUserPresets保存本地设置，不提交私人路径。清理仅针对已确认构建目录，不对源码/工作区根执行宽泛递归删除。
