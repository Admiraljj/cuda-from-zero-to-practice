# CUDA From Zero to Practice

一份用中文讲解、面向零基础读者的 NVIDIA CUDA 实践教材。

学习顺序是：先理解问题与数据，再手算线程分工，阅读完整程序，逐行理解新增语法，运行并验证结果，最后独立修改和练习。这不是速查表，也不是 NVIDIA 官方指南的逐页译本。

> 第 1—26 章主线教材已全部编写完成，附录与案例随书提供。
>
> 文中的示例输出为预期结果；案例尚未完成实机验证，请在自己的开发环境中编译运行并核对。

## 从这里开始

[完整阅读地图](00-阅读地图.md)提供学习顺序、能力关卡与复习入口；真正零基础从第1章开始。

1. [C++ 与 Linux 基础](01-你需要的C++与Linux基础.md)
2. [检查 GPU 与 CUDA 环境](02-检查GPU服务器与CUDA环境.md)
3. [第一段 CUDA 程序与 NVCC](03-第一段CUDA程序与NVCC.md)
4. [线程、Block、Grid、Warp 与 SIMT](04-线程-Block-Grid-Warp与SIMT.md)
5. [一维索引与向量运算](05-一维索引与向量运算.md)
6. [二维索引与图像网格](06-二维索引与图像网格.md)
7. [错误检查与 Compute Sanitizer](07-错误检查与Compute-Sanitizer.md)
8. [显存、拷贝、统一内存与 RAII](08-显存-拷贝-统一内存与RAII.md)
9. [正确计时与带宽基线](09-正确计时与带宽基线.md)
10. [Shared Memory 与线程同步](10-Shared-Memory与线程同步.md)
11. [并行归约：从手算到完整 GPU 归约](11-并行归约-从错误到高效.md)
12. [Warp 通信、掩码与分支](12-Warp原语与分支发散.md)
13. [原子操作与直方图](13-原子操作与直方图.md)
14. [Scan（前缀和）与流压缩](14-Scan前缀和与流压缩.md)
15. [矩阵转置与合并访存](15-矩阵转置与合并访存.md)
16. [矩阵乘法：从一个点积到 Tiled GEMM](16-矩阵乘法-朴素到Tiled-GEMM.md)
17. [二维卷积与 Stencil：手算、Halo 与可分离实现](17-卷积与Stencil.md)
18. [Nsight 性能分析：时间线、指标与可信优化](18-Nsight性能分析与优化方法.md)
19. [Occupancy、寄存器与资源权衡](19-Occupancy-寄存器与资源权衡.md)
20. [Stream、Event、Pinned Memory 与有限槽位流水线](20-Stream-Event-Pinned-Memory与流水线.md)
21. [CUDA Graphs：捕获、参数更新与可复用工作流](21-CUDA-Graphs与重复工作流.md)
22. [Thrust、CUB、cuBLAS：正确调用库与建立基线](22-Thrust-CUB-cuBLAS与库优先原则.md)
23. [CMake、项目结构、测试与可复现构建](23-CMake-项目结构-测试与可复现构建.md)
24. [多GPU入门：归属、分片、依赖与传输](24-多GPU入门.md)
25. [综合项目：一步一步写出稳定GPU Softmax](25-综合项目-GPU-Softmax.md)
26. [毕业工程、独立项目与CUDA学习路线](26-从初学者到CUDA工程师的路线.md)

配套：[练习答案（第 1—26 章）](附录-C-练习答案.md)、[命令](附录-A-命令速查.md)、[诊断](附录-B-常见错误诊断.md)、[资料](附录-D-资料与继续学习.md)、[术语](附录-E-术语表.md)

第 16 章包含 23 节、6 个完整程序、2 张本地暗色图和 24 道练习：从 shape/点积/线性下标，逐步写 naive、shared tiled GEMM，再推导 cuBLAS 布局、精度与统一 benchmark。命令入口见[第 16 章案例目录](code/16_matmul/README.md)。

第 17—20 章新增 87 节、10 个完整程序、1 个数学参考头文件、5 张暗色图和 96 道练习及详细答案。路线是卷积四方法对照 → Systems/Compute 分析 → 资源预算与启动扫描 → 异步往返、事件汇合与串行/流水线对照。各章给出边界测试与可执行的服务器验收命令。

第 21 章包含 26 节、6 个完整程序、2 个辅助头文件、2 张暗色图和 24 道练习及详细答案。先用 CPU 手算依赖和摊销，再学习完整往返 capture、参数值与内容、失败恢复、单节点/整图更新与安全重建、跨 stream fork/join，以及普通排队/Graph/融合的公平对照。命令见[第 21 章案例目录](code/21_graphs/README.md)。

第 22 章包含 31 节、8 个完整程序、2 个辅助头文件、3 张暗色图和 28 道练习及详细答案。从 CPU 契约到 Thrust 容器/范围/函数对象/策略，再到 CUB 归约、Scan、稳定筛选与顺序工作区复用，最后验证 cuBLAS stride、两种标量模式、非零 beta 的 padded 非方阵 GEMM，以及四方法公平对照。命令见[第 22 章案例目录](code/22_libraries/README.md)。

第 23 章包含 32 节、4 个独立递进阶段、25 份完整源码/头文件/构建配置、3 张暗色图和 32 道练习及详细答案。从单文件 CPU 程序开始，逐步拆分声明与定义、建立 CMake 目标与可选 CUDA/cuBLAS 后端，再学习真正的跨 .cu device link、Presets、CTest 与可复现记录。CPU-only 工程无需 CUDA；跳过 GPU 不冒充 GPU 验证通过。命令见[第 23 章案例目录](code/23_project/README.md)。

第 24—26 章新增100节、31份完整源码/头文件/构建配置、8张本地暗色SVG、88道练习与逐题答案。第24章从CPU分片到双卡/peer/中转；第25章用五个完整阶段建立Softmax，比较serial/block/窄行warp；第26章提供完整CUB筛选后平方和工程与三个独立毕业项目的规格和验收。命令分别见[多GPU案例](code/24_multi_gpu/README.md)、[Softmax工程](code/25_softmax_walkthrough/README.md)、[毕业工程](code/26_capstone/README.md)。

第 1 章只需要 C++17 编译器，不需要 GPU。第 2 章起以 Linux、NVIDIA GPU、CUDA Toolkit 为主线；环境检查和命令解释放在正文中。请按顺序阅读，不必预先掌握指针、warp、SIMT 或编译选项。

## 每章包含什么

- 从小问题出发的概念解释，以及必要的 C++ 预备知识。
- 表格、手算示意或 Mermaid 图，帮助理解地址、线程和执行顺序。
- 完整源文件、命令拆解、预期输出、错误诊断和验证方法。
- 从预测结果、修改代码到独立实现的练习与参考答案。

代码按章节放在 `code/`，直接按照正文中的 `g++`、`nvcc` 命令编译运行。第 7 章包含故意错误的教学案例：默认测试不执行错误模式，启用前请阅读该章的保护开关与风险说明。

## 参考资料

技术事实以 NVIDIA 官方资料为主要依据，具体引用放在相应章节中：

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)：执行模型、SIMT、线程层级和内存模型。
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)：接口参数、返回值与行为边界。
- [NVCC 编译器文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)：编译、链接和命令选项。
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：正确性、访存与性能分析。
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) 与 [CUDA-GDB](https://docs.nvidia.com/cuda/cuda-gdb/index.html)：错误诊断和调试。
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)：矩阵计算、布局与精度接口。
- [NVIDIA CCCL](https://github.com/NVIDIA/cccl)：Thrust、CUB 与 CUDA C++ 核心计算库；各章引用匹配版本的接口。

## 下载与阅读

在仓库页面点击 **Code → Download ZIP**，解压后阅读 Markdown；也可以使用 Git：

```bash
git clone https://github.com/Admiraljj/cuda-from-zero-to-practice.git
cd cuda-from-zero-to-practice
```

第一条命令下载仓库，第二条进入教材目录。后续章节中的相对路径与编译命令默认从这个目录出发，除非正文另有说明。

### 在 Obsidian 中阅读

选择“打开文件夹作为仓库”，指向解压或克隆后的本目录。项目提供最小化的暗色外观配置与 `vscode-dark.css`，采用深灰背景和克制的代码配色，不包含个人工作区布局或打开的标签。

如果已有自己的 Obsidian 仓库，只复制教材与 CSS：在“设置 → 外观”选择暗色模式，并启用 CSS 代码片段 `vscode-dark`。GitHub 网页的配色由 GitHub 自己的主题设置决定，Obsidian CSS 不会改变 GitHub 页面。

## 编译与练习

先按各章说明编译程序，预测输出，再逐项核对实际结果。CPU 案例可以单独编译；附带的 CMake 工程也支持关闭 CUDA：

```bash
cmake -S . -B build-cpu -DBUILD_CUDA_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
```

构建 CUDA 示例时，在第 2 章确认工具链后使用新的构建目录：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

需要指定 GPU 架构时，在配置命令中添加经设备查询确认的 `-DCMAKE_CUDA_ARCHITECTURES=数字`，不要照抄不适合自己 GPU 的架构号。CMake 是可选的批量构建入口；初学时先执行正文的单文件命令。

第23/26章根子工程默认CPU-only；启用其GPU后端分别添加BOOK23_ENABLE_CUDA=ON、BOOK26_ENABLE_CUDA=ON（第23章cuBLAS另加BOOK23_ENABLE_CUBLAS=ON），并指定核对后的架构。第24/25章GPU例随BUILD_CUDA_EXAMPLES构建；各章也有独立CPU-only/CUDA工程入口。

## 完成主线之后

主线从C++零基础到独立CUDA项目共26章，已全部提供详细讲解、源码、图与答案。这不是CUDA全部高级特性的逐页译本；Driver API、NCCL、Tensor Core等按项目需求继续，入口见资料附录。

编写与静态检查完成不代表实机验收。当前编辑环境没有C++/CUDA/CMake工具链与GPU，不能声称示例实际编译、CTest/GPU/Sanitizer/profiler或性能已通过；请在兼容开发环境按章验收，保留真实失败/跳过/未执行记录。
