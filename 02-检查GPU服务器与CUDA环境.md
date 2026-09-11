# 02 检查 GPU 服务器与 CUDA 环境

> 本章任务不是安装一切，而是弄清你已经有什么、还缺什么，并用自己的小程序验证到哪一步。
>
> 阅读前提：会完成第 1 章的普通 C++ 编译、运行；理解变量、函数返回值和取地址。GPU 硬件原理不需要预先掌握。

## 2.0 本章的终点：一份可信的环境记录

刚开始学 CUDA 时，人们容易把这些问题混成一句“CUDA 装好了吗”：

- 这台机器有 NVIDIA GPU 吗？
- 操作系统的驱动能访问它吗？
- 当前终端能找到编译器吗？
- 编译器支持我的主机系统和 GPU 吗？
- 当前这个作业或容器有权限用到 GPU 吗？
- 我的 CUDA 程序能否真正执行一次计算？

这是不同层次的问题。一个工具正常，不代表所有层次都正常。本章会依次获得证据，最后写出这样的结论：

> 我在某台计算节点、某个作业环境中，使用这个路径的 nvcc 编译了探针；程序成功枚举可见 CUDA 设备并查询属性。当前还没有测试 kernel 的实际执行，那是下一章的任务。

这比“nvidia-smi 有个表，所以应该都好了”精确得多。

本章有三个完整案例：

| 案例 | 新增能力 | 尚未证明什么 |
|---|---|---|
| `device_count.cu` | 查询当前进程可见的 CUDA 设备数 | 没有启动 kernel |
| `device_info.cu` | 查询每个设备的属性并检查错误 | 没有验证显存分配或计算 |
| `cuda_versions.cu` | 区分 Runtime 和驱动报告的 CUDA 版本 | 版本号不是 GPU 可用性测试 |

配套文件在 `code/02_environment/`。不要求读者先理解错误检查宏、RAII 或复杂项目结构；本章直接写出每一步调用和判断。

## 2.1 五样东西：GPU、驱动、Toolkit、Runtime、主机编译器

### 2.1.1 GPU 是执行设备，不是编译器

NVIDIA GPU 是硬件。你编写的设备代码最终需要以 GPU 支持的形式执行。硬件有自己的计算能力、显存容量和资源限制；安装一个更新的 Toolkit，并不会把旧 GPU 变成新架构 GPU。

你不必现在就记住 SM、warp、Tensor Core 的关系。暂时把 GPU 看作“主机程序能够提交计算工作的另一个执行设备”，到第 4 章再展开硬件与线程映射。

### 2.1.2 驱动使程序能够使用硬件

驱动是系统与 GPU 之间的软件支持，不是某个 `.cu` 源文件。Linux 上既有内核侧组件，也有供用户程序调用的组件。

我们通常把 NVIDIA 驱动当作管理员维护的系统设施。普通学习程序不应该尝试自行加载内核模块、重置 GPU 或杀掉别人的 GPU 进程。

### 2.1.3 Toolkit 是开发工具集合

CUDA Toolkit 包含 CUDA 开发所需的编译工具、头文件、开发库和其他工具。最常遇到的名字是：

```text
nvcc                 协调 CUDA C++ 编译的命令
cuda_runtime.h       本书调用 Runtime API 时包含的头文件
CUDA Runtime         提供本书常用 cuda... 接口的运行时支持
开发库和诊断工具     例如后面会学习的库、检查器和分析工具
```

具体安装内容可能随安装方式而不同。某个环境只有运行既有程序需要的组件，不一定包含用于开发的 nvcc 和头文件。

### 2.1.4 Runtime API 是程序调用的接口

API 可以先理解成“别人已经提供好、你按约定调用的功能入口”。本章调用 `cudaGetDeviceCount` 来问设备数量，后面调用 `cudaMalloc` 来申请设备内存。

CUDA Runtime API 的名字通常以 `cuda` 开头。另有更底层的 CUDA Driver API，名字常以 `cu` 开头。先以 Runtime API 学会计算流程，不要求本章同时掌握两套接口。

### 2.1.5 为什么学 GPU 还需要 g++

典型 CUDA 应用仍然包含很多 CPU 代码：准备输入、分配资源、提交工作、检查错误和处理结果。nvcc 会协调设备代码和主机代码的编译，Linux 上通常需要一个受当前 Toolkit 支持的主机 C++ 编译器。

因此，第 1 章的 g++ 并没有被“GPU 编译器完全替代”。它在构建流程里还有工作。

```text
开发阶段：
你的 .cu 源码
   ├── 主机代码 ── 主机 C++ 编译器 ──┐
   └── 设备代码 ── CUDA 编译工具 ────┤
                                     ↓
                                可执行程序

运行阶段（概念图，不是全部内部调用细节）：
CPU 上运行的主机程序
          ↓ 调用 CUDA Runtime
        驱动支持
          ↓ 提交/管理 GPU 工作
       NVIDIA GPU
```

开发工具链需要支持当前操作系统、主机编译器和目标设备；安装前应核对所选 Toolkit 对应的支持范围。见 [NVIDIA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)。

## 2.2 先确认“我现在在哪”，再检查 GPU

### 2.2.1 本机、登录节点、计算节点可能是三台机器

如果你通过 SSH 使用集群，可能经历：

```text
你的电脑
   ↓ SSH
集群登录节点：编辑文件、提交作业
   ↓ 申请或进入 GPU 作业
计算节点：在获分配的 GPU 上执行程序
```

你在登录节点上看不到 GPU，不一定是整个集群没有 GPU。不要因此在登录节点安装驱动。

执行这些只读命令：

```bash
hostname
pwd
uname -m
head -n 8 /etc/os-release
```

- `hostname`：当前所在机器。
- `pwd`：当前工作目录。
- `uname -m`：主机架构，例如 x86_64 或 aarch64；它不是 GPU Compute Capability。
- `/etc/os-release`：Linux 发行版信息；这里读前几行就够辨认。

请把记录写在自己的学习日志中。发给别人求助时，可以隐藏内网机器名、账号和私有路径，但保留排错需要的工具版本与错误信息。

### 2.2.2 有 GPU 也不代表它现在归你使用

多用户服务器上，当前 GPU 可能属于其他人的作业。看见空闲显存或低利用率，不是获得使用权限的凭证。

本章只查询信息，后续编译和执行实验也应在你获准使用的节点、容器或作业内进行。不要通过改变可见设备列表来访问调度器没有分给你的卡。

## 2.3 第一层检查：nvidia-smi 到底告诉了你什么

先在有权限的环境执行：

```bash
nvidia-smi
```

它是 NVIDIA 的管理与监控工具，通常显示 GPU 列表、版本信息、显存和利用率等。它通过自己的管理路径获取状态，并不是在背后替你编译了一个 CUDA kernel。

### 2.3.1 不必逐格背表，先看五类信息

| 信息 | 先用来回答的问题 | 不应由此推断 |
|---|---|---|
| GPU 名称 | 当前工具看到了什么设备 | 名称相似就有完全相同资源 |
| 驱动相关版本 | 系统使用什么驱动组件 | 它等于 nvcc 版本 |
| CUDA 相关版本 | 驱动报告支持的 CUDA 版本 | 当前 PATH 下有对应 Toolkit |
| 显存已用/总量 | 当前可观察到的显存情况 | 剩余空间都可以由你随意占用 |
| 进程与利用率 | 采样时是否观察到工作 | 0% 表示从来没有执行 kernel |

旧版常见表头是 `Driver Version` 和 `CUDA Version`。较新版本的文档还使用 `KMD Version`、`CUDA UMD Version` 等更具体的名称。名称变化不改变本章最重要的判断：**这里的 CUDA 版本信息属于驱动侧报告，不能当作当前 nvcc 的版本。** 见 [NVIDIA-SMI 官方说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。

### 2.3.2 显存容量和利用率不是同一种“大小”

例如你看到大约 24 GiB 总显存，这是能存放数据等资源的容量量级，不是“GPU 每秒能算多少”。

利用率是某种采样统计。一个只运行很短时间的小 kernel，可能在你手工刷新工具时早已结束；没有看到明显利用率不等于程序没跑。第 9 章会用事件计时验证工作时间，而不是靠监控表跳一下来判断成功。

### 2.3.3 输出太宽时，先列设备名字

```bash
nvidia-smi -L
```

`-L` 请求列出 GPU 信息，常能看到 UUID。UUID 是设备标识，可帮助跨不同编号映射识别同一个设备。这里先保存它，不要求现在做多 GPU 编程。

若命令不存在或执行失败，把完整错误保留下来。例如“命令不存在”和“驱动无法通信”是不同的问题。不要一律归结为“CUDA 没装”。

> **检查点 A**
>
> 请用自己的话解释：nvidia-smi 能显示 GPU，但 nvcc 仍可能不存在，为什么这两件事不矛盾？

## 2.4 第二层检查：终端到底找到了哪一个 nvcc

依次执行：

```bash
command -v nvcc
nvcc --version
command -v g++
g++ --version
```

`command -v` 问 shell 会从哪里找到这个命令。`--version` 让该工具报告自己的版本。它们解决的是两个不同问题：**位置**和**版本**。

### 2.4.1 PATH 是搜索列表，不是显卡配置

终端根据环境中的 `PATH` 搜索可执行命令。如果两个 Toolkit 的 bin 目录都在 PATH，顺序会影响你实际调用的 nvcc。

所以：

```text
磁盘上存在 Toolkit A 和 Toolkit B
                  ↓
当前 PATH 首先命中 Toolkit B 的 nvcc
                  ↓
本次 nvcc --version 报告 B，而不是“系统所有 CUDA 的统一版本”
```

在常见 Linux 安装中，`/usr/local/cuda` 还可能是指向某个版本目录的符号链接。但这不是所有服务器的固定安装位置。

确认 `command -v nvcc` 已找到文件路径后，可以复制实际路径查询：

```bash
readlink -f /actual/path/to/nvcc
```

`/actual/path/to/nvcc` 必须换成你查到的真实路径，不是可照抄的目录。熟悉 shell 后也可以组合成：

```bash
readlink -f "$(command -v nvcc)"
```

`$(...)` 先运行里面的命令，把其输出作为外层参数；双引号让路径作为一个整体参数。仅在前一步确实找到 nvcc 后使用它，不要拿空结果继续猜。

### 2.4.2 Toolkit 版本和 C++ 标准版本没有对应编号关系

`nvcc --version` 可能报告 CUDA 12.x 或 13.x。`-std=c++17` 里的 17 是 C++ 语言标准编号。它们属于不同维度：

```text
C++17：源码使用哪套语言规则
CUDA Toolkit 版本：本次使用哪套开发工具
Compute Capability：GPU 提供哪类计算架构能力
驱动版本：系统 GPU 支持软件的版本
```

不要把 `-std=c++17` 改成 `-std=c++13` 来“匹配 CUDA 13”。

### 2.4.3 找不到 nvcc，先排查环境，不先安装

可能的情况包括：Toolkit 尚未提供、需要加载模块、当前容器只有运行环境、PATH 未包含正确目录。

本章不让你从网上复制一条管理员安装命令。先确认站点已有方案。若是自己维护的专用机器，安装也是一项单独的系统配置工作：按操作系统、GPU 和所选 Toolkit 的官方要求选择组合，然后回到本章验证，而不是“安装命令退出成功就结束”。

## 2.5 模块系统、容器和调度器：环境可能需要进入才看得到

这些不是每个人都必学的工具。只阅读与你实际环境对应的小节。

### 2.5.1 服务器告诉你使用 module

一些系统用 Environment Modules 或类似机制切换软件版本。先查询：

```bash
type module
module avail cuda
module list
```

`module avail cuda` 查可用条目，`module list` 查已加载条目。假设站点确实列出了 `cuda/12.6`，并推荐你用它，才执行：

```bash
module load cuda/12.6
command -v nvcc
nvcc --version
```

上面的模块名只是示例，不保证你的站点提供它。加载模块一般会调整当前 shell 的环境，重新登录或换一个 shell 后不一定仍保持相同设置。

若 `module: command not found`，只能说**当前 shell 中这个命令不可用**：可能站点没采用它，也可能初始化脚本没生效。不能仅凭这一条错误判断整个服务器的部署方式。模块命令的机制可参考 [Environment Modules 文档](https://modules.readthedocs.io/en/latest/module.html)。

### 2.5.2 使用 Slurm 的集群

在需要申请 GPU 的集群，站点可能给出类似的交互作业命令：

```bash
srun --gres=gpu:1 --time=00:10:00 --pty bash
```

这是申请一个带 GPU 的作业环境的示意，不是所有集群通用的一键命令。分区、账号、资源名和时间限制以站点说明为准；提交可能排队，也可能因为参数不完整被拒绝。

成功进入作业之后，再运行本章的 hostname、nvidia-smi 和探针。调度器可能通过设备访问控制和环境变量限定当前作业使用哪些 GPU。见 [Slurm GPU 资源管理](https://slurm.schedmd.com/gres.html)。

不要擅自清空调度器设定的设备限制来“找回所有卡”。

### 2.5.3 使用容器

容器里同时需要考虑两件事：

1. 宿主机的 NVIDIA 驱动和 GPU 访问是否正确接入容器。
2. 容器镜像是否包含本次要用的开发工具，例如 nvcc。

容器能运行 nvidia-smi，不等于镜像里装了 CUDA 开发编译器。反过来，镜像里存在 nvcc，也不等于容器启动时获准访问 GPU。

由管理者配置 NVIDIA Container Toolkit 等集成后，容器才按对应配置获得设备访问。不要在普通学习容器里重新安装宿主机内核驱动。官方示例见 [NVIDIA 容器样例工作负载](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html)。

## 2.6 第一个 CUDA API 实验：只问有几张可见设备

现在开始写自己的探针。所谓“探针”，就是小而明确的诊断程序：只测试一个层次，不把大算法、复杂资源管理和性能优化混进来。

### 2.6.1 完整文件与编译运行

保存为 `device_count.cu`：

```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorName(err) << ": "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "visible CUDA devices: " << count << '\n';
    if (count == 0) {
        std::cerr << "No visible CUDA device; GPU lessons are not ready.\n";
        return 2;
    }

    std::cout << "PASS enumeration (no kernel executed)\n";
    return 0;
}
```

在文件所在目录执行：

```bash
nvcc -std=c++17 device_count.cu -o device_count
./device_count
```

如果直接使用配套代码，从教材根目录先进入：

```bash
cd code/02_environment
```

然后运行上面两条命令。本章没有设备 kernel，只调用主机端的 CUDA API，因此暂时不必手动选择 kernel 目标架构；下一章会真正生成和执行设备代码。

可见一个 CUDA 设备时，预期输出：

```text
visible CUDA devices: 1
PASS enumeration (no kernel executed)
```

设备数随环境变化，不要求必须是 1。如果得到错误，不要删除检查代码来“让它走到 PASS”，而是到 2.12 按错误层次定位。

### 2.6.2 一行调用，实际上有两条返回信息

重点看：

```cpp
int count = 0;
cudaError_t err = cudaGetDeviceCount(&count);
```

这里不是“把设备数量返回给 err”。实际关系是：

```text
主机变量 count，初始为 0
        ↑
        │ 通过 &count 提供写入位置
cudaGetDeviceCount(&count)
        │
        └── 函数返回 cudaError_t 状态 → err
```

- count 保存设备数量。
- `&count` 是 count 变量的地址，让 API 能将结果写到它。
- `cudaError_t` 是 CUDA 定义的错误状态类型。
- err 接收函数的返回状态，而不是设备个数。
- 只有状态表示成功，才按正常结果使用 count。

第 1 章学过“指针让函数能修改调用方的对象”。这里正是在真实 API 中使用那个概念。

### 2.6.3 cudaSuccess 不是函数，CUDA_CHECK 也不是关键字

```cpp
if (err != cudaSuccess)
```

`cudaSuccess` 是 CUDA 定义的成功状态值。`!=` 表示“不等于”。它不是要你实现或调用的函数。

`cudaGetErrorName(err)` 和 `cudaGetErrorString(err)` 才是函数调用：前者取得错误名称，后者取得便于阅读的描述。把两者输出，可以避免只看到一个难懂的整数错误码。

以后可能见到 `CUDA_CHECK(...)`。那是项目作者定义的辅助宏，不是 CUDA C++ 的内建关键字。本章先把判断完整展开，等理解它之后，第 7 章再说明如何消除重复代码。

设备枚举和属性查询的接口、输出参数及可能错误以 [CUDA Runtime 设备管理文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html) 为准。

### 2.6.4 为什么设备数为 0 也不能当作准备完成

程序检查了两种失败路径：

- API 调用失败：打印接口错误，返回 1。
- API 成功但 count 为 0：对本书的 GPU 实验而言还没准备好，返回 2。

不同版本和环境在“没有设备”的情况下可能返回错误，也可能反映为没有可用设备。我们不依赖固定的一种报错文字。

这里 1 和 2 是**本程序自己选的退出码**，不是说 CUDA 官方把“无 GPU”定义成退出码 2。0 表示这个探针完成了设备枚举检查。

> **检查点 B**
>
> 如果有人把代码写成 `int count = cudaGetDeviceCount(&count);`，请先不运行，解释他把哪两种返回信息混淆了。


## 2.7 第二个实验：读取设备属性，不用背显卡参数表

### 2.7.1 为什么已经知道数量，还要查询属性

假设探针告诉你有一张 GPU。下一步仍有很多实际问题：

- 它支持哪种计算架构？
- 一次启动中，每个 block 最多能放多少线程？
- 它有多少显存？
- 本书后面使用的 shared memory 会受到什么设备限制？

这些应由设备属性回答，不靠另一个型号的教程截图猜测。

下面保留第一个案例的数量检查，再在循环中查询每个可见设备。代码不包含复杂封装，任何 API 失败都立即给出具体位置。

**完整文件：`device_info.cu`。**

```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorName(err) << ": "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "device count: " << count << '\n';
    if (count == 0) {
        std::cerr << "No visible CUDA device; GPU lessons are not ready.\n";
        return 2;
    }

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        err = cudaGetDeviceProperties(&p, i);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed for device " << i
                      << ": " << cudaGetErrorName(err)
                      << ": " << cudaGetErrorString(err) << '\n';
            return 1;
        }

        std::cout << '[' << i << "] " << p.name << '\n'
                  << "  compute capability: " << p.major << '.' << p.minor << '\n'
                  << "  global memory: "
                  << p.totalGlobalMem / (1024.0 * 1024 * 1024) << " GiB\n"
                  << "  SM count: " << p.multiProcessorCount << '\n'
                  << "  warp size: " << p.warpSize << '\n'
                  << "  max threads/block: " << p.maxThreadsPerBlock << '\n'
                  << "  max block dimensions: "
                  << p.maxThreadsDim[0] << ','
                  << p.maxThreadsDim[1] << ','
                  << p.maxThreadsDim[2] << '\n'
                  << "  max grid dimensions: "
                  << p.maxGridSize[0] << ','
                  << p.maxGridSize[1] << ','
                  << p.maxGridSize[2] << '\n'
                  << "  shared memory/block (default limit): "
                  << p.sharedMemPerBlock << " bytes\n";
    }

    std::cout << "PASS device query (no kernel executed)\n";
    return 0;
}
```

在文件目录执行：

```bash
nvcc -std=c++17 device_info.cu -o device_info
./device_info
```

输出结构如下。尖括号部分需要由真实设备返回；这不是本书在你服务器上的实测记录：

```text
device count: <当前进程可见数量>
[0] <设备名称>
  compute capability: <主版本>.<次版本>
  global memory: <容量> GiB
  SM count: <SM 数量>
  warp size: <warp 中的线程数量>
  max threads/block: <单个 block 总线程数上限>
  max block dimensions: <x 上限>,<y 上限>,<z 上限>
  max grid dimensions: <x 上限>,<y 上限>,<z 上限>
  shared memory/block (default limit): <字节数> bytes
PASS device query (no kernel executed)
```

如果有多个可见设备，程序会继续打印 `[1]`、`[2]` 等记录。不要拿本章的占位文字当成实际数值填写环境表。

### 2.7.2 cudaDeviceProp p{} 是什么

`cudaDeviceProp` 是 CUDA 提供的结构体类型，里面包含多个字段。可以把一个结构体先理解成“一张有固定栏目名称的表”：

```text
p
├─ name                   设备名
├─ major / minor          Compute Capability
├─ totalGlobalMem         总全局内存字节数
├─ multiProcessorCount    SM 数量
└─ 其他字段               资源上限和能力信息
```

`p{}` 使用值初始化，把这张表先置为初始化状态。但**初始化并不是查询**，此刻还不能把其中的值当成真实硬件属性。

真正查询的是：

```cpp
err = cudaGetDeviceProperties(&p, i);
```

`&p` 提供表格的写入位置；i 指明查询哪个可见逻辑设备；err 仍然接收状态。只有成功后，才使用 `p.name`、`p.major` 等字段。

点号 `.` 表示访问对象的成员。它和命名空间中的双冒号 `::` 用途不同：`p.name` 是这张属性表的 name 栏；`std::cout` 是标准库命名空间里的 cout 名字。

### 2.7.3 每个字段先回答一个具体问题

| 字段 | 读成什么 | 后续哪儿用 |
|---|---|---|
| `name` | 设备名称 | 写实验记录 |
| `major`、`minor` | Compute Capability 主、次版本 | 选择受支持的目标架构和功能 |
| `totalGlobalMem` | 可报告的全局内存总字节数 | 估算数据容量；不是当前空闲字节数 |
| `multiProcessorCount` | 该设备的多处理器数量，通常称 SM 数量 | 第 4、19 章的执行资源讨论 |
| `warpSize` | warp 的线程数 | 第 4、12 章的分组与通信 |
| `maxThreadsPerBlock` | 单个 block 的总线程数上限 | 判断启动参数是否超限 |
| `maxThreadsDim` | block 的 x、y、z 各维上限 | 检查二维和三维 block |
| `maxGridSize` | grid 的各维上限 | 检查整个启动网格 |
| `sharedMemPerBlock` | 常规每 block shared memory 限额 | 第 10 章的块内临时存储 |

本表解释查询字段，不是“任何 kernel 都能无条件使用这些上限”的承诺。kernel 自身的资源使用还会形成限制；较大 shared memory 的 opt-in 等机制也要单独处理。官方字段定义见 [cudaDeviceProp](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)。

### 2.7.4 总线程数限制和每维限制要同时满足

现在只做整数算术，不需要知道调度细节。假设设备报告每个 block 最多 1024 个线程：

```text
block = (32, 8, 1)    总数 = 32 × 8 × 1 = 256
block = (32, 32, 1)   总数 = 1024
block = (64, 32, 1)   总数 = 2048 → 超出总数上限
```

即使 64 和 32 各自不超过对应维度上限，它们的乘积仍然可能超限。因此不能只检查 x、y 各自“看着都不大”。

反过来，总数不超过上限，也要确保每一维合法，并满足该 kernel 的其他资源与语义要求。第 4—6 章会把这些数字变成真正的启动代码。

### 2.7.5 显存、shared memory 和 SM 数量不能互相替代

如果设备有多个 SM，不意味着你的数组每个 SM 都会自动完整复制一份。如果报告单个 block 的 shared memory 限额，也不意味着它等于整张卡的总显存。

本章先把三个概念放在不同格子里：

```text
整张设备的全局内存容量       数据存储容量层面的属性
执行资源中的 SM 数量        处理器组织层面的属性
单个 block 的 shared 限额   一组线程使用临时存储的约束
```

后面会讲它们如何配合。现在不要用“108 个 SM × 32 个线程”去推算 CUDA core 数量，也不要把单个 block 的线程上限当成整张卡只能启动这么多个线程。

### 2.7.6 字节换算不要再混淆

程序使用：

```cpp
p.totalGlobalMem / (1024.0 * 1024 * 1024)
```

因为 1 GiB 等于 1024³ 字节。`1024.0` 让分母按浮点算术计算，输出可以保留小数；这里不是用 GB 的十进制单位。

显卡的商品名称、管理工具显示的 MiB、程序报告的 GiB 可能采用不同表示或可报告容量范围。它们不必字面一模一样。先核对单位和字段含义，不要看到“不是整整数”就认为丢了显存。

## 2.8 Compute Capability：不是 CUDA 版本，也不是核心数量

### 2.8.1 它描述设备架构能力类别

Compute Capability 常译为“计算能力”，通常由主版本和次版本组成，例如 8.0。它用于描述 GPU 的计算架构能力，不是软件包版本号，也不是每秒浮点运算次数。

如果 p.major 为 8、p.minor 为 0，常见目标架构名称会写成 `sm_80`。但不是所有架构目标都只有一个简单数字形式；某些新架构还有后缀及特殊兼容限制。初学阶段先使用设备和所选 Toolkit 明确支持的普通目标，别凭 GPU 名称自行发明编译选项。

### 2.8.2 同样的计算能力，不等于同样的性能

两款 GPU 可以支持相同一类编程功能，却拥有不同数量的执行资源、显存容量和带宽。计算能力主要帮助判断“能否使用这类能力和目标代码”，不能单独回答“快多少”。

本书不会要求你背每个型号的全部参数。首先保存自己设备查询到的值，后面每个功能有要求时，再对照官方说明。

### 2.8.3 新 Toolkit 不一定继续支持非常老的 GPU

这一点尤其不能通过版本号“越大越好”来判断。例如 CUDA Toolkit 13.0 移除了 Maxwell、Pascal、Volta 架构的离线编译和相应库支持；这类目标需要使用仍支持它们的工具链，而不是强行选择新 Toolkit。见 [CUDA 13.0 发布说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)。

这不等于旧机器上的所有既有程序突然不能运行，也不等于驱动兼容性与编译器架构支持是同一个问题。我们下一节把它们分开。

## 2.9 驱动与 Toolkit 兼容：不要写一个“版本大于等于”就下结论

### 2.9.1 三个容易混淆的版本问题

```text
问题 A：硬件是否被工具链支持？
问题 B：系统驱动是否满足运行要求？
问题 C：主机编译器与操作系统是否在 Toolkit 支持范围？
```

它们需要分别核对。就算驱动很新，旧 Toolkit 也未必认识新 GPU 架构；就算 nvcc 可以编译代码，运行时驱动也可能不满足使用的特性。

### 2.9.2 较新驱动通常能支持较旧 CUDA 程序，但不是任意组合

NVIDIA 提供驱动的向后兼容，以及符合条件时的同一 CUDA 大版本族内 minor-version compatibility。后者存在功能与 PTX 等限制，不能推广为“新 Toolkit 总能在旧驱动上正常跑”。

例如，不能仅比较 `cudaDriverGetVersion` 与 `cudaRuntimeGetVersion` 返回的整数，如果前者小一点就断言所有程序都不能运行；也不能因为简单查询成功，就断言任意新功能都可用。应该同时看选用版本的兼容说明、目标代码和实际使用的功能。见 [CUDA 小版本兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

本书入门阶段的做法是：优先使用管理员已经验证过的 GPU、驱动、Toolkit、主机编译器组合，先让小程序完整跑通。

### 2.9.3 主机编译器太新也可能不在支持范围

若 nvcc 报出 `unsupported GNU version` 一类信息，可能是当前 g++ 版本不在所选 Toolkit 支持范围。

不要把 `--allow-unsupported-compiler` 当作第一解决方案。绕过版本检查不等于这个组合得到验证。先用 `command -v g++`、`g++ --version` 确认实际工具，再选择站点提供的兼容编译器。第 3 章会解释 nvcc 怎样通过主机编译器处理 CPU 代码。

## 2.10 第三个实验：用程序打印版本，并理解它的限度

保存为 `cuda_versions.cu`：

```cpp
#include <cuda_runtime.h>
#include <iostream>

void print_version(const char* label, int value) {
    int major = value / 1000;
    int minor = (value % 1000) / 10;
    std::cout << label << ": " << major << '.' << minor
              << " (raw=" << value << ")\n";
}

int main() {
    int runtime_version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
    if (err != cudaSuccess) {
        std::cerr << "cudaRuntimeGetVersion failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }
    print_version("CUDA Runtime API version", runtime_version);

    int driver_version = 0;
    err = cudaDriverGetVersion(&driver_version);
    if (err != cudaSuccess) {
        std::cerr << "cudaDriverGetVersion failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }
    print_version("CUDA version reported by driver", driver_version);

    std::cout << "Version report only; no kernel or allocation tested.\n";
    return 0;
}
```

```bash
nvcc -std=c++17 cuda_versions.cu -o cuda_versions
./cuda_versions
```

若两项报告的值分别为 12060 和 12080，示意输出为：

```text
CUDA Runtime API version: 12.6 (raw=12060)
CUDA version reported by driver: 12.8 (raw=12080)
Version report only; no kernel or allocation tested.
```

这组数值只是帮助读公式，并非你的实际结果。

### 2.10.1 为什么 12060 被打印成 12.6

API 采用“1000 × 主版本 + 10 × 次版本”的编码。于是：

```text
12060 / 1000          = 12       整数除法得到主版本
12060 % 1000         = 60       % 取得余数
60 / 10              = 6        得到次版本
```

这个编码不保留所有发布补丁信息，不能用它代替完整工具版本记录。它也不是 NVIDIA 驱动安装包的版本字符串。

`print_version` 是我们自己写的普通 CPU 函数；`const char* label` 接收一个只读字符串指针，用于告诉读者这行在打印什么。函数里没有 GPU 特有语法。

### 2.10.2 特别注意 cudaRuntimeGetVersion 的版本语义

CUDA 12.0 起，这个 API 的用途是返回按规定格式编码的编译时 Toolkit 版本常量，不再为此初始化 CUDA。因此它返回成功，不能证明驱动初始化成功或 GPU 能执行任务；也不能把它当成对磁盘上所有已安装 Runtime 的扫描。

`cudaDriverGetVersion` 报告的是驱动支持的 CUDA 版本。具体返回规则见 [Runtime 版本管理 API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html)。

这个例子故意不输出“GPU 测试 PASS”。程序只是完成了版本报告。下一章执行一个实际 kernel，才继续补上设备代码加载和执行的证据。

## 2.11 CUDA_VISIBLE_DEVICES：看不见、被重排与没有硬件是不同情况

### 2.11.1 环境变量是传给程序的配置

在 Linux shell 中：

```bash
CUDA_VISIBLE_DEVICES="" ./device_count
```

这条命令只为**这一次启动的程序及其子进程**设置该变量为空，不永久修改当前 shell 的原值。程序会看不到可用的 CUDA 设备，按本章设计应失败退出，可能是 API 错误或无可见设备分支。

随后正常运行：

```bash
./device_count
```

它仍继承原本的环境。如果原本就是被屏蔽的环境，也不会因这条命令自动变成能访问 GPU。

不建议初学实验直接执行 `export CUDA_VISIBLE_DEVICES=`：export 会影响当前 shell 后续启动的程序，容易忘记恢复，还可能覆盖调度器提供的设置。单次命令前缀足以完成这里的实验。

### 2.11.2 程序中的 device 0 是逻辑编号

在你被授权使用的设备集合中，CUDA 可见设备还可以被筛选和重排。用 A、B、C 代表原枚举次序中的三个设备，若某个允许自行选择设备的环境按 C、A 暴露它们：

| 原集合中的设备 | 本次程序看到的逻辑编号 |
|---|---|
| C | 0 |
| A | 1 |
| B | 不可见 |

因此程序里 `cudaSetDevice(0)` 的意思是选择**当前进程可见列表中的第一个 CUDA 设备**，不是“服务器机箱中贴了 0 号标签的物理卡”。

CUDA 的原始枚举次序也不保证与 nvidia-smi 的索引完全一致。需要明确身份时，用可获授权的 UUID 等标识核对，而不是只比较一个数字。

可见性和顺序规则见 [CUDA 环境变量说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)。

### 2.11.3 空值实验是一种安全的负向测试

这次实验只把自己程序能看到的设备减少为零，不会拔掉显卡，不会重置 GPU，也不会让别人的作业失去设备。

观察两点：

1. 程序有没有明确报告“当前无法进行 GPU 实验”？
2. 正常环境下的运行是否保持原样？

不要只要求成功路径能显示漂亮参数。失败时能够准确停下，同样是一个好工具的能力。

> **检查点 C**
>
> 一个作业只获分配一张 GPU，程序显示 device count=1、逻辑编号 0。为什么这并不能证明整台服务器只有一张 GPU？

## 2.12 按证据排错：每次只跨过一层

### 2.12.1 决策顺序

```text
是否在允许使用 GPU 的节点/作业/容器中？
  否 → 先按站点流程进入正确环境
  是 ↓
nvcc 和主机编译器是否可以找到？
  否 → 核对模块、PATH、开发镜像与工具链部署
  是 ↓
探针是否编译成功？
  否 → 看源码路径、头文件、主机编译器兼容及链接错误
  是 ↓
device_count 是否成功并得到正数？
  否 → 看驱动、权限、设备映射与可见性
  是 ↓
device_info 是否成功？
  是 → 记录属性；下一章验证实际 kernel 执行
```

这不是说所有问题都能一眼归因。它让你知道还缺哪层证据，避免遇到任何 CUDA 错误都从重装开始。

### 2.12.2 常见现象与下一步

| 现象 | 可能方向 | 下一步 |
|---|---|---|
| nvidia-smi 不存在 | 当前环境没有管理工具或 PATH 不包含它 | 确认机器/容器和站点配置；不单凭此判断无 GPU |
| nvidia-smi 无法与驱动通信 | 驱动、设备访问或宿主集成问题 | 保存原错误，交由有权限的管理者检查 |
| nvidia-smi 正常、nvcc 不存在 | 开发工具未装/未加载/不在 PATH | 查看已批准的模块或开发镜像 |
| 找不到 cuda_runtime.h | 编译方法或开发头文件问题 | 使用预期 nvcc，检查所选 Toolkit 完整性 |
| unsupported GNU version | 主机编译器组合不受支持 | 选受支持的主机编译器，而非绕过检查 |
| 探针找不到 CUDA 设备 | 作业分配、容器映射、可见性、硬件支持等 | 对照当前环境逐项排查 |
| insufficient driver | 驱动不足以支持当前运行需求等 | 核对对应 Toolkit 的兼容要求 |
| invalid device ordinal | 使用了当前可见范围之外的逻辑编号 | 根据 count 使用 0 到 count-1 |
| 设备忙或不可用 | 计算模式、其他作业或系统状态等 | 不抢占或杀进程；按站点规则协调 |
| 查询成功但下一章 kernel 失败 | 目标代码、加载、启动配置或执行等问题 | 保留启动与同步错误，不能用查询 PASS 抵消 |

“可能方向”不是确定诊断。一个错误码可能对应多个环境原因。提交问题时应附带证据，而不是让对方猜。

### 2.12.3 怎样写一个有用的求助记录

下面是一种记录格式，配套模板位于 `code/02_environment/environment-record.md`：

```text
日期：
执行位置：本机 / 计算节点 / 作业内 / 容器内
主机架构和 Linux 发行版：
GPU 是否由调度器分配：
nvidia-smi：成功/失败，关键版本信息或原错误
nvcc 路径：
nvcc --version：
g++ 路径及版本：
实际编译命令：
实际运行命令：
device_count 完整输出和退出码：
device_info 输出或失败位置：
是否设置 CUDA_VISIBLE_DEVICES（可脱敏）：
已经证明：
尚未验证：
```

不要把密码、SSH 私钥、令牌或与问题无关的个人目录内容写进报告。这里需要的是可复现条件，不是收集机器上一切信息。

## 2.13 本章练习

答案见 [附录-C-练习答案](<附录-C-练习答案.md#第-2-章>)。

1. 分别用一句话解释 GPU、驱动、Toolkit、Runtime API 和主机编译器，不能都写成“CUDA 的一部分”。
2. 某环境 nvidia-smi 报告 CUDA 12.8，nvcc 报告 release 12.6。这一定是安装错误吗？两项分别在说什么？
3. 不看原代码，写出查询设备数并检查状态的核心五到十行。指出哪个变量是结果，哪个变量是状态。
4. 对照自己的真实 device_info 输出填写环境记录，保留单位；不要照抄示意参数。
5. 假设每 block 总线程上限为 1024，各维也允许，(32,8,1)、(32,32,1)、(64,32,1) 哪个因乘积超限？通过这项算术检查就保证能运行任何 kernel 吗？
6. 执行单次 `CUDA_VISIBLE_DEVICES="" ./device_count`，观察非零退出状态；再按原环境正常运行。解释这为何不需要手工恢复 export。
7. 若版本原始值为 13030，按本章函数如何显示？它能否证明当前磁盘只有一种 Toolkit？
8. 解释“属性查询成功”“kernel 启动检查成功”“kernel 同步检查成功”“数值结果正确”为什么是逐步增强的证据，而不是同义词。
9. 把 device_info 的输出再加一项 `maxThreadsPerMultiProcessor`。查清该字段含义：它是每 SM 最大驻留线程数，不是 CUDA core 数，也不是每周期完成的指令数。
10. 整理一个失败案例的报告：可以使用安全的空可见设备实验。写清预期、实际、退出码和未下结论的部分，不要只截最后一行。

## 2.14 自测清单与下一章的边界

本章完成的标准是：

- 知道在哪里运行命令，是否处于合法 GPU 作业环境。
- 能区分驱动侧报告的 CUDA 版本、nvcc 版本和计算能力。
- 能按文件名重新构造探针的编译与运行命令。
- 理解 API 的状态返回值与输出参数。
- 能查询属性并解释最常用的几个字段，不把它们误认为同一类资源。
- 失败时能给出原错误与发生层次，不靠删除检查绕过失败。
- 明确记录“这里没有执行 kernel”。

如果当前没有 GPU，也可以完成前一章 CPU 实验、读懂本章代码和手算练习。但没有实际运行探针，就如实写“未运行”，不要编造设备数据填表。

下一章：[03-第一段CUDA程序与NVCC](<03-第一段CUDA程序与NVCC.md>)。它会在当前已明确的工具链上启动真正的 GPU kernel，并进一步区分主机执行、异步提交、等待完成和输出顺序。

---

### 本章验证与资料说明

本章为零基础学习重新编写，相关命令、字段和兼容性规则在知识点旁链接到 NVIDIA 及相应工具的官方说明。具体支持矩阵和诊断措辞可能随版本变化，应查你所选版本的文档。

当前编辑环境没有可用的 CUDA 编译器及 GPU 实验环境；本轮只在本地进行文档与源码一致性等静态检查，未声称探针在真实 GPU 上运行通过。配套 [案例与检查说明](code/02_environment/README.md) 提供服务器验收入口。

版本：教材化修订第 1 轮，2026-09-11。整体进度见 [00-教材修订进度](<00-教材修订进度.md>)。
