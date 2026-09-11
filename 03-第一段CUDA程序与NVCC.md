# 03 第一段 CUDA 程序与 NVCC

> 本章任务：亲手启动一个 GPU 线程，再启动多个线程；能够解释主机代码、设备代码、启动参数、错误检查和同步。
>
> 先完成第 2 章的环境检查。本章首次执行真正的设备 kernel；如果编译或运行失败，要保留错误，不要把查询探针的 PASS 当成计算也能成功的证明。

## 3.0 从 CPU 程序迈出的新一步

第 1 章中，CPU 在一个循环里处理多个元素。现在先把数组放一边，只问一个更小的问题：

> 怎样明确地让 GPU 执行我写的一段函数，而不是仍由 CPU 执行？

这一章按四个完整案例推进：

| 阶段 | 文件 | 学会的新动作 |
|---|---|---|
| 一个线程 | `hello_one.cu` | 从 CPU 发起 kernel，检查并等它结束 |
| 多个线程 | `hello.cu` | 让每个线程报告自己的编号 |
| 提交与完成 | `launch_order.cu` | 分辨主机走到了哪里，以及 GPU 是否已经完成 |
| 各自计算 | `scalar_tasks.cu` | 传入标量，让每个线程有自己的局部变量和结果 |

它们位于 `code/03_hello/`。每个文件有自己的 main，可以独立编译。不要把四份完整文件粘在一起。

本章的输出用于建立执行模型，不用于性能测试。真正通过显存输入/输出数组的流程在第 5 章完成。

## 3.1 第一个 kernel：一个 GPU 线程打印一句话

### 3.1.1 创建完整文件

在自己的练习目录中创建 `hello_one.cu`。如果使用配套文件，从教材根目录进入：

```bash
cd code/03_hello
```

**完整文件：`hello_one.cu`。**

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void say_hello() {
    printf("GPU: hello from one thread\n");
}

int main() {
    std::cout << "CPU: before launch" << std::endl;
    say_hello<<<1, 1>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "say_hello launch failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "say_hello execution failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "CPU: GPU work finished" << std::endl;
    return 0;
}
```

先编译：

```bash
nvcc -std=c++17 hello_one.cu -o hello_one
```

编译成功后运行：

```bash
./hello_one
```

成功时预期看到：

```text
CPU: before launch
GPU: hello from one thread
CPU: GPU work finished
```

若遇到目标架构或设备代码相关错误，结合第 2 章的计算能力记录，阅读本章 3.9 选择当前工具链支持的架构。不能通过把 kernel 删掉，仍然声称完成本章实验。

### 3.1.2 先给代码划分执行位置

```text
CPU 执行 main：
  输出 before launch
  发起 say_hello 的 GPU 工作
  检查已知启动错误
  等待设备工作完成并检查错误
  输出 GPU work finished

GPU 执行 say_hello：
  输出 hello from one thread
```

最重要的分界不是文件名，而是函数定义和调用方式。`.cu` 文件中可以同时包含 CPU 和 GPU 代码；它并不是“文件里每一行都在 GPU 执行”。

`#include` 是构建时处理的指令，也不是在运行时由某个 GPU 线程执行“包含头文件”。

### 3.1.3 __global__ void say_hello() 怎么读

拆开看：

```cpp
__global__ void say_hello()
```

- `__global__`：CUDA 的函数执行限定符，把这个函数定义为 kernel 入口。
- `void`：kernel 不通过普通函数返回值交回一个结果。
- `say_hello`：我们自己起的函数名字。
- `()`：本例没有传入普通函数参数。
- 后面的大括号：每个执行该 kernel 的线程要执行的代码。

kernel 这个词在本书中指 GPU 计算入口，不是操作系统内核。普通 CUDA SIMT kernel 的函数体描述每个线程的工作；它不是由一个“代表整个 SM 的超级线程”只运行一次。

本书这里由主机发起启动。CUDA 还存在设备端启动等进阶机制，但不影响当前的 CPU→GPU 主线。CUDA 主机与设备执行关系见 [NVIDIA 编程模型说明](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)。

### 3.1.4 为什么 kernel 用 printf，主机用 cout

本例主机端使用 C++ 标准输出流，设备端使用 CUDA 支持的 printf。它们是两种不同位置上的输出方式，不意味着 GPU 可以任意调用全部主机标准库。

`<cstdio>` 提供 printf 相关声明，`<iostream>` 提供 cout/cerr 等。`std::endl` 产生换行并刷新主机输出流，让本例主机提示更容易观察；它不会等待 GPU，也不能代替 CUDA 同步。

## 3.2 两对括号：启动规模与算法参数不要混在一起

### 3.2.1 <<<1, 1>>> 不是函数输入数据

```cpp
say_hello<<<1, 1>>>();
```

可以读成：

```text
say_hello       <<<1, 1>>>                  ()
启动哪个函数     一个 block，每个一个线程     给函数的普通参数：无
```

这里先采用一维组织：

- 第一个 1：这个 grid 有 1 个 block。
- 第二个 1：每个 block 有 1 个 thread。
- 总线程数：1×1=1。

thread 译为线程，block 常译为线程块，grid 常译为网格。它们首先是**这次工作的组织方式**，不是显卡型号，也不是要创建多少块物理 GPU。

### 3.2.2 改成 <<<2, 4>>> 会发生什么

```cpp
say_hello<<<2, 4>>>();
```

它创建 2 个 block，每个 block 4 个线程，共 8 个逻辑线程。因为函数体里没有条件限制，每个线程都会执行一次 printf。

```text
一次 kernel 启动形成的 grid
├─ block 0
│  ├─ thread 0：执行 say_hello 的函数体
│  ├─ thread 1：执行 say_hello 的函数体
│  ├─ thread 2：执行 say_hello 的函数体
│  └─ thread 3：执行 say_hello 的函数体
└─ block 1
   ├─ thread 0：执行 say_hello 的函数体
   ├─ thread 1：执行 say_hello 的函数体
   ├─ thread 2：执行 say_hello 的函数体
   └─ thread 3：执行 say_hello 的函数体
```

函数在源码中只写了一份，但参与执行的线程有多份各自的执行状态。不要在 main 中再手工调用八次，来“配合”这一次八线程启动；那会成为另外八次启动。

逻辑上有 8 个线程，不等于必须有 8 个独占物理核心，也不保证所有操作在同一个时钟周期完成。硬件映射在第 4 章说明。

### 3.2.3 完整启动形式有四个位置，先学会前两个

后面会看到：

```cpp
kernel<<<grid, block, shared_bytes, stream>>>(arguments);
```

| 位置 | 含义 | 本章处理方式 |
|---|---|---|
| grid | block 的组织规模 | 先用一维 block 数量 |
| block | 每个 block 的线程组织规模 | 先用一维线程数量 |
| shared_bytes | 每个 block 额外申请的动态 shared memory 字节数 | 本章不用，省略后为 0 |
| stream | 工作提交的 stream | 本章使用默认 stream |

`arguments` 才是 kernel 函数定义里声明的参数。不要把数组长度 n 塞进第四项，又希望 kernel 自动收到它。

你现在不需要实现 stream 或 shared memory。先知道后两项不是“还有多少线程”，在第 10、20 章再使用它们。

## 3.3 两次检查：提交没有报错，不等于执行没有出错

### 3.3.1 为什么 launch 后不是 err = say_hello<<<...>>>()

kernel 返回类型是 void，三尖括号启动不是一个返回 cudaError_t 的普通函数表达式。本例用两个明确步骤获取状态：

```cpp
say_hello<<<1, 1>>>();
cudaError_t err = cudaGetLastError();
```

然后在确认未报告错误后：

```cpp
err = cudaDeviceSynchronize();
```

这两步承担不同职责，不是“同一句多检查一遍”。

### 3.3.2 第一层：读取当前线程的 CUDA 错误状态

`cudaGetLastError()` 取得当前主机线程的 Runtime 错误状态并重置该记录。紧接本例第一次 launch 使用它，有助于发现启动配置等已知问题。

它**不会为了检查而保证这个 kernel 已经全部运行结束**。在更复杂的程序中，先前异步操作的错误也可能在后续 API 中暴露，不能只根据“这一行拿到错误”断言这行是唯一根因。接口行为见 [CUDA 错误处理 API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html)。

“重置错误记录”也不是修复已经发生的非法访问。不要写一个循环不断读取并清掉错误，假装程序恢复正常。

### 3.3.3 第二层：等待设备工作完成并读取执行结果

`cudaDeviceSynchronize()` 在本章简单的单设备、单启动场景中，让主机等待先前提交的设备工作完成，并报告相应错误。

本例的 `GPU work finished` 只在同步成功后输出。若同步失败，程序打印 `execution failed` 并返回 1，不继续宣称成功。

```text
启动表达式返回
      ↓
读取错误状态：此刻是否已报告错误？
      ↓
同步：等待设备工作结束，是否有执行错误？
      ↓
成功后才能进入依赖 GPU 完成的主机步骤
```

这仍然不能证明复杂算法数值正确。例如 kernel 没越界，却把加法写成乘法，同步可能成功，结果却错误。因此第 5 章会加上输出数组验证。

### 3.3.4 为什么现在先把重复判断写完整

这些案例里会重复出现：

```cpp
if (err != cudaSuccess) {
    std::cerr << ...;
    return 1;
}
```

这里是概念片段，省略号不是可编译语句。完整文件没有省略。

我们暂时保留重复，确保你看得到“哪个 API 返回了什么”。等第 7 章再把它收进函数或宏，减少重复代码，而不是先让你背一个不认识的 CUDA_CHECK。

## 3.4 多个线程：把编号打印出来，再讨论它们怎样分工

**完整文件：`hello.cu`。**

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void hello_kernel() {
    printf("GPU: block=%u thread=%u block_size=%u grid_size=%u\n",
           blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
}

int main() {
    const int blocks = 2;
    const int threads = 4;
    std::cout << "CPU: before launch" << std::endl;
    hello_kernel<<<blocks, threads>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "hello_kernel launch failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "hello_kernel execution failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "CPU: GPU work finished" << std::endl;
    return 0;
}
```

```bash
nvcc -std=c++17 hello.cu -o hello
./hello
```

一种可能的输出是：

```text
CPU: before launch
GPU: block=0 thread=0 block_size=4 grid_size=2
GPU: block=0 thread=1 block_size=4 grid_size=2
GPU: block=0 thread=2 block_size=4 grid_size=2
GPU: block=0 thread=3 block_size=4 grid_size=2
GPU: block=1 thread=0 block_size=4 grid_size=2
GPU: block=1 thread=1 block_size=4 grid_size=2
GPU: block=1 thread=2 block_size=4 grid_size=2
GPU: block=1 thread=3 block_size=4 grid_size=2
CPU: GPU work finished
```

**中间八条 GPU 记录的顺序不保证与这里相同。** 检查的重点是记录集合：两个 block 中各有 thread 0、1、2、3，且没有遗漏或重复，而不是强制要求从上到下排好。

### 3.4.1 四个内建名字分别是谁的编号和大小

| 表达式 | 本例含义 | 可能值 |
|---|---|---|
| `blockIdx.x` | 当前线程所属的 block 在 grid 中的 x 编号 | 0 或 1 |
| `threadIdx.x` | 当前线程在自己 block 中的 x 编号 | 0、1、2、3 |
| `blockDim.x` | 每个 block 的 x 方向线程数量 | 4 |
| `gridDim.x` | grid 的 x 方向 block 数量 | 2 |

它们由 CUDA 为执行中的设备代码提供，和我们自己定义的 `blocks`、`threads` 不同。`.x` 是取 x 分量；二维和三维还会使用 y、z。

`threadIdx.x` 在每个 block 内从 0 重新编号。因此 block 0 的 thread 0 与 block 1 的 thread 0 是不同线程。要唯一标识当前一维 grid 中的线程，需要结合 block 和 thread 编号；第 5 章会推导全局索引。

### 3.4.2 printf 里的 %u 和 %d 不是随便换的

本例 CUDA 内建坐标分量是无符号整数，所以用 `%u`。`%d` 用于对应的有符号 int。格式与传入类型应匹配，不要只因为小数字恰好显示正常就忽略格式。

```cpp
printf("block=%u thread=%u\n", blockIdx.x, threadIdx.x);
```

第一个格式位置对应后面的 blockIdx.x，第二个对应 threadIdx.x。换参数顺序会换显示含义；字符串里写了一个标签，不会自动寻找同名变量。

### 3.4.3 threads=4 不需要等于 warp 大小

这个四线程 block 是为了让输出可以人工核对，不是高性能推荐配置。CUDA 不要求每个合法 block 都必须包含 32 的整数倍线程；不过硬件会按 warp 组织执行，如何影响利用率在下一章讨论。

先让模型正确，再考虑性能。不要因为看见“32”就把所有规模都先改成 32，并跳过理解为什么本例只有 4 个线程。


## 3.5 异步：CPU 不必原地等，但也不能靠打印猜完成时刻

### 3.5.1 先把“异步”翻译成具体事件

本章说 kernel 启动对主机通常是异步的，意思是：

> 主机发起这项工作后，不必等这项 GPU 工作全部完成，才能继续执行后面的主机代码。

它**不保证** GPU 一定在后面的主机语句之前开始，也不保证一定在后面才结束。短小 kernel 可能很快完成；初始化、加载等过程也可能增加主机端调用耗时。

所以“允许 CPU 和 GPU 的进度重叠”，与“每一次都会观察到明显重叠”不是同一句话。

### 3.5.2 用 A、B、C 标出主机进度

**完整文件：`launch_order.cu`。**

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__global__ void report_thread() {
    printf("GPU: thread=%u\n", threadIdx.x);
}

int main() {
    std::cout << "CPU: A before launch" << std::endl;
    report_thread<<<1, 4>>>();
    std::cout << "CPU: B launch call returned" << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "report_thread launch failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "report_thread execution failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "CPU: C synchronization succeeded" << std::endl;
    return 0;
}
```

```bash
nvcc -std=c++17 launch_order.cu -o launch_order
./launch_order
```

你可能看到：

```text
CPU: A before launch
CPU: B launch call returned
GPU: thread=0
GPU: thread=1
GPU: thread=2
GPU: thread=3
CPU: C synchronization succeeded
```

中间四条 GPU 输出顺序可以不同。这个例子不是要证明 GPU 直到 B 之后才开始计算，而是标明我们真正知道的事件：

| 标记 | 现在能确定什么 | 还不能确定什么 |
|---|---|---|
| A | 主机即将走到启动表达式 | GPU 尚未因本次启动执行代码 |
| B | 启动表达式已经返回到主机 | 仅凭这句输出不知道 GPU 是否已完成，也尚未说明错误检查结果 |
| C | 本例的同步调用已经成功 | 不代表任何复杂算法的数值一定正确 |

### 3.5.3 两种合法时间关系

下面把箭头看作时间向右流动，不是同一格代表固定时长：

```text
情形一：主机很快继续，GPU 后面才完成
CPU：A → launch → B → 等待................. → C
GPU：           开始 → 执行 → 完成

情形二：GPU 工作很短，主机走到 B 时已经做完
CPU：A → launch........... → B → 同步返回 → C
GPU：       开始 → 完成
```

两种都不与异步语义矛盾。异步保证的是你不能拿“主机调用返回”当作通用完成通知，而不是承诺一条固定的现实时间线。相关规则见 [CUDA 异步执行说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)。

### 3.5.4 为什么屏幕顺序不是执行时刻的录像

设备 printf 经过缓冲和刷新机制。设备线程执行 printf 的时刻，与文字出现在终端的时刻不是同一回事。

即使每次都看到“CPU A、CPU B、GPU 输出、CPU C”，也不能据此声称程序实际上没有异步。反过来，GPU 行出现在 B 附近，也不能证明每个线程的指令顺序或 warp 调度关系。

第 18 章会用时间线分析工具观察执行；现在只依赖明确的 API 完成条件，而不是把终端当作 profiler。

### 3.5.5 同步不是线程块屏障

本章从 CPU 调用的 `cudaDeviceSynchronize()` 与后面 kernel 里面的 `__syncthreads()` 是不同机制：

```text
cudaDeviceSynchronize：主机等待当前设备的相关先前工作完成
__syncthreads：同一 block 中参与的设备线程进行块内同步
```

它们不是“一个英文长、一个英文短”的同一函数。不要在 kernel 中随意用主机等待接口代替块内屏障。后者的参与条件在第 10 章详细讲。

## 3.6 每个线程有自己的变量：不用先想象物理寄存器

打印编号之后，我们增加一个实际计算：四个线程分别处理 10、11、12、13，并把各自输入乘以 2。

### 3.6.1 先列出每个线程应该算什么

| threadIdx.x | base | input = base + threadIdx.x | output = 2 × input |
|---|---|---|---|
| 0 | 10 | 10 | 20 |
| 1 | 10 | 11 | 22 |
| 2 | 10 | 12 | 24 |
| 3 | 10 | 13 | 26 |

这里所有线程看到相同的 base 参数，但它们的线程编号不同，因而局部 input 和 output 的值不同。

**完整文件：`scalar_tasks.cu`。**

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

__device__ int twice(int x) {
    return 2 * x;
}

__global__ void compute_values(int base) {
    int input = base + static_cast<int>(threadIdx.x);
    int output = twice(input);
    printf("GPU: thread=%u input=%d output=%d\n",
           threadIdx.x, input, output);
}

int main() {
    const int base = 10;
    compute_values<<<1, 4>>>(base);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "compute_values launch failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "compute_values execution failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "CPU: scalar tasks finished" << std::endl;
    return 0;
}
```

```bash
nvcc -std=c++17 scalar_tasks.cu -o scalar_tasks
./scalar_tasks
```

一种可能的输出：

```text
GPU: thread=0 input=10 output=20
GPU: thread=1 input=11 output=22
GPU: thread=2 input=12 output=24
GPU: thread=3 input=13 output=26
CPU: scalar tasks finished
```

仍然检查四组编号和数值是否齐全，不要求 GPU 输出顺序固定。

### 3.6.2 参数写在普通括号里

```cpp
compute_values<<<1, 4>>>(base);
```

前面的 `<<<1,4>>>` 规定 1 个 block、4 个线程；后面的 `(base)` 传入计算参数。kernel 定义中的 `int base` 接收这个标量值。

不要把二者混写成 `compute_values<<<base,4>>>()`。这不只是在“另一种位置传了相同数据”，而是改变了启动规模，并且没有提供函数所需参数。

本例标量按值传递。将来传入指针时，传递的是地址值，不会因此自动复制该指针后面整段数组。第 5 章才处理主机数组和设备数组的存储关系。

### 3.6.3 同一个变量名，可以属于多个不同线程

```cpp
int input = base + static_cast<int>(threadIdx.x);
int output = twice(input);
```

这两行在每个线程各自的执行中发生。不是所有线程轮流修改一个叫 input 的共享格子：

```text
thread 0 的 input=10，output=20
thread 1 的 input=11，output=22
thread 2 的 input=12，output=24
thread 3 的 input=13，output=26
```

一个线程当然可以有多个局部变量，也可以有循环和函数调用。它不是“只能拿着一个数字的小格子”。

这些普通局部变量常由编译器放入寄存器等适当存储位置，但源码层面的重点是它们的线程私有语义。不要凭一条 int 声明断言它必定占用某个固定物理寄存器，更不要由变量名字推断它是整个 block 的 shared memory。

### 3.6.4 __device__ 函数没有再启动新线程

```cpp
__device__ int twice(int x) {
    return 2 * x;
}
```

这是可由设备代码调用的普通设备函数。当前 GPU 线程调用 twice，得到结果后继续执行自己的代码；不会因为调用 twice 就额外创建两个线程，也不需要三尖括号。

它可以返回 int。前面说 kernel 的返回类型是 void，不意味着 GPU 上所有函数都必须是 void。

## 3.7 三类函数先认清，不把所有代码都标成 __global__

| 定义方式 | 本章的典型执行位置 | 典型调用方式 |
|---|---|---|
| 普通函数或 `__host__` 函数 | CPU | `function(args)` |
| `__device__` 函数 | 当前 GPU 线程 | 在设备代码中 `function(args)` |
| `__global__` kernel | GPU 线程执行的入口 | 主机用 `kernel<<<...>>>(args)` 发起 |

还可以把某些函数定义为 `__host__ __device__`，让工具链为相应目标处理它们。那不是“运行时自动猜测选哪边”，也不意味着函数体中能任意混入只在 CPU 可用的操作。

本阶段只需要按职责选择：

- CPU 管理代码留在 main 或普通主机函数里。
- 大量线程执行的入口写成 kernel。
- kernel 内可复用的小计算可以是 device 函数。

例如，把设备的 printf 改成 cout，不会因前面写了 `__global__` 就自动让整个主机 I/O 库在 GPU 上工作。CUDA C++ 的语言与库使用限制要按工具链支持范围判断。

## 3.8 自己写 NVCC 命令：从一个最小模板扩展

### 3.8.1 读懂这条常见命令

```bash
nvcc -std=c++17 hello.cu -o hello.exe
```

| 部分 | 回答的问题 |
|---|---|
| `nvcc` | 用什么驱动编译 CUDA C++？ |
| `-std=c++17` | 使用哪套 C++ 语言规则？ |
| `hello.cu` | 读哪个源文件？ |
| `-o` | 后面的名字用于哪种用途？——输出文件 |
| `hello.exe` | 生成的程序叫什么？ |

命令里的空格分隔参数。`-std=c++17` 是一个选项，不应拆成 `-std = c++17`。`-o` 后面的文件名则是它的值，常写成单独一个参数。

如果你保存的是 `lesson3.cu`，想生成 `run_lesson3`：

```bash
nvcc -std=c++17 lesson3.cu -o run_lesson3
./run_lesson3
```

现在你应能够自己替换名字，而不是只能使用教程中的 hello。

### 3.8.2 .exe 不能改变构建目标平台

在 Linux 本地工具链中输出文件叫 hello.exe，仍按 Linux 构建结果理解，通常执行 `./hello.exe`。文件名后缀不能让它变成 Windows 程序。

Windows 上需要相应受支持的主机工具链和 CUDA 安装；PowerShell 常用 `.\hello.exe` 运行本地程序。不要把 Windows 与 Linux 的命令路径、链接库和编译环境随意混合。

### 3.8.3 .cu 说明输入类型，不是“保证 GPU 被使用”

NVCC 看到 `.cu` 会按 CUDA 源文件路径处理。但如果这个文件只有普通 main 输出文字，依然可能完全没有执行 GPU 工作。

相反，普通 g++ 不负责识别 CUDA 的 `__global__` 和 `<<<...>>>` 扩展。入门阶段，包含这些 CUDA 扩展的案例统一由 nvcc 驱动构建。

## 3.9 目标架构：这次生成的设备代码准备给谁运行

### 3.9.1 一个明确目标的例子

假设第 2 章查询的 GPU 计算能力是 8.0，并且当前 Toolkit 支持这个目标，可以编译：

```bash
nvcc -std=c++17 -arch=sm_80 hello.cu -o hello
```

不要因为看到本行，就在任何服务器都固定写 sm_80。先查询真实设备与编译器支持列表：

```bash
nvcc --list-gpu-code
nvcc --list-gpu-arch
```

前者列出编译器支持的实际 GPU 代码目标，后者列出虚拟架构目标。这两个命令是查看工具链能力，不是枚举当前机器的 GPU；当前机器信息仍看设备查询。

### 3.9.2 sm_80 不是“用 80 个 SM”

sm_80 是目标架构名称，通常对应计算能力 8.0。GPU 可以有不同数量的 SM，而仍属于同一类编译目标。它不表示“最多用 80 个 SM”，不表示线程数，也不是 C++ 标准。

把目标数字改大，不会升级硬件。只包含较新架构机器码的程序，不能假定能在更旧 GPU 上执行。

### 3.9.3 省略 -arch 与使用 native 各有什么边界

省略 -arch 时，NVCC 使用其版本对应的默认目标策略；它不等于保证自动为当前机器所有 GPU 生成最优代码。第一次简单实验可能正常运行，但做可复现实验时应记录实际目标。

受支持的工具链可以使用：

```bash
nvcc -std=c++17 -arch=native hello.cu -o hello
```

它依据**编译时可见 GPU**生成代码，且这种模式不附带 PTX。没有可见的受支持 GPU 时不能依赖它自动猜中远程目标；也不适合作为跨不同 GPU 分发程序的默认承诺。

架构参数和目标形式见 [NVCC 目标生成说明](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch)。后面第 23 章再用 CMake 管理明确的多架构构建。

## 3.10 NVCC、PTX、cubin、SASS：先知道各在哪一层

### 3.10.1 编译驱动不等于单个编译步骤

NVCC 的工作可以先画成：

```text
                 同一个 .cu 文件
                 /            \
          主机代码处理       设备代码处理
                 |            |
            主机目标代码     PTX 等中间表示
                 |            |
                 |       ptxas 等工具生成目标设备代码
                 |            |
                 └────── 打包与链接 ──────┐
                                         ↓
                         包含主机程序和设备代码映像的可执行文件
```

这是一张帮助定位层次的简图，不保证每种编译选项都走完全相同的步骤。

### 3.10.2 四个词的区别

| 名称 | 初学时怎样理解 |
|---|---|
| PTX | NVIDIA 的虚拟指令集表示，不是某张具体 GPU 的最终机器指令 |
| SASS | 描述具体 NVIDIA GPU 机器指令时常用的名称 |
| cubin | 包含目标 GPU 代码等内容的二进制映像，不只是一个“后缀更短的 PTX 文本” |
| fatbin | 可以容纳多个目标代码映像及 PTX 等内容的包装形式 |

工具链可以预先生成适用设备代码；如果程序包含可用 PTX，运行时驱动也可能进行 JIT，即在需要时把它编译为设备可执行形式。实际选择取决于嵌入内容、设备与驱动能力。见 [CUDA 编程指南的 NVCC 章节](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)。

### 3.10.3 一个可选的查看实验

若当前编译器支持 compute_80，可以对标量案例只生成 PTX：

```bash
nvcc -std=c++17 -arch=compute_80 -ptx scalar_tasks.cu -o scalar_tasks.ptx
head -n 30 scalar_tasks.ptx
```

这次输出的是文本形式的设备中间代码，不是完整可执行程序。不能用 `./scalar_tasks.ptx` 来替代正常运行。

你可能看不到源代码中完整的 twice 函数，因为编译器可以内联它。高级语言函数数量、源码行数与机器指令数量，不要求一一对应。

本章只要求知道这些层次的名字和用途，不要求读懂所有 PTX 指令。

## 3.11 调试构建、行号和资源报告：每个选项都有具体目的

### 3.11.1 调试不是打开“自动讲解模式”

对当前设备选择合适目标后，可以使用类似的调试构建：

```bash
nvcc -std=c++17 -O0 -g -G hello.cu -o hello_debug
./hello_debug
```

`-g` 生成主机调试信息；`-G` 生成设备调试信息，并通常关闭设备优化，除非使用了相关的其他控制选项。它们供调试工具使用，不自动输出变量、线程状态或每条执行指令。

因此，同样代码在开启 -g/-G 后打印内容没有增加，是正常现象。想观察变量，自己添加少量输出，或使用后续介绍的调试器。

### 3.11.2 -O0/-O3 与 -G 不能粗暴混为一类

NVCC 的 `-O` 选项指定主机代码优化级别；设备优化还有自己的控制与默认行为。`-O0` 不能单独理解为“一切 GPU 优化都关闭了”。设备调试构建的 -G 才会以它的规则影响设备优化。

常见的优化构建可以保留设备行号：

```bash
nvcc -std=c++17 -O3 -lineinfo hello.cu -o hello_release
```

`-lineinfo` 提供设备代码与源码行的关联，便于工具定位，不等于加入完整设备调试。相关选项以 [NVCC 编译控制文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-specifying-behavior-of-compiler-linker) 为准。

不要用 hello 的打印耗时比较 Debug 与 Release 的“GPU 性能”。打印、初始化和进程启动会占据大量影响，当前案例不是基准测试。

### 3.11.3 -Xptxas=-v 的输出出现在哪个阶段

```bash
nvcc -std=c++17 -O3 -Xptxas=-v hello.cu -o hello_report
```

`-Xptxas` 把选项传给 ptxas；这里的 `-v` 请求资源等信息报告。它通常在**编译时**输出信息，而不是等你运行程序后自动打印寄存器。

现在只认识“编译时报告”和“程序运行输出”不同。第 19 章会结合实际 kernel 解释寄存器和 shared memory 使用量如何影响并发资源。

### 3.11.4 printf 适合少量观察，不适合保留在性能实验中

设备 printf 使用有限缓冲，输出过多可能覆盖之前的信息；同步等特定操作会触发刷新，程序退出本身不保证自动刷新。它还会扰动执行时序和性能。见 [CUDA C++ 支持中的格式化输出说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html)。

若只需要一个线程输出，在 kernel 内写：

```cpp
if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("one representative thread\n");
}
```

这里是片段。两个条件一起限定整个一维示例中 block 0 的 thread 0；只检查 threadIdx.x==0，则每个 block 都可能各打印一次。

## 3.12 常见错误：先分清构建、启动、执行与观察

| 现象 | 首先检查 |
|---|---|
| g++ 不认识 CUDA 语法 | 是否用 nvcc 处理这个 .cu 文件 |
| 编译后还是旧输出 | 是否保存了源文件，编译是否成功，是否运行对了路径 |
| 忘了 kernel 之后的普通圆括号 | 正确形式是 kernel<<<配置>>>(参数)，不能只有三尖括号 |
| 启动参数超出设备或 kernel 限制 | 对照第 2 章属性和当前 kernel 要求 |
| 报 no kernel image / invalid device function 等 | 设备目标、工具链支持及嵌入代码是否匹配；保留完整错误 |
| 删除同步后没有设备输出 | 程序可能在工作完成和输出刷新之前结束；不能期待某个固定现象 |
| GPU 行顺序变了 | 调度与输出顺序本就不保证按编号排序 |
| -g/-G 后没有更多日志 | 调试信息不是主动打印；检查自己添加的输出或调试器操作 |
| 把 __device__ 函数从 main 直接调用 | 主机与设备函数调用范围弄混 |
| 认为 kernel 中一个局部变量由所有线程共享 | 把每线程执行模型误当成单次全局执行 |

故意制造错误时只改自己练习副本；一次改一个因素。不要在共享服务器上用巨大 grid 和海量 printf 来“试试能不能把 GPU 跑满”。

## 3.13 本章练习

答案见 [附录-C-练习答案](<附录-C-练习答案.md#第-3-章>)。

1. 从空文件重写一个线程的 hello，保留启动检查和同步检查。解释主机与设备分别执行哪些行。
2. 把编号案例改成 3 个 block、每个 5 个线程。先列出应出现的 (block,thread) 组合，再运行。是否必须按你写下的顺序显示？
3. 分别把启动设为 (1,8)、(2,4)、(4,2)。线程总数都为 8，为什么打印的编号组合不同？
4. 只让每个 block 的 thread 0 打印；再改成整个 grid 只让 block 0/thread 0 打印。两次输出行数分别是多少？
5. 在标量案例中把 base 改成 100。预测四组输入和输出，重新编译核对。说明 input/output 是每个线程私有变量。
6. 写出 kernel 配置与普通参数的区别。如何用 1 个 block、4 个线程，把整数 7 传给 compute_values？
7. 一段程序先打印主机 A，然后提交 kernel，再打印主机 B，最后同步并打印 C。观察到 A、B、GPU、C，能否证明 GPU 一定在 B 后才执行？为什么？
8. 用自己的话解释 -std=c++17、-arch=sm_80、-o、-g、-G、-lineinfo 分别控制哪一件事。
9. 在自己的 hello 副本中暂时去掉同步及对应检查，多次运行观察，再恢复。输出是否完整只能作为现象记录，为什么不能由十次碰巧完整推断有完成保证？
10. 选做：生成 scalar_tasks.ptx，区分文本输出文件与可执行文件。为什么源函数 twice 可能没有以独立函数形式出现？
11. 从空文件写一个 kernel，每个线程计算自己编号的平方；使用 1 个 block、4 个线程，预期结果为 0、1、4、9。提供运行命令、检查代码，并说明你还没学到的数组输出将在哪章加入。

## 3.14 能独立启动，才开始研究硬件怎样执行

合格的本章成果不是保存一张输出截图，而是你能自己完成：

```text
写出普通 main
      ↓
写出描述每个线程工作的 kernel
      ↓
选择合法的小规模启动配置
      ↓
正确传入普通参数
      ↓
读取启动错误并等待执行完成
      ↓
核对输出中的线程身份和计算结果
```

暂时不需要记住某张卡有多少 CUDA core，也不用先写内存封装库。你已经有了明确的问题可带到下一章：

- 为什么线程要分 block？
- block 如何在 SM 上执行？
- warp 和 SIMT 是什么？
- 很多逻辑线程如何使用有限硬件资源？

下一章：[04-线程-Block-Grid-Warp与SIMT](<04-线程-Block-Grid-Warp与SIMT.md>)。

---

### 验证说明

四份完整案例均有配套源码；[案例目录](code/03_hello/README.md)提供编译与运行方法。核对输出时检查线程记录是否齐全、数值是否正确，不把 GPU 打印顺序写死。

文中的输出为预期或示意；请实际编译运行，并按各节说明观察打印与同步行为。
