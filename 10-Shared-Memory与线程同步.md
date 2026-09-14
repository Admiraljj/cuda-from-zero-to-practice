# 10 Shared Memory 与线程同步

本章从邻居求和的手算出发，逐步建立共享内存、线程协作和同步的完整模型。包含九个独立程序与22道练习。命令默认从教材根目录执行；示例输出为预期结果，实际C++/CUDA编译、GPU运行与Compute Sanitizer验收尚未完成。

## 10.0 从独立计算走向协作

第 5、9 章的向量加法中，一个线程拿到 a[i] 和 b[i]，就能写出 c[i]，通常不用读取另一个线程准备的数据。

这一章换一个问题：每个输出需要当前位置及左右邻居。相邻线程会反复需要同一些输入。我们先正确算出结果，再研究能否协作准备数据。

学习顺序是：

```text
纸笔算邻域 → 普通 CPU 模型 → 直接读 global 的 CUDA 版本
           → 画出 block 的小块数据 → 协作加载 → 等待 → 使用 shared
```

不要一看到 shared memory 就急着把所有数组搬进去。先回答：哪份数据会被谁重复使用？协作带来哪些额外责任？

## 10.1 定义任务：左右各看一个位置

输入有 n 个数，索引范围 0..n-1。定义边界之外的数为 0，输出为：

```text
output[i] = 左边的数 + 自己的数 + 右边的数
```

我们先做求和，不做平均，这样小整数可以精确手算，注意力不会被浮点容差打断。若要做“三点平均”，最后再除以 3.0f；本文的零填充规则下，边界也仍除以 3，不是除以实际存在的邻居数量。

这种“用固定邻域计算每个输出”的模式常叫 stencil。半径 1 表示向左和向右各扩展一个位置，不是只读取一个数。半径 2 才是左右各两个，加上中心共五个位置。

### 10.1.1 用十个数完整算一遍

输入为 1、2、3、4、5、6、7、8、9、10：

| i | 左 | 中 | 右 | 输出 |
|---|---|---|---|---|
| 0 | 0，边界外 | 1 | 2 | 3 |
| 1 | 1 | 2 | 3 | 6 |
| 2 | 2 | 3 | 4 | 9 |
| 3 | 3 | 4 | 5 | 12 |
| 4 | 4 | 5 | 6 | 15 |
| 5 | 5 | 6 | 7 | 18 |
| 6 | 6 | 7 | 8 | 21 |
| 7 | 7 | 8 | 9 | 24 |
| 8 | 8 | 9 | 10 | 27 |
| 9 | 9 | 10 | 0，边界外 | 19 |

最后不是 30，因为数组里没有第 11 个数。边界规则是算法的一部分，不能只在 GPU 越界以后才临时决定。

如果 n=1，唯一输出就是 0+input[0]+0；不能因为线程有左右索引表达式，就真的访问 input[-1] 或 input[1]。

### 10.1.2 为什么输入输出要分开

考虑线程 i 还没有读 input[i+1]，线程 i+1 却已经把新结果写回 input[i+1]。那么线程 i 可能读到更新后的数，而另一次运行又读到旧数，输出不再是上面定义的同一轮 stencil。

因此本章使用独立 input 与 output 分配。所有线程读同一份不变输入，各自写唯一输出。后续需要迭代多轮时再用双缓冲交换角色，不能以为 block 内加一个屏障就能让不同 block 的原地更新变安全。

## 10.2 案例一：不运行 GPU，先把小块数据摆对

先把 10 个输出按每组 4 个分组：

```text
block 0 负责输出： 0  1  2  3
block 1 负责输出： 4  5  6  7
block 2 负责输出： 8  9  -- --
```

block 1 要算 output[4..7]，不仅需要 input[4..7]，还需要左边 input[3] 和右边 input[8]。这两个额外邻接位置叫 halo：本组的输出范围之外，但计算边界输出时需要的数据。

一个宽度 B=4 的小块因此准备 B+2=6 个数：

```text
局部 tile 索引： 0      1   2   3   4      5
含义：         左halo   四个中心位置       右halo
对应全局索引：base-1   base ... base+3    base+4
```

tile 是我们起的变量名，含义是一小块数据，不是 CUDA 自动出现的某种数组对象。下面 CPU 模型里的 tile 还是普通 CPU 数组。

完整文件 code/10_shared/tile_model.cpp：

```cpp
#include <cmath>
#include <iostream>
#include <vector>

float sample(const std::vector<float>& input, int i) {
    return i >= 0 && i < static_cast<int>(input.size()) ? input[i] : 0.0f;
}

int main() {
    constexpr int n = 10, width = 4;
    const int blocks = (n + width - 1) / width;
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i + 1);
    const float expected[n] = {3, 6, 9, 12, 15, 18, 21, 24, 27, 19};

    for (int block = 0; block < blocks; ++block) {
        float tile[width + 2] = {};
        const int base = block * width;
        // Model phase 1: all positions are prepared before phase 2.
        for (int tid = 0; tid < width; ++tid) {
            tile[tid + 1] = sample(input, base + tid);
        }
        tile[0] = sample(input, base - 1);
        tile[width + 1] = sample(input, base + width);
        std::cout << "block=" << block << " tile=";
        for (int j = 0; j < width + 2; ++j) {
            if (j != 0) std::cout << ',';
            std::cout << tile[j];
        }
        std::cout << '\n';
        // Model phase 2: consume the completed local tile.
        for (int tid = 0; tid < width; ++tid) {
            const int i = base + tid;
            if (i < n) output[i] = tile[tid] + tile[tid + 1] + tile[tid + 2];
        }
    }
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(output[i]) || output[i] != expected[i]) return 1;
        std::cout << "i=" << i << " sum=" << output[i] << '\n';
    }
    std::cout << "PASS CPU tile model; no CUDA threads or barriers executed\n";
    return 0;
}
```

运行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra code/10_shared/tile_model.cpp -o tile_model
./tile_model
```

预期前 3 行是：

```text
block=0 tile=0,1,2,3,4,5
block=1 tile=4,5,6,7,8,9
block=2 tile=8,9,10,0,0,0
```

随后输出 10 行 i=索引 sum=结果，数值应与 10.1 表格一致，最后为 PASS CPU tile model; no CUDA threads or barriers executed。

### 10.2.1 为什么中心要放在 tid+1

位置 0 留给左 halo，中心数据从位置 1 开始。所以 tid=0 的中心放 tile[1]，tid=3 的中心放 tile[4]。

对于任意有效 tid：

| 邻域部分 | tile 位置 |
|---|---|
| 左边 | tid |
| 自己 | tid+1 |
| 右边 | tid+2 |

例如 block 1 的 tid=0 对应全局 i=4，从 tile[0]、tile[1]、tile[2] 读取 4、5、6，总和 15。

这里有两套坐标：全局 i 选原数组和输出位置；局部 tid 或 tile 索引选这一个 block 准备的小块。把 input[tid] 误当作 input[i]，就会让后面的 block 反复读取开头。

### 10.2.2 两个 phase 只是 CPU 模型，不是 GPU 同步证据

CPU 代码先完成加载循环，然后进入消费循环，天然体现了“准备完再使用”的顺序。它没有创建 CUDA 线程、没有执行 __syncthreads。

这个模型可以检查索引与数学结果，却不能检测真实线程竞争、warp 调度或 shared memory 访问顺序。下一步仍必须在 GPU 程序中建立相应同步关系。

sample 函数对边界外返回 0，是我们明确的边界处理；普通数组不会自动在越界访问时返回 0。不要删除范围检查，期待运行时替你补零。

## 10.3 案例二：先写正确的朴素 CUDA 版本

先不使用 shared。每个线程直接从 global input 读自己的三个邻域值，写一个 output。

完整文件 code/10_shared/stencil_naive.cu：

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void stencil_naive(const float* input, float* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float left = i > 0 ? input[i - 1] : 0.0f;
        const float center = input[i];
        const float right = i + 1 < n ? input[i + 1] : 0.0f;
        output[i] = left + center + right;
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    constexpr int n = 10, threads = 4;
    const int blocks = (n + threads - 1) / threads;
    const std::size_t bytes = n * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i + 1);
    const float expected[n] = {3, 6, 9, 12, 15, 18, 21, 24, 27, 19};
    float *d_input = nullptr, *d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
        if (!check(cudaMemset(d_output, 0xff, bytes), "poison output")) break;
        stencil_naive<<<blocks, threads>>>(d_input, d_output, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait")) break;
        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            std::cout << "i=" << i << " sum=" << output[i] << '\n';
            if (!std::isfinite(output[i]) || output[i] != expected[i]) ok = false;
        }
    } while (false);
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS naive stencil: 10 elements; input and output are separate\n";
    return 0;
}
```

编译运行：

```bash
nvcc -std=c++17 -O3 -lineinfo code/10_shared/stencil_naive.cu -o stencil_naive
./stencil_naive
```

4 个线程一个 block，10 个输出需要 3 个 block。这里特意用很小的线程数，是为了让表格可手算；不是推荐用 4 线程 block 追求性能。

预期的 10 行 sum 为 3、6、9、12、15、18、21、24、27、19，最后显示 PASS naive stencil: 10 elements; input and output are separate。

### 10.3.1 为什么这里暂时不需要 block 内屏障

每个线程读 input，整个 kernel 不修改 input；每个有效线程写自己独占的 output[i]。一个线程不等待另一个线程产生自己的输入。

所以 __syncthreads 不是每个 kernel 必须添加的固定语法。只有当线程之间真的有需要协调的内存访问或阶段依赖，才分析应该放什么同步。

主机的 cudaDeviceSynchronize 是另外一个层次：它让 CPU 等设备完成，之后才取回和校验 output。它不能放在 CPU 上“代替”一个正在运行的 kernel 内部缺失的线程协作屏障。

### 10.3.2 朴素版本的重复读取在哪里

看中间的 output[4]、output[5]、output[6]：

```text
output[4] 需要 input[3], input[4], input[5]
output[5] 需要 input[4], input[5], input[6]
output[6] 需要 input[5], input[6], input[7]
                              ↑
                    input[5] 被三个线程需要
```

这是引入 shared 的机会，但不是已经证明了 DRAM 必须读同一个数三次。缓存可能已经减少实际显存流量。我们统计的是源码层的需求重叠，优化是否更快要在正确性确认后按第 9 章方法测量。

NaN 输出哨兵用于识别未写出的结果；API 状态与逐元素 finite/期望值检查保留。不能在开始优化时把这些验证删掉。

## 10.4 设计协作：谁加载哪一个位置

现在把 CPU 模型的“先加载再使用”翻译成 block 内线程分工。

每个线程负责中心位置，两个边界线程多做一件事：

| 线程 | 自己的中心加载 | 额外加载 |
|---|---|---|
| tid=0 | input[base] → tile[1] | input[base-1] → tile[0]，边界外写 0 |
| tid=1 | input[base+1] → tile[2] | 无 |
| tid=2 | input[base+2] → tile[3] | 无 |
| tid=3 | input[base+3] → tile[4] | input[base+4] → tile[5]，边界外写 0 |

没有规定每个线程必须只搬一个元素。协作原则是每个需要读取的 tile 位置都有清楚的写入者，且写入完成后才允许使用。

### 10.4.1 Shared memory 是每个 block 的工作区

写 __shared__ float tile[6]，表示这个 kernel 中每个 block 使用一份供其线程共同访问的局部工作区。不是 4 个线程各有一份互不相干的 tile，也不是整个 grid 的所有 block 共用唯一一份 tile。

```text
block 0 的 tile：0,1,2,3,4,5      block 1 的 tile：4,5,6,7,8,9
       ↑ 仅本block线程协作                 ↑ 另一份逻辑工作区
```

相邻 block 的 halo 有数据重复，这完全正常：它们各自从输入准备自己需要的一小块，不依赖另一个 block 先运行。

普通 shared 工作区随 block 的执行使用，不是可在主机长期保存并跨 kernel 继续访问的数组。CPU 不用为这个静态 tile 调用 cudaMalloc，更不能在 kernel 完成后拿它的地址读取结果。需要长期保存的输出仍写入 global output。[CUDA 的 shared memory 说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory)

### 10.4.2 一次协作阶段分成三个动作

```text
线程0：写中心和左halo ─┐
线程1：写中心 ─────────┤
线程2：写中心 ─────────┼→ __syncthreads → 各线程读取自己的三个局部位置
线程3：写中心和右halo ─┘
```

__syncthreads 在这里不是复制函数。复制由前面的赋值完成；屏障负责让合作线程在相应阶段对齐，并为相关访问建立顺序。

也不要把它理解成“每个线程到了以后就单独继续”。本例要求 block 中所有线程到达同一个协作点，确保消费者不会越过准备阶段读取尚未写好的位置。

## 10.5 案例三：固定四线程的 shared 版本

本例先把 B 固定为 4，让你看到大小和索引是一组合同。动态大小放在下一阶段，避免一次引入太多参数。

完整文件 code/10_shared/stencil_static.cu：

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

constexpr int block_width = 4;

__global__ void stencil_static(const float* input, float* output, int n) {
    if (blockDim.x != block_width) return; // Uniform check of this kernel's teaching contract.
    __shared__ float tile[block_width + 2];
    const int tid = threadIdx.x;
    const int base = blockIdx.x * block_width;
    const int i = base + tid;
    tile[tid + 1] = i < n ? input[i] : 0.0f;
    if (tid == 0) tile[0] = base > 0 ? input[base - 1] : 0.0f;
    if (tid == block_width - 1) {
        tile[block_width + 1] = base + block_width < n ? input[base + block_width] : 0.0f;
    }
    __syncthreads();
    if (i < n) output[i] = tile[tid] + tile[tid + 1] + tile[tid + 2];
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    constexpr int n = 10, threads = block_width;
    const int blocks = (n + threads - 1) / threads;
    const std::size_t bytes = n * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i + 1);
    const float expected[n] = {3, 6, 9, 12, 15, 18, 21, 24, 27, 19};
    float *d_input = nullptr, *d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
        if (!check(cudaMemset(d_output, 0xff, bytes), "poison output")) break;
        stencil_static<<<blocks, threads>>>(d_input, d_output, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait")) break;
        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            std::cout << "i=" << i << " sum=" << output[i] << '\n';
            if (!std::isfinite(output[i]) || output[i] != expected[i]) ok = false;
        }
    } while (false);
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS static shared stencil: 10 elements; 4 threads per block\n";
    return 0;
}
```

编译运行：

```bash
nvcc -std=c++17 -O3 -lineinfo code/10_shared/stencil_static.cu -o stencil_static
./stencil_static
```

预期数值应与朴素版逐项相同，最后为 PASS static shared stencil: 10 elements; 4 threads per block。这个例子比较的是正确结果，不输出速度排名。

### 10.5.1 新增语法逐项解释

| 语法 | 本例含义 |
|---|---|
| constexpr int block_width=4 | 编译时可确定的大小，用来声明固定数组 |
| __shared__ float tile[block_width+2] | 每 block 的 6 个 float 工作区 |
| blockIdx.x*block_width | 这个 block 所负责输出的起点 base |
| tile[tid+1] | 本线程准备的中心位置 |
| __syncthreads() | 从协作准备阶段进入消费阶段的 block 屏障 |

最前面的 blockDim.x!=block_width 检查在整个 block 中结果一致。如果误用其他宽度启动，这个教学 kernel 全部线程直接返回，不进入后面的 tile 操作，输出 NaN 哨兵会让结果校验失败。CUDA 启动本身不一定报错：硬件允许的形状和算法要求的形状是两件事。

它与“部分线程因为 i>=n 提前返回”不是同一种控制流。这里的正确启动仍让每个线程参与准备与屏障，最后才按 i<n 筛选输出。

### 10.5.2 尾部两个没有输出的线程为什么还要工作

最后 block 的 base=8。四个线程的全局 i 分别是 8、9、10、11，其中只有前两个有输出。

但 tid=1 计算 output[9] 需要读取 tile[1]、tile[2]、tile[3]。tile[3] 正是 tid=2 对应的中心槽位；它应该提供越界邻居的零值。

```text
最后 block 的 tile 索引：0  1  2  3  4  5
准备好的值：            8  9 10  0  0  0
output[9] 使用：           9 10  0   → 19
```

所以不能只说“tid=2 没有输出，就让它什么都不做”。它仍承担本组局部数据初始化的职责。

程序用条件表达式让有效输入加载真实值，无效位置写 0；所有正确启动的线程到达同一屏障，然后只有有效输出线程执行最后写回。这样把“是否加载有效全局数据”“是否参与协作”“是否写输出”三个问题分开了。

### 10.5.3 为什么声明 shared 不等于初始化它

tile 的每个被读取位置都必须在这次使用前被写入。__shared__ 不会自动把所有槽位清零，也不会从 global input 自动复制内容。

本例四个中心槽位与两个 halo 都有写入责任人。同步只能排序已发生的写入，不能把漏写的位置修好。检查 shared 程序时，要先问“谁写这个槽位”，再问“写和读之间有没有正确顺序”。

### 10.5.4 为什么不凭“都在一个 warp”删除屏障

这里 4 个线程确实属于同一个 warp 的一部分，但这不是允许跨线程随意传递 shared 数据的同步协议。线程执行与编译器对内存访问的处理不能靠“看起来一起走”替代明确的同步。

本章使用 block 屏障表达 block 范围的协作；之后学习 warp 原语时，会另讲参与掩码和 warp 同步的适用条件。不要在没有建立替代协议前直接删除这行。

### 10.5.5 数读取需求，不承诺两倍速度

对不涉及全局边界的 B=4 block，朴素版四个输出各读三个输入，共 12 次源码层输入读取需求；shared 版准备四个中心和两个 halo，共 6 次全局输入读取需求，再从 shared 读取计算。

本例 n=10，朴素版有效全局输入需求为 3n-2=28 次；shared 版十个有效中心加四个有效 halo，共 14 次。边界补零不需要读取全局数组。

但是 shared 版新增了工作区写入、读取、屏障和索引控制。缓存也可能使朴素版实际 DRAM 流量更少。因此 28/14=2 只说明这个记账模型下的需求数量比，绝不证明运行时间恰好减半。

要评价性能，下一阶段应使用足够的规模、相同数据与边界规则、完整验证和正确计时。小案例的价值是能看清每个槽位，不是跑满 GPU。

## 10.6 案例四：把 shared 大小变成启动参数

固定版本的 __shared__ float tile[6] 在编译时就确定了数组长度。如果想对同一个 kernel 选择不同 block 宽度，需要让每 block 工作区大小与启动配置配套变化。

CUDA 的动态 shared 形式是：

```cpp
extern __shared__ float tile[];
```

这里的 extern 不是要求你在另一个 .cpp 中定义同名普通数组，而是这类 CUDA 动态 shared 声明的写法。方括号中不填固定长度，大小来自启动配置的第三项。

```cpp
stencil_dynamic<<<blocks, threads, shared_bytes>>>(d_input, d_output, n);
```

### 10.6.1 第三项究竟为谁申请多少空间

| 启动项 | 本例含义 |
|---|---|
| blocks | 多少个 block |
| threads | 每 block 有多少个线程 |
| shared_bytes | 每 block 的动态 shared 字节数 |

半径 1 需要 threads+2 个 float，所以 shared_bytes=(threads+2)*sizeof(float)。

例如 threads=32，长度为 34 个 float，每 block 需要 136 字节，不是 34 字节，也不是 32*4=128 字节。后一种漏掉两个 halo，会让程序越过申请的动态工作区。

不要乘 blocks 再传入第三项。它本来就是每个 block 的申请量。全 grid 有很多 block，不代表所有 block 同时驻留，也不要求你在主机为整个 grid 一次手工拼出 shared 数组。

动态 shared 与静态 shared 可以同时存在，但第三项仅指定动态部分。本案例不另外声明静态 shared，先避免两种预算混淆。

### 10.6.2 完整代码：主动覆盖边界

完整文件 code/10_shared/stencil_dynamic.cu：

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void stencil_dynamic(const float* input, float* output, int n) {
    extern __shared__ float tile[];
    const int tid = threadIdx.x;
    const int width = blockDim.x;
    const int base = blockIdx.x * width;
    const int i = base + tid;
    tile[tid + 1] = i < n ? input[i] : 0.0f;
    if (tid == 0) tile[0] = base > 0 && base - 1 < n ? input[base - 1] : 0.0f;
    if (tid == width - 1) {
        tile[width + 1] = base + width < n ? input[base + width] : 0.0f;
    }
    __syncthreads();
    if (i < n) output[i] = tile[tid] + tile[tid + 1] + tile[tid + 2];
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool run_case(int n, int threads) {
    if (n == 0) {
        std::cout << "EMPTY n=0 threads=" << threads << "; no allocation or kernel for this case\n";
        return true;
    }
    const int blocks = n / threads + (n % threads != 0);
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t shared_bytes = static_cast<std::size_t>(threads + 2) * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i % 7 - 3);
    float *d_input = nullptr, *d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
        if (!check(cudaMemset(d_output, 0xff, bytes), "poison output")) break;
        stencil_dynamic<<<blocks, threads, shared_bytes>>>(d_input, d_output, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait")) break;
        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            const int left = i > 0 ? (i - 1) % 7 - 3 : 0;
            const int center = i % 7 - 3;
            const int right = i + 1 < n ? (i + 1) % 7 - 3 : 0;
            const float expected = static_cast<float>(left + center + right);
            if (!std::isfinite(output[i]) || output[i] != expected) {
                std::cerr << "mismatch n=" << n << " threads=" << threads << " i=" << i << '\n';
                ok = false;
                break;
            }
        }
    } while (false);
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (ok) {
        std::cout << "PASS n=" << n << " threads=" << threads << " blocks=" << blocks
                  << " shared_bytes=" << shared_bytes << '\n';
    }
    return ok;
}

int main() {
    try {
        int device = -1;
        cudaDeviceProp properties = {};
        if (!check(cudaGetDevice(&device), "current device")) return 1;
        if (!check(cudaGetDeviceProperties(&properties, device), "device properties")) return 1;
        const int widths[] = {1, 4, 7, 32};
        const int sizes[] = {0, 1, 3, 4, 5, 6, 7, 8, 10, 17, 31, 32, 33};
        int cases = 0;
        for (int threads : widths) {
            const std::size_t shared_bytes = static_cast<std::size_t>(threads + 2) * sizeof(float);
            if (threads > properties.maxThreadsPerBlock || threads > properties.maxThreadsDim[0] ||
                shared_bytes > properties.sharedMemPerBlock) {
                std::cerr << "teaching configuration exceeds current device limits\n";
                return 1;
            }
            for (int n : sizes) {
                if (!run_case(n, threads)) return 1;
                ++cases;
            }
        }
        std::cout << "PASS dynamic matrix: " << cases << " cases, including 4 empty cases\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
```

编译运行：

```bash
nvcc -std=c++17 -O3 -lineinfo code/10_shared/stencil_dynamic.cu -o stencil_dynamic
./stencil_dynamic
```

程序不接受终端参数，使用源码中明确列出的测试矩阵。这样先练“怎样挑边界”，再在后续扩展中引入更多可调参数。

widths 的四个值是 1、4、7、32，sizes 的十三个值是 0、1、3、4、5、6、7、8、10、17、31、32、33。组合得到 52 个案例，其中 4 个 n=0 不申请数组、不启动 kernel，其余 48 个执行逐元素检查。

main 在矩阵之前查询设备，所以程序整体仍然需要 CUDA 环境；不能因为存在空案例就把整次运行当成不需要 GPU 的实验。

### 10.6.3 为什么特别选择这些尺寸

- 1 个线程时，tid=0 同时也是最后一个线程，要负责中心、左 halo、右 halo 三个不同槽位；
- n=1 时，两边都用零，不能有任何真实越界读取；
- n=B-1、B、B+1 跨过一个 block 的边界；
- B=7 说明索引和 halo 逻辑不是只对 2 的幂成立；
- n=33、B=32 时，最后 block 只有一个有效输出，但协作准备仍必须完整。

这些尺寸不是充分证明所有可能输入正确，而是一组有针对性的结构边界测试。大规模随机数据、不同输入和实际检测工具还可以继续增加。

### 10.6.4 输入不再只用递增正数

input[i]=i%7-3 循环产生 -3、-2、-1、0、1、2、3。参考值用主机上的整数表达式独立计算左右中三个位置，再转换为 float。

这让负值、零值和跨周期变化都参与检验，减少只对等差正数成立的错误恰好通过。因为数值很小，这些和仍可精确表示；仍保留 finite 检查防止 NaN 蒙混过关。

如果后来加平均除以 3.0f，某些结果不能被 float 精确表示，就要重新选择合理的误差判断，而不是删除检查。

### 10.6.5 如何核对输出

例如以下配置的元数据可以提前算出：

```text
EMPTY n=0 threads=1; no allocation or kernel for this case
PASS n=1 threads=1 blocks=1 shared_bytes=12
...
PASS n=33 threads=32 blocks=2 shared_bytes=136
PASS dynamic matrix: 52 cases, including 4 empty cases
```

省略号表示中间其他配置行，不是程序真的只检查这两项。每一条 PASS n=... 都在对应案例全部元素校验和资源释放检查之后打印。

不要只看最后有一个 PASS 就忽略前面的失败诊断；本程序遇到任一失败会返回非零，正常情况下不应出现 mismatch、CUDA API 错误或异常文本。

### 10.6.6 先用工具确认边界，再谈速度

在具备 Compute Sanitizer 的机器上，可以对这个正常矩阵运行：

```bash
compute-sanitizer --tool memcheck ./stencil_dynamic
compute-sanitizer --tool racecheck ./stencil_dynamic
compute-sanitizer --tool synccheck ./stencil_dynamic
```

分别关注非法访存、shared 数据竞争和同步使用问题。它们不是数学参考值校验的替代；也不要把检测工具下的时间用于性能排名。

本轮编辑环境尚未执行这些命令。它们是你在 CUDA 环境中的验收入口，不能提前记录为“检测通过”。

## 10.7 屏障不是“让 GPU 慢一点”：它建立了什么关系

先不要记 API 名字，回到两个线程的具体问题：

```text
T0：把 17 写入 tile[0]     T1：读取 tile[0]
```

如果没有建立先后关系，T1 可能先读，T0 随后才写。“T0 的编号比较小”不是顺序保证；“上一段代码在文件里比较靠上”也不是跨线程顺序保证。代码的上下顺序首先属于执行它的那个线程，不能自动扩展到其他线程。

### 10.7.1 同步点把两个阶段接起来

本章使用的完整协议是：

```text
每个线程写好自己负责的位置
                 ↓
      全 block 的同一个屏障
                 ↓
每个线程读取其他线程准备的位置
```

这里有两类责任，缺一不可：

1. **数据责任**：每一个将被读取的位置都有人写，而且不是两个线程无保护地争写。
2. **同步责任**：这些写入与后面的跨线程读取之间有正确的顺序。

屏障能解决第二类，不能替你解决第一类。假设大家都等到了，但没有任何线程写 tile[5]，屏障之后 tile[5] 仍然没有合法初值；假设两个人同时争写 tile[2]，在后面加屏障也不能追溯地消除前面的竞争。

现代 CUDA 文档对 __syncthreads() 的说明还涉及已经退出的线程：同一 block 中尚未退出的线程需要到达相应同步点或者退出。它同时为参与线程建立内存顺序。入门算法不要据此把“提前 return”当成修复尾部的诀窍——即使某种提前退出没有挂起，也可能漏掉邻居依赖的加载。我们在 10.5 的尾部例子中已经看到：没有输出的 tid=2，仍必须把一个 shared 槽位写成 0。[官方同步原语说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#synchronization-primitives)

### 10.7.2 条件屏障：问整个 block 是否作出相同选择

下面是理解条件的局部示意，不是完整程序：

```cpp
if (blockIdx.x == 0) {
    __syncthreads();
}
```

同一个 block 内 blockIdx.x 相同，因此该 block 要么所有线程都进入，要么所有线程都不进入。这种条件是一致的。但是它**不让其他 block 等待 block 0**。

换成以下写法就不同了：

```cpp
// 不要采用：条件可能在同一 block 内不同。
if (threadIdx.x < 16) {
    __syncthreads();
}
```

如果 block 有 64 个线程，只有前 16 个进入。不能用“只同步前 16 个”解释这段代码；__syncthreads 不是一个自动根据分支缩小参与集合的屏障。官方要求条件在 block 内一致；不满足要求可能挂起或产生非预期行为。不要断言每次都会永久死锁，也不要因为一次输出正确就认定它可移植。

对尾部有效性判断，推荐的结构是：

```cpp
// 示意：先处理每个槽位的有效性，再由所有线程同步。
tile[tid] = i < n ? input[i] : 0.0f;
__syncthreads();
if (i < n) {
    // 在这里使用已经准备好的数据。
}
```

“条件赋值”与“条件屏障”不是同一个概念。前者只决定写入真实数据还是填充值；后者决定线程是否参与协议。

循环也一样：如果每个线程的循环次数不同，却把 block 屏障放在循环里，就要重新检查各线程是否按相同阶段到达对应屏障。不要靠屏障在源文件中出现了相同次数来推断正确性，执行路径和每一轮的含义才是重点。

### 10.7.3 fence 为什么不能代替等待

fence 可以译作“内存栅栏”。它约束调用线程的内存操作顺序，作用范围可以是 block、device 或 system；它本身不组织一群线程集合等待，也不会自动让消费者知道生产者已经完成。

把早先例子改成如下局部代码，仍不是正确的共享通信协议：

```cpp
// 错误思路的示意，勿作为程序运行。
tile[tid] = static_cast<float>(tid);
__threadfence_block();
float value = tile[(tid + 1) % blockDim.x];
```

tid=0 到达自己的 fence，只说明它执行了自己的顺序约束，不能据此断定 tid=1 已写完 tile[1]。其他线程也执行 fence，不会自动把所有 fence 合成一道集合屏障。正确版本在写和读之间使用本章的 __syncthreads()。

不要把 fence 描述成“强行刷新所有缓存”或“全 GPU 同步”。复杂的生产者/消费者协议还会涉及原子操作、发布与获取语义及作用域；它们适合在后续原子与并发章节逐步学习，而不是在这里用普通 flag 自旋猜一个协议。[官方内存 fence 说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#memory-fence-functions)

下面只比较本章涉及的用法，不是完整的并发接口大全：

| 接口 | 谁调用 | 当前最该记住的作用 | 不能推断什么 |
|---|---|---|---|
| __syncthreads() | kernel 内同一 block 的线程 | 按 block 协作协议等待并建立内存顺序 | 其他 block 或 CPU 也到达了 |
| __threadfence_block() | kernel 内某个线程 | 约束该线程内存操作在 block 作用域中的顺序 | 邻居已经完成计算 |
| cudaDeviceSynchronize() | 本章的主机代码 | 等待此前提交的设备工作并检查执行错误 | 修复 kernel 内本来就存在的竞争 |

CPU 等到一个错误 kernel 完成，得到的是“它已执行完”的事实，不是“内部算法变正确了”。

### 10.7.4 同一个 warp 也不是免同步凭证

SIMT 描述执行模型，不等于一个 C++ 局部变量可以任意跨线程通信。尤其不能把 warp 内程序隐含地当成一段永远锁步的串行脚本。

如果通信只发生在一个 warp 内，可以在明确参与集合和掩码的前提下学习 __syncwarp 或 shuffle；但不能将一个 64 线程 block 中的 __syncthreads 全部替换为 __syncwarp()。前者覆盖整个 block 的协作，后者不能替第二个 warp 与第一个 warp 建立所需的完整关系。第 12 章再专门讲掩码和 warp 通信，本章保留清楚的 block 协议。

## 10.8 为什么循环复用 shared 时经常需要两次屏障

上面的 stencil 每次 kernel 只准备一个 tile，用完就结束。接下来做一个更小的协议实验：同一个 block 在三个阶段反复覆盖同一块 shared 空间。

这不是三轮邻域平滑，也不是性能基准；先用极简单的数值，把“写完才能读”和“读完才能覆盖”分开看清楚。

### 10.8.1 用四个线程预测一轮

为了纸上好算，先假设 B=4。第 phase 阶段，线程 tid 写：

```text
tile[tid] = 100 × phase + tid
```

接着每个线程读取右边线程准备的值，最后一个绕回第 0 个：

| phase | 准备好的 tile | 四个线程读到的值 |
|---|---|---|
| 0 | 0, 1, 2, 3 | 1, 2, 3, 0 |
| 1 | 100, 101, 102, 103 | 101, 102, 103, 100 |
| 2 | 200, 201, 202, 203 | 201, 202, 203, 200 |

这里的绕回是这个实验特意定义的环状读取，不是前面 stencil 的零边界规则。改变边界规则必须明确说出来，不能把两个模型混在一起。

第一次屏障放在“准备”与“读取”之间，保证这一阶段所有生产者完成。第二次屏障放在“读取”与“下一阶段准备”之间，保证没有线程还需要旧 tile。

```text
阶段 p：写 tile
        │
        A 屏障：等生产者
        │
        读 tile → 写本阶段独立的 output
        │
        B 屏障：等消费者
        │
阶段 p+1：覆盖同一个 tile
```

### 10.8.2 少了第二次会怎样

设 T0 本阶段要读 T1 的槽位。如果没有 B 屏障，可能出现这样的合法调度交错：

| 时刻 | T0 | T1 |
|---|---|---|
| 1 | 完成 A 屏障，暂时尚未读取 tile[1] | 完成 A 屏障 |
| 2 | 尚未读取 | 读完自己的邻居 |
| 3 | 尚未读取 | 进入下一轮，把 tile[1] 覆盖成 101 |
| 4 | 读 tile[1]，却拿到了下一轮的数据 | 已经进入下一轮 |

这张表不是在声称硬件一定按表执行，而是提供一个协议没有排除的坏交错。只要算法的正确性依赖“这种交错碰巧没有发生”，就不能称为可靠同步。

第一道屏障只隔开了本轮的“写→读”，没有阻止快线程跨入下一轮。每一轮都有一道 A 屏障也不够：覆盖发生在下一道 A 屏障之前。

### 10.8.3 完整安全程序

实际程序使用 64 个线程、3 个阶段，避免把实验限定为恰好一个 warp。保存为 code/10_shared/reuse_tile.cu：

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>

constexpr int width = 64;
constexpr int phases = 3;

__global__ void reuse_tile(float* output) {
    if (blockDim.x != width) return;
    __shared__ float tile[width];
    const int tid = threadIdx.x;
    for (int phase = 0; phase < phases; ++phase) {
        tile[tid] = static_cast<float>(100 * phase + tid);
        __syncthreads();  // All producers finished this phase.
        output[phase * width + tid] = tile[(tid + 1) % width];
        __syncthreads();  // All consumers finished before tile is overwritten.
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    try {
        constexpr int count = width * phases;
        constexpr std::size_t bytes = count * sizeof(float);
        std::vector<float> output(count);
        float* d_output = nullptr;
        bool ok = false;
        do {
            if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
            if (!check(cudaMemset(d_output, 0xff, bytes), "poison output")) break;
            reuse_tile<<<1, width>>>(d_output);
            if (!check(cudaGetLastError(), "launch")) break;
            if (!check(cudaDeviceSynchronize(), "wait")) break;
            if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
            ok = true;
            for (int phase = 0; phase < phases; ++phase) {
                for (int tid = 0; tid < width; ++tid) {
                    const float expected = static_cast<float>(100 * phase + (tid + 1) % width);
                    const float actual = output[phase * width + tid];
                    if (!std::isfinite(actual) || actual != expected) {
                        std::cerr << "mismatch phase=" << phase << " tid=" << tid << '\n';
                        ok = false;
                    }
                }
            }
        } while (false);
        if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
        if (!ok) return 1;
        for (int phase = 0; phase < phases; ++phase) {
            std::cout << "phase=" << phase << " first=" << output[phase * width]
                      << " last=" << output[phase * width + width - 1] << '\n';
        }
        std::cout << "PASS shared reuse: 3 phases, 64 threads, 192 checked values\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
```

新增表达式逐个看：

- phase 从 0 到 2，是阶段编号，不是线程编号。
- (tid+1)%width 让最后一个线程的邻居索引绕回 0。
- phase*width+tid 把三行输出压成一维数组。每个阶段占 64 个位置，每个线程只写自己的位置。
- 两个 __syncthreads 都在所有线程执行的循环体中；phases 对整个 block 相同。
- 开头的 blockDim.x 检查对同一 block 一致，是固定形状的算法契约检查。本程序只启动一个 width=64 的 block。
- 输出独立保存每一阶段的结果，否则只验证最后一轮会漏掉早期读错。

### 10.8.4 编译、运行与预期结果

从教材根目录执行：

```bash
nvcc -std=c++17 -lineinfo code/10_shared/reuse_tile.cu -o reuse_tile
./reuse_tile
compute-sanitizer --tool racecheck ./reuse_tile
compute-sanitizer --tool synccheck ./reuse_tile
```

第一条编译 CUDA C++17 源码，-lineinfo 保留与源码行关联的信息，-o 指定可执行文件名。后两条分别检查共享内存竞争与同步使用；不是测量性能。

数值输出应为：

```text
phase=0 first=1 last=0
phase=1 first=101 last=100
phase=2 first=201 last=200
PASS shared reuse: 3 phases, 64 threads, 192 checked values
```

这仍是预期输出，当前编辑环境没有执行 CUDA 程序。程序实际检查全部 192 个值，不只是显示的六个数。

默认示例始终保留正确同步，不提供会自动启动错误 kernel 的模式。若今后在自己的隔离测试环境研究故意错误，应显式建立副本，避免覆盖正确基线，并遵守第 7 章对故障实验的运行限制。一次 racecheck 没报错也不是任意并发程序正确性的证明。

### 10.8.5 是否每段 shared 代码都机械加两次

不需要。判断方法是问：“读取结束以后，这块存储会不会在其他线程读完前被人覆盖？”

本程序最后一阶段之后就结束，最后的 B 屏障可以经过证明后省掉；教学版本为了保持每轮同一结构而保留它。前面的单次 stencil 在写回 output 后不再覆盖 tile，也不需要凭空加第二次屏障。

反过来，下一轮使用双缓冲的另一块 shared 区域时，等待位置也可能改变。所谓“两次屏障”不是咒语，而是当前单缓冲协议对两个依赖方向的具体实现。

## 10.9 半径变大：一个线程可以加载多个位置

前面的加载代码为半径 1 专门指定了两个 halo 的负责人。如果半径是 8，而 block 只有 4 个线程，还能“只让头尾两个线程各加载一个”吗？不能，因为所需数据位置已经远远超过两个。

解决方法不是不断加特殊 if，而是把“加载多少个槽位”与“有多少个线程”分开。

### 10.9.1 从输出范围推导输入范围

设：

- B：一个 block 的线程数，也是它负责的输出位置数上限。
- base：这个 block 的第一个输出索引。
- R：邻域半径，允许为 0。
- tid：线程在 block 中的编号，0..B-1。
- j：shared 中的槽位编号，不是 global 数组索引。

block 的输出范围是 base..base+B-1。左右各扩展 R，所需输入范围就是：

```text
base-R .. base+B-1+R
```

包含 B+2R 个槽位。把第一个输入位置放在 tile[0]，自然得到：

```text
global = base + j - R
```

注意这里的 global 只是程序中的变量名字，表示全局数组索引，不是 CUDA 的 __global__ 修饰符。

线程 tid 对应的中心在 tile[tid+R]；向左右各读 R 个位置，最左槽位为 tid，最右槽位为 tid+2R。因此输出循环可以直接累加 tile[tid+offset]，其中 offset 从 0 到 2R。

### 10.9.2 B=4、R=3：十个槽位怎样分给四个线程

所有线程按步长 B 领取槽位：

```cpp
for (int j = tid; j < B + 2 * R; j += B) {
    // 加载 tile[j]。
}
```

手算得到：

| 线程 | 第一次 j | 第二次 j | 第三次 j |
|---|---|---|---|
| T0 | 0 | 4 | 8 |
| T1 | 1 | 5 | 9 |
| T2 | 2 | 6 | 无，10 已越过槽位范围 |
| T3 | 3 | 7 | 无，11 已越过槽位范围 |

每个有效 j 都能唯一写成“tid 加上若干个 B”。因此不会遗漏，也不会有两个线程领取同一个槽位。

线程数并不是工作项数。“一个线程只能搬一个数”从来不是 CUDA 的规定。上一章的 grid-stride 思想在这里缩小到 block 内的协作加载。

加载循环的次数可以不同，因为循环内部没有 block 屏障。退出各自的加载循环后，所有线程在循环外执行同一个屏障，然后开始消费。

```text
T0 加载 3 个槽位 ──┐
T1 加载 3 个槽位 ──┤
T2 加载 2 个槽位 ──┼─ 屏障 ─ 输出阶段
T3 加载 2 个槽位 ──┘
```

如果把屏障放入每次加载的循环体，本例不同线程的迭代次数就不一致，破坏了清楚的阶段协议。

### 10.9.3 左边界为什么要特别留意类型

第一个 block 的 base=0，当 R=3、j=0 时，global=-3，应该填 0。若无意中把所有索引混成 unsigned，减法可能绕成极大的正数，导致边界逻辑难以理解。

本例限制 n≤1048576、R≤64、B≤1024，并用 int 表示这些索引。这样 base+j-R 可以合理表达负数，且所有中间值远小于 int 上限。分配字节数再转换成 std::size_t。

这不是说任意规模都能用 int。规模扩大后必须重新证明加法、乘法、网格计算和字节数不溢出；仅仅把 n 改成 long long 而漏改其他表达式，不算完成大规模支持。

### 10.9.4 完整程序与受限参数

code/10_shared/stencil_radius.cu：

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

__global__ void stencil_radius(const float* input, float* output, int n, int radius) {
    extern __shared__ float tile[];
    const int tid = threadIdx.x;
    const int width = blockDim.x;
    const int base = blockIdx.x * width;
    const int slots = width + 2 * radius;
    for (int j = tid; j < slots; j += width) {
        const int global = base + j - radius;
        tile[j] = global >= 0 && global < n ? input[global] : 0.0f;
    }
    __syncthreads();
    const int i = base + tid;
    if (i < n) {
        float sum = 0.0f;
        for (int offset = 0; offset <= 2 * radius; ++offset) {
            sum += tile[tid + offset];
        }
        output[i] = sum;
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool parse_decimal(const char* text, int minimum, int maximum, int& result) {
    if (text == nullptr || *text == '\0') return false;
    int value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return false;
        const int digit = *p - '0';
        if (value > (maximum - digit) / 10) return false;
        value = value * 10 + digit;
        if (value > maximum) return false;
    }
    if (value < minimum) return false;
    result = value;
    return true;
}

bool run_case(int n, int radius, int threads) {
    if (n == 0) {
        std::cout << "EMPTY n=0 radius=" << radius << " threads=" << threads
                  << "; no CUDA call for this case\n";
        return true;
    }
    int device = -1;
    cudaDeviceProp prop = {};
    if (!check(cudaGetDevice(&device), "current device")) return false;
    if (!check(cudaGetDeviceProperties(&prop, device), "device properties")) return false;
    const int blocks = n / threads + (n % threads != 0);
    const std::size_t shared_bytes = static_cast<std::size_t>(threads + 2 * radius) * sizeof(float);
    if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
        blocks > prop.maxGridSize[0] || shared_bytes > prop.sharedMemPerBlock) {
        std::cerr << "configuration exceeds device limits\n";
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = static_cast<float>(i % 7 - 3);
    float *d_input = nullptr, *d_output = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMalloc(&d_input, bytes), "allocate input")) break;
        if (!check(cudaMalloc(&d_output, bytes), "allocate output")) break;
        if (!check(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
        if (!check(cudaMemset(d_output, 0xff, bytes), "poison output")) break;
        stencil_radius<<<blocks, threads, shared_bytes>>>(d_input, d_output, n, radius);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait")) break;
        if (!check(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            int expected = 0;
            for (int offset = -radius; offset <= radius; ++offset) {
                const int neighbor = i + offset;
                if (neighbor >= 0 && neighbor < n) expected += neighbor % 7 - 3;
            }
            if (!std::isfinite(output[i]) || output[i] != static_cast<float>(expected)) {
                std::cerr << "mismatch i=" << i << " actual=" << output[i]
                          << " expected=" << expected << '\n';
                ok = false;
                break;
            }
        }
    } while (false);
    if (d_output != nullptr && !check(cudaFree(d_output), "free output")) ok = false;
    if (d_input != nullptr && !check(cudaFree(d_input), "free input")) ok = false;
    if (ok) {
        std::cout << "PASS n=" << n << " radius=" << radius << " threads=" << threads
                  << " blocks=" << blocks << " shared_bytes=" << shared_bytes << '\n';
    }
    return ok;
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--matrix") {
            const int sizes[] = {0, 1, 3, 4, 5, 17, 65};
            const int radii[] = {0, 1, 2, 8};
            const int widths[] = {1, 4, 7, 32};
            int cases = 0;
            for (int radius : radii) {
                for (int threads : widths) {
                    for (int n : sizes) {
                        if (!run_case(n, radius, threads)) return 1;
                        ++cases;
                    }
                }
            }
            std::cout << "PASS radius matrix: " << cases << " cases, including 16 empty cases\n";
            return 0;
        }
        int n = 10, radius = 2, threads = 4;
        if (argc != 1 && (argc != 4 ||
            !parse_decimal(argv[1], 0, 1048576, n) ||
            !parse_decimal(argv[2], 0, 64, radius) ||
            !parse_decimal(argv[3], 1, 1024, threads))) {
            std::cerr << "Usage: stencil_radius [n radius threads] | --matrix\n"
                      << "n: 0..1048576, radius: 0..64, threads: 1..1024\n";
            return 2;
        }
        return run_case(n, radius, threads) ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
```

主机校验没有照抄 shared 索引公式，而是对每个输出直接遍历 global 邻域。这样 CPU 与 GPU 更不容易共享同一个 tile 偏移错误。

输入取 i%7-3，包含负数、0 和正数。最多累加 129 个很小的整数，结果在 float 可以精确表示的整数范围内，因此这里可以用精确相等；不是说所有浮点程序都应如此验证。

parse_decimal 只接受非空十进制数字文本，在执行 value*10 前检查上界。它拒绝负号、字母、超范围参数以及溢出的长数字字符串。上限是教材主动设置的实验边界，不是 GPU 支持的最大问题规模。

### 10.9.5 自己改命令，而不是只会复制

从根目录运行：

```bash
nvcc -std=c++17 -lineinfo code/10_shared/stencil_radius.cu -o stencil_radius
./stencil_radius
./stencil_radius 17 8 4
./stencil_radius 0 2 4
./stencil_radius --matrix
```

参数按 n、radius、threads 顺序排列。例如 17 8 4 表示 17 个输入元素，左右各看 8 个邻居，每个 block 启动 4 个线程。这时每个 block 需要 4+2×8=20 个 float，共 80 字节动态 shared。

默认配置为 10 2 4，预期配置行是：

```text
PASS n=10 radius=2 threads=4 blocks=3 shared_bytes=32
```

n=0 的单独运行不会申请数组或调用 CUDA API，打印 EMPTY 后返回成功。--matrix 包含非空案例，因此整体仍需要 CUDA 环境；这与“其中的空案例不调用 CUDA”并不矛盾。

矩阵包含 7 种长度、4 种半径和4种线程数，共112项，其中16项为空输入。它覆盖 R=0、R>n、R>B、B=1、非2的幂线程数和尾部不足一个 block。动态版本 10.6 的52项继续保留，两套测试关注的参数维度不同。

负面输入测试例如：

```bash
./stencil_radius 17 -1 4
./stencil_radius 17 2 0
./stencil_radius 999999999999999999999999 2 4
```

这些命令应在主机参数解析阶段以退出码2拒绝，不会故意提交非法 GPU 工作。检查程序退出码而不只观察屏幕是否出现文字。

在真实 CUDA 环境进一步运行：

```bash
compute-sanitizer --tool memcheck ./stencil_radius --matrix
compute-sanitizer --tool racecheck ./stencil_radius --matrix
compute-sanitizer --tool synccheck ./stencil_radius --matrix
```

这些验收仍未在当前编辑环境执行。较大的测试矩阵会增加工具运行时间，不宜当作 benchmark。

### 10.9.6 如何检查自己的索引推导

以最后一个有效线程 tid=B-1 为例，它读取的最大槽位是：

```text
(B-1)+2R = (B+2R)-1
```

正好等于最后一个合法 shared 索引。这种“检查最小值和最大值”的推导，比只运行中间线程有用。

再检查全局边界：加载时同时判断 global>=0 和 global<n。shared 的所有槽位都会被赋真实值或0，不论对应线程最终是否有有效输出。

最后检查空输入：没有输出就不需要启动 block。不能把 blocks=0 直接交给 kernel 启动来假装完成空操作。

## 10.10 多轮邻域平均：两个全局数组怎样轮流当输入

10.8 的 shared 复用实验中，各阶段输入直接由 phase 和 tid 生成。现在真正做“上一轮输出成为下一轮输入”的迭代算法。

请先区分两个不同层次：

```text
一个 kernel 内：
global 输入 → block 自己的 shared tile → global 输出

多个 kernel 之间：
第 0 轮数组 → kernel 1 → 第 1 轮数组 → kernel 2 → 第 2 轮数组
```

shared tile 是单个 block 的临时工作区。它不是保存全局迭代状态的地方，更不是跨 block 的公共同步空间。

### 10.10.1 把原来的三点平均定义完整

仍使用零填充边界，每个输出都是“左+中+右”除以3。第一轮的输入是原始数组，第二轮的输入是完整的第一轮输出。

例如输入 [3,6,9]：

| 轮次 | 第 0 个 | 第 1 个 | 第 2 个 |
|---|---|---|---|
| 原始 | 3 | 6 | 9 |
| 第 1 轮 | (0+3+6)/3=3 | (3+6+9)/3=6 | (6+9+0)/3=5 |
| 第 2 轮 | (0+3+6)/3=3 | (3+6+5)/3=14/3 | (6+5+0)/3=11/3 |

边界处也固定除以3。如果除以有效邻居数量，第一轮会变成 [4.5,6,7.5]，是另一种边界算法。两种都可以研究，但不能拿一种 GPU 实现对照另一种 CPU 参考。

零填充的平均不保证全数组总和保持不变，边界的影响会随迭代向内传播。单元素 [-8] 的每一轮都会再除以3，而不是永远保持 -8。

### 10.10.2 为什么不能每个 block 自己循环所有轮

设两个 block 分别负责输出0..3和4..7。第一轮之后，block 0 的输出3在第二轮需要第一轮的输出4，也就是 block 1 的结果。

block 0 内执行 __syncthreads，只能等待本 block 的参与线程，不能保证 block 1 已完成第一轮。即使 block 0 已经完成，它也不能提前读取还没完成的邻块结果。

本章用最容易审查的方法：**每轮单独启动 kernel，并在主机等待和检查这一轮完成**，然后才进行下一轮。之后学习 stream 时，可以让同一 stream 的有序 kernel 提交表达这种依赖，把不必要的逐轮主机等待移出性能路径；这里先保留每轮验证，方便定位第一次出错的轮次。

不要用所有 block 自旋等待一个全局计数器来临时模仿 grid 屏障。未驻留的 block 可能还无法开始工作；执行模型和资源条件会让这种方案失败。协作式 grid 同步是后续专题，不是 __syncthreads 的隐含功能。

### 10.10.3 双缓冲：交换指针，不是复制数组

我们申请两个设备数组 A、B：

```text
开始：read → A（原始输入），write → B
第1轮：A → B；完成后交换角色
第2轮：B → A；完成后交换角色
第3轮：A → B；完成后交换角色
```

代码中的 read 与 write 都只是保存设备地址的主机变量。std::swap(read, write) 交换两个地址值，不搬运 n 个元素。

回忆第1章：“指针赋值不是数组复制”。这里正好利用这一点，以很小的主机操作切换角色。

原始分配地址 d_a、d_b 不被交换，用于最后各释放一次；不要释放 read 和 write 以后又释放 d_a、d_b，否则会把同一份分配释放两次。

CPU 参考同样有 reference 和 next 两个 vector。每轮先从完整 reference 计算 next，最后 reference.swap(next) 交换两个容器的内容存储角色。这种容器交换与裸指针交换不是同一个函数实现，但在此处都避免了为了进入下一轮逐元素复制数组。

### 10.10.4 完整程序

code/10_shared/stencil1d.cu 保留三点平均案例的名字，并扩展为逐轮验证的双缓冲程序。

```cpp
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

__global__ void stencil_average(const float* input, float* output, int n) {
    extern __shared__ float tile[];
    const int tid = threadIdx.x;
    const int width = blockDim.x;
    const int base = blockIdx.x * width;
    for (int j = tid; j < width + 2; j += width) {
        const int global = base + j - 1;
        tile[j] = global >= 0 && global < n ? input[global] : 0.0f;
    }
    __syncthreads();
    const int i = base + tid;
    if (i < n) {
        output[i] = (tile[tid] + tile[tid + 1] + tile[tid + 2]) / 3.0f;
    }
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool parse_decimal(const char* text, int minimum, int maximum, int& result) {
    if (text == nullptr || *text == '\0') return false;
    int value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return false;
        const int digit = *p - '0';
        if (value > (maximum - digit) / 10) return false;
        value = value * 10 + digit;
        if (value > maximum) return false;
    }
    if (value < minimum) return false;
    result = value;
    return true;
}

int main(int argc, char** argv) {
    try {
        int n = 10, steps = 3, threads = 4;
        if (argc != 1 && (argc != 4 ||
            !parse_decimal(argv[1], 0, 1048576, n) ||
            !parse_decimal(argv[2], 0, 32, steps) ||
            !parse_decimal(argv[3], 1, 1024, threads))) {
            std::cerr << "Usage: stencil1d [n steps threads]\n"
                      << "n: 0..1048576, steps: 0..32, threads: 1..1024\n";
            return 2;
        }
        if (n == 0) {
            std::cout << "EMPTY n=0; no CUDA calls\n";
            return 0;
        }
        std::vector<float> input(n), output(n);
        std::vector<double> reference(n), next(n);
        for (int i = 0; i < n; ++i) {
            input[i] = static_cast<float>(i % 17 - 8);
            reference[i] = input[i];
        }
        if (steps == 0) {
            output = input;
            if (output != input) return 1;
            std::cout << "PASS identity: n=" << n << " steps=0; no CUDA calls\n";
            return 0;
        }
        int device = -1;
        cudaDeviceProp prop = {};
        if (!check(cudaGetDevice(&device), "current device")) return 1;
        if (!check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
        const int blocks = n / threads + (n % threads != 0);
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        const std::size_t shared_bytes = static_cast<std::size_t>(threads + 2) * sizeof(float);
        if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
            blocks > prop.maxGridSize[0] || shared_bytes > prop.sharedMemPerBlock) {
            std::cerr << "configuration exceeds device limits\n";
            return 1;
        }
        float *d_a = nullptr, *d_b = nullptr;
        bool ok = false;
        do {
            if (!check(cudaMalloc(&d_a, bytes), "allocate A")) break;
            if (!check(cudaMalloc(&d_b, bytes), "allocate B")) break;
            if (!check(cudaMemcpy(d_a, input.data(), bytes, cudaMemcpyHostToDevice), "H2D")) break;
            float* read = d_a;
            float* write = d_b;
            bool all_steps_ok = true;
            for (int step = 0; step < steps; ++step) {
                for (int i = 0; i < n; ++i) {
                    const double left = i > 0 ? reference[i - 1] : 0.0;
                    const double right = i + 1 < n ? reference[i + 1] : 0.0;
                    next[i] = (left + reference[i] + right) / 3.0;
                }
                if (!check(cudaMemset(write, 0xff, bytes), "poison destination")) {
                    all_steps_ok = false;
                    break;
                }
                stencil_average<<<blocks, threads, shared_bytes>>>(read, write, n);
                if (!check(cudaGetLastError(), "launch") ||
                    !check(cudaDeviceSynchronize(), "wait") ||
                    !check(cudaMemcpy(output.data(), write, bytes, cudaMemcpyDeviceToHost), "D2H")) {
                    all_steps_ok = false;
                    break;
                }
                double max_error = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double error = std::fabs(static_cast<double>(output[i]) - next[i]);
                    const double tolerance = 1e-5 + 1e-5 * std::fabs(next[i]);
                    if (!std::isfinite(output[i]) || !std::isfinite(next[i]) || error > tolerance) {
                        std::cerr << "mismatch step=" << step + 1 << " i=" << i << '\n';
                        all_steps_ok = false;
                        break;
                    }
                    max_error = std::max(max_error, error);
                }
                if (!all_steps_ok) break;
                std::cout << "step=" << step + 1 << " max_abs_error=" << max_error
                          << " first=" << output.front() << " last=" << output.back() << '\n';
                std::swap(read, write);
                reference.swap(next);
            }
            ok = all_steps_ok;
        } while (false);
        if (d_b != nullptr && !check(cudaFree(d_b), "free B")) ok = false;
        if (d_a != nullptr && !check(cudaFree(d_a), "free A")) ok = false;
        if (!ok) return 1;
        std::cout << "PASS average: n=" << n << " steps=" << steps
                  << " threads=" << threads << "; every step checked\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
```

这里新增的主机逻辑较多，可以分四遍读，而不是从第一行一路硬读到底：

1. 先找 kernel：仍是加载 B+2 个槽位、一次屏障、有效线程输出。
2. 再找 read/write：看每轮输入输出分别指向哪份分配。
3. 找 std::swap：确认交换发生在本轮完成、复制回主机并校验之后。
4. 最后看错误分支：任一轮失败就停止，不把后面的轮次继续算完掩盖问题，仍检查释放结果。

每轮 cudaMemset 把目标数组填为 NaN 哨兵，便于发现漏写。必须填 write，不能填 read，否则会毁掉本轮输入。两份设备数组的地址始终不同，所以这一步不会覆盖当前输入。

### 10.10.5 为什么这次不再精确比较 float

除以3通常不能得到二进制可精确表示的有限小数，多轮运算会积累舍入误差。CPU参考用double累加，GPU结果为float，使用：

```text
允许误差 = 1e-5 + 1e-5 × |参考值|
```

同时显式拒绝 NaN 和无穷大。这个容差适用于本例受限输入范围和最多32轮，不是所有迭代算法的通用标准。若换输入范围、轮数或精度，要重新研究误差传播。

打印的 max_abs_error 只在本轮所有元素通过有限数及容差检查时有意义；不得以最大误差变量的初值0推断算法正确。

### 10.10.6 运行并预测奇偶轮输出位置

从教材根目录执行：

```bash
nvcc -std=c++17 -lineinfo code/10_shared/stencil1d.cu -o stencil1d
./stencil1d
./stencil1d 1 3 4
./stencil1d 65 4 32
./stencil1d 1000 5 256
./stencil1d 0 3 4
./stencil1d 10 0 4
```

三个参数依次是元素数 n、迭代轮数 steps、每 block 线程数 threads。注意与上一节的 n/radius/threads 不同；命令名字代表不同程序，不能只看参数都为三个整数就认为含义一样。

单元素 [-8]、三轮的理论结果依次为 -8/3、-8/9、-8/27；float 打印值约为 -2.66667、-0.888889、-0.296296。max_abs_error 随实际编译执行结果填写，不在这里捏造固定误差。

steps=0 定义为恒等操作，主机将输入复制为输出，不调用 CUDA。n=0 定义为空结果，也不调用 CUDA。它们验证的是主机控制流边界，不是GPU上运行了“零轮kernel”。

只看设备存储：奇数轮的结果落在 B，偶数轮落在 A；在交换之后，read 总指向最新结果。程序本轮先从 write 复制结果再交换，因此不需要事后根据奇偶轮猜最终从哪份分配下载。

### 10.10.7 正确性流程不是性能流程

本程序每轮执行 CPU 参考、NaN填充、GPU等待、D2H和全数组验证，因此它是一份教学验证程序，不是高吞吐的迭代求解器基准。

后续性能实验可以把 CPU 参考与检查移到计时之外、按同一 stream 顺序提交各轮，最后才下载结果，但必须保留原先证明过的轮间依赖。不能以“去掉等待后速度快了”解释一个读取未完成结果的程序。

在真实设备上可执行 memcheck、racecheck、synccheck，每次先用65个元素、4轮、32线程的受限案例：

```bash
compute-sanitizer --tool memcheck ./stencil1d 65 4 32
compute-sanitizer --tool racecheck ./stencil1d 65 4 32
compute-sanitizer --tool synccheck ./stencil1d 65 4 32
```

当前环境尚未执行上述 CUDA 验收。

## 10.11 Bank：共享内存“大家一起读”也有布局问题

前面讨论“谁先谁后”，现在假设算法和同步已经正确，研究布局可能带来的性能差别。bank conflict 是性能问题，不是“发生 conflict 就一定算错”的数值错误。

### 10.11.1 一个够用但有边界的模型

先只分析以下情形：一个 warp 的32个线程，各执行一次标量32-bit shared读取；地址按4字节字对齐。对于这里讨论的现代常见映射，可把连续32-bit字依次映射到32个banks。

用word表示“第几个4字节字”，而不是字节偏移，则相对映射为：

```text
bank = word % 32
```

若数组起始位置不是bank 0，只是所有bank编号整体平移，不改变这些例子的冲突程度。以下以起始bank 0便于手算。

这是一张理解地址关系的模型，不是对每一种机器指令、数据类型、访问宽度和架构的完整模拟器。[官方 shared memory 与 bank 说明](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)

### 10.11.2 连续、跨步与广播

假设 lane 是 warp 内编号：

| 每个lane读取的word | lane 0,1,2,3的地址 | 映射bank | 本标量读模型的结论 |
|---|---|---|---|
| lane | 0,1,2,3 | 0,1,2,3 | 32个lane分布到32个bank |
| 2×lane | 0,2,4,6 | 0,2,4,6 | lane0和16落到同bank不同字，形成2路冲突 |
| 32×lane | 0,32,64,96 | 0,0,0,0 | 32个不同字挤到一个bank |
| 33×lane | 0,33,66,99 | 0,1,2,3 | bank分散 |
| 常数0 | 0,0,0,0 | 0,0,0,0 | 同一个字的读取可广播，不是32路不同字冲突 |

“同一个bank”不够，需要接着问“是不是同一个地址”。多个线程读同一个值可以共享一次读取的结果；多个线程读同一bank内不同位置则是另一回事。

多个线程无同步地写同一地址也不是可以套用“广播”的正确算法。广播在这里说的是读取；写入是否竞争，仍要回到所有权和同步分析。

### 10.11.3 为什么二维数组加一列可能有用

C++二维数组按行连续存储。对 float tile[32][32]，tile[row][col] 的字索引为 row×32+col。

同一个warp读一列 tile[lane][0]，字索引是 lane×32，恰好是上表的最拥挤模式。如果改成 float tile[32][33]，读相同逻辑列的字索引变成 lane×33，bank随lane变化。

但这不等于“任何数组都加1就加速”。必须确认真实线程分工、实际读写方向以及生成的指令。加padding还会占额外shared空间。第15章做矩阵转置时，再把这个推导放进完整GPU性能实验。

对本章的半径R求和，固定某个offset时，相邻tid读取 tile[tid+offset]，是连续字索引。一个线程循环读取多个邻居，并不等价于32个lane同时争读同一bank；分析时要先固定同一条访问指令，再比较各lane地址。

### 10.11.4 不依赖 GPU 的地址模型程序

code/10_shared/bank_model.cpp：

```cpp
#include <algorithm>
#include <array>
#include <iostream>
#include <set>

int main() {
    constexpr int lanes = 32;
    const int strides[] = {0, 1, 2, 32, 33};
    const int expected[] = {1, 1, 2, 32, 1};
    for (int case_id = 0; case_id < 5; ++case_id) {
        const int stride = strides[case_id];
        std::array<std::set<int>, 32> words_by_bank;
        for (int lane = 0; lane < lanes; ++lane) {
            const int word = stride * lane;
            const int bank = word % 32;
            words_by_bank[bank].insert(word);
        }
        int busiest = 0;
        int used = 0;
        for (const auto& words : words_by_bank) {
            busiest = std::max(busiest, static_cast<int>(words.size()));
            if (!words.empty()) ++used;
        }
        if (busiest != expected[case_id]) return 1;
        std::cout << "stride=" << stride << " used_banks=" << used
                  << " max_distinct_words_per_bank=" << busiest << '\n';
    }
    std::cout << "PASS scalar 32-bit read model; not a GPU transaction measurement\n";
    return 0;
}
```

std::array<std::set<int>,32> 可以分成两层理解：

- 外层array有32个元素，对应32个bank。
- 每个元素是set<int>，保存该bank被请求的不同word编号；重复insert同一个word只保留一个。
- busiest取每个bank中不同word数量的最大值，区分广播与不同地址挤在一起。

这里只在CPU整理整数地址，没有模拟硬件调度。编译运行：

```bash
g++ -std=c++17 -O2 code/10_shared/bank_model.cpp -o bank_model
./bank_model
```

预期：

```text
stride=0 used_banks=1 max_distinct_words_per_bank=1
stride=1 used_banks=32 max_distinct_words_per_bank=1
stride=2 used_banks=16 max_distinct_words_per_bank=2
stride=32 used_banks=1 max_distinct_words_per_bank=32
stride=33 used_banks=32 max_distinct_words_per_bank=1
PASS scalar 32-bit read model; not a GPU transaction measurement
```

模型输出32，不能据此宣称某个真实kernel恰好慢32倍。kernel还有其他指令、缓存行为、调度、吞吐和资源限制。

### 10.11.5 double、float4 和多次服务不能生搬硬套

一个double占两个32-bit字，一个float4占四个；一个warp的请求总数据量也随之增加。实际生成的指令及其服务拆分必须一起考虑。

即便不存在额外bank冲突，宽访问也可能需要多个服务步骤。不能看见“分了多次”就全部归因于冲突，也不能把float的lane索引公式不加修改地套在所有类型上。

学习路线是：先用本节标量模型预测，接着看生成指令和Nsight Compute的实际指标定义，最后结合时间实测。模型是提出假设的工具，不是 profiler 的替代品。

## 10.12 Shared 不是越多越好：把每个 block 的预算算清楚

将global数据搬进shared可能减少重复加载，也会增加加载、同步、索引计算和shared容量开销。性能优化必须同时考虑收益和成本。

### 10.12.1 每 block 与每 SM 不是同一个数字

每个block有自己的逻辑shared工作区；一个SM可能同时驻留多个block。这些block的shared分配共同消耗该SM的资源。

假设某个简化模型中，SM能给驻留block使用64 KiB shared，每个block需要20 KiB。只按shared算，最多容纳 floor(64/20)=3个block。第四个需要80 KiB，放不下。

这只是一个**容量上界**，还没有考虑线程、warp、寄存器、block数量上限、分配粒度和具体硬件配置。不能把这个假设中的64 KiB当成读者GPU的固定参数。

更不能把grid里1000个block的shared总需求一次性减去某个SM容量。未驻留的block随后分批执行；每个block的动态shared字节数不乘grid大小填入启动第三参数。

### 10.12.2 Occupancy 的含义

理论occupancy可以理解为：

```text
某配置预计可驻留的warp数 / SM允许的最大驻留warp数
```

它描述能同时驻留多少warp，不是SM当前计算单元忙碌百分比，不是达到峰值FLOPS的百分比，也不是程序速度排名。

较低occupancy可能削弱隐藏延迟的能力；但更高occupancy也可能以额外指令、更多global流量或寄存器溢出为代价。应该用第9章可信计时和后续profiler判断，而不是只追求表中最大百分数。

### 10.12.3 查询实际设备与实际编译的 kernel

下面的程序有一个用于资源查询的kernel，但不会启动它。它查询设备限制、编译后kernel属性，以及不同动态shared预算对应的理论驻留数量。

code/10_shared/shared_resources.cu：

```cpp
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

__global__ void shared_probe(float* output) {
    extern __shared__ float tile[];
    const int tid = threadIdx.x;
    tile[tid] = static_cast<float>(tid);
    __syncthreads();
    if (tid == 0) output[blockIdx.x] = tile[blockDim.x - 1];
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    int device = -1;
    cudaDeviceProp prop = {};
    cudaFuncAttributes attr = {};
    if (!check(cudaGetDevice(&device), "current device")) return 1;
    if (!check(cudaGetDeviceProperties(&prop, device), "device properties")) return 1;
    if (!check(cudaFuncGetAttributes(&attr, shared_probe), "kernel attributes")) return 1;
    std::cout << "device=" << device << " name=" << prop.name << '\n';
    std::cout << "shared_per_block=" << prop.sharedMemPerBlock
              << " shared_per_SM=" << prop.sharedMemPerMultiprocessor << '\n';
    std::cout << "kernel_static_shared=" << attr.sharedSizeBytes
              << " kernel_max_dynamic_shared=" << attr.maxDynamicSharedSizeBytes
              << " registers_per_thread=" << attr.numRegs << '\n';
    const int widths[] = {32, 64, 128, 256};
    const int extras[] = {0, 4096, 16384};
    const int max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    if (max_warps <= 0) {
        std::cerr << "invalid device warp capacity\n";
        return 1;
    }
    int queried = 0;
    std::cout << std::fixed << std::setprecision(2);
    for (int threads : widths) {
        for (int extra : extras) {
            const std::size_t dynamic_bytes = static_cast<std::size_t>(threads) * sizeof(float) + extra;
            if (threads > prop.maxThreadsPerBlock || threads > prop.maxThreadsDim[0] ||
                threads > attr.maxThreadsPerBlock ||
                dynamic_bytes + attr.sharedSizeBytes > prop.sharedMemPerBlock ||
                dynamic_bytes > static_cast<std::size_t>(attr.maxDynamicSharedSizeBytes)) {
                std::cout << "SKIP threads=" << threads << " dynamic_bytes=" << dynamic_bytes << '\n';
                continue;
            }
            int active_blocks = 0;
            if (!check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active_blocks, shared_probe, threads, dynamic_bytes),
                       "occupancy query")) return 1;
            const int warps_per_block = (threads + prop.warpSize - 1) / prop.warpSize;
            const int active_warps = active_blocks * warps_per_block;
            const double percent = 100.0 * active_warps / max_warps;
            std::cout << "threads=" << threads << " dynamic_bytes=" << dynamic_bytes
                      << " active_blocks_per_SM=" << active_blocks
                      << " theoretical_occupancy_percent=" << percent << '\n';
            ++queried;
        }
    }
    if (queried == 0) {
        std::cerr << "no supported teaching configuration\n";
        return 1;
    }
    std::cout << "PASS resource query: " << queried << " configurations; no kernel launched\n";
    return 0;
}
```

逐项理解查询：

- cudaGetDeviceProperties给设备层面的上限。
- cudaFuncGetAttributes给这个已编译kernel的属性，其中numRegs是每线程寄存器数量，sharedSizeBytes是静态shared字节数。
- maxDynamicSharedSizeBytes约束该函数动态shared的当前允许大小；本例不主动启用超默认预算的opt-in。
- cudaOccupancyMaxActiveBlocksPerMultiprocessor接收具体kernel、block线程数和每block动态shared字节数，返回该配置理论可驻留的block数。[Occupancy API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html)

不要手动给它再加一次静态shared作为动态参数；它知道kernel自身的编译资源属性。程序中的总量检查才会把静态与动态相加，确认没有越过选用的默认每block预算。

extras={0,4096,16384}是在实际需要的threads×4字节之上，额外预留不同大小的动态shared，以观察资源预算的影响。probe并不使用这些额外位置，也不启动，因此这不是“多读16 KiB”的访存实验。

### 10.12.4 运行后该记录什么

```bash
nvcc -std=c++17 -lineinfo code/10_shared/shared_resources.cu -o shared_resources
./shared_resources
```

输出含设备名称、shared_per_block、shared_per_SM、kernel静态与动态限制、寄存器数量，以及每种配置的active_blocks_per_SM与theoretical_occupancy_percent。具体数值依实际设备和编译结果而定，本教材不编造通用数表。

预期最后一行的形式是：

```text
PASS resource query: <支持的配置数> configurations; no kernel launched
```

尖括号是说明占位符，不是程序字面输出。遇到不支持的组合，会打印SKIP并继续其他组合；API错误或全部组合都不支持时返回失败。

即使查询成功，也不表示probe执行通过。理论occupancy没有考虑当前grid是否足够大，更不能代表实际运行期间的平均活跃warp比例。

### 10.12.5 将结果用于选择，而不是宣判

可以先挑两三组合法B，把它们代入已通过正确性检查的stencil。再独立设计重复次数、计时范围和验证方式，比较naive与shared。

需要保持相同输入、边界规则、输出、GPU、编译选项和计时范围。不能用B=4的教学小例子直接代表生产配置，也不能比较一个版本只测kernel、另一个版本连分配和D2H一起测的时间。

如果shared版本没变快，这是有价值的实验结果。可能复用程度不足，或者global读取被缓存有效服务，或者同步/索引成本增加，或者驻留资源受限。先提出可验证的原因，再做profile；不要为了符合“shared更快”的预设而删掉不合意的测量。

## 10.13 把正确性检查组织成一条可重复的路线

读完概念以后，不要马上把几个优化同时塞进一个kernel。先让每次实验回答一个明确问题。

### 10.13.1 本章九个程序各自负责什么

| 文件，位于code/10_shared | 类型 | 主要问题 |
|---|---|---|
| tile_model.cpp | CPU模型 | global输入怎样对应到shared槽位 |
| stencil_naive.cu | CUDA | 不协作也能先正确得到邻域和 |
| stencil_static.cu | CUDA | 固定B时谁写哪个槽位、何时等待 |
| stencil_dynamic.cu | CUDA | B变化时动态字节数和尾部是否正确 |
| reuse_tile.cu | CUDA | 同一shared工作区复用的两种依赖 |
| stencil_radius.cu | CUDA | 任意受限R和一个线程加载多个位置 |
| stencil1d.cu | CUDA | 多轮平均、全局双缓冲及逐轮校验 |
| bank_model.cpp | CPU模型 | 标量地址怎样映射到bank |
| shared_resources.cu | Runtime查询 | 设备/kernel资源与理论驻留，未启动kernel |

CPU模型不执行CUDA；资源查询需要CUDA环境，但不启动kernel。不要把所有文件成功返回统称为“9个GPU实验通过”。

本章命令清单位于 [代码目录说明](code/10_shared/README.md)。每个源文件独立编译，里面各有main；不能把整个目录的所有.cu一起交给一个链接命令，否则会出现多个main及重复辅助函数定义。

### 10.13.2 一次只排查一层

| 现象 | 先检查 | 不应直接作出的结论 |
|---|---|---|
| nvcc找不到文件 | 当前目录、文件名、相对路径 | GPU型号不支持shared |
| 启动报告配置错误 | block维度、设备/函数限制、动态字节数 | 算法公式一定错 |
| memcheck越界 | global边界与tile最大索引分别推导 | 多加一道屏障能修复 |
| racecheck报告shared冲突 | 写者唯一性、写读及复用覆盖的先后关系 | 改成volatile就正确 |
| 尾部数值错误 | 无输出线程是否仍完成填充职责 | n必须是block整数倍 |
| 首轮对、第二轮错 | read/write交换、每轮参考、跨block轮间依赖 | 容差太严格 |
| 平均边界不一致 | 零填充、固定除以3还是按有效数归一化 | 浮点误差 |
| shared版本更慢 | 同范围计时、输入规模、同步成本与资源 | 测试必然失败 |

volatile不会自动带来互斥、原子性或完整通信协议。遇到竞态先画出依赖图，不要靠修改类型修饰符碰运气。

### 10.13.3 用CMake选择本章，不重跑全书

完成第2章工具链检查后，从根目录配置：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target shared_tile_model shared_bank_model shared_naive shared_static shared_dynamic shared_reuse shared_radius stencil1d shared_resources
ctest --test-dir build -C Release -R "^(shared_|stencil1d)" --output-on-failure
```

--target后面是CMake目标名，不必等于源码文件名。ctest的-R按测试名正则筛选，这里选择shared_开头或stencil1d开头的测试；引号让shell把整个表达式当成一个参数。

如果机器只有C++编译器，可以直接编译两个.cpp模型。是否支持关闭CUDA的CMake选项，请按所用教材目录的根README操作；单独编译CPU模型不需要配置CUDA工程。

CMake能组织命令，不会安装CUDA或驱动，也不能使没有GPU的机器运行CUDA kernel。这里给出的是可执行验收路线，不是声称当前编辑机器完成了这些构建。

### 10.13.4 保存实验记录

至少记录：

```text
GPU / compute capability：
驱动 / nvcc版本：
源文件及修改点：
完整编译命令：
n / R或steps / B：
数学边界规则：
程序退出码：
全数组或逐轮检查结果：
memcheck / racecheck / synccheck结果：
尚未执行的项目：
```

没有测量性能，就把时间栏留空；没有运行sanitizer，就写“未运行”。可解释、可复现的未完成记录，比一个没有上下文的PASS截图更有价值。

## 10.14 练习：从画分工到独立实现

先独立思考，再对照 [第10章练习答案](附录-C-练习答案.md#第-10-章)。需要运行CUDA的题目在有工具链和授权GPU的环境进行；纸笔题和CPU模型题不要求GPU。

### 第一组：索引与数据责任

1. 输入[1,2,3,4,5]、B=4、R=2。画出两个block的8个shared槽位，并算出5个邻域和。
2. 在n=10、B=4、R=1的最后一个block里，tid=2没有输出，为什么仍要完成一次零填充？具体哪个线程会读它准备的槽位？
3. n=1、输入[-3]、R=8、B=4时，邻域和是多少？三点平均做两轮又是多少？解释为什么这不是同一道题。
4. B=4、R=3时各线程加载哪些j？如果将屏障放进加载循环，为什么协议不合适？
5. 推导最后一个线程读取的最大shared索引。R=0时程序应退化为什么操作？

### 第二组：同步协议

6. 分别判断以blockIdx.x==0、threadIdx.x<16、i<n为条件包围__syncthreads的风险。你的判断应说明block配置和条件是否一致。
7. 解释“所有线程都调用__threadfence_block，因此相当于__syncthreads”错在哪里。
8. 按10.8的环状读取实验，写出省略第二次屏障导致读到下一轮值的交错过程。
9. 为什么单次stencil末尾无需机械增加第二次屏障？为什么最后一轮复用实验的第二次屏障可在证明后省略？
10. CPU调用cudaDeviceSynchronize能否修复kernel内部已经发生的数据竞争？同一个warp能否作为无需通信协议的理由？

### 第三组：参数、迭代与精度

11. 执行stencil_radius 17 8 4，推导blocks和shared_bytes。换成B=1时，是否还可以覆盖所有槽位？
12. 列出stencil_radius对空输入、负半径、threads=0、超长数字分别怎样处理；哪些情况不调用CUDA？
13. 双缓冲初始read=A、write=B。分别写出第1、2、3轮提交时的角色，以及最后交换之后最新结果的位置。
14. 输入[3,6,9]做两轮零填充三点平均，手算结果。若边界改成按有效邻居数量除法，第一轮会怎样？
15. 为什么半径求和例可以精确比较，而多轮平均用容差？只判断fabs(actual-expected)>tol有什么遗漏？

### 第四组：布局与资源

16. 在32-bit标量读模型中，word=4×lane和word=8×lane各使用多少bank？每个bank最多几个不同word？
17. 所有lane读tile[0]和所有lane写tile[0]，能否都用广播说明正确？
18. 假设SM的shared预算64 KiB，每block需要24 KiB。仅凭shared得到的驻留block上界是多少？为什么不能据此直接给真实GPU的occupancy或速度结论？
19. shared_resources打印PASS，是否说明kernel已经在GPU执行？它的理论occupancy与实测吞吐有什么不同？

### 第五组：从模仿走向独立设计

20. 设计一个naive与shared半径求和的性能对照方案：明确相同项、计时范围、预热、样本、验证及应记录的未知项，不要求提前得到shared更快的结论。
21. 从半径R=2版本改写五点加权平均，权重从左到右为[1,2,3,2,1]，固定除以9，仍用零边界。先写CPU参考，再修改GPU消费循环，保留协作加载与错误检查。至少测试n=1、5、17。
22. 有人想在一个普通kernel里，让所有block循环执行10轮平均，只在轮末调用__syncthreads。指出缺失的依赖，提出本章范围内可以验证的替代方案。

## 10.15 本章结束时，你应该能自己写出什么

给定一个邻域问题，你应能从空白文件开始按下面顺序搭建，而不只是记住某份kernel：

1. 定义输入输出、边界、半径和轮次。
2. 用CPU和小数值算出可信参考。
3. 决定每个block负责哪些输出，再反推它需要哪些输入。
4. 给shared槽位分配唯一加载者，处理左右越界与尾部无输出线程。
5. 在真正的跨线程依赖之间放屏障，不把fence、warp或CPU等待混为一谈。
6. 写回独立输出并检查每个结果；若多轮迭代，维护完整轮间依赖。
7. 在正确性成立之后，再用bank模型、资源查询和实际计时提出优化实验。

本章没有承诺shared总能变快，也没有涵盖所有高级共享内存机制。异步拷贝、pipeline、较新架构的专用搬运与矩阵指令，需要更深入的资源和执行模型知识，不能用几个API名替代这里的基本功。

参考入口：

- [Programming Guide：编写CUDA kernel](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)。
- [同步、warp同步与内存fence](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#synchronization-primitives)。
- [Best Practices：共享内存与访问布局](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)。
- [Runtime：Occupancy API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html)。
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

下一章：[并行归约（尚未发布）](README.md#后续章节)。届时不再是每个线程独立写一个邻域输出，而是让一组线程逐轮合并中间结果。先画“哪一轮谁读谁写”，就能把本章的协作与同步知识继续用下去。
