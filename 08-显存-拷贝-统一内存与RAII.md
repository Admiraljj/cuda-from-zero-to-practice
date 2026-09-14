# 第 8 章：显存、拷贝、统一内存与 RAII

前面你已经能启动 kernel、安排索引并检查错误。本章把这些程序里的“内存”拆开看：地址是什么，数据在哪里，谁能访问，什么时候能访问，最后谁释放。

建议分四次学习，不必一口气读完：8.0—8.3 看字节与 managed；8.4—8.7 做能力查询、pinned、显存快照和预取；8.8—8.11 补内存层级与基础 RAII；8.12—8.17 再学移动、模板并完成练习。暂时看不懂模板时，先把固定 float 版本运行起来，再回来扩展。

本章提供 9 个完整小程序，其中 2 个只需要 CPU，以及 2 个供案例使用的头文件。数值很小，目的是让每一步都能手算；它们不是性能基准。

路径约定：8.1—8.9 的命令从教材根目录执行；8.11 会明确进入 code/08_memory，之后的 RAII 与移动案例在该目录执行。另有 [本章代码与命令清单](code/08_memory/README.md)，可随时按清单重新确定当前目录。

## 8.0 先把五个问题分开

看到一个指针，先问：

| 问题 | 例子 |
|---|---|
| 地址值是什么 | 指针当前保存哪个位置 |
| 谁能访问 | CPU、当前GPU、其他GPU是否有合法访问能力 |
| 数据实际在哪里 | 当前在主机内存、设备内存，还是经某种映射访问 |
| 什么时候能访问 | 上一次写入或复制是否完成，是否与别人冲突 |
| 谁负责释放 | 谁拥有分配，最后一次使用何时结束 |

以前把它们混成“这是个GPU指针”会掩盖很多细节。第5章的d_a变量在主机代码中保存设备地址，但它不自动拥有CPU解引用权限；第6章的pitch告诉你地址怎样走；第7章的同步告诉你何时读取才有明确顺序。

本章不从一大段模板类开始。先观察真实内存行为，直到你能指出“手工管理哪一步容易遗漏”，再引入RAII。封装是把已经理解的责任写进类型，不是让未知行为消失。

## 8.1 内存单位先补一课：填充字节不等于填充float元素

第7章提到cudaMemset按字节填充。先不用GPU，在CPU上用std::memset观察同一类单位差异。

float的1.0不是“四个字节都等于1”。本实验限定在常见IEEE-754单精度表示，通过static_assert让不满足前提的环境拒绝编译，而不是假装任何C++实现的float都相同。

完整文件code/08_memory/byte_fill.cpp：

```cpp
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

int main() {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                  "This representation experiment requires IEEE-754 binary32 float");
    float values[2] = {};
    std::memset(values, 1, sizeof(values));
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(values);
    bool all_bytes_one = true;
    for (std::size_t i = 0; i < sizeof(values); ++i) {
        if (bytes[i] != 1) all_bytes_one = false;
    }
    std::cout << "after memset(1): bytes_all_one=" << all_bytes_one
              << " value_is_one=" << (values[0] == 1.0f) << '\n';
    if (!all_bytes_one || values[0] == 1.0f || !std::isfinite(values[0])) return 1;
    std::cout << "numeric_value=" << std::scientific << values[0] << '\n';

    std::fill(values, values + 2, 1.0f);
    if (values[0] != 1.0f || values[1] != 1.0f) return 1;
    std::cout << "after fill(1.0f): value_is_one=" << (values[0] == 1.0f) << '\n';
    std::cout << "PASS byte fill versus element fill\n";
    return 0;
}
```

编译：

```bash
g++ -std=c++17 -Wall -Wextra code/08_memory/byte_fill.cpp -o byte_fill
./byte_fill
```

第一行应显示bytes_all_one=1，但value_is_one=0。第二行显示按浮点数解释后的值；本实验不靠它的字符串格式验收。std::fill则按float元素赋1.0f，最后value_is_one=1。

reinterpret_cast<const unsigned char*>用于查看对象的字节表示，不是把float数值转换为0—255的整数。std::fill(values,values+2,1.0f)处理从首元素到尾后位置之间的两个元素，尾后位置本身不写。

把这个结论带回CUDA：cudaMemset(d,1,bytes)的1是字节值，不是你希望的浮点初值。要填任意float，可以先在主机构造再复制，或写一个每线程赋值的kernel。全零的常见IEEE浮点表示另当别论，不能据此推断其他数也能按字节重复填入。

## 8.2 Managed Memory：一个地址可在不同阶段交给CPU和GPU

第5章显式分配主机与设备两份数组，再通过复制传递数据。managed方案允许我们使用同一个分配的地址，在规定的同步条件下分别从CPU和GPU访问。

先看程序，注意CPU初始化、GPU计算、CPU检查三个阶段之间的顺序。

完整文件code/08_memory/managed_scale.cu：

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void scale(float* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] *= 2.0f;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    const int n = 10;
    float* values = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMallocManaged(&values, n * sizeof(float)), "allocate managed")) break;
        for (int i = 0; i < n; ++i) values[i] = static_cast<float>(i + 1);
        scale<<<1, 32>>>(values, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait before CPU access")) break;

        ok = true;
        for (int i = 0; i < n; ++i) {
            const float expected = 2.0f * (i + 1);
            std::cout << "i=" << i << " value=" << values[i] << '\n';
            if (!std::isfinite(values[i]) || values[i] != expected) ok = false;
        }
    } while (false);
    if (values != nullptr && !check(cudaFree(values), "free managed")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS managed scale: 10 elements; no explicit memcpy\n";
    return 0;
}
```

编译：

```bash
nvcc -std=c++17 code/08_memory/managed_scale.cu -o managed_scale
./managed_scale
```

应得到10个值2、4、6、8、10、12、14、16、18、20。cudaMallocManaged成功后CPU初始化；kernel启动后CPU不读写values；同步成功后才进入CPU检查。

没有显式cudaMemcpy，不代表数据从未移动，不代表所有系统都在同一片物理内存上执行，也不代表CPU和GPU可以随便同时修改同一元素。访问能力、物理放置、迁移与并发限制因平台而异，后续能力查询实验会把差异列出来。[CUDA统一与系统内存](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)

程序故意不加预取。先确保同步和结果正确，再将预取作为可选性能策略，而不是“只要用了managed就必需背下的三行API”。

## 8.3 地址统一，不等于物理内存合并

现在回到上一节的 `values`。CPU初始化时用它，GPU计算时也用它，是否意味着CPU内存和GPU显存已经变成同一块硬件？不是。先把两个相似的词分开。

### 8.3.1 UVA首先讨论怎样表示地址

UVA是Unified Virtual Addressing，统一虚拟寻址。这里的关键词是“虚拟地址”，不是“物理存储”。在支持UVA的进程中，主机内存和设备内存使用统一的虚拟地址空间；这不自动赋予每个处理器访问所有分配的权限。

一个方便的类比是门牌号：统一给不同楼里的房间编号，不等于把不同楼拆掉合并，更不等于拿到某个门牌号就有那间房的钥匙。这个类比只解释编号与访问权限的区别，不用于推导GPU内部实现。

所以，第5章的下面两个操作仍然不同：

```cpp
float* alias = d_values;  // 复制地址值；没有读取设备数组
// float first = d_values[0];  // 不要据“有UVA”就从CPU这样读取cudaMalloc分配
```

这是解释用片段，不是完整程序；`d_values`指此前成功取得的普通设备分配。第一行只让另一个主机指针变量保存相同地址，不复制数组、不迁移数据，也不改变释放责任。

### 8.3.2 Managed Memory讨论怎样让不同处理器访问同一分配

`cudaMallocManaged`建立可由CPU和GPU访问的managed分配。它在不同平台上的物理放置、迁移和并发访问限制可能不同。UVA与managed不能互换：前者回答地址如何组织，后者还涉及分配的访问和管理行为。[NVIDIA：统一虚拟地址与统一内存](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)

对于本章已经写出的两个程序，可以先画出下面的区别，而不猜测某台机器的页面到底放在哪里：

```text
显式复制程序：
CPU数组 h ──复制内容──> 设备数组 d ──GPU修改──> d
CPU数组 h <──复制内容──────────────────────── d
有两份分配；h与d不是同一个数组的两个名字。

managed_scale：
managed分配 values
  CPU初始化 → 提交GPU计算 → 等待GPU完成 → CPU验证 → 释放
程序使用同一个分配地址；不据此推断没有数据传输。
```

### 8.3.3 用四个阶段检查上一节程序

| 阶段 | 主机代码正在做什么 | 是否可以开始CPU验证 |
|---|---|---|
| 初始化 | 写入1、2、……、10 | 尚未计算，没有结果可验证 |
| kernel提交返回 | CPU已经提交任务，GPU可能仍在运行 | 不可以直接假定结果已就绪 |
| 同步成功返回 | 本例等待的GPU工作已完成 | 可以开始逐项读取 |
| 释放成功后 | 分配生命周期已结束 | 不可以继续使用旧地址 |

注意最后一行：如果释放后普通指针变量里还保留旧数值，它也不是一张仍有效的访问凭证。后面学习RAII时，正是要把“哪个对象负责释放、何时不再允许借用地址”写清楚。

### 8.3.4 先做三道判断题

1. 把设备指针赋给另一个指针变量，是否已经完成D2H复制？
2. 使用managed分配后，能否为了少写一行代码直接去掉本例CPU验证前的同步？
3. 两种实现都得到2、4、……、20，是否证明它们的性能相同？

答案：第一题没有，只复制地址值。第二题不能，本例需要这一步建立GPU写入完成后CPU再读的顺序。第三题不能，数值相同只说明这组输入的结果一致；分配、数据传输、首次访问和同步成本需要按明确范围测量，计时方法放到第9章。

你现在不需要背出所有内存属性。下一步用能力查询确认当前设备支持什么，再学习pinned内存与预取；不要用一次运行结果推断所有CUDA平台。

## 8.4 不靠猜测：查询当前设备的内存能力

上一节说明“行为随平台变化”，但这句话不能代替操作方法。真正写程序时，需要把不确定性变成一次查询：查询成功了吗？返回的属性值是多少？该值允许我们作出什么判断？

### 8.4.1 四个属性不是四个速度分数

| 属性名 | 本章怎样理解 | 不能推出什么 |
|---|---|---|
| managedMemory | 是否支持managed分配 | 支持不等于比显式复制快 |
| concurrentManagedAccess | 是否具备完整的managed并发访问能力 | 不等于CPU和GPU可以无同步地修改同一元素 |
| pageableMemoryAccess | 是否支持直接访问系统pageable内存的相应能力 | 不能仅凭UVA就假定普通数组可传给kernel |
| pageableMemoryAccessUsesHostPageTables | 系统内存访问是否使用主机页表；结合前两项区分一致性机制 | 不等于所有代码都不需要同步 |

表中名字对应下面程序打印的标签，实际传给接口的是带cudaDevAttr前缀的枚举值。应结合属性读，而不是看到一个1就跳过其他判断。[NVIDIA统一内存能力说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)

### 8.4.2 完整程序

文件：code/08_memory/memory_capabilities.cu。

```cpp
#include <cuda_runtime.h>
#include <iostream>

bool query(int& value, cudaDeviceAttr attribute, const char* name, int device) {
    const cudaError_t status = cudaDeviceGetAttribute(&value, attribute, device);
    if (status != cudaSuccess) {
        std::cerr << name << ": " << cudaGetErrorString(status) << '\n';
        return false;
    }
    std::cout << name << '=' << value << '\n';
    return true;
}

int main() {
    int device = -1;
    const cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        std::cerr << "current device: " << cudaGetErrorString(status) << '\n';
        return 1;
    }
    int managed = -1, concurrent = -1, pageable = -1, host_tables = -1;
    std::cout << "device=" << device << '\n';
    if (!query(managed, cudaDevAttrManagedMemory, "managedMemory", device)) return 1;
    if (!query(concurrent, cudaDevAttrConcurrentManagedAccess,
               "concurrentManagedAccess", device)) return 1;
    if (!query(pageable, cudaDevAttrPageableMemoryAccess,
               "pageableMemoryAccess", device)) return 1;
    if (!query(host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables,
               "pageableMemoryAccessUsesHostPageTables", device)) return 1;

    if (!managed) std::cout << "mode=no managed allocation support\n";
    else if (!concurrent) std::cout << "mode=limited managed support\n";
    else if (!pageable) std::cout << "mode=full support for managed allocations\n";
    else if (!host_tables) std::cout << "mode=system memory, software coherence\n";
    else std::cout << "mode=system memory, hardware coherence\n";
    std::cout << "PASS capability query; no kernel or allocation experiment\n";
    return 0;
}
```

在教材根目录执行：

```bash
nvcc -std=c++17 code/08_memory/memory_capabilities.cu -o memory_capabilities
./memory_capabilities
```

这里没有kernel启动语法，也没有分配数组。它只回答“当前运行环境告诉我哪些能力”。即使最后输出PASS，也不是managed_scale已经运行通过。

### 8.4.3 query函数的四个参数怎样配合

以第一次调用为例：

```cpp
query(managed, cudaDevAttrManagedMemory, "managedMemory", device)
```

四个参数依次是：接收结果的变量、要查哪个属性、打印时用的标签、查哪个逻辑设备。第二个参数是枚举常量，第三个是供人阅读的字符串，二者不是同一种东西；不能把字符串直接塞到属性枚举的位置。

query声明里的int&让value引用外层的变量。进入函数后，&value取得该整数的地址，把它交给cudaDeviceGetAttribute写入查询结果。所以这里有两层：

```text
managed：保存属性值的整数
    ↓ 通过引用传给query
value：同一整数的另一个名字
    ↓ &value
CUDA接口得到这个整数的地址，写回0或1
```

函数自己的返回值bool用于告诉main查询是否成功；它和写回managed的0/1没有相同含义。比如，查询成功且结果为0时，query返回true，但managed为0。若混淆二者，会把“不支持该能力”误判成“接口调用失败”。

### 8.4.4 如何读输出而不伪造设备属性

一种用于练习阅读的假设输出是：

```text
device=0
managedMemory=1
concurrentManagedAccess=1
pageableMemoryAccess=0
pageableMemoryAccessUsesHostPageTables=0
mode=full support for managed allocations
PASS capability query; no kernel or allocation experiment
```

这些数字不是某个指定GPU的实测。逐步判断：managed分配可用；完整managed支持可用；不能据此推广到所有普通系统分配。上节cudaMallocManaged方案可以继续研究，但把std::vector::data()直接传给kernel不是由这组属性授予的通用做法。

再手算两组，不运行也能检查分支理解：

- managed=1、concurrent=0：进入limited分支；本章继续使用保守的CPU初始化—GPU工作—同步—CPU读取。
- managed=1、concurrent=1、pageable=1、host_tables=1：输出硬件一致性分支；仍然需要遵守数据依赖与同步协议。

接口调用失败时程序立即返回1，并打印哪个查询失败。不要为了“输出看起来完整”在失败后把默认-1当成真实能力继续推断。

## 8.5 Pinned主机缓冲区：它仍然不是设备数组

现在有一个具体任务：CPU生成8个浮点数，GPU分别加10，然后CPU检查。先沿用第5章的两份分配，只把主机缓冲区的申请方式改掉。

### 8.5.1 pageable和page-locked在说哪一层

普通主机内存通常由操作系统分页管理。CUDA提供cudaMallocHost申请用于主机—设备传输的主机缓冲区，通常称为pinned或page-locked memory。传统离散GPU环境下，固定这些页面有利于设备传输；分配过多会占用宝贵的系统资源，不应把所有CPU数组一律改成pinned。

这是对常见平台的起点解释，不是每种一致性系统的实现定律；当前官方接口特别说明，在使用主机页表的系统上，cudaMallocHost可能不实际锁页。[cudaMallocHost与cudaFreeHost接口](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

本章最重要的区别不是背术语，而是看到两份分配：

| 变量 | 本例申请接口 | 本例谁直接读写它 | 释放接口 |
|---|---|---|---|
| h | cudaMallocHost | CPU初始化与验证；作为复制的一端 | cudaFreeHost |
| d | cudaMalloc | GPU计算；作为复制的另一端 | cudaFree |

本例没有使用mapped host memory，也没有把h传入kernel。pinned与zero-copy不是应当无条件互换的词；mapped访问还涉及设备能力、映射和访问成本，不是将所有cudaMemcpy删除就完成了优化。

### 8.5.2 完整往返实验

文件：code/08_memory/pinned_roundtrip.cu。

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

__global__ void add_ten(float* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] += 10.0f;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    const int n = 8;
    const std::size_t bytes = n * sizeof(float);
    float* h = nullptr;
    float* d = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMallocHost(&h, bytes), "allocate pinned host")) break;
        if (!check(cudaMalloc(&d, bytes), "allocate device")) break;
        for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i);
        if (!check(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice), "H2D")) break;
        add_ten<<<1, 32>>>(d, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(cudaDeviceSynchronize(), "wait")) break;
        if (!check(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost), "D2H")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            std::cout << "i=" << i << " value=" << h[i] << '\n';
            if (!std::isfinite(h[i]) || h[i] != static_cast<float>(i + 10)) ok = false;
        }
    } while (false);
    if (d != nullptr && !check(cudaFree(d), "free device")) ok = false;
    if (h != nullptr && !check(cudaFreeHost(h), "free pinned host")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS pinned roundtrip: 8 elements\n";
    return 0;
}
```

编译与运行：

```bash
nvcc -std=c++17 code/08_memory/pinned_roundtrip.cu -o pinned_roundtrip
./pinned_roundtrip
```

应输出8条记录，i从0到7，value依次为10到17，最后打印PASS pinned roundtrip: 8 elements。这一预期可以不依赖GPU手算：第i个元素输入为i，kernel加10，结果就是i+10。

### 8.5.3 不要漏看变量前的&

float* h=nullptr创建的是一个主机指针变量，它暂时不指向任何分配。cudaMallocHost(&h,bytes)把这个指针变量的地址交给接口，让接口把新分配的地址写进h；这不是给GPU传一个float数值。

CUDA C++头文件提供相应的类型化便利接口，因此这里可以使用float**。阅读C API文档时可能看到void**原型；不要因此到处添加未经理解的强制转换。按本文包含cuda_runtime.h并用nvcc编译这一完整C++案例。

bytes使用n*sizeof(float)，是整个缓冲区的字节数。n=8且float为4字节时，bytes为32，不是8，也不是32个float。

### 8.5.4 数据何时变化

H2D之后，h与d的初始内容相同，但仍是独立分配。GPU执行add_ten时只修改d；它不会因为h与d曾经拷贝过，就持续把h自动更新。D2H才把结果写回h。

```text
h: 0 1 2 3 4 5 6 7
       │ H2D
d: 0 1 2 3 4 5 6 7
       │ kernel每元素加10
d:10 11 12 13 14 15 16 17
       │ 等待完成，再D2H
h:10 11 12 13 14 15 16 17
```

程序保留显式cudaDeviceSynchronize用于区分执行错误与后续复制错误，也方便你把步骤读清楚。本章没有启动多stream流水线，因此不能凭使用pinned就声称H2D、kernel、D2H已经发生重叠。

### 8.5.5 失败以后仍然要释放

若h申请成功、d申请失败，break离开do块后，d仍为空，h会通过cudaFreeHost释放。若最后一个结果不正确，ok变为false，但两份分配仍会执行清理。释放返回错误也会使程序最终返回1。

本例只讨论显式代码路径；不要把它当成覆盖进程被强杀、硬件故障等情形的保证。错误检查不会神奇地恢复已经损坏的CUDA执行状态。

### 8.5.6 现在先不改成Async

“异步”意味着提交后可能尚未完成，不意味着更快。以后用cudaMemcpyAsync时，还要规定：CPU何时可以重新写入H2D源缓冲？何时可以读取D2H目的缓冲？这个缓冲何时可释放？多个stream是否存在依赖？

这些问题会在第20章用明确时间线和完整程序展开。本章先做到分配配对、结果正确、顺序清楚。8个元素的实验不是传输带宽基准。

## 8.6 观察显存统计：这是快照，不是精确账本

学习cudaMalloc之后，很自然会想：“申请4MiB前后，可用显存是否恰好减少4MiB？”可以观察，但不应把这个等式写成程序必然成立的验收条件。

cudaMemGetInfo写回当前上下文可见的总量与操作系统报告的空闲量，单位为字节。它不保证你能一次申请全部报告空闲的内存；其他进程或线程的活动也可能让查询与使用之间的状态变化。[cudaMemGetInfo说明](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

### 8.6.1 完整的三次快照

文件：code/08_memory/memory_snapshot.cu。

```cpp
#include <cuda_runtime.h>
#include <iostream>

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

bool snapshot(const char* stage) {
    std::size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "memory snapshot")) return false;
    std::cout << stage << " free_bytes=" << free_bytes
              << " total_bytes=" << total_bytes << '\n';
    return true;
}

int main() {
    const std::size_t requested = 4 * 1024 * 1024;
    unsigned char* device_data = nullptr;
    if (!snapshot("before")) return 1;
    std::cout << "requested_bytes=" << requested << '\n';
    if (!check(cudaMalloc(&device_data, requested), "allocate")) return 1;
    bool ok = snapshot("allocated");
    if (!check(cudaFree(device_data), "free")) return 1;
    device_data = nullptr;
    if (!snapshot("released")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS snapshots; differences are not asserted\n";
    return 0;
}
```

```bash
nvcc -std=c++17 code/08_memory/memory_snapshot.cu -o memory_snapshot
./memory_snapshot
```

输出包含before、allocated、released三行统计，中间打印requested_bytes=4194304。不要照抄别人机器的free_bytes。这里的PASS只表示各接口按程序路径成功执行，不表示差值满足某个固定公式。

4*1024*1024=4194304字节称为4MiB；十进制4MB则是4000000字节。比较记录时先统一单位，不能把不同单位的数字直接相减。

### 8.6.2 snapshot函数没有申请一个新数组

它的两个size_t变量只是接收统计结果；&free_bytes与&total_bytes给接口两个写回位置。stage是描述这次查询发生在哪一阶段的字符串，不会改变显存状态。

真正的分配在main中的cudaMalloc。unsigned char作为字节类型，让requested直接对应分配大小；本例不读取这段未初始化内存，也不启动kernel。

### 8.6.3 为什么不用free_before-free_after判断成功

设一次练习观察到申请前有1000MiB空闲，申请后有1002MiB空闲。不能直接下结论“cudaMalloc没有分配”：在两次观察之间，其他使用者可能释放了更多内存。反之，下降超过4MiB也不自动说明你的数组越界。

而且size_t是无符号类型；如果后一个值更大，直接做前减后可能产生一个很大的回绕值。最先应该记录的是两份原始数据和接口状态，而不是急着计算一个看似精确的“自己用了多少”。

本例的实际申请结果以cudaMalloc返回值为依据，释放结果以cudaFree返回值为依据。不要为了验证统计等式反复申请接近全部空闲容量；共享环境下这可能影响其他任务。

## 8.7 预取：提前提出数据放置请求，不是省略同步

上一节把统计与分配分开，本节再把“数据可访问”与“希望它放在哪里”分开。managed_scale已经能在合适的环境中完成CPU—GPU—CPU计算；预取是进一步告诉运行时，接下来希望在哪个处理器访问这段数据。

### 8.7.1 先读一条队列

本例所有GPU操作都由同一主机线程提交到默认stream。暂时把stream理解为有顺序的工作队列：

```text
CPU完成初始化
    ↓
默认stream：预取到当前GPU → triple kernel → 预取回CPU
    ↓ 等待GPU工作完成
CPU读取并验证
```

箭头表示本程序规定的先后关系，不表示每个箭头都让整个CPU停住。预取API名称中的Async提示它对主机是异步的；它返回后，不意味着预取已经完成。最后的同步仍然保留。

### 8.7.2 两代接口不能混用

| Toolkit接口版本 | 目的地参数 | 本例调用形态 |
|---|---|---|
| 12.x，本书核对12.9.1文档 | GPU整数编号；主机用cudaCpuDeviceId | cudaMemPrefetchAsync(ptr, bytes, destination) |
| 13.0起 | cudaMemLocation结构体 | cudaMemPrefetchAsync(ptr, bytes, location, 0) |

13.x调用中的最后一个0是flags，不是stream；stream使用下一个参数的默认值0。若以后显式传stream，新版应写五个参数。GPU目的地设置Device类型与逻辑设备编号，主机目的地设置Host类型，此时id不用于选择GPU。[CUDA 12.9.1](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-runtime-api/group__CUDART__MEMORY.html)、[CUDA 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__MEMORY.html)

CUDART_VERSION来自编译时头文件；它不是运行时驱动版本。条件编译选择的是“当前编译器看见的接口长什么样”，不能修复驱动与Toolkit本身不兼容的问题。

### 8.7.3 完整的有条件预取实验

文件：code/08_memory/managed_prefetch.cu。

```cpp
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#if CUDART_VERSION < 12000
#error This teaching example targets CUDA Toolkit 12 or later
#endif

cudaError_t prefetch(float* values, std::size_t bytes, int destination) {
#if CUDART_VERSION >= 13000
    cudaMemLocation location = {};
    if (destination == cudaCpuDeviceId) {
        location.type = cudaMemLocationTypeHost;
    } else {
        location.type = cudaMemLocationTypeDevice;
        location.id = destination;
    }
    return cudaMemPrefetchAsync(values, bytes, location, 0);
#else
    return cudaMemPrefetchAsync(values, bytes, destination);
#endif
}

__global__ void triple(float* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] *= 3.0f;
}

bool check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

int main() {
    int device = -1, managed = 0, concurrent = 0;
    if (!check(cudaGetDevice(&device), "current device")) return 1;
    if (!check(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, device),
               "managed support")) return 1;
    if (!check(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device),
               "prefetch support")) return 1;
    if (!managed || !concurrent) {
        std::cout << "SKIP prefetch: managed=" << managed
                  << " concurrent=" << concurrent << '\n';
        return 77;
    }
    const int n = 10;
    const std::size_t bytes = n * sizeof(float);
    float* values = nullptr;
    bool ok = false;
    do {
        if (!check(cudaMallocManaged(&values, bytes), "allocate")) break;
        for (int i = 0; i < n; ++i) values[i] = static_cast<float>(i + 1);
        if (!check(prefetch(values, bytes, device), "prefetch to GPU")) break;
        triple<<<1, 32>>>(values, n);
        if (!check(cudaGetLastError(), "launch")) break;
        if (!check(prefetch(values, bytes, cudaCpuDeviceId), "prefetch to CPU")) break;
        if (!check(cudaDeviceSynchronize(), "wait before CPU read")) break;
        ok = true;
        for (int i = 0; i < n; ++i) {
            std::cout << "i=" << i << " value=" << values[i] << '\n';
            if (!std::isfinite(values[i]) || values[i] != 3.0f * (i + 1)) ok = false;
        }
    } while (false);
    if (values != nullptr && !check(cudaFree(values), "free")) ok = false;
    if (!ok) return 1;
    std::cout << "PASS managed prefetch: 10 elements\n";
    return 0;
}
```

```bash
nvcc -std=c++17 code/08_memory/managed_prefetch.cu -o managed_prefetch
./managed_prefetch
```

程序面向Toolkit 12及以后版本。较早头文件会触发#error，明确停止构建，而不是让下面的兼容分支被误称为支持所有历史版本。

若能力满足，预期输出i=0—9，value=3、6、……、30，最后是PASS managed prefetch: 10 elements。若能力不满足，会输出SKIP以及实际属性值，并返回77。77是这份教学程序约定的“未执行该实验”，不是CUDA统一规定的错误码。

### 8.7.4 为什么先查concurrentManagedAccess

向GPU预取要求目标GPU具备相应concurrentManagedAccess能力；提交操作所用stream所属设备也有这一要求。本例查询当前设备，将它作为目的地，同时使用该设备的默认stream，因而可以用同一次能力检查约束两端。不要在修改成多GPU程序后仍假定这两个设备一定相同。[预取前提](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__MEMORY.html)

跳过是诚实的结果，不是程序损坏；不要为了让末尾打印PASS，忽略属性或把不支持的接口错误吞掉。如果查询接口自身失败，则返回1，和能力不足的77不同。

### 8.7.5 什么结果才能说明预取有用

这份10元素程序只检查顺序、接口和数值，不做性能比较。预取本身也有代价；任务太小、访问位置反复改变或没有合适时间隐藏迁移成本时，不应期待它自动加速。

合理比较需要固定输入、计算、正确性标准与计时范围，并区分首次访问和后续重复。这些内容放在第9章。不要把“没有写cudaMemcpy”当作没有数据移动，也不要用“加了预取后程序可以跑”代替性能证据。

## 8.8 在kernel里看到的global、shared、local、register、constant

前几节主要从主机侧管理分配。进入kernel以后，还会碰到另一组分类。它们不与pinned/managed简单排成一张“从快到慢”的直线，因为讨论角度不同：有的说主机分配策略，有的说设备代码的存储空间。

| 空间 | 本章先记住的可见范围 | 生命周期或管理方式 | 不能误解成什么 |
|---|---|---|---|
| global | 合法访问它的GPU线程 | 分配持续到释放，可跨kernel使用 | 可见不代表并发写入自动安全 |
| shared | 常规用法为同一个block的线程 | 随该block工作存在 | 不是所有block天然共用一份数组 |
| register | 单个线程的值 | 由编译器为线程执行分配 | 源码变量不保证总有独立寄存器 |
| local | 逻辑上属于单个线程 | 编译器管理的线程私有存储 | local不等于片上快速内存 |
| constant | kernel只读的设备常量空间 | CUDA C++常量对象属于相应context | const float*不自动声明constant空间 |

shared与寄存器使用SM上的资源；local空间背后位于设备内存，并可能经过缓存。大局部数组、动态索引或寄存器溢出可能导致local访问。不要把表格当成固定延迟承诺。[NVIDIA设备存储空间说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)

### 8.8.1 同一个值可以经过不同空间

假设kernel中写：

```cpp
float value = input[i];
value *= 2.0f;
output[i] = value;
```

input和output指向global分配；value是该线程自己的逻辑变量。编译器可以把它放入寄存器，也可以消去中间步骤或采取其他实现。这个变量与数组元素不是同一份存储，仅在执行读写时传递数值。

### 8.8.2 shared为什么要等第10章再深入

本章的逐元素缩放每个线程独立工作，不需要先把数据放进shared。shared往往用来让同block线程复用数据或交换中间结果；一旦发生跨线程依赖，就要安排写入、同步和读取。

用本章的概念可以先画出责任分层：

```text
主机：安排分配、传输、启动、等待、释放
  └─ global分配：保存跨kernel需要的数据
       └─ 某个block：可以合作使用本block的shared
            └─ 某个线程：有自己的局部变量/寄存器值
```

“作用域小”不等于“物理位置近”。第10章会用需要邻居元素的计算说明shared为什么有用，第19章再结合工具观察寄存器与资源限制。

## 8.9 在谈RAII之前，先看一个对象什么时候结束

手工清理已经出现很多次：申请、检查、计算、检查、释放。程序分支越多，越容易漏掉某条提前退出路径。C++允许把资源的释放责任绑定到对象生命周期，这种做法叫RAII，Resource Acquisition Is Initialization。

名字很长，先记一条工作原则：**对象成功建立后，由这个对象负责在其生命周期结束时执行清理。** 这不等于对象一定申请GPU内存，也不等于任何终止方式都会自动清理。

### 8.9.1 一个只打印顺序的完整CPU程序

文件：code/08_memory/scope_trace.cpp。

```cpp
#include <cstdio>
#include <stdexcept>

class Trace {
public:
    explicit Trace(const char* name) : name_(name) {
        std::printf("construct %s\n", name_);
    }
    ~Trace() noexcept {
        std::printf("destroy %s\n", name_);
    }
    Trace(const Trace&) = delete;
    Trace& operator=(const Trace&) = delete;
private:
    const char* name_;
};

void work(bool fail) {
    Trace outer("outer");
    {
        Trace inner("inner");
        if (fail) throw std::runtime_error("planned failure");
    }
    std::puts("normal work completed");
}

int main() {
    std::puts("NORMAL");
    work(false);
    std::puts("EXCEPTION");
    try {
        work(true);
    } catch (const std::runtime_error& error) {
        std::printf("caught: %s\n", error.what());
    }
    std::puts("PASS scope trace");
    return 0;
}
```

```bash
g++ -std=c++17 -Wall -Wextra code/08_memory/scope_trace.cpp -o scope_trace
./scope_trace
```

预期输出：

```text
NORMAL
construct outer
construct inner
destroy inner
normal work completed
destroy outer
EXCEPTION
construct outer
construct inner
destroy inner
destroy outer
caught: planned failure
PASS scope trace
```

这个程序不申请CUDA资源，也不验证GPU清理。它只让C++对象生命周期变得可见。

### 8.9.2 class、public、private分别是什么

class Trace定义一种类型，并没有立即创建outer或inner。public下的成员可以从类外使用；private下的name_由类自身管理。末尾的分号是类定义语法的一部分。

Trace outer("outer")才创建一个对象。与类同名的Trace函数是构造函数，不写返回类型；它在对象初始化时执行。~Trace是析构函数，在这里负责打印销毁信息。下划线只是命名习惯，用来提示这是成员变量，不是C++特殊语法。

构造函数冒号后的name_(name)叫成员初始化列表：用传入的name初始化成员name_。name是参数，name_是对象内部保存的字段。它们名字相近，但不是同一个变量。

本例保存的是指向字符串字面量的指针，而不是复制任意外部字符串。字符串字面量在程序运行期间持续存在，所以这里不会因为work返回而悬空。若换成某个临时字符串的c_str()，就需要另行处理所有权。

### 8.9.3 正常结束与异常展开

正常路径中，inner的花括号先结束，因此先析构inner；之后打印normal work completed，work结束时再析构outer。

异常路径中，throw把正常控制流转向匹配的catch，未执行normal work completed。离开相应作用域时，已成功构造的局部对象按规则析构，这就是栈展开。最终catch收到异常并打印说明。[C++异常展开规则](https://eel.is/c++draft/except.ctor)

这里不是“程序崩溃后系统帮忙善后”，而是受C++异常处理规则约束的程序路径。未捕获异常、进程被强杀、崩溃等不能套用“必定完整展开”的结论。

### 8.9.4 const引用与what

catch中的const std::runtime_error& error以只读引用接收异常对象，避免不必要复制。error.what()提供异常说明字符串。你暂时不用设计异常继承体系；只需看懂：throw报告原因，catch在边界统一处理。

显式提前return也会离开正常作用域并析构相应局部对象；但std::exit不按同样方式析构调用链上的自动局部对象。因此不能在释放资源的教学中，一边用RAII，一边默认所有错误都用exit宏结束。[C++终止函数规则](https://eel.is/c++draft/support.start.term)

## 8.10 第一层封装：只负责float设备缓冲区

现在把Trace中“析构时打印”换成“析构时尝试释放所拥有的设备分配”。先只处理float，不要求你立即学模板和移动。

这份类的约定是：只用于当前单个CUDA设备，不在拥有内存期间切换或重置设备；对象可以给kernel借出地址，但不能复制或移动。先建立这样一个小而明确的责任范围。

### 8.10.1 完整头文件

文件：code/08_memory/float_buffer.cuh。

```cpp
#ifndef CUDA_BOOK_FLOAT_BUFFER_CUH
#define CUDA_BOOK_FLOAT_BUFFER_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

inline void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Single-device teaching owner: do not change/reset the current device while it owns memory.
class FloatBuffer {
public:
    explicit FloatBuffer(std::size_t count) : count_(count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::length_error("FloatBuffer byte count overflow");
        }
        if (count != 0) require_cuda(cudaMalloc(&ptr_, count * sizeof(float)), "allocate buffer");
    }

    ~FloatBuffer() noexcept {
        const cudaError_t status = close();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "buffer cleanup failed: %s\n", cudaGetErrorString(status));
        }
    }

    FloatBuffer(const FloatBuffer&) = delete;
    FloatBuffer& operator=(const FloatBuffer&) = delete;
    FloatBuffer(FloatBuffer&&) = delete;
    FloatBuffer& operator=(FloatBuffer&&) = delete;

    float* get() noexcept { return ptr_; }
    const float* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(float); }

    // One release attempt. On error, report it and end the operation; do not retry blindly.
    cudaError_t close() noexcept {
        float* old = ptr_;
        ptr_ = nullptr;
        count_ = 0;
        return old == nullptr ? cudaSuccess : cudaFree(old);
    }

private:
    float* ptr_ = nullptr;
    std::size_t count_ = 0;
};

#endif
```

.cuh是CUDA项目中常见的头文件命名习惯，不意味着它可以独立执行。下一节的.cu文件通过#include包含它。头文件里的保护宏防止同一翻译单元重复定义类；require_cuda前的inline允许这种函数定义放在头文件中供多个翻译单元使用，不是保证编译器必须把机器码内联。

### 8.10.2 构造成功后，类保证什么

创建FloatBuffer device(8)后，如果没有异常，device要么持有8个float的未初始化设备存储，要么在count=0时是空对象。成功分配不等于数组内容已初始化；下一节仍会执行H2D复制。

构造前先用max()/sizeof(float)检查乘法范围，防止count*sizeof(float)溢出。max()来自numeric_limits，表示size_t能表示的最大值。这个检查只防算术溢出，不保证设备有足够内存；真实容量不足仍由cudaMalloc报错。

如果构造函数中的cudaMalloc失败，require_cuda抛出异常，整个FloatBuffer对象没有成功构造，因此它自己的析构函数不会作为“已完成对象”被调用。这份构造函数只申请一份资源，成功后没有其他可能抛异常的业务步骤；如果以后扩展为申请多个资源，需要保证后续失败也会清理前面成功的部分。

### 8.10.3 为什么构造函数前有explicit

explicit让“创建资源拥有者”这件事必须明确。我们允许：

```cpp
FloatBuffer device(8);
```

但不允许把整数在需要FloatBuffer的地方隐式变成一次设备分配。例如FloatBuffer device=8会被拒绝。explicit不是GPU性能选项，而是C++接口约束。[explicit构造函数](https://eel.is/c++draft/class.conv.ctor)

### 8.10.4 get、size、bytes都没有复制数组

| 调用 | 返回的东西 | 单位或意义 |
|---|---|---|
| device.get() | 底层地址 | 借用，不转移释放责任 |
| device.size() | count_ | float元素数量 |
| device.bytes() | count_*sizeof(float) | 字节数 |

不要把get当成“从GPU取回数组”。它只返回指针值；CPU要读普通设备内存内容仍需相应复制或其他合法访问机制。

get有两个版本，后面的const约束被调用的对象：从const FloatBuffer取得const float*，不能通过这个指针修改目标。它仍不是硬件constant memory。

### 8.10.5 为什么先禁用复制和移动

普通指针复制只复制地址。若两个对象都保存同一个ptr_并都认为自己有权释放，析构时就可能对同一分配释放两次。用=delete明确禁止复制构造与复制赋值，可以在编译时阻止这类无意操作。

第一版连移动也禁止，让你先只面对一个拥有者。后面再增加一个“转交所有权、让原对象变空”的动作，而不是通过删除这些限制强行让代码编译。

### 8.10.6 close与析构的错误责任

正常路径可以显式调用close并检查返回值。析构则提供兜底清理，它不能随便把错误继续抛出：尤其在另一个异常正在展开时，再抛出异常可能导致程序终止。noexcept明确声明这里不向外抛出C++异常。

本教学实现的close先取出地址，再把对象清空，然后只尝试一次cudaFree。这样析构不会在显式close失败后盲目重复释放。代价是：如果释放API报错，类不声称资源已成功回收，也不提供继续使用或自动重试该地址的恢复机制。调用者应记录错误、终止该操作，而不是因size()变成0就宣称清理成功。

这一设计是可读的教学取舍，不是通用生产分配器。后面需要多设备、异步释放、内存池或复杂故障恢复时，所有权协议也要扩展。不要把“使用了析构函数”误读为“任何故障下都没有资源问题”。

## 8.11 案例六：把 RAII 用进真正的 CUDA 程序

封装完成不等于学会使用。现在把第 5 章的“分配—复制—计算—取回—释放”重新写一遍。算法仍然很小：输入 1 到 8，每个元素乘以 2。这样你可以把注意力放在资源责任上。

完整文件 code/08_memory/raii_scale.cu：

```cpp
#include "float_buffer.cuh"
#include <cmath>
#include <iostream>
#include <string>

__global__ void scale(float* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] *= 2.0f;
}

int main(int argc, char** argv) {
    const bool fail = argc == 2 && std::string(argv[1]) == "--fail-after-allocation";
    if (argc > 2 || (argc == 2 && !fail)) {
        std::cerr << "usage: raii_scale [--fail-after-allocation]\n";
        return 2;
    }
    try {
        const int n = 8;
        float host[n] = {};
        for (int i = 0; i < n; ++i) host[i] = static_cast<float>(i + 1);
        FloatBuffer device(n);
        if (fail) throw std::runtime_error("planned failure after allocation");
        require_cuda(cudaMemcpy(device.get(), host, device.bytes(), cudaMemcpyHostToDevice), "H2D");
        scale<<<1, 32>>>(device.get(), n);
        require_cuda(cudaGetLastError(), "launch");
        require_cuda(cudaDeviceSynchronize(), "wait");
        require_cuda(cudaMemcpy(host, device.get(), device.bytes(), cudaMemcpyDeviceToHost), "D2H");
        for (int i = 0; i < n; ++i) {
            std::cout << "i=" << i << " value=" << host[i] << '\n';
            if (!std::isfinite(host[i]) || host[i] != 2.0f * (i + 1)) {
                throw std::runtime_error("numeric mismatch");
            }
        }
        require_cuda(device.close(), "explicit release");
        std::cout << "PASS RAII scale: 8 elements\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
```

### 8.11.1 从编译到运行

先进入本章代码目录。两个文件 float_buffer.cuh 和 raii_scale.cu 必须位于同一目录；引号形式的 include 会先在相应的源文件位置查找头文件。

```bash
cd code/08_memory
nvcc -std=c++17 -lineinfo raii_scale.cu -o raii_scale
./raii_scale
```

这里再拆一次命令，而不是要求你记住一串字符：

| 部分 | 本例中的作用 | 自己写程序时怎样替换 |
|---|---|---|
| nvcc | CUDA 编译驱动，协调主机与设备代码编译 | CUDA 源码通常仍用它 |
| -std=c++17 | 选择 C++17 语言标准 | 本书示例保持此项 |
| -lineinfo | 给设备代码加入源代码行号信息，便于诊断 | 不是调试版 -G，也不是运行命令 |
| raii_scale.cu | 本次编译的源文件 | 换成你的 .cu 文件名 |
| -o raii_scale | 指定输出程序名 | 名字可以自行决定 |
| ./raii_scale | 在 Linux 当前目录执行刚生成的程序 | 必须与输出名对应 |

Windows 在配置好 CUDA 与主机编译器的终端中可写 nvcc -std=c++17 -lineinfo raii_scale.cu -o raii_scale.exe，随后在 PowerShell 中运行 .\raii_scale.exe。扩展名 .exe 并不表示 CUDA 在编译时运行了；编译与执行是两步。

头文件不会单独变成一个需要手工运行的程序。预处理时 include 把它的声明和定义引入当前编译单元。不要执行 nvcc float_buffer.cuh -o buffer 来尝试“先运行类”。

正常结果应为：

```text
i=0 value=2
i=1 value=4
i=2 value=6
i=3 value=8
i=4 value=10
i=5 value=12
i=6 value=14
i=7 value=16
PASS RAII scale: 8 elements
```

这是由输入和计算规则推导的预期结果，不是本书编辑环境的 GPU 实测记录。

### 8.11.2 类在 CPU 上，所拥有的数组在设备内存中

device 这个 C++ 对象本身是主机函数的局部变量。它的成员 ptr_ 保存地址，count_ 保存数量。被 cudaMalloc 申请的数组才是设备存储。两者不是同一块东西：

```text
CPU 上的 device 对象                 CUDA 分配的设备存储
┌───────────────────┐                ┌────────────────────┐
│ ptr_ ─────────────┼── 保存地址 ───→ │ 8 个 float 的位置  │
│ count_ = 8        │                │ 初始内容未定义     │
└───────────────────┘                └────────────────────┘
       生命周期结束                         需要 cudaFree
```

主机调用 device.get() 只取得地址，device.bytes() 只算出 32 字节。kernel 得到的是底层 float* 和 n，而不是整个资源管理类。这个类没有 __device__ 成员函数，我们也没有把它复制进 GPU，让每个线程都负责释放一次数组。

因此，RAII 改变的是“主机代码如何管理资源”，没有改变 kernel 如何通过 i 访问元素。

### 8.11.3 require_cuda 怎样把 API 失败连接到清理

以 H2D 那行为例，执行顺序是：

1. 先调用 cudaMemcpy，得到 cudaError_t 状态；
2. 把状态和文字 "H2D" 交给 require_cuda；
3. 成功则继续；失败则构造异常并抛出；
4. 控制流离开 try 中已建立对象的作用域，执行相应析构；
5. 到达 catch，打印错误，main 返回 1。

传入的 "H2D" 只是定位标签，不是 CUDA API 的参数。它让“invalid argument”之类错误不至于脱离发生位置。

catch(const std::exception& error) 可以接住这里使用的 runtime_error 和 length_error，因为它们都派生自 std::exception。const 引用避免复制异常对象，并表明这里只读取它；what() 返回说明文本。

注意，返回码是给 shell 或测试工具看的。return 0 表示该程序认定成功；return 1 表示失败；不认识命令行参数时返回 2。这些约定不能替代检查输出内容，但可以让脚本不用猜某行中文是不是错误。

### 8.11.4 故意在分配后抛出异常

运行：

```bash
./raii_scale --fail-after-allocation
echo $?
```

应打印 ERROR: planned failure after allocation，并返回 1。Windows PowerShell 运行程序后用 $LASTEXITCODE 查看退出码。

这个分支只在分配成功后、任何 kernel 提交前抛出一个普通 C++ 异常。它不是越界实验，也不是制造 GPU 非法访问。FloatBuffer 已构造，所以离开作用域时其析构函数会尝试清理；没有走到正常路径的显式 close 也不应直接遗漏这一步。

但是，看到 ERROR 行并不能单凭日志证明 GPU 上绝无泄漏。若要验证真实清理结果，还应在有 CUDA 的机器上结合第 7 章的方法检查：

```bash
compute-sanitizer --tool memcheck --leak-check full ./raii_scale
compute-sanitizer --tool memcheck --leak-check full ./raii_scale --fail-after-allocation
```

第二个程序本来就返回非零；应区分“预设异常导致的退出”和工具实际报告的非法访问、泄漏。不要为了把终端变成零退出码而删掉异常检查。

[Compute Sanitizer 的泄漏检查说明](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#leak-checking)给出了 --leak-check full 的用途：在上下文销毁时检查仍未释放的设备分配。

如果连 cudaMalloc 都失败，程序可能先报告 allocate float buffer 相关错误，当然也不会到达预设异常。这说明测试前提不成立，而不是证明故障分支通过。

## 8.12 为什么要移动：转交所有权，不是移动 GPU 数据

现在假设要把某个已经申请好的缓冲区交给另一个对象管理。重新申请并复制数组是一种办法，但如果目的只是换一个负责释放的对象，就没有必要搬运数据。

先区分三种操作：

| 操作 | 数组内容是否复制 | 谁负责释放 |
|---|---|---|
| float* p = owner.get() | 不复制，只借用地址 | 仍是 owner |
| 用 cudaMemcpy 复制到另一份分配 | 复制数据；需要独立目标空间 | 各自的拥有者 |
| 移动构造新的拥有者 | 本例不复制数组 | 新对象接手，旧对象变空 |

移动后的状态是本教学类明确规定的协议：

```text
移动前： first {地址 A, 数量 4} ───拥有──→ 分配 A
         second 尚未构造

移动构造 second(std::move(first))

移动后： first {nullptr, 0}
         second {地址 A, 数量 4} ──拥有──→ 原来的分配 A
         分配 A 的 4 个数没有因移动而改变
```

如果只把地址赋给 second，却没有清空 first，两个对象就都认为要释放 A，这不是正确的所有权转交。

这里“移动后源对象为空”是我们的类保证的，不应推广为任何 C++ 类型被 std::move 后都变成空。不同类型有自己的移动后状态约定。

std::move 本身也没有 CUDA 功能。它把表达式转换为允许匹配移动操作的形式；真正转交指针的是接下来选中的移动构造函数。单独写 std::move(first); 并丢弃结果，不会执行我们的移动构造，也不会清空 first。[C++ 移动构造规则](https://eel.is/c++draft/class.copy.ctor)

## 8.13 案例七：逐步读懂带类型参数的设备缓冲区

### 8.13.1 模板解决哪个重复问题

FloatBuffer 固定保存 float。如果还要保存 int，不必复制整个类并机械替换所有 float。用类型参数 T 表示“本次缓冲区保存的元素类型”：

```cpp
template<class T>
class DeviceBuffer;
```

template 是模板声明；class T 在这里声明的是类型参数，写 typename T 也可以。T 不是运行时输入的字符串，也不是数组里的数。

写 DeviceBuffer<float> 时，T 对应 float；写 DeviceBuffer<int> 时，T 对应 int。因此 get() 分别返回 float*、int*，bytes() 分别使用 sizeof(float)、sizeof(int)。不同模板实参形成不同类型，编译器会检查指针是否匹配。

不要把“可写任意 T”理解为“可以把任意 C++ 对象直接复制到设备”。std::string 等对象还可能持有主机指针、拥有动态资源，并有构造析构要求。本教材版本只允许 float 和 int，先把标量数组所有权学清楚。

完整文件 code/08_memory/device_buffer.cuh：

```cpp
#ifndef CUDA_BOOK_DEVICE_BUFFER_CUH
#define CUDA_BOOK_DEVICE_BUFFER_CUH

#include "float_buffer.cuh"
#include <type_traits>

// Deliberately restricted to the two scalar types used by this lesson.
template<class T>
class DeviceBuffer {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, int>,
                  "This teaching buffer supports only float and int");
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::length_error("DeviceBuffer byte count overflow");
        }
        if (count != 0) require_cuda(cudaMalloc(&ptr_, count * sizeof(T)), "allocate typed buffer");
    }
    ~DeviceBuffer() noexcept {
        const cudaError_t status = close();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "typed buffer cleanup failed: %s\n", cudaGetErrorString(status));
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    cudaError_t close() noexcept {
        T* old = ptr_;
        ptr_ = nullptr;
        count_ = 0;
        return old == nullptr ? cudaSuccess : cudaFree(old);
    }
private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

#endif
```

这个头文件复用 float_buffer.cuh 中的 require_cuda，因此编译本例时两个头文件都要保留。它是为了教学逐步扩展的组织方式；后续大型项目可以把错误工具单独拆成公共头文件。

### 8.13.2 两句看起来相似的代码，含义并不相同

```cpp
DeviceBuffer<int> second(std::move(first)); // 构造一个新对象
// second = std::move(first);              // 给已存在对象赋值：本类禁止
```

移动构造时，second 之前不存在，没有旧资源需要处理。把 first 的成员拿过来，再清空 first 即可。

移动赋值时，second 可能已拥有另一块分配。先释放旧资源可能失败，释放时还可能存在未完成工作，调用者要能理解错误策略。本章不把这些问题隐藏在一行赋值里，明确 =delete 禁用移动赋值。需要时先学会显式结束旧资源的生命周期，再设计相应接口。

移动构造上的 noexcept 是一个承诺：这个构造只转移指针和数量，不申请资源、不发起复制，不向外抛出异常。它并不是“设备执行绝不出错”的承诺，因为这里根本没有提交设备工作。

### 8.13.3 用数据和地址一起验证

完整文件 code/08_memory/move_buffer.cu：

```cpp
#include "device_buffer.cuh"
#include <iostream>
#include <utility>

static_assert(!std::is_copy_constructible_v<DeviceBuffer<int>>);
static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer<int>>);
static_assert(!std::is_move_assignable_v<DeviceBuffer<int>>);

int main() {
    try {
        const int input[4] = {9, 8, 7, 6};
        int output[4] = {};
        DeviceBuffer<float> empty(0);
        if (empty.get() != nullptr || empty.size() != 0 || empty.bytes() != 0) return 1;
        DeviceBuffer<int> first(4);
        require_cuda(cudaMemcpy(first.get(), input, first.bytes(), cudaMemcpyHostToDevice), "H2D");
        int* original = first.get();
        DeviceBuffer<int> second(std::move(first));
        if (first.get() != nullptr || first.size() != 0 || second.get() != original ||
            second.size() != 4 || second.bytes() != sizeof(input)) {
            throw std::runtime_error("ownership transfer failed");
        }
        std::cout << "source_size=" << first.size()
                  << " destination_size=" << second.size() << " same_address=1\n";
        require_cuda(cudaMemcpy(output, second.get(), second.bytes(), cudaMemcpyDeviceToHost), "D2H");
        for (int i = 0; i < 4; ++i) {
            std::cout << "i=" << i << " value=" << output[i] << '\n';
            if (output[i] != input[i]) throw std::runtime_error("copy-back mismatch");
        }
        require_cuda(second.close(), "release moved buffer");
        require_cuda(second.close(), "close empty owner");
        std::cout << "PASS move ownership; no kernel launched\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
```

编译运行：

```bash
nvcc -std=c++17 -lineinfo move_buffer.cu -o move_buffer
./move_buffer
```

预期输出：

```text
source_size=0 destination_size=4 same_address=1
i=0 value=9
i=1 value=8
i=2 value=7
i=3 value=6
PASS move ownership; no kernel launched
```

这个 CUDA 程序没有 kernel。它使用 Runtime API 管理和复制设备内存，足以观察所有权。不是所有 .cu 程序都必须写 __global__ 函数。

按时间读一次：

| 阶段 | first | second | 数据动作 |
|---|---|---|---|
| 创建 first(4) | 拥有 4 个 int 的分配 | 不存在 | 分配，内容尚未初始化 |
| H2D | 仍拥有 | 不存在 | 复制 9、8、7、6 到设备 |
| 移动构造 | 空 | 接手相同地址 | 无数组传输 |
| D2H | 空 | 仍拥有 | 从原分配复制回 output |
| 第一次 close | 空 | 放弃地址并尝试释放 | 正常情况下释放成功 |
| 第二次 close | 空 | 已空 | 不再调用 cudaFree |

original 只是保存地址供比较，不是第三个拥有者。等 second 释放后，这个地址不能再用来访问数据或释放。变量里还残留一个数字，不代表那块内存仍属于你。

程序同时检查了空缓冲 DeviceBuffer<float>(0)。我们自己规定零元素代表空对象，并跳过分配；这使它有清楚的 size=0、bytes=0、get=nullptr 状态，而不是依赖某个运行时对零字节分配的具体处理。

开头三个 static_assert 是编译期检查：

- 不能复制构造；
- 能够无异常地移动构造；
- 不能移动赋值。

它们不是运行时的 GPU 测试。若你不小心删除禁拷贝限制，或者修改了移动接口，这些检查有机会在编译阶段提醒你。

## 8.14 资源生命周期不等于 GPU 工作生命周期

### 8.14.1 “已经提交”不是“已经用完”

作用域是 C++ 控制流概念，kernel 执行是设备工作概念。下面是一个需要你警惕的代码形态，不是本章建议的运行例：

```cpp
{
    FloatBuffer device(n);
    // 初始化数据……
    kernel<<<grid, block>>>(device.get(), n);
} // 这里发生析构，但上面没有明确检查执行完成
```

不要把 cudaFree 的潜在同步行为当成程序的完成协议，更不要把它当作检查 kernel 正确完成的替代品。不同分配/释放机制的同步语义不同；未来使用异步分配或内存池时，靠偶然等待来维持正确性的程序更容易出错。

本章采用保守、清楚的顺序：

```text
分配 → 初始化/复制 → 提交 kernel → 检查启动状态
     → 等待并检查执行结果 → CPU 使用结果 → 显式检查释放
```

第 20 章才会把“全设备等待”细化为 stream 和 event 的依赖关系。现在宁可把阶段写明显，也不要为了少一行同步让 CPU 提前复用仍在传输的输入数组。

RAII 自动化的是释放责任，不是自动推导所有读写依赖。如果一个裸指针被别的对象、线程或未完成的 GPU 工作继续借用，拥有者就不能随便提前结束生命周期。

### 8.14.2 借来的地址会过期

```cpp
float* borrowed = nullptr;
{
    FloatBuffer owner(8);
    borrowed = owner.get();
}
// borrowed 不会自动变成 nullptr，但原资源已被尝试释放。
// 不能再把它当作有效输入传给 kernel，也不能再 cudaFree(borrowed)。
```

这叫悬空借用。get() 返回的不是一张“永久有效通行证”。本教学类不会跟踪有多少裸指针副本，所以必须由调用者保证借用不越过有效期。

移动后原始分配仍然存在时，先前借出的地址可能仍指向那份分配，但释放权和生命周期已经由新拥有者决定。为了减少初学阶段的混乱，先养成移动后重新从新拥有者 get() 的习惯。

### 8.14.3 哪些退出方式会执行析构

| 情形 | 对本例已构造的局部对象意味着什么 |
|---|---|
| 正常走出作用域 | 执行析构 |
| return 离开所在函数 | 对相应局部对象执行析构 |
| 异常被外层 catch 接住，途中展开作用域 | 清理已构造完成的对象 |
| 构造函数尚未成功就抛出 | 该完整对象未建立；已构造的成员/基类另按规则清理 |
| std::exit 直接终止 | 不按普通返回方式析构当前栈上的自动局部对象 |
| 进程被强制终止、系统崩溃 | 不应依赖 C++ 析构代码得到执行机会 |

因此，在依赖 RAII 的作用域里遇到错误就调用 std::exit 的宏，不等于会安全地执行所有局部清理。第 7 章的错误检查方法解决“错误有没有被看见”，这里还需要选择与生命周期相容的传播方式。这个示例用异常在外层统一处理。[C++ 异常展开](https://eel.is/c++draft/except.ctor)、[终止函数规则](https://eel.is/c++draft/support.start.term)

### 8.14.4 教学封装的边界

这里的 owner 假定在同一设备上下文约定下创建、使用和销毁。它没有保存设备编号，也没有在析构时切换设备；使用期间不要重置设备，也不要把它直接当作多 GPU 通用封装。

严重的设备执行错误不是“catch 住就能恢复正常”的普通业务失败。本例在 API 失败后报告错误并退出当前操作；析构可能只能尝试清理、报告清理失败。我们不承诺能够从任意损坏状态中恢复，也不把 size=0 当作物理资源必然释放成功的证据。

正常路径显式 close 能把释放状态交给调用者检查。析构兜底只能记录错误，不能修改已经准备好的 main 返回值来让所有清理错误自动传播。生产系统需要更完整的错误报告、上下文、异步依赖和分配器策略；先知道边界，比把一个短类误认成通用容器更重要。

## 8.15 怎样选择内存，而不是“哪一种永远最快”

### 8.15.1 把选择对应到需求

| 你当前要解决的事 | 合理起点 | 仍要承担的责任 |
|---|---|---|
| 想先专心写 kernel，减少显式复制语句 | 支持时使用 managed memory | CPU/GPU 访问阶段、同步、迁移成本 |
| 想明确画出每次 H2D/D2H | 普通主机数组 + cudaMalloc/cudaMemcpy | 字节数、方向、边界、释放 |
| 大量主机与设备传输，希望进一步优化 | 先测量；有需要再用 pinned host buffer | 主机资源占用、重叠条件 |
| 希望复制与计算重叠 | pinned + 合适的 stream/依赖设计 | 不提前复用缓冲、验证实际重叠 |
| 一个 block 内许多线程反复使用同一块数据 | shared memory | 协作装载、同步、容量和 bank conflict |
| 频繁申请释放临时设备空间 | 先复用已分配缓冲，再学内存池 | 生命周期与 stream ordering |
| 只想减少遗漏的释放 | 为相应资源写 owner | 正确配对释放、禁止错误复制、错误策略 |

不要因为 managed 示例最短就认定它最快，也不要因为显式拷贝多几行就认定它更慢。数据规模、访问方式、平台模型和是否反复迁移都会改变结果。性能需要测量，第 9 章会专门讲“测到了哪段时间”。

“复杂指针结构方便移植”也有前提：结构内部指向的每个对象都必须可被使用它的处理器合法访问。把外层结构放入 managed memory，不会自动把其成员指向的普通主机分配变成所有平台都能访问的设备数据。

### 8.15.2 传输方向与释放配对速查

| 创建方式 | 典型用途 | 对应结束方式 |
|---|---|---|
| 栈数组 float a[8] | 小型主机例子 | 离开有效作用域 |
| new float[n] | 主机动态数组 | delete[] |
| std::malloc | 主机原始存储 | std::free |
| cudaMalloc | 设备分配 | cudaFree |
| cudaMallocManaged | managed 分配 | cudaFree |
| cudaMallocHost / cudaHostAlloc | CUDA 管理的主机分配 | cudaFreeHost |
| cudaHostRegister 对既有存储进行注册 | 本章未展开的注册方式 | 先 cudaHostUnregister，再按原方式释放原存储 |

RAII 类必须使用与所拥有资源匹配的释放方式。不能把 FloatBuffer 的 cudaMalloc 简单换成 cudaMallocHost，却仍在 close 中调用 cudaFree。[Runtime 内存 API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)

cudaMemcpy 的方向也是合同的一部分：

| 方向常量 | 含义 | 本书缩写 |
|---|---|---|
| cudaMemcpyHostToDevice | 主机到设备 | H2D |
| cudaMemcpyDeviceToHost | 设备到主机 | D2H |
| cudaMemcpyDeviceToDevice | 设备到设备 | D2D |
| cudaMemcpyHostToHost | 主机到主机 | H2H |
| cudaMemcpyDefault | 在支持统一虚拟寻址时按地址推断 | 初学阶段优先写明方向 |

“目标在左、来源在右、大小是字节”仍然适用。自动方向推断不能检查你本意究竟想复制哪个数组，也不能把错误的大小自动修好。

### 8.15.3 一个节省传输的手算例子

假设一个 100 MB 数组依次经过三个 GPU kernel A、B、C，CPU 只需要最后结果。这里 MB 按十进制计算，且假设三步输入输出大小相同。

方案一：每一步都传入、算一次、传出。

```text
CPU ─H2D→ A ─D2H→ CPU ─H2D→ B ─D2H→ CPU ─H2D→ C ─D2H→ CPU
                    6 次 × 100 MB = 600 MB
```

方案二：一次传入，让中间结果留在 GPU，最后传出。

```text
CPU ─H2D→ [A → B → C] ─D2H→ CPU
                2 次 × 100 MB = 200 MB
```

减少的是不必要的主机—设备往返，不是 kernel 内部的设备内存访问。若 A、B、C 需要不同输入输出缓冲，还应正确安排独立数组或双缓冲，不能为了省分配覆盖仍要使用的数据。

由此得到一套实用顺序：

1. 先检查 CPU 是否真的需要中间结果；
2. 能连续在 GPU 处理，就让数据保留在那里；
3. 能复用分配，就不要在紧密循环里不断申请释放；
4. 再考虑合并小传输、使用 pinned、构建异步管线；
5. 分开测量传输、kernel 和端到端时间，确认瓶颈。

这比一开始寻找“最快的内存类型”更容易得到可解释的改进。

## 8.16 练习：从读懂到独立写出

不要求一次全部完成。先做 A 组，再做 B 组；C 组用于确认你真的理解所有权。答案见 [附录 C：练习答案](附录-C-练习答案.md) 的第 8 章。

### A 组：纸笔与预测

1. 有 12 个 float，假设 sizeof(float)=4。n 与 bytes 各是多少？cudaMemcpy 的第三参数应传哪个？如果只传 12，会完整复制几个元素？
2. std::memset(values,1,sizeof(values)) 为什么不能把 float 数组变成 1.0？请分别解释字节值 1 与数值 1.0。
3. managed_scale 在 kernel 之后有一行 cudaDeviceSynchronize。删除这行后立刻由 CPU 读结果，为什么不能凭“一般算得很快”保证正确？
4. UVA、managed、pinned 分别主要回答地址、访问管理、主机分配中的哪个问题？为什么“两个指针数值相同”不能证明 CPU 可以解引用它？
5. 内存能力查询函数返回 cudaSuccess，同时 managedMemory=0，是“API 失败”还是“查询成功但能力没有”？为什么属性值不能只靠默认初始化为 0 来判断？
6. cudaMemGetInfo 显示空闲 6 GiB。申请 4 MiB 后空闲减少了 5 MiB，是否就证明申请函数多分配了 1 MiB 给你的数组？能否据此保证接着申请全部剩余空闲空间？
7. “local 变量一定在片上高速内存”“const float* 就是 constant memory”“shared 可以让不同 block 随意交换数据”，逐句判断并改写。

### B 组：改程序并验证

8. 把 managed_scale 改为加 5：输入仍是 1 到 10。改 kernel 的同时，修改期望值和 PASS 文本；写出首末输出。
9. 把 pinned_roundtrip 从 8 个 float 改为 17 个。保持一个 32 线程 block，说明为什么边界检查不能删。列出应修改和不应修改的量。
10. 运行 memory_capabilities，再尝试 managed_prefetch。记录工具链版本、四项属性与退出码；如果程序返回 77，应怎样写实验结论？没有 GPU 时可先写结果记录模板，不能编造属性。
11. 在 scope_trace 中让内层作用域正常结束以后再抛出异常。预测 inner 是否会被析构两次，再修改代码验证。
12. 运行 raii_scale 的正常路径和 --fail-after-allocation 路径。分别写出预期退出码，说明第二条如何帮助检查“已分配但尚未计算时失败”的路径。

### C 组：所有权与设计

13. FloatBuffer a(8); 后面写 FloatBuffer b=a; 为什么不允许？若想让 b 有相同内容但独立存储，应该有哪些步骤？
14. DeviceBuffer<int> first(4); 然后移动构造 second。画出移动前后两个对象保存的地址和数量。指出 H2D、移动、D2H 中哪些真的搬运数组数据。
15. 为什么 second(std::move(first)) 可以，而 second=std::move(first) 不可以？这个限制来自 CUDA，还是我们写的 C++ 类接口？
16. 借出 float* p=owner.get() 后让 owner 离开作用域，再给 kernel 传 p，有什么问题？把 p 自己设为 nullptr 能修复已经提交的工作吗？
17. 为固定 float 版本写出 bytes() 的表达式，并解释构造函数为什么还要检查乘法溢出。零元素对象应保证哪些状态？释放 API 报错后 size=0 能否证明显存已经回收？
18. 一个 64 MB 数组连续经过 4 个 GPU 步骤，CPU 只需要最终结果。每步往返与只在首尾传输各是多少 MB？请同时写出这种简化成立的输入输出假设。

## 8.17 本章验收与回顾

本章一共提供 9 个可执行示例、2 个头文件，按学习顺序如下：

| 顺序 | 程序 | 你应能解释的现象 |
|---|---|---|
| 1 | byte_fill.cpp | 填充字节与赋元素值不同；只需 CPU |
| 2 | managed_scale.cu | 同一 managed 指针不取消访问阶段 |
| 3 | memory_capabilities.cu | 查询状态与设备属性值不同 |
| 4 | pinned_roundtrip.cu | pinned 主机缓冲与设备缓冲是两份分配 |
| 5 | memory_snapshot.cu | 空闲显存只是瞬时统计 |
| 6 | managed_prefetch.cu | 版本与能力影响预取接口及是否可运行 |
| 7 | scope_trace.cpp | 构造、析构与异常展开；只需 CPU |
| 8 | raii_scale.cu | 一个对象接管一份 float 设备分配 |
| 9 | move_buffer.cu | 只转交所有权，不移动设备数组内容 |

具体编译命令集中在 code/08_memory/README.md。没有 GPU 的读者先完成两个 CPU 实验和纸笔练习；GPU 程序不应通过删除错误检查或伪造输出来“通过”。

继续下一章以前，至少能不看代码回答：

- 我现在拿到的是地址、元素数量还是字节数？
- 这个处理器此刻是否允许访问目标数据？
- 上一次使用该缓冲的异步操作是否已经完成？
- 当前只有谁负责释放？借出的指针什么时候过期？
- 这个案例验证了数值、所有权、能力，还是性能？

最后一个问题尤其重要。本章没有证明 managed 比显式复制快，也没有证明 pinned 一定带来重叠。我们先把正确性和资源责任写清楚，下一章再学习如何可信地测量时间。

> 验证边界：本次编辑环境没有可用的 C++/CUDA 编译器或 GPU。代码经过文本一致性与离线规则检查，但未在这里完成编译、GPU 运行或 Compute Sanitizer 验证。文中输出是预期结果；请在自己的 CUDA 环境按命令运行并核对。

下一章：[09-正确计时与带宽基线](09-正确计时与带宽基线.md)。
