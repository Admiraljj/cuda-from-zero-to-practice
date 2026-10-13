# 附录B：先定位错误阶段，再修改一件事

这不是“按报错复制一条修复命令”的表。先确定失败发生在配置、编译、host/device链接、GPU提交、实际执行、数值验证还是清理。日志保留第一次关键错误，后续错误可能只是它的连锁结果。

## B.1 一条可重复的排查路线

1. 保存最小输入、完整构建命令、实际运行二进制路径。
2. 无法构建：先读第一条有意义诊断，区分缺头、缺定义、CUDA编译与链接。
3. 可以运行：检查API/launch/完成点返回值，不只看printf。
4. 用小输入与CPU参考逐项比较；查shape/字节/索引/所有权。
5. 先memcheck，再按需要initcheck/racecheck/synccheck。
6. 正确后才查性能：先Systems找整体空档，再Compute看指定kernel。

不得默认执行故意坏kernel、无界spin或reset共享GPU来“试试看”。[第7章](07-错误检查与Compute-Sanitizer.md)有受控错误练习与保护规则。

## B.2 配置、编译与链接

| 症状 | 先检查 | 不要先做什么 |
|---|---|---|
| nvcc找不到 | Toolkit/PATH/module，which nvcc | 随便改源码 |
| CMake CUDA识别失败 | host compiler/Toolkit/架构、完整配置日志 | 混用旧cache与新compiler |
| header not found | -I/目标include、路径与大小写 | include一个.cpp补定义 |
| function not declared | 正确声明/namespace/头文件 | 靠偶然间接include |
| undefined reference普通函数 | 定义源是否参与链接、签名是否一致 | 只多include一次头 |
| multiple definition | 头中非inline定义、重复编译、include实现文件 | 误以为guard跨所有.cpp |
| cudart/cuBLAS符号缺失 | Runtime/库链接与imported target | 认为find_package安装了库 |
| device符号未解析 | 跨.cu引用、RDC/device link、架构一致 | 把普通host跨文件与RDC混为一谈 |

模板错误从实例化使用位置回溯，别只盯库内部最后一行。具体构建关系见[第23章](23-CMake-项目结构-测试与可复现构建.md)。

## B.3 提交与异步执行错误

| 症状 | 可能原因/第一检查 |
|---|---|
| invalid configuration | grid/block/shared参数、算法线程要求与设备限制 |
| too many resources | 寄存器/shared/线程/launch限制，编译资源报告 |
| invalid device ordinal | 本进程可见编号，CUDA_VISIBLE_DEVICES是否重排 |
| invalid device function/no kernel image | 编译目标与实际GPU、Toolkit支持 |
| invalid resource handle | stream/event归属、是否已销毁 |
| illegal memory access | 越界、坏地址、生命周期、错误设备 |
| misaligned address | 类型转换、向量加载、偏移对齐 |
| unspecified launch failure | 保留更早错误，用小例与工具定位 |
| 后续API持续失败 | 可能严重异步错误；不要吞错继续算 |

launch检查与完成检查分开。非法访存可能在后面的同步/拷贝处暴露，不证明那条拷贝本身一定是根因。调试时终止进程、修复再重跑通常比在未知context状态重试安全；不默认cudaDeviceReset所有资源。

## B.4 内存的六个问题

- allocation容量是元素数乘sizeof(T)吗？乘法先检查范围了吗？
- 索引在本片local范围还是host global范围？二维row*cols、pitch/leading dimension单位正确吗？
- 尾部threads是否有边界？负索引是否先转size_t变成巨大正数？
- 指针属于host/device/managed/哪张GPU？部分重叠是否违反API前提？
- 最后消费者完成前，host/device缓冲是否已复用/释放？
- error catch展开是否先销毁仍被Async操作使用的host vector/栈变量？

guard和input snapshot分别查边界与只读约定，不替代工具。指针赋值只复制地址，不复制数组。

## B.5 竞争与同步的六个问题

- 哪些执行者访问同一个位置？是否至少一个写？
- atomic是否保护需要的单地址操作？它是否被误用为全局barrier？
- block屏障是不是全体该阶段参与者都会到达？
- tree每轮是否在消费前轮结果前会合？
- shared复用前，旧阶段读者是否都读完？
- producer→event→wait→consumer顺序与资源生命周期是否匹配？

__syncthreads不跨block/设备；stream有序不等于不同stream自动有序；peer可访问不等于源数据已经ready。warp mask描述参与者，不只是有效数据列。具体反例见第10/12/20/24/25章。

## B.6 数值不一致先排除算法错

先打印4—10项或一个输出元素的完整计算，再看误差。求和次序、FMA、fast math、低精度都可能影响舍入，但不能用“浮点不同”解释漏项、错row、旧输出、错beta、未初始化或NaN。

使用double参考、finite/非负/逐元素绝对+相对误差、不变量、guard/input checks。行和正确未必Softmax位置正确；平方和正确未必稳定选择顺序正确；总像素和正确未必边界正确。

容差必须说明尺度/类型/算法，不为让错误程序通过随意放宽。先测试精确可表示小整数，再一般分数与极端范围。

## B.7 性能异常不是API失败

kernel host计时几乎0可能只测提交；Event跨不同GPU不合法；copy不重叠可能受pageable、依赖、硬件/工作规模限制，而非少写一个“async”。

低occupancy不自动坏，高occupancy不自动快。检查工作量、内存访问/复用、同步、CPU供给、传输和库配置；报告优化是否改变数学/范围/初始化/验证成本。看第9/18/19/20章。

## B.8 请求帮助的最小证据

提供最小源文件/输入、完整配置与编译命令、实际二进制路径、第一条到后续完整错误、GPU/driver/Toolkit/host compiler/CMake、shape/threads/shared、退出码、Sanitizer状态、预期与实际。

删去密码/令牌/私有登录地址；不要只给“报错了”的截图。若有SKIP/NOT RUN，说明什么没执行，不能把它改写成PASS。
