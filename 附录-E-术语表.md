# 附录E：术语、单位与首次学习位置

用于回查，不要求先背完。名字相似不表示处于同一层；请同时看“是谁的对象、单位是什么、在哪执行”。

## E.1 执行与线程组织

| 术语 | 含义/边界 | 详细章节 |
|---|---|---|
| Host / Device | CPU提交一侧 / GPU执行一侧，不由变量名h_/d_自动决定 | 2、3、8 |
| Kernel | 一个launch产生许多逻辑thread执行实例 | 3、4 |
| Grid / Block / Thread | 一次launch全部blocks / 可shared+block barrier协作组 / 一个逻辑实例 | 4—6 |
| SM | 承载/调度blocks、warps的硬件单元，不等于软件block | 4、19 |
| Warp / Lane | 当前CUDA的32thread执行组 / 组内位置；按算法声明支持并核对warpSize | 4、12 |
| SIMT | 单指令多线程的执行组织，仍保持thread程序语义 | 4 |
| Divergence | 同warp线程走不同控制流，不等于不同blocks分支 | 4、12 |
| Mask | collective参与lane位图，不是元素数量 | 12、25 |
| Occupancy | resident warps相对架构最大warps比例，不是速度评分 | 19 |
| Eligible / Issued | 可发射 / 实际发射，不能与resident混同 | 19 |

## E.2 地址与内存

| 术语 | 含义/单位与边界 | 详细章节 |
|---|---|---|
| 元素数 / 字节数 | n / n*sizeof(T)，索引也不是容量 | 1、5、8 |
| Pointer | 地址值；赋值不搬数据，解引用需有效范围/内存类型 | 1、8 |
| Global memory | device范围的大容量存储；线程可访问哪些地址由代码/权限决定 | 8 |
| Register / Local memory | 每thread逻辑值的存储；local可能在显存，不因名字就片上 | 8、19 |
| Shared memory | 每block逻辑实例，普通blocks不因同SM就互访 | 10、25 |
| Constant memory | device只读路径的特定对象，适合相应广播访问 | 17 |
| Pinned memory | 页锁定host存储，支持适当Async传输，不保证重叠 | 8、20 |
| Managed Memory | 统一指针与系统管理放置/迁移；仍有平台/同步/性能边界 | 8、24 |
| UVA | 统一虚拟地址表示，不自动合并多卡显存/建立同步 | 24 |
| Stride / Pitch / Leading dimension | 跨步/行容量等布局量，必须说清元素或字节单位 | 6、16、22 |
| Coalescing | warp地址访问以较少memory transactions服务的模式 | 15、18 |
| Bank conflict | shared bank服务不同地址访问出现分批，具体按规则判断 | 10、15 |
| Guard / Sentinel | 保护区标记 / 输出初始标记，辅助发现越界或漏写 | 7、16、25、26 |
| Workspace | 算法临时device工作区，容量/所有权/并发复用需明确 | 22、26 |

## E.3 顺序与完成

| 术语 | 含义/不能替代什么 | 详细章节 |
|---|---|---|
| Stream | 提交有序工作队列，不是线程层级，也不保证别的stream顺序 | 20 |
| Event | 队列进度/依赖/计时对象，创建/记录/等待/测时各有条件 | 9、20、24 |
| Async | host可能不等实际完成，不自动保证并行重叠 | 3、8、20 |
| Barrier | 对明确参与集合的阶段会合，不跨普通block/设备 | 10、11 |
| Fence | 内存顺序/可见性约束，不让其他线程自动等待 | 10、12 |
| Atomic | 特定地址操作不可分割，不提供通用全局barrier | 13 |
| Race | 冲突访问缺少适当同步；偶尔数值正确也可能有race | 7、10 |
| Completion point | 已要求的工作完成/结果可读/资源可复用的证据点 | 3、20、24 |
| Graph / Capture / Exec | 工作流描述 / 记录操作 / 可launch的实例，不是捕获就执行 | 21 |

## E.4 算法与数值

| 术语 | 含义/边界 | 详细章节 |
|---|---|---|
| Reduction | 多项合成少量结果；单位元、范围、顺序需定义 | 11、12 |
| Scan | 每位置前缀累计，inclusive/exclusive不同 | 14 |
| Stable select | 保留满足predicate的项且相对顺序不变 | 14、22、26 |
| Histogram | 分bin计数，冲突更新与容量/总数有不变量 | 13 |
| Transpose | 交换矩阵维度与元素位置，不把reshape当transpose | 15 |
| GEMM | C=alpha*A*B+beta*C_old，布局/精度/旧C重要 | 16、22 |
| Stencil / Halo | 局部邻域计算 / 分片或tile需要的边界输入 | 10、17、24 |
| Softmax | 指数归一化的每行权重，本章稳定max-shift FP32契约 | 25 |
| CPU oracle | 独立参考与小手算，不是让GPU实现验证自己 | 全书、25、26 |
| atol / rtol | 绝对/相对误差允许项，不替代finite与算法不变量 | 9、16、25 |
| Arithmetic intensity / Roofline | 运算与流量比例 / 上界分析模型，不是实测吞吐 | 9、18 |
| Warm-up / Median | 预热 / 中位样本；与范围、样本数一起解释 | 9、25 |
| H2D / D2H / D2D | host→device / device→host / device之间的数据传输 | 5、8、24 |

## E.5 工程、库与多设备

| 术语 | 含义/边界 | 详细章节 |
|---|---|---|
| Declaration / Definition | 告诉类型签名 / 提供实体实现，不是多include就补定义 | 1、23 |
| Host lambda / Capture list | C++可调用对象 / 借用或保存外层变量，不等于CUDA Graph capture或kernel | 22、25、26 |
| Translation unit | 预处理后参与独立编译的源单元，guard作用不跨全部源 | 23 |
| NVCC | 协调host/device编译链接的驱动，不是GPU运行命令 | 3、23 |
| Object / Host link / Device link | 对象文件 / host定义解析 / device跨单元引用解析 | 23 |
| RDC | 可重定位device代码；不因host wrapper跨文件就必需 | 23 |
| Configure / Build / Run / Test | 生成构建体系 / 编译链接 / 执行 / 检查行为 | 23 |
| Runtime / Driver API | 常用执行管理层 / 更低层加载与context等控制 | 2、22 |
| Thrust / CUB / cuBLAS | 并行C++算法抽象 / primitives / 线性代数库，各自契约不同 | 22 |
| Device / Rank / Root | CUDA设备号 / 通信组参与者号 / collective根，不必相同 | 24 |
| Peer access / Peer copy | 直接访问权限 / 数据传输；都不自动证明producer完成 | 24 |
| AllReduce / AllGather | 按op合并并分发 / 收集数据，不是同一操作 | 24 |
| PASS / FAIL / SKIP / NOT RUN | 对应已验证成功 / 错误 / 约定跳过 / 未执行，不能混写 | 23—26 |

回到[阅读地图](00-阅读地图.md)，根据实际缺口查原章节，不需要将此表当作新概念第一次讲解。
