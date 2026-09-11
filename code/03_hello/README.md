# 第 3 章配套案例

正文见 [第 3 章](../../03-第一段CUDA程序与NVCC.md)。本章开始执行真正的 CUDA kernel。需要已获准使用的 NVIDIA GPU，以及支持它的驱动、Toolkit 和主机编译器。

| 文件 | 任务 |
|---|---|
| `hello_one.cu` | 一个 GPU 线程输出；主机检查启动与同步 |
| `hello.cu` | 2 个 block × 4 个线程，各自输出身份和启动规模 |
| `launch_order.cu` | 标记 CPU 提交前、提交返回、同步成功三个位置 |
| `scalar_tasks.cu` | 四个线程各自计算；调用普通 device 函数 |

在本目录先单独编译运行：

```bash
nvcc -std=c++17 hello_one.cu -o hello_one
./hello_one
```

正式记录应包含目标架构。若确认设备与 Toolkit 支持 sm_80，可以加 `-arch=sm_80`；不要把示例目标照搬到所有机器。其他参数解释见正文。

## 配套检查

从教材根目录执行：

```bash
python3 scripts/verify_cuda_intro.py --chapter 03 --static-only
```

这是文档和源码一致性检查，不编译或运行 GPU 程序。

具备正确开发环境时执行：

```bash
python3 scripts/verify_cuda_intro.py --chapter 03
```

可用 `--arch sm_80` 指定经过确认的目标，或 `--nvcc /实际路径/nvcc` 指定编译器。`--build-only` 只验证构建，不运行 kernel。

运行检查按 GPU 记录的多重集合验证，允许线程输出重排，但拒绝缺失、重复和错误数值。它只要求主机自己的 A/B/C 顺序与同步后的完成标记，不要求 GPU 必须在 B 之后开始输出。所有案例都必须编译并运行成功，才报告 GPU 测试通过。

## 本地离线测试

```bash
python3 scripts/test_cuda_intro_checks.py
```

这里使用人工构造的输出文本检验检查器本身，覆盖线程重排、B 与设备输出交错、重复记录、漏记录、错误数值和缺失完成标记。**它不执行 CUDA，也不证明真实 GPU 行为已经通过测试。**

## 本轮验证状态

2026-09-11：文档/源码静态检查及检查器离线单元测试通过。编辑环境没有可用 nvcc/GPU，真实构建与执行仍待开发环境验收。
