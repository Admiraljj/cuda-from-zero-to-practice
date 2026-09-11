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

## 运行其余案例

```bash
nvcc -std=c++17 hello.cu -o hello
./hello
nvcc -std=c++17 launch_order.cu -o launch_order
./launch_order
nvcc -std=c++17 scalar_tasks.cu -o scalar_tasks
./scalar_tasks
```

按正文中的公式核对每条线程记录，允许线程输出顺序变化，但不能缺失、重复或出现错误数值。观察主机A/B/C顺序与同步后的完成标记，不要求GPU一定在B之后开始输出。
