# 第 1 章配套案例

这些案例是普通 C++17 程序，只使用 CPU，不需要 CUDA 或 GPU。教学正文见 [第 1 章](../../01-你需要的C++与Linux基础.md)。

建议先逐个编译、预测输出并修改，不要第一次就批量运行后认为已经学会。

| 学习顺序 | 文件 | 本次只增加什么 |
|---|---|---|
| 1 | `01_hello.cpp` | 主函数、输出、退出码 |
| 2 | `02_values.cpp` | 类型、数值、元素数和字节数 |
| 3 | `03_loop_array.cpp` | 数组索引、循环边界 |
| 4 | `04_functions.cpp` | 函数、按值传参、引用 |
| 5 | `05_pointer_alias.cpp` | 地址、解引用、别名、指针参数 |
| 6 | `06_vector.cpp` | 容器复制、data、范围循环 |
| 7 | `07_vector_add_cpu.cpp` | 准备—计算—验证的闭环 |
| 8 | `08_multi_file/` | 声明、定义、头文件和链接 |
| 9 | `solutions/scale_cpu.cpp` | 从空文件写缩放程序的答案 |

## 单个案例

在本目录的 Linux 终端执行：

```bash
g++ -std=c++17 -Wall -Wextra -O0 -g 01_hello.cpp -o hello
./hello
```

换案例时同时换输入文件名和输出程序名。具体输出与逐行解释在正文。

## 批量验收

这不是学习必需的步骤，也不要求初学者先学 Python。完成手工实验后，可以在安装 Python 3 与 g++ 的服务器上，从教材根目录执行：

```bash
python3 scripts/verify_chapter01.py --compiler g++
```

检查器先检查文档与源码是否一致，再在新建的临时目录编译 9 个程序并核对输出；另外检查“漏掉 add.cpp 应链接失败”这一负向构建案例。它不依赖 GPU，也不使用全书的 CUDA CMake 工程。编译和运行均设置超时；临时构建产物在检查结束时自动清理，原始源码不修改。

可选的 CPU 内存检查构建：

```bash
python3 scripts/verify_chapter01.py --compiler g++ --sanitize
```

这需要编译器支持 AddressSanitizer、UndefinedBehaviorSanitizer 及其运行库。此模式不是 GPU 设备端检查。

仅检查文档与源码一致性、不编译：

```bash
python3 scripts/verify_chapter01.py --static-only
```

没有 C++ 编译器时，完整检查会明确失败退出，不会把“没有运行”写成“运行通过”。静态检查通过也不能替代实际编译和执行。

## 验证记录（2026-09-11）

本轮编辑环境只有 Python，没有找到可用的 g++、clang++ 或 cl，WSL 也未安装。因此本地仅完成文档/源码静态一致性检查；正文中的输出为预期结果，尚未在 C++ 编译器上实测。请在具备 C++17 编译器的 Linux 环境执行上面的完整检查。
