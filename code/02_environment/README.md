# 第 2 章：CUDA 环境探针

正文见 [第 2 章](../../02-检查GPU服务器与CUDA环境.md)。三个程序只查询信息，不启动 kernel，不分配用户数据显存，不测试 GPU 数值计算。不能把探针通过当成全书示例均可运行。

在本目录的 Linux 终端逐个执行：

```bash
nvcc -std=c++17 device_count.cu -o device_count
./device_count
nvcc -std=c++17 device_info.cu -o device_info
./device_info
nvcc -std=c++17 cuda_versions.cu -o cuda_versions
./cuda_versions
```

`device_count` 和 `device_info` 返回 0 表示查询成功且至少有一个可见 CUDA 设备；API 失败返回 1；成功查询但设备数为 0 返回 2。这是本程序的退出码约定，不是 CUDA 错误码定义。

`cuda_versions` 仅报告版本，不证明设备可用。CUDA 12.0 起，`cudaRuntimeGetVersion` 用于返回编译时 Toolkit 版本常量，不能当作初始化或执行测试。

## 记录与检查

用 [环境记录模板](environment-record.md) 记录真实结果，避免把正文示意参数当成实测。

从教材根目录运行静态检查：

```bash
python3 scripts/verify_cuda_intro.py --chapter 02 --static-only
```

在已经获准使用 NVIDIA GPU、具备 nvcc 和主机编译器的环境完整编译运行：

```bash
python3 scripts/verify_cuda_intro.py --chapter 02
```

只有编译工具、没有可访问 GPU 的构建机可以使用 `--build-only`。该模式会明确标记未运行设备测试。

完整检查包含正常查询与单次屏蔽设备后的失败路径检查。它只修改自己创建的子进程环境，不修改 shell 的设备限制，也不访问作业之外的设备。构建产物位于新建的临时目录，结束后自动清理。

## 本轮验证状态

2026-09-11：本地静态一致性检查通过。编辑环境没有可用 nvcc，真实编译、驱动查询和 GPU 测试未执行；预期结果不能当成服务器验收结果。
