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

逐项核对程序返回值与输出。若当前没有可访问的 CUDA 设备，记录实际错误或零设备状态，不把正文里的示意参数当成自己的设备信息。
