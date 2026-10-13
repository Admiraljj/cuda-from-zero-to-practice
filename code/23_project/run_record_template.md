# 第 23 章运行记录模板

此文件是空白学习模板，不是已执行的结果。复制到自己的记录文件后填写，缺少的实验写“未执行”。不要写账号、令牌或服务器密码。

## 源码与环境

- 仓库 commit / 是否有未提交修改：
- 本次工程阶段与目录：
- OS / CPU 编译器及版本：
- CMake 版本 / generator：
- nvcc / Toolkit / Runtime / cuBLAS 版本（若使用）：
- GPU 型号 / compute capability / driver（若使用）：

## 完整命令

- 配置命令与 options：
- source / binary 目录：
- build 命令与 Debug/Release 配置：
- CUDA 架构 / 是否使用 -G / 实际 verbose flags：
- 程序命令：
- CTest / Sanitizer / profiler 命令（若执行）：

## 输入与结果

- backend / n / seed / alpha / beta：
- oracle、长度、finite、输入不变性：
- 正常测试数量 / 拒绝测试数量 / skipped 数量：
- 每个 skip 的具体原因：
- 失败的首条关键诊断：

## 证据与边界

- 保存的构建日志 / 测试输出：
- 哪些只做了静态检查，哪些实际编译/运行：
- GPU / Sanitizer 是否真正执行：
- 若测量：计时范围、预热、样本、同机负载与配置：
- 本章默认案例是正确性工程，不把整体运行耗时称为 kernel-only benchmark。
