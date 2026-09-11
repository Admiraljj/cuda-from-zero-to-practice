# 第 4 章实验：执行身份与分支路径

先读教材第 4 章，再预测输出；这里不测 GPU 调度周期或分歧性能。

| 文件 | 启动 | 应验证的记录 |
|---|---|---|
| inspect_layout.cu | 2 个 block，每个 40 线程 | 80 条身份记录；warp/lane 从各自 block 内编号计算 |
| branch_paths.cu | 两次各 1 个 block、64 线程 | 128 条路径记录；mode 0 按 warp 分组，mode 1 按奇偶分组 |

在当前目录编译运行：

```bash
nvcc -std=c++17 inspect_layout.cu -o inspect_layout
./inspect_layout
nvcc -std=c++17 branch_paths.cu -o branch_paths
./branch_paths
```

在教材根目录执行检查：

```bash
python3 scripts/verify_cuda_intro.py --chapter 04 --static-only
python3 scripts/verify_cuda_intro.py --chapter 04 --build-only
python3 scripts/verify_cuda_intro.py --chapter 04
python3 scripts/test_cuda_intro_checks.py
```

前三条依次是静态检查、仅编译、编译并实际运行；第四条仅用人工构造输出测试检查器，不执行 CUDA。需要指定架构时给前三条中的编译模式加上设备与 Toolkit 都支持的 --arch 参数。

同一次 kernel 的线程打印顺序不固定；检查器按完整身份集合核对，拒绝缺失、重复或算错的记录。源码中的 PASS launch 仅证明启动及同步未报告错误，不代替内容验证。改线程数以后，原检查器的固定预期不再适用。

编写本章时当前编辑环境无 nvcc 和可用 CUDA 运行环境；此处提供的是待服务器验收的源码，不声称实际 GPU 测试已通过。
