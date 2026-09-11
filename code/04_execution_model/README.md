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

同一次 kernel 的线程打印顺序不固定。按完整身份集合核对，确认记录没有缺失、重复或计算错误。源码中的 `PASS launch` 仅表示启动及同步未报告错误，不能替代内容验证。修改线程数量后，要重新计算总记录数、warp数与lane。
