# 第 5 章配套实验

按顺序完成 index_map.cpp、memory_roundtrip.cu、vector_add.cu。前者是 CPU 编号模型，后两者需要 NVIDIA CUDA。完整代码逐份嵌入第 5 章正文。

## 在这个目录单独编译

```bash
g++ -std=c++17 -Wall -Wextra index_map.cpp -o index_map
./index_map
nvcc -std=c++17 memory_roundtrip.cu -o memory_roundtrip
./memory_roundtrip
nvcc -std=c++17 vector_add.cu -o vector_add
./vector_add 10 4
./vector_add 1000 256
./vector_add 10 4 --inject-nan
```

最后一次故意破坏主机验证输入，应该报告 mismatch、返回 1，不是成功运行测试。它没有故意制造设备非法访问。默认无参数运行向量加法使用 n=1048576、threads=256。

参数范围：n 为 0..1048576，threads 为 1..1024，并进一步受实际设备限制。n=0 不做设备查询、分配或 kernel 启动，其成功不能用作 GPU 验收。

## 结果核对

小数组按正文公式逐项核对，大数组由C++程序内部逐项比较。再测试空任务、非整除规模和非法参数；故障注入应报告失败。结果正确不代表性能最优，本章不做性能结论。

需要批量编译时可使用仓库根目录的 CMake。只练习CPU案例时，配置 `-DBUILD_CUDA_EXAMPLES=OFF`，无需CUDA工具链；具体命令见仓库首页。
