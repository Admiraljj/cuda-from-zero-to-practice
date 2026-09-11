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

## 教材根目录中的自动验收

```bash
python3 scripts/verify_chapter05.py --static-only
python3 scripts/test_chapter05_checks.py
python3 scripts/verify_chapter05.py --cpu-only
python3 scripts/verify_chapter05.py --build-only
python3 scripts/verify_chapter05.py
```

- static-only：教材/源码镜像、围栏、章节链接。
- test_chapter05_checks.py：人工输出样例，只测试 Python 检查器。
- cpu-only：用 g++ 实际编译运行 CPU 模型。可用 --cxx clang++ 或编译器完整路径替换，要求兼容 GCC 风格命令。
- build-only：用 nvcc 构建所有三个程序，不运行。
- 默认：构建并运行；验证复制链路、小数组逐项输出、多种规模的程序自检、非法输入与 NaN 注入。
- --nvcc 指定编译器路径；--arch 指定设备和 Toolkit 均支持的目标，如 sm_80。不要直接照抄不匹配的架构。
- 缺少编译器返回 2，明确 NOT RUN；不安装驱动，不持久修改环境，不购买或配置远程资源。

默认设备验收选择最高 256 线程/block 的测试组；512 线程是正文中建议手动尝试、需要确认设备限制的扩展。

## 全书 CMake

保留 vector_add 原目标，新增 vector_index_model 与 vector_memory_roundtrip；CTest 增加小规模、空任务和两个前置实验。全书 CMake 配置需要 CUDA，即使只构建其中的 CPU 目标也如此；纯 CPU 学习请用上面的单独 g++ 命令。

## 验证记录

当前编辑环境没有 C++/CUDA 编译器；已执行的只有静态检查和 Python 检查器离线测试。服务器编译与 GPU 运行尚未执行。大数组仅打印最终自检标志，逐项比较在 C++ 程序内部完成；Python 检查器不会凭空获得未打印的设备数据。这里不做性能结论。
