# 第 6 章配套实验

按正文顺序运行。每个文件都自带 main，独立编译，不需要第三方图片库。

| 文件 | 工作 | 输出证据 |
|---|---|---|
| grid_map.cpp | CPU 枚举二维身份 | 32 个线程位置，15 个有效、17 个跳过 |
| invert.cu | 八组灰度图测试 | 空任务、单行/列、整除与非整除、两种 block 形状 |
| rgb_invert.cu | 固定 5×3 交错 RGB | 全部 15 像素、45 个分量 |
| pitched_scale.cu | 5×3 float 行跨度实验 | 实际两份 pitch 与全部矩阵结果 |

## 单独编译

```bash
g++ -std=c++17 -Wall -Wextra grid_map.cpp -o grid_map
./grid_map
nvcc -std=c++17 invert.cu -o invert
./invert
nvcc -std=c++17 rgb_invert.cu -o rgb_invert
./rgb_invert
nvcc -std=c++17 pitched_scale.cu -o pitched_scale
./pitched_scale
```

程序不解析运行参数。修改尺寸请改源码，并同步修改启动配置和检查器预期，再重新编译。RGB 的 grid(2,2) 对应固定5×3输入，不能在扩大尺寸后照搬。pitch 数值不固定，也不要求输入输出 pitch 相等。

## 自动验收（在教材根目录）

```bash
python3 scripts/verify_chapter06.py --static-only
python3 scripts/test_chapter06_checks.py
python3 scripts/verify_chapter06.py --cpu-only
python3 scripts/verify_chapter06.py --build-only
python3 scripts/verify_chapter06.py
```

前两条不执行 CUDA；分别检查源码镜像/链接，以及用人工输出检验 Python 检查器。cpu-only 使用 g++ 构建运行坐标模型；build-only 用 nvcc 构建全部四个程序；默认才会实际运行 GPU 案例。支持 --cxx、--nvcc 指定编译器；CUDA 构建可加与硬件和 Toolkit 匹配的 --arch。

脚本从同目录 verify_chapter05.py 复用进程调用逻辑，两份脚本应一起保留。默认 CMake 保留 invert，新增 image_grid_model、rgb_invert、pitched_scale 目标与测试；纯 CPU 学习不需要全书 CUDA CMake。

## 验证边界

本地已通过四份源码镜像检查与九项检查器/数学模型离线测试。当前无 C++/CUDA 编译器，尚未取得 CPU 程序编译运行或 GPU 测试结果。灰度大图由 C++ 程序逐像素自检；Python 只校验其输出元数据与打印的小矩阵。没有报告性能结果，也没有把打印的顺序当成设备调度顺序。
