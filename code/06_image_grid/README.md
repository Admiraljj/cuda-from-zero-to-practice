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

程序不解析运行参数。修改尺寸请改源码，并同步修改启动配置和结果预期，再重新编译。RGB 的 grid(2,2) 对应固定5×3输入，不能在扩大尺寸后照搬。pitch 数值不固定，也不要求输入输出 pitch 相等。

## 结果核对

CPU模型应输出32个位置，其中15个有效、17个跳过，并且0—14每个有效下标恰好出现一次。灰度图程序会逐像素验证八组输入；RGB实验核对15个像素的全部45个分量；pitch实验核对三行数值与地址跨度。

不要要求输入输出pitch相等，也不要把固定的打印顺序当成GPU线程的执行顺序。
