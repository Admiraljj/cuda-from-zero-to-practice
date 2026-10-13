# 第26章指导项目：稳定选择正数后平方和

详细教学见[第26章](../../26-从初学者到CUDA工程师的路线.md)。固定输入[-2,0,3,-1,4]应得到selected=[3,4]、sum=25。保留序列顺序，空/无正数时sum=0。输入int范围[-1000,1000]，最多1048576项，64-bit signed累加。

从教材根目录CPU-only构建，不需要CUDA：

```bash
cmake -S code/26_capstone -B build-book26-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-book26-cpu --config Release
ctest --test-dir build-book26-cpu -C Release --output-on-failure
./build-book26-cpu/book26_app 1031 mixed cpu
./build-book26-cpu/book26_app 1031 mixed cub
```

最后一条应NOT RUN/77，不能视为GPU通过。GPU构建添加BOOK26_ENABLE_CUDA=ON与核对后的CMAKE_CUDA_ARCHITECTURES，使用新build目录。全书根工程也提供此子工程；启用CUB后端使用BOOK26_ENABLE_CUDA=ON，并指定已确认架构。

```bash
nvcc -std=c++17 -O3 -lineinfo -Icode/26_capstone/include code/26_capstone/src/main.cpp code/26_capstone/src/core.cpp code/26_capstone/src/cub_backend.cu -o capstone
./capstone 1031 mixed cub
./capstone 1031 none cub
./capstone 1031 all cub
compute-sanitizer --tool memcheck --error-exitcode 99 ./capstone 1031 mixed cub
compute-sanitizer --tool initcheck --error-exitcode 99 ./capstone 1031 none cub
```

参数n、mode、backend；0完成、1真实错误、2CLI拒绝、77已定义不可用。n0没有GPU执行；正n/none实际执行select后得到count0。所有GPU结果仍与CPU稳定序列与sum全量核对，并检查selected guard和device input snapshot。

此后端选择逻辑GPU0，完成整个host往返，count回传后按实际count查询归约workspace。不是完全异步/任意上下文透明接口。打印的backend_call_sample_ms是一次正确性运行的完整函数范围，不能给CPU/GPU做公平性能排名。

本机未配置/编译/运行工程，未验收GPU/Sanitizer/profiler；请填写[运行记录模板](project_record_template.md)，不要把示例命令写成实测成功。
