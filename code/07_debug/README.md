# 第 7 章：受控错误与修复对照

主实验debug_cases.cu有五组模式：oob、init、race、sync、launch，各有-good和-bad。默认运行oob-good；任何坏模式必须显式加--run-broken。只在已授权练习GPU上运行故意错误，不用于生产负载。

numeric_check.cpp是普通CPU数值比较器实验。broken_oob.cu保留最小越界案例，默认拒绝执行。后者不加入默认CMake正常验收。

## 编译（教材根目录）

```bash
g++ -std=c++17 -Wall -Wextra code/07_debug/numeric_check.cpp -o numeric_check
nvcc -std=c++17 -lineinfo code/07_debug/debug_cases.cu -o debug_cases
nvcc -std=c++17 -lineinfo code/07_debug/broken_oob.cu -o broken_oob
```

## 好版正常运行

```bash
./numeric_check
./debug_cases oob-good
./debug_cases init-good
./debug_cases race-good
./debug_cases sync-good
./debug_cases launch-good
```

好版对输出逐项比较，launch-good观察四个API状态。CMake只加入这些修复版和CPU检查器，不加入坏版执行。

## 对照检测

```bash
compute-sanitizer --tool memcheck --padding 32 --error-exitcode 99 ./debug_cases oob-good
compute-sanitizer --tool initcheck --error-exitcode 99 ./debug_cases init-good
compute-sanitizer --tool racecheck --racecheck-report all --error-exitcode 99 ./debug_cases race-good
compute-sanitizer --tool synccheck --error-exitcode 99 ./debug_cases sync-good
```

需要观察坏版时，明确把good改成bad并追加--run-broken；必须使用本章说明的对应工具。launch-bad只提交超限配置以观察拒绝与错误状态，使用：

```bash
./debug_cases launch-bad --run-broken
```

不要把程序BAD CASE completed或退出0当作计算正确。坏版即便产生看似合理结果，其执行仍然错误。

## 自动检查

```bash
python3 scripts/verify_chapter07.py --static-only
python3 scripts/test_chapter07_checks.py
python3 scripts/verify_chapter07.py --cpu-only
python3 scripts/verify_chapter07.py --build-only
python3 scripts/verify_chapter07.py
```

只有默认模式运行CUDA修复版与工具。显式执行故意错误：

```bash
python3 scripts/verify_chapter07.py --with-intentional-errors
```

支持--cxx、--nvcc、--sanitizer指定工具路径，--arch指定支持的架构。脚本复用同目录verify_chapter05.py的进程执行函数，不应单独移动它。

工具检测验收要求：有工具启动证据、可识别汇总、相应错误类别；坏版不能只凭任意非零退出码通过。若工具版本的报告格式改变或该架构/构建未复现预期诊断，脚本会失败并打印日志，需人工核对，不自动弱化检查。

每个子命令设有超时。超时不是目标错误被成功检测的证据；工具/目标进程的退出与设备恢复应另行确认，不能保证所有平台上的子进程均已被终止。只处理自己启动的练习进程，不重置共享GPU或终止其他用户任务。

## 工具能力与版本

本章基于NVIDIA官方Compute Sanitizer说明。initcheck基础案例使用默认global空间；较新工具提供shared/all选项，本地旧版本可能没有。racecheck主要观察shared访问hazard，不覆盖所有global竞争。synccheck不是任意并发协议的完整证明器。先修memcheck错误，再分析其他工具结果。

## 当前验证记录

三份源码与正文镜像检查、10项合成日志/输出离线测试已通过。无可用C++/CUDA编译器及GPU实测环境，未执行真实编译、Compute Sanitizer或CUDA-GDB。本书中的预期诊断不是伪造的实测日志。需要在你的练习环境完成编译与设备验收。
