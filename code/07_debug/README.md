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

## 如何判断实验结果

需要同时核对程序输出、工具是否正常启动、诊断类别与错误汇总。坏版不能仅凭任意非零退出码算作复现成功；修复版则需要数值正确，并且对应工具未报告错误。

如果工具或程序卡住，超时本身不是成功检测目标错误的证据。只处理自己启动的练习进程，不重置共享GPU或终止其他用户任务。

## 工具能力与版本

本章基于NVIDIA官方Compute Sanitizer说明。initcheck基础案例使用默认global空间；较新工具提供shared/all选项，本地旧版本可能没有。racecheck主要观察shared访问hazard，不覆盖所有global竞争。synccheck不是任意并发协议的完整证明器。先修memcheck错误，再分析其他工具结果。
