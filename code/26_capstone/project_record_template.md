# 项目交付记录模板

这是空白模板，不是已运行结果。填写实际平台与证据；不要填写账号/令牌/私人登录地址。

## 规格

- 数学任务与布局：
- 输入/输出类型、shape、范围与空语义：
- 所有权、完成点与错误状态：
- 未支持范围：

## 构建环境

- commit：
- GPU、Compute Capability、可见设备：
- OS、driver、Toolkit、库、host compiler、CMake：
- 配置/构建命令、flags与构建类型：

## 验收（每项填写PASS / FAIL / SKIP / NOT RUN与原因）

- 固定手算/CPU参考：
- 编译/链接：
- 正常/空/尾部/非法参数：
- 数值/序列/guard/device input不变：
- GPU测试通过数、失败数、跳过数：
- memcheck/racecheck/initcheck/synccheck适用性、命令、状态：
- profiler命令、报告文件、观察：
- 仍未执行：

## 性能（没有实测留空，不猜数字）

- baseline与候选、同一任务/数据/精度：
- kernel-only / device workflow / host-complete / full roundtrip范围：
- 预热、样本、方法轮换、统计规则：
- 实际统计与离散情况：
- 控制实验、适用结论与限制：

## 他人复现

- 干净构建结果：
- README缺口与修正：
- 下一步：
