# Docs

## 定位
- 根目录 `AGENTS.md` 只保留入口导航和硬约束。
- `docs/` 用来承接需要按需加载的深层说明，避免把所有信息都堆在入口文档里。
- 文档默认只记录长期有效、可复用、能减少重复沟通的内容。

## 按需加载
- 想快速理解项目为什么存在、主链路是什么：读 [`01-business-goal-and-architecture.md`](01-business-goal-and-architecture.md)
- 想知道训练、推理、编码、解码入口和产物约定：读 [`02-entrypoints-and-artifacts.md`](02-entrypoints-and-artifacts.md)
- 想确认代码应该改在哪、何时允许动 `helios/`、输出怎么写：读 [`03-change-boundaries-and-collaboration.md`](03-change-boundaries-and-collaboration.md)
- 想评估一个方案是否真的服务压缩恢复主线：读 [`04-quality-checklist.md`](04-quality-checklist.md)

## 推荐阅读顺序
- 第一次进入仓库：`AGENTS.md -> 01 -> 02`
- 准备动代码：`AGENTS.md -> 02 -> 03`
- 评审方案或复盘结果：`01 -> 03 -> 04`
- 只在 `reconstruct/` 工作：先看 [`../reconstruct/AGENTS.md`](../reconstruct/AGENTS.md)

## 文档维护原则
- 若文档与代码冲突，以当前主链路代码和入口脚本为准。
- 发现新的长期稳定事实后，优先补 `docs/`，只把最关键的结论回写到 `AGENTS.md`。
- 实验参数、一次性路径、临时 checkpoint 和临时结论不进入这里。
