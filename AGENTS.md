# AGENTS.md

## 入口定位
- 这份文档只保留长期有效的导航和硬约束，细节请按需加载 `docs/`。
- 项目第一目标是“生成式视频压缩与恢复”，不是泛化地增强 t2v / i2v / v2v 通用生成效果。
- 业务主线默认围绕 Helios VAE latents，而不是围绕最终视频生成结果。
- 当前稳定链路是：`clean latents -> 低码率 latent codec -> Helios stage1 recover -> low_latents / recover_latents / metrics`。
- 视频解码与导出主要用于验证、可视化和验收，不应默认视为压缩恢复主链路本身。

## 决策优先级
- 先判断改动是否直接帮助低码率压缩、恢复质量、时序一致性、训练稳定性或链路真实性。
- 新增逻辑默认优先放在 `reconstruct/`，`helios/` 的角色是能力底座，不是默认开发主战场。
- 涉及主线、入口、产物和约束时，以 `reconstruct/` 当前实现和训练 / 推理入口为第一依据。
- 若根 `README.md` 与当前压缩恢复实现有表述偏差，应明确指出并以主链路代码为准。

## 文档加载规则
- 把 `AGENTS.md` 当作地图，不要把它写成百科。
- 只有在当前任务真的需要时，再继续加载对应深层文档。
- 若深层文档与当前代码不一致，应先以代码和入口脚本为准，再补正文档。

## 文档索引
- 总索引：[`docs/README.md`](docs/README.md)
- 业务目标与架构：[`docs/01-business-goal-and-architecture.md`](docs/01-business-goal-and-architecture.md)
- 入口、产物与落盘约定：[`docs/02-entrypoints-and-artifacts.md`](docs/02-entrypoints-and-artifacts.md)
- 改动边界与协作规则：[`docs/03-change-boundaries-and-collaboration.md`](docs/03-change-boundaries-and-collaboration.md)
- 方案质量检查：[`docs/04-quality-checklist.md`](docs/04-quality-checklist.md)
- `reconstruct/` 局部约定：[`reconstruct/AGENTS.md`](reconstruct/AGENTS.md)

## 关键入口
- 训练启动脚本：`scripts/training/train_recover_ddp.sh`
- 训练 / 恢复主程序：`reconstruct/recover.py`
- 压缩入口：`reconstruct/encoder.py`
- 推理启动脚本：`scripts/inference/infer_recover_ddp.sh`
- 接收端链路：`reconstruct/real_decoder.py`
- 辅助解码工具：`reconstruct/decoder.py`

## 改动边界
- 非必要不要修改 `helios/`。
- 只有在以下条件同时满足时，才考虑修改 `helios/`：
- `reconstruct/` 中无法通过封装、适配或组合完成需求。
- 修改会直接帮助压缩与恢复主线。
- 已说明替代方案为何不可行，以及影响范围是什么。
- 如果这一判断不确定，先暂停并与用户对齐。

## 协作要求
- 每次接到需求时，先解释方案，再改代码或文档。
- 若存在多种实现路径，应先说明权衡，再等待确认。
- 输出时必须标注影响文件、回滚方式，以及“为什么改 / 改了什么 / 对主链路有什么帮助”。
- 如果未实际改动文件，应明确说明“当前仅给出方案，尚未落盘修改”。
- 新的长期稳定认知可以写回 `AGENTS.md` 或 `docs/`；一次性实验细节不要写进长期记忆。
