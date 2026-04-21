# AGENTS.md

- Act like a pragmatic coding agent: concise, direct, and focused on getting the task done end-to-end.
- For coding work, inspect the codebase first, avoid assumptions, and prefer reasonable execution over stopping to ask unless the ambiguity is genuinely risky.
- Keep changes minimal and surgical: match existing style, avoid unrelated refactors, and never revert or overwrite user changes unless explicitly asked.
- Prefer simplicity over abstraction. No speculative features, unnecessary configurability, or cleanup outside the requested scope.
- Before implementing, surface assumptions and tradeoffs when they matter. For multi-step work, define a short plan with verification steps.
- Verify results where feasible, typically with tests or direct checks, and report any gaps if verification is incomplete.
- Prefer rg/rg --files for search, and use apply_patch for manual file edits.
- Provide short progress updates while working, then give a concise final summary.
- If asked for a code review, prioritize findings, risks, regressions, and missing tests, with file/line references first.
- Respect environment constraints: workspace-write filesystem access, restricted network, no destructive commands without explicit approval.
- Follow project-specific guidance from AGENTS.md / CLAUDE.md: think before coding, favor the minimum code that solves the problem, make only necessary edits, and define success in verifiable terms.

## 项目定位
- 项目名称：高效视频传输、压缩与恢复
- 本仓库是基于开源视频生成框架 Helios 进行二次开发的“生成式视频压缩与恢复系统”
- 项目的第一目标不是泛化地增强 t2v / i2v / v2v 通用生成效果，而是让 Helios 的 latent 建模与生成能力服务于低码率压缩、传输与高质量恢复
- 目标价值是利用生成模型的 latent 表达能力，实现更低码率的视频表示、更贴近真实传输场景的恢复链路，以及更高质量的细节与时序重建

## 当前第一优先级
- 优先服务“生成式视频压缩与恢复”业务，而不是通用视频生成能力扩展
- 所有训练、推理、数据处理、评测和工程改动，都应优先回答以下问题：
  - 是否有助于低码率压缩与恢复
  - 是否有助于提升重建质量、清晰度、细节恢复、时序一致性或码率效率
  - 是否更贴近真实的压缩-传输-恢复链路
  - 是否能在尽量少改 Helios 源码的前提下复用其现有能力
- 若某项改动主要提升通用生成效果，但不能明确帮助压缩与恢复主线，则不应作为优先项

## 业务主线理解
- 项目真实主线默认围绕 Helios VAE latents 展开，而不是直接围绕最终视频生成结果展开
- 当前稳定主链路是：
  - 输入 clean latents
  - 进行低码率 latent codec 处理
  - 复用 Helios stage1 recover 能力进行条件恢复
  - 输出 `low_latents`、`recover_latents` 和 `metrics`
- 视频解码与导出主要用于验证、可视化和验收，不应默认被视为压缩恢复主链路本身
- 当“通用生成效果优化”和“压缩恢复目标”冲突时，应优先选择压缩恢复目标

## 源码事实优先级
- 涉及项目真实主线、入口、产物和约束时，以 `reconstruct/` 当前实现及其训练/推理入口为第一依据
- 不应仅依据仓库 `README.md` 中对上游 Helios 通用生成能力的描述来判断本项目主线
- 若 README 叙述与 `reconstruct/` 主链路实现存在表述偏差，应明确指出并以当前压缩恢复链路实现为准

## 核心原则
- `helios/` 的主要角色是能力底座，不是默认开发主战场
- `reconstruct/` 是压缩与恢复相关新增逻辑的默认落点
- 优先通过封装、调用、适配、组合 Helios 现有能力来完成需求，而不是直接改写 Helios 源码
- 默认优先做最小必要修改，避免无关重构
- 如果需求或当前实现偏离“压缩与恢复主线”，需要明确指出

## 关键入口
- 训练启动脚本：`scripts/training/train_recover_ddp.sh`
- 训练主程序：`reconstruct/recover.py`
- 压缩入口：`reconstruct/encoder.py`
- 恢复入口：`reconstruct/recover.py`
- 推理启动脚本：`scripts/inference/infer_recover_ddp.sh`
- 辅助解码工具：`reconstruct/decoder.py` 保留，但不是压缩恢复主线入口
- `reconstruct/AGENTS.md` 记录该目录下更具体的主链路约定与产物约定

## 目录职责理解
- `reconstruct/`：压缩、latent 表达、恢复、实验验证、可视化，以及与压缩恢复链路直接相关的新逻辑
- `helios/`：Helios 的核心模型、pipeline、scheduler、dataset、utils 等底层实现
- `scripts/training/`：训练脚本与配置；压缩恢复主线训练入口为 `train_recover_ddp.sh`
- `scripts/inference/`：推理脚本；压缩恢复主线入口为 `infer_recover_ddp.sh`，其余脚本主要服务 t2v / i2v / v2v 等通用生成能力

## 改动边界
- 新增代码默认优先放在 `reconstruct/`
- 非必要尽量不要修改 `helios/` 源码
- 只有在以下条件同时满足时，才考虑修改 `helios/`：
  - `reconstruct/` 中无法通过外部封装或适配完成需求
  - 修改会直接帮助压缩与恢复主线
  - 已说明为何替代方案不可行
- 一旦需要修改 `helios/`，必须先说明：
  - 为什么必须改
  - 影响范围是什么
  - 为什么不能把逻辑放在 `reconstruct/`
  - 对压缩与恢复链路的直接收益是什么
- 如果上述判断存在不确定性，先暂停并与用户对齐，再继续执行

## 评估与表达导向
- 训练、推理、评测和方案说明，默认优先围绕以下维度展开：
  - 码率效率
  - 恢复误差
  - 时序连续性
  - 训练稳定性
  - 压缩-传输-恢复链路真实性
- 若提出训练、推理、数据或评测改动，应优先说明其对码率效率、恢复质量、时序一致性或训练稳定性的帮助
- 所有建议都应尽量先映射到压缩与恢复主线，而不是停留在“生成效果更好”的泛化表述

## 协作要求
- 每次接到需求时，先解释方案，再改代码
- 若发现需求存在多种实现路径，应先说明权衡，再等待确认选择哪条实现路径
- 若我对需求理解、链路目标、改动边界或 Helios 修改必要性存在不确定，应立即暂停并与你对齐，再执行下一步
- 输出时必须标注：
  - 影响文件
  - 回滚方式，并给我一个按钮点击即可回滚到之前版本，如果需要 git 请你帮我完成
- 修改说明应尽量清楚指出：
  - 为什么改
  - 改了什么
  - 对压缩与恢复链路有什么帮助
- 如果未实际改代码，应明确说明“当前仅给出方案，尚未落盘修改”

## 记忆更新规则
- 每次对话后，可根据新的项目理解更新当前 `AGENTS.md`
- 如有必要，可在相关子目录新增局部 `AGENTS.md`，用于记录该目录的职责、约束和注意事项
- 记忆内容应以“长期有效、可复用、能减少重复沟通”为标准
- 不写临时噪音信息，不记录一次性无价值细节
- `reconstruct/AGENTS.md` 负责补充该目录下的压缩恢复主链路约定；后续若 `helios/` 持续出现局部协作成本，再考虑单独新增该目录下的局部 `AGENTS.md`

## 输出偏好
- 方案说明应优先围绕业务目标、实现思路、影响范围、风险点展开
- 所有建议都应尽量先映射到压缩与恢复主线，而不是停留在通用生成优化表述
- 若提出训练、推理、数据或评测改动，应优先说明其对码率效率、恢复质量、时序一致性或训练稳定性的帮助
