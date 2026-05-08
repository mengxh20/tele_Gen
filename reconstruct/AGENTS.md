# AGENTS.md

## 目录定位
- `reconstruct/` 是本项目压缩与恢复主线的默认落点。
- 这里承接低码率 latent 表达、恢复训练、恢复推理、接收端验证和结果落盘相关的新逻辑。
- 除非确有必要，不应把这里的业务逻辑回写到 `helios/`。

## 先读哪些文档
- 全局入口：[`../AGENTS.md`](../AGENTS.md)
- 主链路架构：[`../docs/01-business-goal-and-architecture.md`](../docs/01-business-goal-and-architecture.md)
- 入口与产物约定：[`../docs/02-entrypoints-and-artifacts.md`](../docs/02-entrypoints-and-artifacts.md)
- 改动边界与协作规则：[`../docs/03-change-boundaries-and-collaboration.md`](../docs/03-change-boundaries-and-collaboration.md)

## 本目录关键文件
- `encoder.py`：把视频编码为 Helios VAE latents。
- `recover.py`：压缩与恢复主入口，同时承担训练与推理。
- `real_decoder.py`：接收端链路入口，优先走 `low_latents -> recover -> video`。
- `decoder.py`：把 `recover_latents` 解码回视频，主要用于可视化、验证和验收。
- `latent_io.py`：latent 载荷读写、校验、shape / 分块元信息和 bpp 计算等通用能力。

## 产物约定
- 训练产物默认包括：
  - `transformer_full.pt`
  - `recover_config.json`
  - `train_metrics.json`
  - `checkpoints/latest`：始终指向最近一次完整可推理的 checkpoint，可在训练过程中直接用于恢复推理
- 推理输出目录默认包括：
  - `low_latents/`
  - `recover_latents/`
  - `metrics/`
- `entropy_metrics/` 默认保留，用于记录外层熵编后的运输码率统计
- `enc_latents/` 仅在显式开启 `--save_entropy_bin` 时作为外层运输码流落盘；默认不保存 `.bin`
- 当需要更贴近真实接收端链路、但又不想落盘 `.bin` 时，优先使用 `real_decoder.py` 从 `low_latents` 在内存中模拟运输码流输入，再进入 recover
- 若需要导出视频，应视为验证或展示环节，而不是替代上述主产物
- 当需要更贴近真实接收端链路时，优先使用 `real_decoder.py` 走 `low_latents -> recover -> video` 路线

## 与 Helios 的关系
- 本目录优先复用 Helios 已有能力，但当修改有助于释放模型能力时，可以直接修改 `helios/` 中的任何部分
- 当前稳定复用点主要包括：
  - stage1 条件拼装
  - transformer
  - scheduler
  - 空 prompt 编码
- 局部记忆只描述这些复用关系，不展开 Helios 内部通用生成细节

## 改动约束
- 新增压缩恢复逻辑优先继续放在 `reconstruct/`
- `helios/` 可以直接修改，当修改有助于释放模型能力、提升压缩恢复效果时，不需要特殊审批
- 默认优先做最小必要修改，避免无关重构

## 不固化到长期记忆的内容
- 不把以下内容写成长期固定规则：
  - `history_sizes`
  - `latent_window_size`
  - loss 权重
  - 训练轮次
  - 一次性实验路径
  - 临时 checkpoint 或临时输出结果
- 这些内容默认视为实验参数或阶段性实现细节，应以当前代码和配置为准
