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

## 本目录稳定约定
- 当前稳定链路默认是：`clean latents -> low_latents -> recover_latents -> metrics`。
- 若需要导出视频，应视为验证或展示环节，而不是替代上述主产物。
- 修改这里的逻辑时，优先说明它如何帮助低码率压缩、恢复质量、时序一致性、训练稳定性或链路真实性。

## 不写进长期记忆的内容
- `history_sizes`
- `latent_window_size`
- loss 权重
- 训练轮次
- 一次性实验路径
- 临时 checkpoint 或临时输出结果

- 这些内容默认以当前代码、配置和实验记录为准，不固化到 `AGENTS.md`。
