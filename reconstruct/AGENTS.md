# AGENTS.md

## 目录定位
- `reconstruct/` 是本项目压缩与恢复主线的默认落点
- 这里承接与生成式视频压缩、低码率 latent 表达、恢复训练、恢复推理、实验验证和结果落盘直接相关的新逻辑
- 除非确有必要，不应把该目录下的业务逻辑回写到 `helios/`

## 主链路职责
- `encoder.py`：负责把视频编码为 Helios VAE latents，作为后续压缩与恢复链路的输入准备
- `recover.py`：压缩与恢复主入口，同时承担训练与推理
- `decoder.py`：把 latent 解码回视频，主要用于可视化、验证和验收，不是压缩恢复主线入口
- `latent_io.py`：承接 latent 载荷读写、格式校验、shape 与分块元信息处理等通用能力

## 稳定业务链路
- 当前稳定链路默认是：
  - 读取 clean latents
  - 保留首帧 keyframe
  - 对 remainder 做时空下采样与量化，形成低码率 latent 表示
  - 得到 low latents baseline
  - 按 section / window 组织历史条件
  - 复用 Helios stage1 transformer 与 scheduler 做恢复
  - 落盘 `low_latents`、`recover_latents` 与 `metrics`
- 推理默认不以直接生成视频为目标，而是优先验证低码率压缩与恢复链路本身

## 产物约定
- 训练产物默认包括：
  - `transformer_full.pt`
  - `recover_config.json`
  - `train_metrics.json`
- 推理输出目录默认包括：
  - `low_latents/`
  - `recover_latents/`
  - `metrics/`
- 若需要导出视频，应视为验证或展示环节，而不是替代上述主产物

## 与 Helios 的关系
- 本目录优先复用 Helios 已有能力，而不是重写其底层逻辑
- 当前稳定复用点主要包括：
  - stage1 条件拼装
  - transformer
  - scheduler
  - 空 prompt 编码
- 局部记忆只描述这些复用关系，不展开 Helios 内部通用生成细节

## 改动约束
- 新增压缩恢复逻辑优先继续放在 `reconstruct/`
- 若需求主要是服务低码率压缩、恢复质量、时序一致性或链路真实性，应先考虑在该目录内通过外封装、适配和组合实现
- 若判断必须修改 `helios/`，需要先说明：
  - 为什么 `reconstruct/` 内无法完成
  - 影响哪些 Helios 模块
  - 对压缩恢复主线的直接收益是什么

## 不固化到长期记忆的内容
- 不把以下内容写成长期固定规则：
  - `history_sizes`
  - `latent_window_size`
  - loss 权重
  - 训练轮次
  - 一次性实验路径
  - 临时 checkpoint 或临时输出结果
- 这些内容默认视为实验参数或阶段性实现细节，应以当前代码和配置为准
