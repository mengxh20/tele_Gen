# 入口、产物与落盘约定

## 关键入口
- `scripts/training/train_recover_ddp.sh`
  训练启动脚本，使用 Accelerate + DeepSpeed ZeRO-3 组织多卡训练。
- `reconstruct/recover.py train`
  压缩恢复训练主程序，围绕 clean latents 与 low latents 的差异训练 recover transformer。
- `reconstruct/encoder.py`
  把视频编码为 Helios VAE latents，作为压缩恢复链路的输入准备。
- `scripts/inference/infer_recover_ddp.sh`
  推理启动脚本，使用 `torchrun` 按样本分片执行恢复推理。
- `reconstruct/recover.py infer`
  压缩恢复推理主程序，输出 `low_latents/`、`recover_latents/` 和 `metrics/`。
- `reconstruct/real_decoder.py`
  接收端验证入口，优先走 `low_latents -> recover -> video` 的真实链路。
- `reconstruct/decoder.py`
  辅助解码工具，主要把 `recover_latents` 解码回视频用于展示和验收。
- `reconstruct/latent_io.py`
  latent payload 的读写、校验、shape 元信息、分块信息和 bpp 计算等通用能力。

## 训练阶段产物
- 一个可推理 checkpoint 目录默认包含：
- `transformer_full.pt`：recover transformer 权重。
- `recover_config.json`：推理依赖的结构化配置。
- `train_metrics.json`：训练过程指标。
- `learned_codec.pt`：仅在使用 learned tail codec 时出现。
- 管理型 checkpoint 目录还可能附带 `latest` 或 `latest_checkpoint.txt`，用于把“checkpoint 根目录”解析到最新可推理版本。

## 推理阶段产物
- `output_dir/low_latents/`
  低码率 latent 载荷，贴近压缩和传输环节。
- `output_dir/recover_latents/`
  恢复后的高保真 latent 结果。
- `output_dir/metrics/`
  每个样本的恢复指标、checkpoint 来源和推理关键配置。

## 解码与展示
- `reconstruct/decoder.py` 默认读取 `recover_latents/` 做视频导出。
- `reconstruct/real_decoder.py` 默认读取 `low_latents/`，内部先恢复成 `recover_latents`，再解码成视频。
- 视频目录属于验证和展示层，不应替代对 latent 产物和指标文件的管理。

## 快速任务映射
- 想改训练逻辑：优先看 `reconstruct/recover.py train` 和训练脚本。
- 想改压缩 / 编码格式：优先看 `reconstruct/encoder.py`、`reconstruct/latent_io.py` 和 `reconstruct/recover.py` 里的 codec 逻辑。
- 想改接收端恢复流程：优先看 `reconstruct/recover.py infer` 与 `reconstruct/real_decoder.py`。
- 想改展示或验收视频：优先看 `reconstruct/decoder.py`，但要确认没有偏离主链路。
