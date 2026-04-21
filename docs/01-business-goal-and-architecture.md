# 业务目标与架构

## 项目目标
- 本仓库是基于 Helios 二次开发的“生成式视频压缩与恢复系统”。
- 第一优先级不是增强通用视频生成能力，而是让 Helios 的 latent 建模能力服务低码率压缩、传输与高质量恢复。
- 默认评价标准是：码率效率、恢复误差、细节恢复、时序一致性、训练稳定性，以及链路是否贴近真实传输场景。

## 主链路
- 当前稳定业务链路围绕 Helios VAE latents 展开：

```text
clean latents
  -> 低码率 latent codec
  -> low_latents
  -> stage1 recover（结合首帧和历史条件）
  -> recover_latents
  -> metrics
  -> video export（可选，仅验证/展示）
```

- 在这条链路里，`low_latents` 表示更贴近传输端的低码率载荷，`recover_latents` 表示接收端恢复后的高保真 latent。
- 视频解码不是默认主产物，而是主链路验证完成后的附加步骤。

## 模块分工
- `reconstruct/`：压缩、latent 表达、恢复训练、恢复推理、实验验证、可视化，以及与压缩恢复直接相关的新逻辑。
- `helios/`：底层模型、pipeline、scheduler、dataset、utils 等能力底座。
- `scripts/training/`：训练启动脚本和分布式启动配置。
- `scripts/inference/`：推理启动脚本，以及接收端验证链路的组织入口。

## 源码事实优先级
- 涉及真实主线、入口、产物和约束时，以 `reconstruct/` 当前实现为第一依据。
- 根 `README.md` 主要承接上游 Helios 的通用生成说明，不能单独当作本项目主线定义。
- 如果 README 与当前压缩恢复链路实现有偏差，应在沟通中明确指出，并以当前代码为准。

## 非主线内容
- 仅提升通用生成效果、但无法映射到压缩恢复收益的改动，不是默认优先项。
- 直接把“导出视频更好看”当作唯一目标，也不应替代对 `low_latents`、`recover_latents` 和 `metrics` 的关注。
