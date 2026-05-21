#!/bin/bash

# 遇到错误即停止运行
set -e

cd /data/heli/code/tele_Gen

echo "==========================================="
echo "1/4: 运行 Head sf 5 (headsoft5)"
echo "==========================================="
CKPT_DIR="/data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_headsoft_tailp_section7_anchor1_steps24/checkpoints"
OUT_DIR="/data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_headsoft_tailp_x0int8_probe_section7_anchor1_steps24/recover_outputs_x0int8sf1_headsoft5_tailp4"

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 reconstruct/recover.py infer \
  --checkpoint_dir "$CKPT_DIR" \
  --output_dir "$OUT_DIR" \
  --max_samples 4 \
  --steps 24 \
  --predict_only_steps 2 \
  --single_refresh_steps 8 \
  --dual_refresh_steps 12 \
  --global_keyframe_codec_type int8_refresh_like \
  --global_keyframe_spatial_factor 1 \
  --dual_head_codec_type soft_pool \
  --dual_head_soft_pool_spatial_factor 5 \
  --dual_tail_codec_type p_delta \
  --dual_tail_p_delta_spatial_factor 4

echo "==========================================="
echo "2/4: 运行 Tail sf 5 (tailp5)"
echo "==========================================="
CKPT_DIR="/data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_headsoft_tailp_section7_anchor1_steps24/checkpoints"
OUT_DIR="/data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_headsoft_tailp_x0int8_probe_section7_anchor1_steps24/recover_outputs_x0int8sf1_headsoft4_tailp5"

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 reconstruct/recover.py infer \
  --checkpoint_dir "$CKPT_DIR" \
  --output_dir "$OUT_DIR" \
  --max_samples 4 \
  --steps 24 \
  --predict_only_steps 2 \
  --single_refresh_steps 8 \
  --dual_refresh_steps 12 \
  --global_keyframe_codec_type int8_refresh_like \
  --global_keyframe_spatial_factor 1 \
  --dual_head_codec_type soft_pool \
  --dual_head_soft_pool_spatial_factor 4 \
  --dual_tail_codec_type p_delta \
  --dual_tail_p_delta_spatial_factor 5

echo "==========================================="
echo "3/4: 解视频 (headsoft5)"
echo "==========================================="
python /data/heli/code/tele_Gen/reconstruct/decoder.py -R /data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_headsoft_tailp_x0int8_probe_section7_anchor1_steps24/recover_outputs_x0int8sf1_headsoft5_tailp4

echo "==========================================="
echo "4/4: 解视频 (tailp5)"
echo "==========================================="
python /data/heli/code/tele_Gen/reconstruct/decoder.py -R /data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_headsoft_tailp_x0int8_probe_section7_anchor1_steps24/recover_outputs_x0int8sf1_headsoft4_tailp5

echo "==========================================="
echo "所有任务已执行完成！"
echo "==========================================="