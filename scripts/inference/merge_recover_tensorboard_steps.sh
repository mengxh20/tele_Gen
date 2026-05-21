#!/usr/bin/env bash
set -euo pipefail

cd /data/heli/code/tele_Gen

PYTHON_BIN="${PYTHON_BIN:-/data/heli/miniconda3/envs/teleai/bin/python}"

OUTPUT_LOGDIR="${OUTPUT_LOGDIR:-/data/heli/code/tele_Gen/reconstruct/experiments2/tensorboard_merged_base_safeaux_cumulative_0_60000_continuous}"

RUN_SPECS=(
  "1w|0|/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux10k_section7_anchor1_steps24/tensorboard"
  "3w_continue|10000|/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux_continue20k_from_step10000_section7_anchor1_steps24/tensorboard"
  "5w_continue|30000|/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux_continue20k_from_step30000_section7_anchor1_steps24/tensorboard"
  "6w_continue|50000|/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux_continue10k_from_step50000_section7_anchor1_steps24/tensorboard"
)

echo "[INFO] output_logdir=${OUTPUT_LOGDIR}"

"${PYTHON_BIN}" - "${OUTPUT_LOGDIR}" "${RUN_SPECS[@]}" <<'PY'
import shutil
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


output_logdir = Path(sys.argv[1])
run_specs = sys.argv[2:]
KEY_TAGS = {
    "train/loss",
    "train/raw_loss",
    "train/flow_loss",
    "train/x_loss",
    "train/noise_loss",
    "train/temporal_delta_loss",
    "train/aux_scale",
    "train/loss_ema",
    "train/sigma_mean",
    "train/flow_target_rms",
    "train/flow_pred_rms",
    "train/x_target_rms",
    "train/x_pred_rms",
    "train/noise_target_rms",
    "train/noise_pred_rms",
}

if output_logdir.exists():
    shutil.rmtree(output_logdir)
output_logdir.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(str(output_logdir / "continuous"))
total_scalars = 0
for spec in run_specs:
    label, offset_text, logdir_text = spec.split("|", 2)
    offset = int(offset_text)
    logdir = Path(logdir_text)
    if not logdir.exists():
        raise FileNotFoundError(f"TensorBoard logdir not found for {label}: {logdir}")

    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalar_tags = [tag for tag in accumulator.Tags().get("scalars", []) if tag in KEY_TAGS]
    copied = 0
    for tag in scalar_tags:
        for event in accumulator.Scalars(tag):
            merged_step = int(event.step) + offset
            writer.add_scalar(tag, event.value, merged_step, walltime=float(merged_step))
            copied += 1

    writer.flush()
    total_scalars += copied
    print(f"[merge] {label}: offset={offset} tags={len(scalar_tags)} scalars={copied} source={logdir}", flush=True)

writer.close()
print(f"[merge] done output={output_logdir} total_scalars={total_scalars}", flush=True)
PY

echo "[INFO] TensorBoard command:"
echo "tensorboard --logdir ${OUTPUT_LOGDIR} --host 0.0.0.0 --port 6006"
