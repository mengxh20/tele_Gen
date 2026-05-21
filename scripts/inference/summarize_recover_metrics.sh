#!/usr/bin/env bash
set -euo pipefail

cd /data/heli/code/tele_Gen

PYTHON_BIN="${PYTHON_BIN:-/data/heli/miniconda3/envs/teleai/bin/python}"

DEFAULT_DIRS=(
  "/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux10k_section7_anchor1_steps24/recover_outputs_videoSRC01_04_nochunk149_step10000_x0int8sf1_headsoft5_tailp4"
  "/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux_continue20k_from_step10000_section7_anchor1_steps24/recover_outputs_videoSRC01_04_nochunk149_step11000_continue_from_step10000_x0int8sf1_headsoft5_tailp4"
  "/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux_continue20k_from_step10000_section7_anchor1_steps24/recover_outputs_videoSRC01_04_nochunk149_step20000_cumulative30000_x0int8sf1_headsoft5_tailp4"
  "/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux_continue20k_from_step30000_section7_anchor1_steps24/recover_outputs_videoSRC01_04_nochunk149_step10000_from_step30000_x0int8sf1_headsoft5_tailp4"
  "/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux_continue20k_from_step30000_section7_anchor1_steps24/recover_outputs_videoSRC01_04_nochunk149_step20000_cumulative50000_x0int8sf1_headsoft5_tailp4"
)

if [[ "$#" -gt 0 ]]; then
  METRIC_DIRS=("$@")
else
  METRIC_DIRS=("${DEFAULT_DIRS[@]}")
fi

"${PYTHON_BIN}" - "${METRIC_DIRS[@]}" <<'PY'
import json
import sys
from pathlib import Path
from statistics import mean


def first_section_events(events):
    by_section = {}
    for event in events:
        idx = event.get("section_index")
        if idx not in by_section:
            by_section[idx] = event
    return by_section


def average(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def fmt(value, digits=6):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


for output_dir_arg in sys.argv[1:]:
    output_dir = Path(output_dir_arg)
    metrics_dir = output_dir / "metrics"
    metric_paths = sorted(metrics_dir.glob("*.json"))
    if not metric_paths:
        print(f"\n[WARN] no metrics json found: {metrics_dir}")
        continue

    rows = []
    for path in metric_paths:
        payload = json.loads(path.read_text())
        decoded_metrics = payload.get("decoded_metrics") or {}
        section_events = first_section_events(payload.get("streaming_frame_group_ready_events") or [])
        section0 = section_events.get(0, {})
        section1 = section_events.get(1, {})
        rows.append(
            {
                "file": path.stem,
                "low_bpp": payload.get("low_bpp"),
                "psnr": decoded_metrics.get("rgb_psnr", decoded_metrics.get("psnr")),
                "lpips": decoded_metrics.get("lpips"),
                "section0_time": section0.get("section_elapsed_seconds"),
                "section1_time": section1.get("section_elapsed_seconds"),
                "section0_mode": section0.get("section_mode"),
                "section1_mode": section1.get("section_mode"),
            }
        )

    print(f"\n{output_dir}")
    print(f"samples: {len(rows)}")
    print(f"avg low_bpp: {fmt(average(rows, 'low_bpp'), 9)}")
    print(f"avg PSNR: {fmt(average(rows, 'psnr'), 6)}")
    print(f"avg LPIPS: {fmt(average(rows, 'lpips'), 6)}")
    print(f"first section time: {fmt(average(rows, 'section0_time'), 6)} s")
    print(f"second section time: {fmt(average(rows, 'section1_time'), 6)} s")
    print("per sample:")
    for row in rows:
        print(
            "  {file}: low_bpp={low_bpp} psnr={psnr} lpips={lpips} "
            "s0={s0}s({s0_mode}) s1={s1}s({s1_mode})".format(
                file=row["file"],
                low_bpp=fmt(row["low_bpp"], 6),
                psnr=fmt(row["psnr"], 4),
                lpips=fmt(row["lpips"], 6),
                s0=fmt(row["section0_time"], 4),
                s0_mode=row.get("section0_mode") or "NA",
                s1=fmt(row["section1_time"], 4),
                s1_mode=row.get("section1_mode") or "NA",
            )
        )
PY
