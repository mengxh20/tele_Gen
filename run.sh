#!/bin/bash
export SAVEDIR=reconstruct/gen2recon_runs_scale_Mar06
bash scripts/training/train_recover_ddp.sh
# bash scripts/inference/infer_recover_ddp.sh