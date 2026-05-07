#!/bin/bash
export SAVEDIR=reconstruct/gen2recon_runs_HE2E_Mar06
bash scripts/training/train_recover_ddp.sh
bash scripts/inference/infer_recover_ddp.sh