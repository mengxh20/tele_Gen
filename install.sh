#!/bin/bash

pip install -r requirements.txt

rm -rf ~/.triton/cache/
rm -rf /tmp/torchinductor_*

pip uninstall triton torchao xformers wandb tensorflow tensorflow-cpu -y 
pip install wandb==0.23.0 triton==3.3.1 -i http://172.31.0.9:8800/root/pypi/ --trusted-host 172.31.0.9 --timeout 1000

rm -rf ~/.triton/cache/
rm -rf /tmp/torchinductor_*