#!/bin/bash

set -x

conda create -n verl python==3.12 -y
conda activate verl

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

pip install --no-deps -e .
pip install math-verify