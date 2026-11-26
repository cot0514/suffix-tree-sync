#!/bin/bash
# GPU 실측 실행기. 사용: [PC=프롬프트글자수] [MT=최대토큰] bash vllm_path/gpu_run.sh [모델] [배치]
# 먼저 nvidia-smi로 GPU가 비어 있는지 확인할 것.
set -e
export VLLM_USE_FLASHINFER_SAMPLER=0  # nvcc 없어서 flashinfer JIT 불가
source ~/venv_vllm/bin/activate
cd ~/suffix-tree-sync
MODEL=${1:-Qwen/Qwen2.5-1.5B}
B=${2:-64}
PC=${PC:-2000}
MT=${MT:-192}
mkdir -p results
for spec in none suffix; do
  python vllm_path/gpu_measure.py --model "$MODEL" --spec $spec --batch $B \
     --prompt-chars $PC --max-tokens $MT \
     --out results/gpu_${spec}_b${B}_pc${PC}.json 2>&1 | grep -v -E "^W1115|deprecated|INFO" | tail -20
done
