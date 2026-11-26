#!/bin/bash
# 3050 로컬 추가 실험: (1) 추측 토큰 수 스윕, (2) 적중률 높은 반복 프롬프트.
# 사용: bash vllm_path/gpu_sweep.sh [배치=64]     (GPU가 비어 있을 때. 약 25~35분, 8회)
# 보는 것: suffix 스텝 시간이 추측 토큰 수(k)에 따라 늘면 -> 검증 연산 낭비 가설,
#          거의 그대로면 -> 구현 오버헤드 쪽. 반복 프롬프트에서 채택 토큰이 늘면 suffix 이득 여부 확인.
export VLLM_USE_FLASHINFER_SAMPLER=0   # nvcc 없음
source ~/venv_vllm/bin/activate
cd ~/suffix-tree-sync
B=${1:-64}
MODEL=Qwen/Qwen2.5-1.5B
mkdir -p results
run() { # 프롬프트종류 spec k
  local P=$1 S=$2 K=$3 out=results/sweep_${1}_${2}_k${3}_b${B}.json
  echo "-- $P $S k=$K b=$B"
  timeout 600 python vllm_path/gpu_measure.py --model $MODEL --spec $S --spec-tokens $K --prompts $P \
     --batch $B --prompt-chars 2000 --max-tokens 192 --gpu-util 0.85 --out $out 2>&1 \
     | grep -E "경고|Error|Traceback" | head -3
  [ ${PIPESTATUS[0]} -eq 124 ] && echo "TIMEOUT: $P $S k=$K"
}
for P in code repeat; do
  run $P none 0
  for K in 4 8 16; do run $P suffix $K; done
done
python - <<PY
import json,glob,os
rows=[]
for f in sorted(glob.glob("results/sweep_*_b${B}.json")):
    r=json.load(open(f)); rows.append(r)
print(f"{'prompts':8}{'spec':8}{'k':>3}{'step_ms':>9}{'prop_ms':>8}{'G_ms':>8}{'draft/req':>10}{'tok/req':>8}{'tok/ms':>8}{'n':>4}")
for r in rows:
    tp=r['peak_running']*r['tokens_per_req_step']/r['step_ms']
    print(f"{r.get('prompts','?'):8}{r['spec']:8}{r.get('spec_tokens',0) if r['spec']!='none' else 0:>3}{r['step_ms']:>9.1f}{r['propose_ms']:>8.2f}{r['gpu_side_ms']:>8.1f}{r.get('drafted_per_req_step',0):>10.2f}{r['tokens_per_req_step']:>8.2f}{tp:>8.2f}{r['steps_steady']:>4}")
PY
