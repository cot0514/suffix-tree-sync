#!/bin/bash
# 3050 추가 실험 3: 같은 반복 프롬프트를 b=32로 (검증 토큰 수를 500 아래로). 약 10~15분, 3회.
export VLLM_USE_FLASHINFER_SAMPLER=0
source ~/venv_vllm/bin/activate
cd ~/suffix-tree-sync
B=${1:-32}
MODEL=Qwen/Qwen2.5-1.5B
run() { # 태그 spec k
  local tag=$1 S=$2 K=$3 out=results/sweep3_${1}_b${B}.json
  echo "-- $tag"
  timeout 420 python vllm_path/gpu_measure.py --model $MODEL --spec $S --spec-tokens $K --prompts repeat \
     --batch $B --prompt-chars 500 --max-tokens 192 --gpu-util 0.85 --out $out 2>&1 | grep -E "경고|Error|Traceback" | head -3
  [ ${PIPESTATUS[0]} -eq 124 ] && echo "TIMEOUT: $tag"
}
run rep_none none 0
run rep_k8 suffix 8
run rep_k16 suffix 16
python - <<PY
import json,glob,os
print(f"{'tag':10}{'step_ms':>9}{'prop_ms':>8}{'G_ms':>8}{'draft/req':>10}{'tok/req':>8}{'tok/ms':>8}{'n':>4}")
for f in sorted(glob.glob("results/sweep3_*_b${B}.json")):
    r=json.load(open(f)); tag=os.path.basename(f)[7:-len("_b${B}.json")]
    tp=r['peak_running']*r['tokens_per_req_step']/r['step_ms']
    print(f"{tag:10}{r['step_ms']:>9.1f}{r['propose_ms']:>8.2f}{r['gpu_side_ms']:>8.1f}{r.get('drafted_per_req_step',0):>10.2f}{r['tokens_per_req_step']:>8.2f}{tp:>8.2f}{r['steps_steady']:>4}")
PY
