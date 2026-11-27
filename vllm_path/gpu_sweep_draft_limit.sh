#!/bin/bash
# 3050 추가 실험: (A) 추측 길이를 실제로 늘려 스텝 시간과의 관계 확인, (B) 높은 적중률 프롬프트.
# 사용: bash vllm_path/gpu_sweep_draft_limit.sh [배치=64]   (약 25분, 7회)
export VLLM_USE_FLASHINFER_SAMPLER=0
source ~/venv_vllm/bin/activate
cd ~/suffix-tree-sync
B=${1:-64}
MODEL=Qwen/Qwen2.5-1.5B
mkdir -p results
run() { # 태그 프롬프트 글자수 spec k 추가옵션...
  local tag=$1 P=$2 PC=$3 S=$4 K=$5; shift 5
  local out=results/draft_limit_${tag}_b${B}.json
  echo "-- $tag"
  timeout 600 python vllm_path/gpu_measure.py --model $MODEL --spec $S --spec-tokens $K --prompts $P \
     --batch $B --prompt-chars $PC --max-tokens 192 --gpu-util 0.85 --out $out "$@" 2>&1 \
     | grep -E "경고|Error|Traceback" | head -3
  [ ${PIPESTATUS[0]} -eq 124 ] && echo "TIMEOUT: $tag"
}
# A) code 프롬프트, 추측 길이 제한 풀기 (min_token_prob=0, max_spec_factor=4)
run code_none code 2000 none 0
for K in 4 8 16; do run code_relaxed_k$K code 2000 suffix $K --min-token-prob 0 --max-spec-factor 4; done
# B) 반복 프롬프트(500자), 기본 설정
run rep_none repeat 500 none 0
for K in 8 16; do run rep_default_k$K repeat 500 suffix $K; done
python - <<PY
import json,glob,os
print(f"{'tag':22}{'step_ms':>9}{'prop_ms':>8}{'G_ms':>8}{'draft/req':>10}{'tok/req':>8}{'tok/ms':>8}{'n':>4}")
for f in sorted(glob.glob("results/draft_limit_*_b${B}.json")):
    r=json.load(open(f)); tag=os.path.basename(f)[len("draft_limit_"):-len("_b${B}.json")]
    tp=r['peak_running']*r['tokens_per_req_step']/r['step_ms']
    print(f"{tag:22}{r['step_ms']:>9.1f}{r['propose_ms']:>8.2f}{r['gpu_side_ms']:>8.1f}{r.get('drafted_per_req_step',0):>10.2f}{r['tokens_per_req_step']:>8.2f}{tp:>8.2f}{r['steps_steady']:>4}")
PY
