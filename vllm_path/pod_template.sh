#!/bin/bash
# RunPod용 GPU 실측 한 줄 실행기 (pod_run.sh는 make_pod_script.py로 생성, 직접 고치지 말 것)
# 사용: bash pod_run.sh quick   # 1.5B, b=64 한 점 (설치 확인용, 약 15~25분)
#       bash pod_run.sh full    # 1.5B + 7B, b=64/128/256 (약 1~1.5시간)
# 끝나면 /workspace/results.tar.gz 를 받아 가고, 파드는 반드시 삭제(Terminate)할 것.
MODE=${1:-quick}
W=/workspace/st_gpu
mkdir -p $W/results && cd $W
exec > >(tee -a $W/run.log) 2>&1
echo "== $(date) mode=$MODE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi | grep -i "CUDA Version"
echo "python: $(python3 --version)  nvcc: $(command -v nvcc || echo 없음)  CPU: $(nproc)개"

# 0) 도구 (build-essential 없으면 arctic-inference 빌드 실패)
if ! command -v g++ >/dev/null; then apt-get update -y && apt-get install -y build-essential; fi

# 1) 환경 (이미 있으면 건너뜀)
if [ ! -f $W/venv/bin/activate ]; then
  python3 -m venv $W/venv || { echo "venv 실패: apt-get install -y python3-venv 후 재시도"; exit 1; }
  source $W/venv/bin/activate
  pip install -U pip
  pip install vllm && pip install "arctic-inference[vllm]" || { echo "설치 실패 -> 위 로그 확인"; exit 1; }
else
  source $W/venv/bin/activate
fi
export HF_HOME=/workspace/hf
python -c "import vllm,torch; print('vllm',vllm.__version__,'torch',torch.__version__,'cuda_ok',torch.cuda.is_available())" || exit 1
command -v nvcc >/dev/null || export VLLM_USE_FLASHINFER_SAMPLER=0

# 2) 측정 스크립트 (로컬 vllm_path/gpu_measure.py와 같음)
cat > $W/gpu_measure.py <<'PYEOF'
@@GPU_MEASURE@@
PYEOF

# 3) 실행
run() { # 모델 배치 [프롬프트글자]
  local M=$1 B=$2 PC=${3:-2000} tag
  tag=$(echo $M | tr '/' '_')
  for spec in none suffix; do
    echo "-- $M b=$B pc=$PC $spec"
    timeout 900 python $W/gpu_measure.py --model $M --spec $spec --batch $B \
      --prompt-chars $PC --max-tokens 256 --gpu-util 0.85 --max-len 2048 \
      --out $W/results/${tag}_${spec}_b${B}_pc${PC}.json 2>&1 | grep -v -E "^W[0-9]{4}|deprecated|INFO|Capturing|Loading|JIT" | tail -20
    [ ${PIPESTATUS[0]} -eq 124 ] && echo "TIMEOUT(900s): $M b=$B $spec"
  done
}
if [ "$MODE" = quick ]; then
  run Qwen/Qwen2.5-1.5B 64
else
  for B in 64 128 256; do run Qwen/Qwen2.5-1.5B $B; done
  for B in 64 128 256; do run Qwen/Qwen2.5-7B $B; done
fi

# 4) 요약표
python - <<'PY'
import json,glob,os
rows=[json.load(open(f))|{"f":os.path.basename(f)} for f in sorted(glob.glob("/workspace/st_gpu/results/*_b*_pc*.json")) if "_rows" not in f]
print(f"{'model':22}{'spec':8}{'b':>5}{'peak':>6}{'step_ms':>10}{'prop_ms':>9}{'G_ms':>9}{'tok/req':>9}{'tok/ms':>8}")
for r in rows:
    tp=r['batch']*r['tokens_per_req_step']/r['step_ms']
    print(f"{r['model'][-20:]:22}{r['spec']:8}{r['batch']:>5}{r['peak_running']:>6}{r['step_ms']:>10.1f}{r['propose_ms']:>9.2f}{r['gpu_side_ms']:>9.1f}{r['tokens_per_req_step']:>9.2f}{tp:>8.2f}")
PY
tar czf /workspace/results.tar.gz -C $W results run.log
echo "== 끝. /workspace/results.tar.gz 를 받아 가고 파드를 삭제하세요."
