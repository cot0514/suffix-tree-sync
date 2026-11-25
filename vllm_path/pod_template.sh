#!/bin/bash
# RunPod용 GPU 실측 한 줄 실행기 (pod_run.sh는 make_pod_script.py로 생성, 직접 고치지 말 것)
# 사용: bash pod_run.sh quick   # 1.5B, b=64 한 점 (설치 확인용)
#       bash pod_run.sh full    # 1.5B + 7B, b=64/128/256 (빈 트리, 파이썬 소스 프롬프트)
#       bash pod_run.sh tree    # 전역 트리를 채운 상태의 GPU 측정 (일반 / suffix 빈 트리 / suffix 10000개)
#       bash pod_run.sh cpu     # 파드 CPU 에서 요청별 호출과 묶음 호출 비교 (방식들이 합쳐진 뒤)
#       bash pod_run.sh batch   # vLLM 의 propose 를 묶음 호출로 바꿔 끼워 원본과 비교 (vllm_path 가 합쳐진 뒤)
# 바꿀 수 있는 값(환경 변수): POINTS REPEAT MAX_TOKENS CACHED / CPU_BATCHES CPU_CACHED BRANCH
#                             / THREADS BATCH_MODE (batch 모드)
# 끝나면 /workspace/results.tar.gz 를 받아 가고, 파드는 반드시 삭제(Terminate)할 것.
MODE=${1:-quick}
W=${WORK_DIR:-/workspace/st_gpu}
VLLM_VERSION=0.33.0
ARCTIC_VERSION=0.3.0
REPO_URL=https://github.com/scj7565/suffix-tree-sync.git
BRANCH=${BRANCH:-main}
case $MODE in quick|full|tree|cpu|batch) ;; *) echo "모드는 quick, full, tree, cpu, batch 중 하나"; exit 1 ;; esac
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
  pip install "vllm==$VLLM_VERSION" && pip install "arctic-inference[vllm]==$ARCTIC_VERSION" \
    || { echo "설치 실패 -> 위 로그 확인"; rm -rf $W/venv; exit 1; }
else
  source $W/venv/bin/activate
fi
export HF_HOME=${HF_HOME:-/workspace/hf}
python -c "import vllm,torch; print('vllm',vllm.__version__,'torch',torch.__version__,'cuda_ok',torch.cuda.is_available())" || exit 1
command -v nvcc >/dev/null || export VLLM_USE_FLASHINFER_SAMPLER=0

# 2) 측정 스크립트 (로컬 vllm_path/gpu_measure.py와 같음)
cat > $W/gpu_measure.py <<'PYEOF'
@@GPU_MEASURE@@
PYEOF

# 저장소 받기: 토큰 데이터(data/)와 cpu 모드의 소스. 이미 있으면 그대로 쓴다
fetch_repo() {
  if [ ! -d $W/repo ]; then
    command -v git >/dev/null || { apt-get update -y && apt-get install -y git; }
    git clone --depth 1 --branch $BRANCH $REPO_URL $W/repo || { echo "저장소 받기 실패"; exit 1; }
  fi
  [ -f $W/repo/data/tokens_code.bin ] || { echo "데이터 파일이 없습니다"; exit 1; }
}
finish() {
  tar czf $W/../results.tar.gz -C $W results run.log
  echo "== 끝. $(cd $W/.. && pwd)/results.tar.gz 를 받아 가고 파드를 삭제하세요."
}

# 3-1) 전역 트리를 채운 상태의 GPU 측정
run_point() { # 반복번호 모델 배치 조건이름 추가옵션...  (SCRIPT: 측정 스크립트, PREFIX: 결과 이름 앞부분)
  local rep=$1 M=$2 B=$3 name=$4; shift 4
  local out=$W/results/${PREFIX}_$(echo $M | tr '/' '_')_b${B}_${name}_run${rep}.json
  echo "-- $(date +%T) $M b=$B $name ($rep/$REPEAT)"
  timeout 1200 python $SCRIPT --model $M --batch $B --prompts data --data $W/repo/data/tokens_code.bin \
    --max-tokens $MAX_TOKENS --gpu-util 0.85 --max-len 2048 --out $out "$@" > $W/last_run.log 2>&1
  local code=$?
  if [ $code -ne 0 ]; then
    echo "FAIL exit=$code (124 면 시간 초과)"; grep -v -E "^W[0-9]{4}|INFO" $W/last_run.log | tail -20
  fi
  return $code
}
summarize() { # 결과 이름 앞부분
  W=$W PREFIX=$1 python - <<'PY'
import glob, json, os, re, statistics
prefix = os.environ["PREFIX"] + "_"
groups = {}
for f in glob.glob(os.environ["W"] + "/results/" + prefix + "*_run*.json"):
    if f.endswith("_rows.json"):
        continue
    key = re.sub(r"_run\d+\.json$", "", os.path.basename(f))[len(prefix):]
    groups.setdefault(key, []).append(json.load(open(f)))
print(f"{'point':46}{'n':>3}{'step_ms':>9}{'prop_ms':>9}{'draft/req':>10}{'tok/req':>8}{'tok/ms':>8}{'steady':>7}")
for key in sorted(groups):
    runs = groups[key]
    med = lambda field: statistics.median(r[field] for r in runs)
    speed = statistics.median(r["peak_running"] * r["tokens_per_req_step"] / r["step_ms"] for r in runs)
    print(f"{key:46}{len(runs):>3}{med('step_ms'):>9.1f}{med('propose_ms'):>9.2f}"
          f"{med('drafted_per_req_step'):>10.2f}{med('tokens_per_req_step'):>8.2f}{speed:>8.2f}{med('steps_steady'):>7.0f}")
PY
}
if [ $MODE = tree ]; then
  fetch_repo
  POINTS=${POINTS:-"Qwen/Qwen2.5-1.5B:256 Qwen/Qwen2.5-1.5B:128 Qwen/Qwen2.5-7B:256"}
  REPEAT=${REPEAT:-3}
  MAX_TOKENS=${MAX_TOKENS:-512}
  CACHED=${CACHED:-10000}
  SCRIPT=$W/gpu_measure.py PREFIX=tree
  # 중간에 멈춰도 모든 점이 한 번씩은 남도록 반복을 바깥에 둔다
  for rep in $(seq 1 $REPEAT); do
    for point in $POINTS; do
      M=${point%:*}; B=${point##*:}
      run_point $rep $M $B none --spec none
      run_point $rep $M $B cached0 --spec suffix
      run_point $rep $M $B cached$CACHED --spec suffix --n-cached $CACHED
    done
  done
  summarize tree
  finish; exit 0
fi

# 묶음 모듈(tree_batch) 빌드. 결과: $W/repo/bin/vllm_path
build_module() {
  local build_venv=${BUILD_VENV:-$W/venv}
  if [ $build_venv = $W/venv ]; then
    dpkg -s python3-dev >/dev/null 2>&1 || { apt-get update -y && apt-get install -y python3-dev; }
    pip install cmake ninja "nanobind==2.9.2" numpy || exit 1
  fi
  (cd $W/repo && sed -i 's/\r$//' vllm_path/*.sh && VENV=$build_venv bash vllm_path/build.sh) \
    || { echo "묶음 모듈 빌드 실패"; exit 1; }
}

# 3-2) 파드 CPU 에서 요청별 호출과 묶음 호출 비교
if [ $MODE = cpu ]; then
  fetch_repo
  [ -f $W/repo/vllm_path/run_compare.py ] || { echo "vllm_path/ 가 없습니다 (BRANCH 확인)"; exit 1; }
  build_module
  cd $W/repo
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core" | tee $W/results/cpu_info.txt
  MODES=$(python -c "import sys; sys.path.insert(0, 'bin/vllm_path'); import tree_batch; print(' '.join(tree_batch.modes()))") || exit 1
  echo "방식: $MODES"
  # 먼저 작은 설정으로 모든 방식의 결과가 맞는지 확인
  for mode in $MODES; do
    python vllm_path/run_compare.py data/tokens_code.bin --quick --mode=$mode > $W/last_run.log 2>&1 \
      || { echo "확인 실패: $mode"; tail -20 $W/last_run.log; [ $mode = subtree_lock ] && exit 1; continue; }
    echo "확인 통과: $mode"
  done
  compare() { # 방식 배치 트리크기
    echo "-- $(date +%T) mode=$1 b=$2 n_cached=$3"
    python vllm_path/run_compare.py data/tokens_code.bin --mode=$1 --b=$2 --n_cached=$3 --request_start=10000 \
      --threads=1,4,8 --csv=$W/results/cpu_compare.csv > $W/last_run.log 2>&1
    echo "exit=$?"; tail -8 $W/last_run.log
  }
  for B in ${CPU_BATCHES:-256 128 64}; do
    for N in ${CPU_CACHED:-10000 0}; do compare subtree_lock $B $N; done
  done
  # 다른 방식은 가장 큰 배치와 첫 트리 크기에서만
  set -- ${CPU_BATCHES:-256}; first_batch=$1
  set -- ${CPU_CACHED:-10000}; first_cached=$1
  for mode in $MODES; do
    case $mode in serial|subtree_lock) ;; *) compare $mode $first_batch $first_cached ;; esac
  done
  finish; exit 0
fi

# 3-3) vLLM 의 propose 를 묶음 호출로 바꿔 끼워 원본과 비교
if [ $MODE = batch ]; then
  fetch_repo
  [ -f $W/repo/vllm_path/batch_proposer.py ] || { echo "vllm_path/batch_proposer.py 가 없습니다 (BRANCH 확인)"; exit 1; }
  build_module
  POINTS=${POINTS:-"Qwen/Qwen2.5-1.5B:256 Qwen/Qwen2.5-1.5B:128 Qwen/Qwen2.5-7B:256"}
  REPEAT=${REPEAT:-3}
  MAX_TOKENS=${MAX_TOKENS:-512}
  CACHED=${CACHED:-10000}
  THREADS=${THREADS:-"4 8"}
  BATCH_MODE=${BATCH_MODE:-subtree_lock}
  SCRIPT=$W/repo/vllm_path/gpu_measure.py PREFIX=batch
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core" | tee $W/results/cpu_info.txt
  # 먼저 첫 점에서 묶음 추측이 맞는지 확인하고, 틀리면 측정하지 않는다 (check 결과는 속도에 쓰지 않음)
  set -- $POINTS; first=$1
  last_thread=${THREADS##* }
  PREFIX=check
  run_point 0 ${first%:*} ${first##*:} threads$last_thread --spec suffix --n-cached $CACHED \
    --proposer check --mode $BATCH_MODE --threads $last_thread || exit 1
  python - $W/results/check_*_run0.json <<'PY' || { echo "묶음 추측 확인 실패: 측정 중단"; finish; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
print({k: r[k] for k in ("compared", "match_phase_pct", "match_vllm_pct", "restarted", "integrity")})
sys.exit(0 if r["match_phase_pct"] > 99 and r["restarted"] == 0 and not r["integrity"] else 1)
PY
  PREFIX=batch
  for rep in $(seq 1 $REPEAT); do
    for point in $POINTS; do
      M=${point%:*}; B=${point##*:}
      run_point $rep $M $B vllm --spec suffix --n-cached $CACHED
      for T in $THREADS; do
        run_point $rep $M $B threads$T --spec suffix --n-cached $CACHED --proposer batch --mode $BATCH_MODE --threads $T
      done
    done
  done
  summarize batch
  finish; exit 0
fi

# 3-4) 이전과 같은 GPU 측정 (quick, full)
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
rows=[json.load(open(f))|{"f":os.path.basename(f)} for f in sorted(glob.glob("results/*_b*_pc*.json")) if "_rows" not in f]
print(f"{'model':22}{'spec':8}{'b':>5}{'peak':>6}{'step_ms':>10}{'prop_ms':>9}{'G_ms':>9}{'tok/req':>9}{'tok/ms':>8}")
for r in rows:
    tp=r['batch']*r['tokens_per_req_step']/r['step_ms']
    print(f"{r['model'][-20:]:22}{r['spec']:8}{r['batch']:>5}{r['peak_running']:>6}{r['step_ms']:>10.1f}{r['propose_ms']:>9.2f}{r['gpu_side_ms']:>9.1f}{r['tokens_per_req_step']:>9.2f}{tp:>8.2f}")
PY
finish
