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
"""GPU 한 스텝 시간 G와 suffix 추측기 CPU 시간 분리 측정 (vLLM 0.33, 엔진 같은 프로세스).

실행 예: python vllm_path/gpu_measure.py --spec none   --batch 64
         python vllm_path/gpu_measure.py --spec suffix --batch 64
         python vllm_path/gpu_measure.py --spec suffix --batch 64 --prompts data \
                --data data/tokens_code.bin --n-cached 10000     # 전역 트리를 채우고 측정
         ... --proposer batch --mode subtree_lock --threads 8     # propose 를 묶음 호출로 바꿔 끼움
         ... --proposer check                                     # 묶음 호출의 추측이 맞는지 비교
         (batch, check 는 먼저 bash vllm_path/build.sh 로 tree_batch 모듈을 만들어야 함)
스텝마다 (전체 시간, propose 시간, 활성 요청 수, 새 토큰 수)를 재고,
모든 요청이 디코딩 중인 구간(활성 요청 수 >= 0.9*batch)만 모아 중앙값을 낸다.
워밍업에는 측정에 쓰지 않는 프롬프트를 쓰고, 끝나면 suffix 캐시를 비운다
(측정 요청의 정답이 미리 트리에 들어가면 적중률이 부풀려진다).
"""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # propose를 같은 프로세스에서 재려고

import argparse, glob, json, statistics, struct, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--spec", choices=["none", "suffix"], default="suffix")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max-tokens", type=int, default=192)
ap.add_argument("--prompt-chars", type=int, default=2000)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--gpu-util", type=float, default=0.85)
ap.add_argument("--spec-tokens", type=int, default=16)
ap.add_argument("--prompts", choices=["code", "repeat", "data"], default="code",
                help="code=설치된 파이썬 소스, repeat=반복 패턴(적중률 높은 경우), data=--data 의 프롬프트")
ap.add_argument("--data", default=None, help="토큰 데이터 파일 (예: data/tokens_code.bin, Qwen2.5 토큰)")
ap.add_argument("--n-cached", type=int, default=0,
                help="측정 전에 suffix 전역 트리에 넣는 지난 응답 수 (--data 의 앞에서부터)")
ap.add_argument("--min-token-prob", type=float, default=None, help="suffix_decoding_min_token_prob (기본 0.1)")
ap.add_argument("--max-spec-factor", type=float, default=None, help="suffix_decoding_max_spec_factor (기본 1.0)")
ap.add_argument("--proposer", choices=["vllm", "batch", "check"], default="vllm",
                help="vllm=원래 propose, batch=묶음 호출로 교체, check=원래 결과를 쓰면서 묶음 추측을 비교")
ap.add_argument("--mode", default="subtree_lock", help="--proposer batch/check 의 전역 트리 방식")
ap.add_argument("--threads", type=int, default=8, help="--proposer batch/check 의 스레드 수")
ap.add_argument("--module-dir", default=None, help="tree_batch 모듈 폴더 (기본: 저장소 bin/vllm_path)")
ap.add_argument("--out", default=None, help="결과 JSON 경로")
a = ap.parse_args()
if a.proposer != "vllm" and a.spec != "suffix":
    ap.error("--proposer batch/check 는 --spec suffix 에서만 씁니다")

REQUEST_START = 10000   # --data 에서 측정 요청으로 쓰는 첫 기록 번호 (트리에 넣는 기록과 겹치지 않게)
WARMUP_COUNT = 4
if (a.prompts == "data" or a.n_cached) and not a.data:
    ap.error("--prompts data 와 --n-cached 에는 --data 가 필요합니다")
if not 0 <= a.n_cached <= REQUEST_START:
    ap.error(f"--n-cached 는 0 이상 {REQUEST_START} 이하입니다")

import vllm
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind

# propose 시간 재기
propose_ms = []
drafted = []   # propose 호출마다 만든 추측 토큰 총수 (다음 스텝에서 검증됨)
if a.spec == "suffix":
    from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
    _orig = SuffixDecodingProposer.propose
    replacement = []   # --proposer batch/check 일 때 propose 대신 부를 객체 (LLM 을 만든 뒤에 넣는다)
    def _timed(self, *args, **kw):
        t = time.perf_counter()
        r = replacement[0].propose(*args, **kw) if replacement else _orig(self, *args, **kw)
        propose_ms.append((time.perf_counter() - t) * 1e3)
        drafted.append(sum(len(x) for x in r))
        return r
    SuffixDecodingProposer.propose = _timed

# suffix 추측기 객체 붙잡기 (전역 트리를 비우거나 채울 때 씀)
proposers = []
if a.spec == "suffix":
    _orig_init = SuffixDecodingProposer.__init__
    def _keep(self, *args, **kw):
        _orig_init(self, *args, **kw)
        proposers.append(self)
    SuffixDecodingProposer.__init__ = _keep


def load_records(path):
    """KTOK 형식: 'KTOK', 개수, 이어서 (프롬프트 길이, 응답 길이, 프롬프트 토큰, 응답 토큰)."""
    import numpy as np
    raw = open(path, "rb").read()
    if raw[:4] != b"KTOK":
        sys.exit("데이터 파일 형식이 아닙니다: " + path)
    records, off = [], 8
    for _ in range(struct.unpack_from("<i", raw, 4)[0]):
        plen, rlen = struct.unpack_from("<ii", raw, off)
        off += 8
        records.append((np.frombuffer(raw, "<i4", plen, off).astype(np.int32),
                        np.frombuffer(raw, "<i4", rlen, off + 4 * plen).astype(np.int32)))
        off += 4 * (plen + rlen)
    return records


records = load_records(a.data) if a.data else []

# 프롬프트: 앞의 batch 개는 측정용, 뒤의 WARMUP_COUNT 개는 워밍업용
import site
wanted = a.batch + WARMUP_COUNT
prompts = []
if a.prompts == "code":
    # 설치된 파이썬 소스 파일 앞부분 (실제 코드 문맥, 재현 가능)
    files = sorted(glob.glob(os.path.join(site.getsitepackages()[0], "**", "*.py"), recursive=True))
    for f in files:
        try:
            s_ = open(f, encoding="utf-8").read()
        except Exception:
            continue
        if len(s_) >= a.prompt_chars:
            prompts.append(s_[: a.prompt_chars])
        if len(prompts) == wanted:
            break
elif a.prompts == "data":
    picked = records[REQUEST_START : REQUEST_START + wanted]
    prompts = [{"prompt_token_ids": prompt.tolist()} for prompt, _ in picked]
else:
    # 반복 패턴: 요청마다 이름만 다른 한 줄 블록을 그대로 반복 -> 이어쓰기가 앞 줄을 그대로 복사하기 쉬움
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    for i in range(a.batch):
        blk = f"{names[i % 10]} report: the quick brown fox jumps over the lazy dog and keeps running home.\n"
        prompts.append((blk * (a.prompt_chars // len(blk) + 1))[: a.prompt_chars])
    for i in range(WARMUP_COUNT):
        blk = f"warmup {i}: a slow green turtle walks under the old bridge and sleeps near the water.\n"
        prompts.append((blk * (a.prompt_chars // len(blk) + 1))[: a.prompt_chars])
assert len(prompts) == wanted, f"프롬프트 {len(prompts)}개뿐"
prompts, warmup_prompts = prompts[: a.batch], prompts[a.batch :]

kw = dict(model=a.model, dtype="float16", max_model_len=a.max_len,
          gpu_memory_utilization=a.gpu_util, max_num_seqs=a.batch,
          enforce_eager=False)
if a.spec == "suffix":
    sc = {"method": "suffix", "num_speculative_tokens": a.spec_tokens}
    if a.min_token_prob is not None:
        sc["suffix_decoding_min_token_prob"] = a.min_token_prob
    if a.max_spec_factor is not None:
        sc["suffix_decoding_max_spec_factor"] = a.max_spec_factor
    if a.n_cached:
        # 기본 상한(10000)을 넘으면 측정 중인 propose 안에서 오래된 응답 삭제가 일어난다
        sc["suffix_decoding_max_cached_requests"] = a.n_cached + a.batch + 100
    kw["speculative_config"] = sc
llm = LLM(**kw)
eng = llm.llm_engine

swapped = None
if a.proposer != "vllm":
    assert len(proposers) == 1, f"추측기 객체 {len(proposers)}개"
    from batch_proposer import BatchProposer, CheckProposer
    swapped = BatchProposer(proposers[0], a.mode, a.threads, a.module_dir)
    if a.proposer == "check":
        swapped = CheckProposer(swapped, _orig)
    replacement.append(swapped)

sp = SamplingParams(temperature=0, max_tokens=a.max_tokens, ignore_eos=True,
                    output_kind=RequestOutputKind.DELTA)
# 워밍업(그래프 캡처·컴파일 영향 제거)
for i, p in enumerate(warmup_prompts):
    eng.add_request(f"w{i}", p, sp)
while eng.has_unfinished_requests():
    eng.step()

# 워밍업이 남긴 캐시를 비우고, 지난 응답을 vLLM 과 같은 호출 순서로 채운다 (시간 측정 밖)
cache, fill_s, cached_after_fill = None, 0.0, None
if a.spec == "suffix":
    assert len(proposers) == 1, f"추측기 객체 {len(proposers)}개"
    cache = proposers[0].suffix_cache
    for rid in list(cache.active_requests):
        cache.stop_request(rid)
    for rid in list(cache.cached_requests):
        cache.evict_cached_response(rid)
    t = time.perf_counter()
    for k in range(a.n_cached):
        prompt, resp = records[k]
        cache.start_request(f"cached{k}", prompt)
        cache.add_active_response(f"cached{k}", resp)
        cache.stop_request(f"cached{k}")
    fill_s = time.perf_counter() - t
    cached_after_fill = len(cache.cached_requests)
    assert cached_after_fill == a.n_cached, f"캐시 {cached_after_fill}개"
if swapped:
    # 워밍업이 남긴 것을 비우고 같은 지난 응답을 묶음 쪽 트리에도 채운다
    swapped.reset()
    swapped.fill(records[: a.n_cached])
propose_ms.clear()
drafted.clear()

for i, p in enumerate(prompts):
    eng.add_request(f"r{i}", p, sp)
rows = []
while eng.has_unfinished_requests():
    n0 = len(propose_ms)
    d0 = len(drafted)
    t = time.perf_counter()
    outs = eng.step()
    dt = (time.perf_counter() - t) * 1e3
    pm = sum(propose_ms[n0:])
    new = sum(len(o.outputs[0].token_ids) for o in outs)
    rows.append((dt, pm, len(outs), new, sum(drafted[d0:])))

# 최대 동시 실행 수(KV가 모자라면 batch보다 작을 수 있음)의 90% 이상인 스텝만, 프리필 끝난 뒤
peak = max(r[2] for r in rows)
i0 = next(i for i, r in enumerate(rows) if r[2] >= peak) + 1
steady = [r for r in rows[i0:] if r[2] >= 0.9 * peak and r[3] > 0]
if not steady:
    print("경고: 안정 구간 없음, 전체 스텝 사용"); steady = rows
med = lambda i: statistics.median(r[i] for r in steady)
res = dict(min_token_prob=a.min_token_prob, max_spec_factor=a.max_spec_factor, prompts=a.prompts, spec_tokens=a.spec_tokens,
           drafted_per_req_step=sum(r[4] for r in steady) / max(1, sum(r[2] for r in steady)),
           peak_running=peak, prompt_chars=a.prompt_chars, model=a.model, spec=a.spec, batch=a.batch, steps_total=len(rows),
           steps_steady=len(steady), step_ms=med(0), propose_ms=med(1),
           gpu_side_ms=med(0) - med(1),
           tokens_per_req_step=sum(r[3] for r in steady) / max(1, sum(r[2] for r in steady)),
           n_cached=a.n_cached if cache else 0, fill_s=fill_s, cached_after_fill=cached_after_fill,
           cached_at_end=len(cache.cached_requests) if cache else None,
           proposer=a.proposer,
           vllm=vllm.__version__)
if swapped:
    batch = swapped.batch if a.proposer == "check" else swapped
    res.update(mode=a.mode, threads=a.threads, restarted=batch.restarted, integrity=batch.tree.check(),
               batch_cached=batch.cached_requests)
    if a.proposer == "check":
        res.update(compared=swapped.compared,
                   match_phase_pct=100.0 * swapped.same_phase / max(1, swapped.compared),
                   match_vllm_pct=100.0 * swapped.same_vllm / max(1, swapped.compared))
print(json.dumps(res, ensure_ascii=False, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    json.dump(rows, open(a.out.replace('.json', '_rows.json'), 'w'))
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
