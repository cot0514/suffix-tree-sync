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
