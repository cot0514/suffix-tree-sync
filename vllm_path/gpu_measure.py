"""GPU 한 스텝 시간 G와 suffix 추측기 CPU 시간 분리 측정 (vLLM 0.33, 엔진 같은 프로세스).

실행 예: python vllm_path/gpu_measure.py --spec none   --batch 64
         python vllm_path/gpu_measure.py --spec suffix --batch 64
스텝마다 (전체 시간, propose 시간, 활성 요청 수, 새 토큰 수)를 재고,
모든 요청이 디코딩 중인 구간(활성 요청 수 >= 0.9*batch)만 모아 중앙값을 낸다.
"""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # propose를 같은 프로세스에서 재려고

import argparse, glob, json, statistics, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--spec", choices=["none", "suffix"], default="suffix")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max-tokens", type=int, default=192)
ap.add_argument("--prompt-chars", type=int, default=2000)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--gpu-util", type=float, default=0.85)
ap.add_argument("--spec-tokens", type=int, default=16)
ap.add_argument("--prompts", choices=["code", "repeat"], default="code",
                help="code=설치된 파이썬 소스, repeat=반복 패턴(적중률 높은 경우)")
ap.add_argument("--min-token-prob", type=float, default=None, help="suffix_decoding_min_token_prob (기본 0.1)")
ap.add_argument("--max-spec-factor", type=float, default=None, help="suffix_decoding_max_spec_factor (기본 1.0)")
ap.add_argument("--out", default=None, help="결과 JSON 경로")
a = ap.parse_args()

import vllm
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind

# propose 시간 재기
propose_ms = []
drafted = []   # propose 호출마다 만든 추측 토큰 총수 (다음 스텝에서 검증됨)
if a.spec == "suffix":
    from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
    _orig = SuffixDecodingProposer.propose
    def _timed(self, *args, **kw):
        t = time.perf_counter()
        r = _orig(self, *args, **kw)
        propose_ms.append((time.perf_counter() - t) * 1e3)
        drafted.append(sum(len(x) for x in r))
        return r
    SuffixDecodingProposer.propose = _timed

# 프롬프트
import site
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
        if len(prompts) == a.batch:
            break
else:
    # 반복 패턴: 요청마다 이름만 다른 한 줄 블록을 그대로 반복 -> 이어쓰기가 앞 줄을 그대로 복사하기 쉬움
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    for i in range(a.batch):
        blk = f"{names[i % 10]} report: the quick brown fox jumps over the lazy dog and keeps running home.\n"
        prompts.append((blk * (a.prompt_chars // len(blk) + 1))[: a.prompt_chars])
assert len(prompts) == a.batch, f"프롬프트 {len(prompts)}개뿐"

kw = dict(model=a.model, dtype="float16", max_model_len=a.max_len,
          gpu_memory_utilization=a.gpu_util, max_num_seqs=a.batch,
          enforce_eager=False)
if a.spec == "suffix":
    sc = {"method": "suffix", "num_speculative_tokens": a.spec_tokens}
    if a.min_token_prob is not None:
        sc["suffix_decoding_min_token_prob"] = a.min_token_prob
    if a.max_spec_factor is not None:
        sc["suffix_decoding_max_spec_factor"] = a.max_spec_factor
    kw["speculative_config"] = sc
llm = LLM(**kw)
eng = llm.llm_engine

sp = SamplingParams(temperature=0, max_tokens=a.max_tokens, ignore_eos=True,
                    output_kind=RequestOutputKind.DELTA)
# 워밍업(그래프 캡처·컴파일 영향 제거)
for i, p in enumerate(prompts[:4]):
    eng.add_request(f"w{i}", p, sp)
while eng.has_unfinished_requests():
    eng.step()
propose_ms.clear()

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
           vllm=vllm.__version__)
print(json.dumps(res, ensure_ascii=False, indent=1))
if a.out:
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    json.dump(rows, open(a.out.replace('.json', '_rows.json'), 'w'))
