#!/usr/bin/env python3
"""vLLM 실제 경로 측정 (README 7장 1·2단계).

같은 작업량(tree_bench 의 phase 순서와 같음)을 두 방식으로 돌려 파이썬 호출 비용을 비교한다.
  1) 요청별 호출 : 진짜 SuffixDecodingCache. 요청마다 파이썬이 C++ 를 부른다
  2) 묶음 호출   : tree_batch 모듈. 스텝마다 갱신 한 번, 추측 한 번만 부른다 (전역 트리는 --mode)
                   --threads 1,4,... 로 스레드 수마다 한 줄씩 낸다 (3단계: 묶음 N스레드)
묶음 호출에는 요청별 호출이 스텝마다 받은 입력을 그대로 다시 넣는다. 그래서 추측이 일부 달라도
이후 입력과 트리 내용은 갈라지지 않고, 모든 실행이 같은 작업량을 처리한다.

판정 (PASS, PASS*, FAIL 은 tree_bench 의 --check 와 같은 뜻)
  PASS            트리 내용·구조가 정상이고 추측이 요청별 호출과 비트 단위로 같음
  PASS* spec N%   트리 내용·구조는 정상, 추측은 N% 만 같음 (count 가 같은 형제의 순서 차이일 수 있음)
  FAIL tree@stepN 트리 내용이 묶음 serial 1스레드와 다름
  FAIL integrity  트리 구조가 깨짐
  FAIL batch      묶음 serial 1스레드의 추측이 요청별 호출과 다름 (묶음 모듈 자체의 문제)
  n/a             그 방식에 digest() 가 없어서 트리 내용을 비교하지 못함
트리 내용은 --check (또는 --quick) 면 매 스텝, 아니면 마지막 스텝만 비교한다.
매 스텝 비교는 시간 측정 밖이지만 캐시에 영향을 주므로 속도를 잴 때는 --check 를 빼고 돌린다.
종료 코드: FAIL 이 하나라도 있으면 2.

실행 (WSL2):  source ~/venv_arctic/bin/activate
              python vllm_path/run_compare.py data/tokens_code.bin [--quick]
시간에서 빼는 것: 사전 채우기, 정답 비교(GPU 역할), 요청별 입력 만들기, 결과 풀어 읽기.
시간에 넣는 것 : 호출 자체 + (묶음 호출은) 입력을 한 줄 배열로 묶는 비용.
"""
import argparse
import os
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCTIC_STAGE = os.environ.get("ARCTIC_STAGE", str(Path.home() / "arctic_stage"))
sys.path.insert(0, str(REPO_ROOT / "bin" / "vllm_path"))
sys.path.insert(0, ARCTIC_STAGE)

now = time.perf_counter


# ---------------------------------------------------------------- 데이터
def load_records(path):
    """KTOK 형식: 'KTOK', 개수, 이어서 (프롬프트 길이, 응답 길이, 프롬프트 토큰, 응답 토큰)."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"KTOK":
        sys.exit("데이터 파일 형식이 아닙니다: " + str(path))
    (n,) = struct.unpack_from("<i", raw, 4)
    off = 8
    recs = []
    for _ in range(n):
        plen, rlen = struct.unpack_from("<ii", raw, off)
        off += 8
        prompt = np.frombuffer(raw, dtype="<i4", count=plen, offset=off).astype(np.int32)
        off += 4 * plen
        resp = np.frombuffer(raw, dtype="<i4", count=rlen, offset=off).astype(np.int32)
        off += 4 * rlen
        recs.append((prompt, resp))
    return recs


def pack(arrays):
    """배열 여러 개를 한 줄로 잇고 시작 위치(오프셋) 배열을 만든다."""
    offsets = np.zeros(len(arrays) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(a) for a in arrays])
    flat = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int32)
    return flat, offsets


def draft_key(tokens, parents, probs, score, match_len):
    """추측 결과 하나를 비교용 값으로 만든다 (정수·실수 튜플의 해시는 실행마다 같다)."""
    return hash((tuple(tokens), tuple(parents), tuple(probs), score, match_len))


# ---------------------------------------------------------------- 작업량 (tree_bench 의 Sim 과 같은 규칙)
class Workload:
    def __init__(self, recs, cfg):
        self.cfg = cfg
        self.seq, self.plen, self.pos, self.pending = [], [], [], []
        for i in range(cfg.b):
            prompt, resp = recs[cfg.n_cached + i]
            s = np.concatenate([prompt, resp]).astype(np.int32)
            self.seq.append(s)
            self.plen.append(len(prompt))
            self.pending.append(s[len(prompt):len(prompt) + 1])   # 첫 샘플 토큰
            self.pos.append(len(prompt) + 1)
        self.accepted = self.drafted = self.req_steps = 0

    def live_requests(self):
        return [i for i in range(self.cfg.b) if self.pos[i] < len(self.seq[i]) - 1]

    def context(self, i):
        lo = max(0, self.pos[i] - self.cfg.depth)
        return self.seq[i][lo:self.pos[i]]

    def verify(self, live, draft_tokens):
        """정답(teacher forcing)과 비교해 수락 길이를 정하고 다음에 붙일 토큰을 만든다 (GPU 역할)."""
        for i, d in zip(live, draft_tokens):
            s, pos = self.seq[i], self.pos[i]
            a = 0
            while a < len(d) and pos + a < len(s) - 1 and d[a] == s[pos + a]:
                a += 1
            self.accepted += a
            self.drafted += len(d)
            self.req_steps += 1
            n_new = min(a + 1, len(s) - pos)
            self.pending[i] = s[pos:pos + n_new]
            self.pos[i] = pos + n_new

    def tokens_per_step(self):
        return (self.accepted + self.req_steps) / max(1, self.req_steps)


class RunResult:
    def __init__(self):
        self.update_ms = []     # 스텝별 갱신 시간
        self.spec_ms = []       # 스텝별 추측 시간
        self.live = []          # 스텝별 살아 있는 요청 수
        self.keys = []          # 스텝별 추측 결과 비교값 (요청 순서대로)
        self.inputs = []        # 요청별 호출의 스텝별 입력 (요청 번호, 붙일 토큰, 문맥)
        self.digests = {}       # 묶음 호출의 스텝 번호 -> 전역 트리 내용 요약값
        self.integrity = ""     # 묶음 호출의 전역 트리 구조 검사 결과 (정상이면 빈 문자열)
        self.tok_per_step = 0.0
        self.local_s = self.global_s = self.spec_s = self.lockw_s = 0.0   # 묶음 호출의 구간별 합 (초)
        self.wall_s = 0.0

    def mean(self, xs):
        return sum(xs) / max(1, len(xs))


# ---------------------------------------------------------------- 1) 요청별 호출
def run_per_request(recs, cfg):
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    cache = SuffixDecodingCache(max_tree_depth=cfg.depth, max_cached_requests=-1)
    # 사전 채우기: 지난 응답을 vLLM 과 같은 호출 순서로 넣는다 (시간에서 제외)
    for k in range(cfg.n_cached):
        prompt, resp = recs[k]
        cache.start_request(k, prompt)
        cache.add_active_response(k, resp)
        cache.stop_request(k)
    wl = Workload(recs, cfg)
    for i in range(cfg.b):
        cache.start_request(cfg.n_cached + i, recs[cfg.n_cached + i][0])

    out = RunResult()
    for _ in range(cfg.steps):
        live = wl.live_requests()
        if not live:
            break
        ids = [cfg.n_cached + i for i in live]
        pend = [wl.pending[i] for i in live]
        ctxs = [wl.context(i) for i in live]
        out.inputs.append((live, pend, ctxs))

        t0 = now()
        for rid, tok in zip(ids, pend):
            cache.add_active_response(rid, tok)
        t1 = now()
        drafts = [cache.speculate(rid, ctx, cfg.spec) for rid, ctx in zip(ids, ctxs)]
        t2 = now()

        out.update_ms.append((t1 - t0) * 1e3)
        out.spec_ms.append((t2 - t1) * 1e3)
        out.live.append(len(live))
        out.keys.append([draft_key(d.token_ids, d.parents, d.probs, d.score, d.match_len)
                         for d in drafts])
        wl.verify(live, [d.token_ids for d in drafts])
    out.tok_per_step = wl.tokens_per_step()
    return out


# ---------------------------------------------------------------- 2) 묶음 호출
def run_batch(recs, cfg, threads, inputs, mode):
    import tree_batch

    bt = tree_batch.BatchTree(mode, cfg.depth, threads)
    # 사전 채우기 (시간에서 제외)
    for k in range(cfg.n_cached):
        bt.prepare_seq(k, len(recs[k][1]))
    for i in range(cfg.b):
        bt.prepare_seq(cfg.n_cached + i, len(recs[cfg.n_cached + i][1]))
    for k in range(cfg.n_cached):
        bt.extend_global(k, recs[k][1])
    flat, off = pack([recs[cfg.n_cached + i][0] for i in range(cfg.b)])
    bt.start_requests(flat, off)
    bt.reset_stats()

    out = RunResult()
    for step, (live, pend, ctxs) in enumerate(inputs):
        req_idx = np.array(live, dtype=np.int32)
        seq_ids = req_idx + np.int32(cfg.n_cached)

        t0 = now()
        tok_flat, tok_off = pack(pend)
        bt.update(req_idx, seq_ids, tok_flat, tok_off)
        t1 = now()
        ctx_flat, ctx_off = pack(ctxs)
        toks, parents, probs, doff, scores, mlens = bt.speculate(
            req_idx, ctx_flat, ctx_off, cfg.spec)
        t2 = now()

        bt.end_step()   # 스텝 끝 정리 (update 시간에 넣는다)
        t3 = now()

        out.update_ms.append((t1 - t0 + t3 - t2) * 1e3)
        out.spec_ms.append((t2 - t1) * 1e3)
        out.live.append(len(live))
        keys = []
        toks_l, parents_l, probs_l = toks.tolist(), parents.tolist(), probs.tolist()
        scores_l, mlens_l, doff_l = scores.tolist(), mlens.tolist(), doff.tolist()
        for k in range(len(live)):
            a, b = doff_l[k], doff_l[k + 1]
            keys.append(draft_key(toks_l[a:b], parents_l[a:b], probs_l[a:b],
                                  scores_l[k], mlens_l[k]))
        out.keys.append(keys)
        if cfg.check or step == len(inputs) - 1:   # 트리 내용 요약 (시간 측정 밖)
            out.digests[step] = bt.digest()
    out.local_s, out.global_s, out.spec_s, out.lockw_s = bt.stats()
    out.wall_s = sum(out.update_ms + out.spec_ms) / 1e3
    out.integrity = bt.check()
    return out


# ---------------------------------------------------------------- 비교·출력
def spec_match(keys_ref, keys_run):
    """요청별 호출과 같은 추측의 비율(%). 모두 같을 때만 100 이 된다."""
    total = sum(len(step_keys) for step_keys in keys_ref)
    same = sum(x == y for ref_step, run_step in zip(keys_ref, keys_run) for x, y in zip(ref_step, run_step))
    return 100.0 * same / max(1, total)


def judge(run, ref_keys, ref_tree):
    """묶음 실행 하나를 판정한다. (순위, 판정, 추측 일치율) 을 돌려준다. 순위가 클수록 나쁘다."""
    match = spec_match(ref_keys, run.keys)
    unknown = False
    for step, digest in run.digests.items():
        if digest == 0 or ref_tree.digests[step] == 0:
            unknown = True
        elif digest != ref_tree.digests[step]:
            return 3, f"FAIL tree@step{step}", match
    if run.integrity:
        return 3, "FAIL integrity", match
    if unknown:
        return 2, "n/a", match
    if match < 100.0:
        shown = min(match, 99.99)   # 반올림 때문에 100.00% 로 보이지 않게
        return 1, f"PASS* spec {shown:.2f}%", match
    return 0, "PASS", match


def parse_args():
    p = argparse.ArgumentParser(description="vLLM 실제 경로 측정 (요청별 호출 vs 묶음 호출)")
    p.add_argument("tokens", help="데이터 파일 (예: data/tokens_code.bin)")
    p.add_argument("--b", type=int, default=256, help="동시에 처리하는 요청 수")
    p.add_argument("--n_cached", type=int, default=10000, help="전역 트리에 미리 넣는 지난 응답 수")
    p.add_argument("--request_start", type=int,
                   help="처리하는 요청이 시작하는 기록 번호 (기본: n_cached). 고정하면 트리 크기를 바꿔도 같은 요청")
    p.add_argument("--steps", type=int, default=100, help="스텝 수")
    p.add_argument("--spec", type=int, default=16, help="추측 토큰 최대 개수")
    p.add_argument("--depth", type=int, default=24, help="트리 최대 깊이")
    p.add_argument("--repeat", type=int, default=3, help="반복 횟수 (중앙값 사용)")
    p.add_argument("--mode", default="subtree_lock", help="묶음 호출의 전역 트리 방식")
    p.add_argument("--threads", default="1,2,4,6,8", help="묶음 호출의 스레드 수 목록 (쉼표로 구분)")
    p.add_argument("--csv", help="결과를 한 줄씩 덧붙일 CSV 파일")
    p.add_argument("--check", action="store_true", help="트리 내용을 매 스텝 비교 (속도 측정에는 쓰지 않기)")
    p.add_argument("--quick", action="store_true", help="작은 설정으로 빠르게 확인 (--check 포함)")
    cfg = p.parse_args()
    if cfg.quick:
        cfg.b, cfg.n_cached, cfg.steps, cfg.repeat = 32, 300, 40, 1
        cfg.check = True
        if cfg.threads == "1,2,4,6,8":
            cfg.threads = "1,4"
    if cfg.request_start is None:
        cfg.request_start = cfg.n_cached
    elif cfg.request_start < cfg.n_cached:
        p.error("--request_start 는 --n_cached 이상이어야 합니다 (트리에 넣은 기록과 겹치면 안 됨)")
    cfg.threads = [int(x) for x in cfg.threads.split(",") if x]
    return cfg


CSV_HEADER = ("variant,mode,threads,b,n_cached,steps,repeat,update_ms,spec_ms,total_ms,speedup,"
              "calls_per_step,local_ms,global_ms,spec_in_ms,lockw_pct,check,spec_match_pct,request_start")


def main():
    cfg = parse_args()
    # 열 구성이 다른 옛 CSV 에 덧붙이면 열이 어긋나므로 측정 전에 멈춘다
    if cfg.csv and Path(cfg.csv).exists():
        lines = Path(cfg.csv).read_text(encoding="utf-8").splitlines()
        if lines and lines[0] != CSV_HEADER:
            sys.exit("CSV 의 열 구성이 달라서 덧붙일 수 없습니다. 새 파일 이름을 쓰세요: " + cfg.csv)
    recs = load_records(cfg.tokens)
    need = cfg.request_start + cfg.b
    if len(recs) < need:
        sys.exit(f"need {need} records, have {len(recs)}: --n_cached, --request_start, --b 중에서 줄이세요")
    print(f"file={cfg.tokens} records={len(recs)} B={cfg.b} N_CACHED={cfg.n_cached} REQUEST_START={cfg.request_start} "
          f"STEPS={cfg.steps} SPEC={cfg.spec} DEPTH={cfg.depth} mode={cfg.mode} repeat={cfg.repeat}")

    # 트리에 넣을 기록 바로 뒤에 처리할 요청이 오도록 이어 붙인다
    recs = recs[:cfg.n_cached] + recs[cfg.request_start:cfg.request_start + cfg.b]

    names = ["per_request"] + [f"batch{t}" for t in cfg.threads]
    runs = {n: [] for n in names}
    ref_tree = None
    for r in range(cfg.repeat):
        print(f"[{r + 1}/{cfg.repeat}] 요청별 호출 ...", flush=True)
        runs["per_request"].append(run_per_request(recs, cfg))
        inputs = runs["per_request"][0].inputs
        if ref_tree is None:
            print("기준 트리: 묶음 serial 1스레드 ...", flush=True)
            ref_tree = run_batch(recs, cfg, 1, inputs, "serial")
        for t in cfg.threads:
            print(f"[{r + 1}/{cfg.repeat}] 묶음 호출 {t}스레드 ...", flush=True)
            runs[f"batch{t}"].append(run_batch(recs, cfg, t, inputs, cfg.mode))

    def med(name, field):
        return statistics.median(runs[name][i].mean(getattr(runs[name][i], field))
                                 for i in range(cfg.repeat))

    def med_attr(name, field):
        return statistics.median(getattr(x, field) for x in runs[name])

    # 정확성: 반복마다 판정해서 가장 나쁜 판정과 가장 낮은 추측 일치율을 보고한다
    ref_keys = runs["per_request"][0].keys
    ref_match = spec_match(ref_keys, ref_tree.keys)
    ref_broken = ref_match < 100.0 or bool(ref_tree.integrity)
    if ref_broken:
        print(f"기준 트리 이상: 추측 일치 {ref_match:.4f}%, 구조 검사 '{ref_tree.integrity}'")
    verdicts, matches = {}, {}
    for t in cfg.threads:
        results = [judge(x, ref_keys, ref_tree) for x in runs[f"batch{t}"]]
        verdicts[t] = "FAIL batch" if ref_broken else max(results, key=lambda v: v[0])[1]
        matches[t] = min(v[2] for v in results)
        broken = next((x.integrity for x in runs[f"batch{t}"] if x.integrity), None)
        if broken:
            print(f"{t}스레드 구조 검사: {broken}")
    any_fail = ref_broken or any(v.startswith("FAIL") for v in verdicts.values())

    first_batch = "batch" + str(cfg.threads[0])
    live_avg = statistics.mean(runs[first_batch][0].live) if runs[first_batch][0].live else 0
    rows = []   # (name, label, upd, spc, total, speedup, calls, check, local, global, spec, lockw%)
    base_total = None
    for name in names:
        if name == "per_request":
            label, t, calls, chk = "1) per-request call", 1, 2 * live_avg, "-"
        else:
            t = int(name[5:])
            label, calls, chk = f"2) batch call, {t} thread{'s' if t > 1 else ''}", 2, verdicts[t]
        upd, spc = med(name, "update_ms"), med(name, "spec_ms")
        total = upd + spc
        if base_total is None:
            base_total = total
        nsteps = max(1, len(runs[name][0].live))
        if name == "per_request":
            loc = glo = sp_in = lw = 0.0
        else:   # 구간별 시간은 스텝당 ms, 락 대기는 (스레드 수 x 전체 시간) 대비 %
            loc = med_attr(name, "local_s") / nsteps * 1e3
            glo = med_attr(name, "global_s") / nsteps * 1e3
            sp_in = med_attr(name, "spec_s") / nsteps * 1e3
            wall = med_attr(name, "wall_s")
            lw = 100.0 * med_attr(name, "lockw_s") / max(1e-9, t * wall)
        rows.append((name, label, upd, spc, total, base_total / total, calls, chk, loc, glo, sp_in, lw))

    print()
    print("ms per full-batch step (median of repeats). phase order: update all -> speculate all")
    print(f"{'variant':26} | {'update':>8} {'spec':>8} {'total':>8} | {'speedup':>7} | {'py calls/step':>13} | "
          f"{'loc':>6} {'glob':>6} {'spec':>6} {'lockw%':>6} | check")
    for name, label, upd, spc, total, sp, calls, chk, loc, glo, sp_in, lw in rows:
        det = "" if name == "per_request" else f"{loc:6.3f} {glo:6.3f} {sp_in:6.3f} {lw:6.1f}"
        print(f"{label:26} | {upd:8.3f} {spc:8.3f} {total:8.3f} | {sp:6.2f}x | {calls:13.0f} | "
              f"{det:>27} | {chk}")
    one = next((r for r in rows if r[0] == "batch1"), None)
    if one:
        print("speedup vs batch 1 thread: " + ", ".join(
            f"{r[0][5:]}t x{one[4] / r[4]:.2f}" for r in rows if r[0].startswith("batch")))
    print(f"live requests/step (avg) = {live_avg:.1f}, tok/st = {runs['per_request'][0].tok_per_step:.3f}  "
          f"mode={cfg.mode}, tree check = {'every step' if cfg.check else 'last step'}")

    if cfg.csv:
        new = not Path(cfg.csv).exists()
        with open(cfg.csv, "a", encoding="utf-8") as f:
            if new:
                f.write(CSV_HEADER + "\n")
            for name, label, upd, spc, total, sp, calls, chk, loc, glo, sp_in, lw in rows:
                t = 1 if name == "per_request" else int(name[5:])
                match = "" if name == "per_request" else f"{matches[t]:.4f}"
                f.write(f"{'per_request' if name == 'per_request' else 'batch'},{cfg.mode},{t},{cfg.b},"
                        f"{cfg.n_cached},{cfg.steps},{cfg.repeat},{upd:.4f},{spc:.4f},{total:.4f},{sp:.3f},"
                        f"{calls:.0f},{loc:.4f},{glo:.4f},{sp_in:.4f},{lw:.2f},{chk},{match},{cfg.request_start}\n")
    return 2 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
