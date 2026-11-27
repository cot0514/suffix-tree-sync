"""run_all.bat 결과(CSV)로 그래프와 요약 표를 만든다 (lazy 담당).

  python scripts/plot.py                              # results/all_scale.csv, all_mixed.csv
  python scripts/plot.py results/quick_scale.csv      # CSV 를 직접 지정 (여러 개 가능)
  python scripts/plot.py --out results/plots --at 8   # 저장 폴더, 단계 분해·삭제 그래프의 스레드 수

필요: pip install pandas matplotlib

만드는 것 (--out 폴더)
  1_step_ms.png      스레드 수 → 스텝 시간 (행 = 데이터, 열 = 삭제 수). 주 결과
  2_speedup.png      스레드 수 → serial 1스레드 대비 속도 향상
  3_breakdown.png    --at 스레드에서 단계별 시간 (개인 갱신·전역 갱신·삭제·추측·end) 쌓은 막대
  4_lockwait.png     스레드 수 → 락 기다린 비율
  5_evict.png        --at 스레드에서 삭제 수 → 스텝 시간
  6_mixed.png        mixed 순서의 스레드 수 → 스텝 시간 (mixed CSV 가 있을 때)
  summary.md         방식별 1스레드 시간, 가장 빠른 시간과 그때 스레드 수, serial 1스레드 대비 배수

같은 조건(데이터·방식·순서·스레드·설정)이 여러 줄이면 CSV 에서 마지막 줄만 쓴다 (다시 잰 값).
색은 방식마다 고정 (serial, global_lock, subtree_lock, node_lock, optimistic, lazy 순서, 그 밖의 방식은 그 뒤).
"""
import argparse
import pathlib
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import pandas as pd
except ImportError:
    sys.exit("pandas, matplotlib 가 필요합니다:  pip install pandas matplotlib")

ROOT = pathlib.Path(__file__).resolve().parent.parent
if not sys.stdout.isatty():   # 파이프·파일로 보낼 때는 UTF-8 (Windows 기본 cp949 로 한글이 깨지는 것 방지)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- 모양
# 범주 색 8개 (색약 구분 검사 통과한 팔레트, 고정 순서). 같은 방식은 모든 그래프에서 같은 색.
PALETTE = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]   # 색만으로 구분하지 않도록 모양도 다르게
KNOWN_MODES = ["serial", "global_lock", "subtree_lock", "node_lock", "optimistic", "lazy"]
# 단계 = 처리 순서가 있는 값 → 파란색 한 색의 밝기 단계 (순서형)
PHASES = ["ph_local_ms", "ph_global_ms", "ph_remove_ms", "ph_spec_ms", "ph_end_ms"]
PHASE_COLORS = ["#84b9f3", "#519ced", "#257de0", "#1b64be", "#0e3d7e"]
TEXT, TEXT2, GRID, SURFACE = "#191919", "#666666", "#e3e3e3", "#ffffff"
KEY = ["data", "mode", "sched", "threads", "B", "N_CACHED", "STEPS", "EVICT", "local_par"]

KO = {
    "threads": "스레드 수", "step_ms": "스텝 시간 (ms, 작을수록 좋음)", "speedup": "serial 1스레드 대비 (배)",
    "lockwait": "락 기다린 비율 (%)", "evict": "스텝마다 지우는 답변 수", "ms": "스텝 시간 (ms)",
    "phases": ["개인 트리 갱신", "전역 트리 갱신", "삭제", "추측", "end (스텝 끝 정리)"],
    "t1": "스레드 수에 따른 스텝 시간 (phase)", "t2": "속도 향상 (serial 1스레드 = 1)",
    "t3": "{t}스레드에서 단계별 시간", "t4": "락 기다린 비율", "t5": "{t}스레드에서 삭제 양에 따른 스텝 시간",
    "t6": "읽기·쓰기가 섞일 때 (mixed)", "evict_n": "삭제 {e}", "base": "기준",
    "note3": "삭제를 갱신과 동시에 할 수 있는 방식(caps의 par_remove=true)은 측정 틀이 삭제를 전역 갱신 구간 안에서 돌리므로, 삭제 시간이 '전역 트리 갱신'에 들어간다.",
}
EN = {
    "threads": "threads", "step_ms": "step time (ms, lower is better)", "speedup": "x vs serial 1 thread",
    "lockwait": "lock wait (%)", "evict": "responses evicted per step", "ms": "step time (ms)",
    "phases": ["local tree update", "global tree update", "remove", "speculate", "end (step cleanup)"],
    "t1": "Step time vs threads (phase)", "t2": "Speedup (serial 1 thread = 1)",
    "t3": "Time per phase at {t} threads", "t4": "Lock wait", "t5": "Step time vs eviction at {t} threads",
    "t6": "Reads and writes mixed (mixed)", "evict_n": "evict {e}", "base": "baseline",
    "note3": "Modes with caps par_remove=true run removals inside the global update phase, so their removal time is counted in 'global tree update'.",
}


def setup_fonts():
    """한글 글꼴이 있으면 한글 라벨, 없으면 영어 라벨 (Linux/Colab 에서 네모 글자 방지)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans KR", "Noto Sans CJK KR"]:
        if name in names:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return KO
    return EN


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def mode_styles(modes):
    """방식 → (색, 모양). 알려진 방식은 고정 칸, 나머지는 이름순으로 남은 칸."""
    order = [m for m in KNOWN_MODES if m in modes] + sorted(m for m in modes if m not in KNOWN_MODES)
    slots = {m: KNOWN_MODES.index(m) for m in order if m in KNOWN_MODES}
    free = [i for i in range(len(PALETTE)) if i not in slots.values()]
    for m in order:
        if m not in slots:
            slots[m] = free.pop(0) if free else None
    styles = {}
    for m in order:
        i = slots[m]
        if i is None:   # 8개를 넘으면 색을 새로 만들지 않고 회색 점선으로 (방식 수를 줄이는 게 맞음)
            print(f"[WARN] 방식이 8개를 넘어 '{m}' 은 회색 점선으로 그림")
            styles[m] = ("#9c9c9c", "x", "--")
        else:
            styles[m] = (PALETTE[i], MARKERS[i], "-")
    return order, styles


def load(paths):
    frames = []
    for p in paths:
        p = pathlib.Path(p)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["_src"] = p.name
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["data"] = df["data"].astype(str).str.replace(r"^tokens_", "", regex=True).str.replace(r"\.bin$", "", regex=True)
    df = df.drop_duplicates(subset=KEY, keep="last")   # 같은 조건을 다시 쟀으면 마지막 값
    return df


def mode_handles(order, styles):
    return [plt.Line2D([], [], color=styles[m][0], marker=styles[m][1], linestyle=styles[m][2],
                       linewidth=2, markersize=6, markeredgecolor=SURFACE) for m in order]


def finish(fig, title, handles, labels, out, note=None):
    """제목(맨 위) → 범례 → 그림 순서로 배치하고 저장. 위·아래 여백은 인치 단위로 고정."""
    h = fig.get_figheight()
    top_in = 0.75                      # 제목 + 범례 높이 (인치)
    bottom_in = 0.35 if note else 0.0
    fig.suptitle(title, y=1 - 0.12 / h, fontsize=11, color=TEXT, va="top")
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.38 / h), fontsize=9, labelcolor=TEXT)
    if note:
        fig.text(0.01, 0.08 / h, note, fontsize=7.5, color=TEXT2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, bottom_in / h, 1, 1 - top_in / h))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("saved", out)


def grid_figure(df, L, value, ylabel, title, order, styles, out, transform=None, hline=None):
    """행 = 데이터, 열 = 삭제 수, 선 = 방식, x = 스레드 수."""
    datas, evicts = sorted(df["data"].unique()), sorted(df["EVICT"].unique())
    fig, axes = plt.subplots(len(datas), len(evicts), figsize=(3.6 * len(evicts), 2.9 * len(datas) + 0.8),
                             squeeze=False)
    for r, d in enumerate(datas):
        row_max = 0.0
        for c, e in enumerate(evicts):
            ax = axes[r][c]
            style_axes(ax)
            sub = df[(df["data"] == d) & (df["EVICT"] == e)]
            for m in order:
                s = sub[sub["mode"] == m].sort_values("threads")
                if s.empty:
                    continue
                y = transform(s, sub) if transform else s[value]
                if y is None:
                    continue
                col, mk, ls = styles[m]
                ax.plot(s["threads"], y, color=col, marker=mk, linestyle=ls, linewidth=2, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1)
                row_max = max(row_max, float(pd.Series(y).max()))
            if hline is not None:
                ax.axhline(hline, color=TEXT2, linewidth=1, linestyle=":")
            ax.set_xticks(sorted(sub["threads"].unique()))
            ax.set_title(f"{d} · {L['evict_n'].format(e=e)}", fontsize=9, color=TEXT)
            if r == len(datas) - 1:
                ax.set_xlabel(L["threads"], fontsize=8, color=TEXT2)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=TEXT2)
        top = max(row_max, hline or 0) * 1.08 or 1.0
        for c in range(len(evicts)):   # 한 행은 같은 눈금 (삭제 수끼리 비교하기 쉽게)
            axes[r][c].set_ylim(0, top)
    finish(fig, title, mode_handles(order, styles), order, out)


def pick_threads(df, want):
    ts = sorted(df["threads"].unique())
    below = [t for t in ts if t <= want]
    return below[-1] if below else ts[-1]


def breakdown_figure(df, L, at, order, out):
    datas, evicts = sorted(df["data"].unique()), sorted(df["EVICT"].unique())
    t = pick_threads(df, at)
    sub_all = df[df["threads"] == t]
    fig, axes = plt.subplots(len(datas), len(evicts), figsize=(3.6 * len(evicts), 3.0 * len(datas) + 1.0),
                             squeeze=False)
    for r, d in enumerate(datas):
        row_tops = []
        for c, e in enumerate(evicts):
            ax = axes[r][c]
            style_axes(ax)
            sub = sub_all[(sub_all["data"] == d) & (sub_all["EVICT"] == e)].set_index("mode")
            ms = [m for m in order if m in sub.index]
            bottom = [0.0] * len(ms)
            for p, col in zip(PHASES, PHASE_COLORS):
                vals = [float(sub.loc[m, p]) for m in ms]
                ax.bar(range(len(ms)), vals, bottom=bottom, color=col, width=0.7,
                       edgecolor=SURFACE, linewidth=1)   # 조각 사이 2px 틈
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels(ms, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{d} · {L['evict_n'].format(e=e)}", fontsize=9, color=TEXT)
            if c == 0:
                ax.set_ylabel(L["ms"], fontsize=8, color=TEXT2)
            row_tops.append(max(bottom) if bottom else 0.0)
        for c in range(len(evicts)):
            axes[r][c].set_ylim(0, (max(row_tops) or 1.0) * 1.08)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_COLORS]
    finish(fig, L["t3"].format(t=t), handles, L["phases"], out, note=L["note3"])


def evict_figure(df, L, at, order, styles, out):
    datas = sorted(df["data"].unique())
    t = pick_threads(df, at)
    sub_all = df[df["threads"] == t]
    if sub_all["EVICT"].nunique() < 2:
        print("skip", out, "(삭제 수가 하나뿐)")
        return
    fig, axes = plt.subplots(1, len(datas), figsize=(4.2 * len(datas), 3.6), squeeze=False)
    for c, d in enumerate(datas):
        ax = axes[0][c]
        style_axes(ax)
        sub = sub_all[sub_all["data"] == d]
        for m in order:
            s = sub[sub["mode"] == m].sort_values("EVICT")
            if s.empty:
                continue
            col, mk, ls = styles[m]
            ax.plot(s["EVICT"], s["step_ms"], color=col, marker=mk, linestyle=ls, linewidth=2, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1)
        ax.set_xticks(sorted(sub["EVICT"].unique()))
        ax.set_title(d, fontsize=9, color=TEXT)
        ax.set_xlabel(L["evict"], fontsize=8, color=TEXT2)
        if c == 0:
            ax.set_ylabel(L["step_ms"], fontsize=8, color=TEXT2)
        ax.set_ylim(bottom=0)
    finish(fig, L["t5"].format(t=t), mode_handles(order, styles), order, out)


def speedup(s, sub):
    base = sub[(sub["mode"] == "serial") & (sub["threads"] == 1)]
    if base.empty:
        return None
    return float(base["step_ms"].iloc[0]) / s["step_ms"]


def summary(df, order, out):
    lines = ["| 데이터 | 삭제 | 방식 | 1스레드 ms | 가장 빠른 ms (스레드) | serial 1스레드 대비 | 락 대기 % (그때) |",
             "|---|---|---|---|---|---|---|"]
    for d in sorted(df["data"].unique()):
        for e in sorted(df["EVICT"].unique()):
            sub = df[(df["data"] == d) & (df["EVICT"] == e)]
            base = sub[(sub["mode"] == "serial") & (sub["threads"] == 1)]
            b = float(base["step_ms"].iloc[0]) if not base.empty else None
            for m in order:
                s = sub[sub["mode"] == m]
                if s.empty:
                    continue
                one = s[s["threads"] == 1]
                best = s.loc[s["step_ms"].idxmin()]
                one_ms = f"{float(one['step_ms'].iloc[0]):.3f}" if not one.empty else "-"
                x = f"x{b / best['step_ms']:.2f}" if b else "-"
                lines.append(f"| {d} | {e} | {m} | {one_ms} | {best['step_ms']:.3f} ({int(best['threads'])}) | {x} | "
                             f"{best['lockwait_pct']:.1f} |")
    text = "\n".join(lines) + "\n"
    pathlib.Path(out).write_text(text, encoding="utf-8")
    print(text)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description="run_all.bat 결과 그래프")
    ap.add_argument("csv", nargs="*", help="CSV 파일 (기본: results/all_scale.csv, results/all_mixed.csv)")
    ap.add_argument("--out", default=str(ROOT / "results" / "plots"), help="저장 폴더")
    ap.add_argument("--at", type=int, default=8, help="단계 분해·삭제 그래프에 쓸 스레드 수 (없으면 그 이하 중 가장 큰 값)")
    a = ap.parse_args()
    paths = a.csv or [ROOT / "results" / "all_scale.csv", ROOT / "results" / "all_mixed.csv"]
    df = load(paths)
    if df is None or df.empty:
        sys.exit(f"읽을 CSV 가 없습니다: {[str(p) for p in paths]}  (먼저 scripts\\run_all.bat)")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = setup_fonts()
    order, styles = mode_styles(set(df["mode"]))

    ph = df[df["sched"] == "phase"]
    if not ph.empty:
        grid_figure(ph, L, "step_ms", L["step_ms"], L["t1"], order, styles, out / "1_step_ms.png")
        if (ph["mode"] == "serial").any():
            grid_figure(ph, L, None, L["speedup"], L["t2"], order, styles, out / "2_speedup.png",
                        transform=speedup, hline=1.0)
        else:
            print("skip 2_speedup.png (serial 결과 없음)")
        breakdown_figure(ph, L, a.at, order, out / "3_breakdown.png")
        grid_figure(ph, L, "lockwait_pct", L["lockwait"], L["t4"], order, styles, out / "4_lockwait.png")
        evict_figure(ph, L, a.at, order, styles, out / "5_evict.png")
        summary(ph, order, out / "summary.md")
    mx = df[df["sched"] == "mixed"]
    if not mx.empty:
        grid_figure(mx, L, "step_ms", L["step_ms"], L["t6"], order, styles, out / "6_mixed.png")


if __name__ == "__main__":
    main()
