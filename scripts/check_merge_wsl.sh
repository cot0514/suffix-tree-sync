#!/bin/bash
# 방식 하나를 합칠 때마다 돌리는 점검의 WSL 부분. 보통은 scripts\check_merge.bat 이 부른다.
# 직접 실행 (저장소 폴더에서): bash scripts/check_merge_wsl.sh [tsan]
# 필요: ~/venv_arctic (nanobind, numpy), ~/arctic_stage (vllm_path/stage.sh 로 만듦)
# 저장소를 임시 폴더에 복사해서 빌드하므로 원본에는 아무것도 만들지 않는다.
# 하는 일: 묶음 모듈 빌드 -> 등록된 모든 방식을 run_compare.py --quick (code, chat, 1·4·8스레드)
#          tsan 을 주면 ThreadSanitizer 로 모든 방식을 검사 (phase, 4스레드, 삭제 0·2)
# 종료 코드: 모두 통과 0, 하나라도 실패 1. PASS* 는 표시만 하고 실패로 치지 않는다.
set -u
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
venv="${VENV:-$HOME/venv_arctic}"
stage="${ARCTIC_STAGE:-$HOME/arctic_stage}"
if [ ! -f "$venv/bin/activate" ] || [ ! -d "$stage" ]; then
    echo "[FAIL] 준비 안 됨: $venv 또는 $stage 가 없음"
    exit 1
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
(cd "$repo_root" && tar --exclude=./.git --exclude=./bin --exclude=./results -cf - .) | tar -xf - -C "$work_dir"
cd "$work_dir" || exit 1
find . -name '*.sh' -exec sed -i 's/\r$//' {} +
failed=0

if ! bash vllm_path/build.sh > build.log 2>&1; then
    tail -30 build.log
    echo "[FAIL] vllm_path build"
    exit 1
fi
echo "[ok] vllm_path build"
source "$venv/bin/activate"
modes="$(python -c 'import sys; sys.path.insert(0, "bin/vllm_path"); import tree_batch; print(*tree_batch.modes())')"
if [ -z "$modes" ]; then
    echo "[FAIL] 방식 목록을 읽지 못함"
    exit 1
fi
echo "modes: $modes"

for mode in $modes; do
    for data in code chat; do
        timeout 900 python vllm_path/run_compare.py "data/tokens_$data.bin" --quick --mode="$mode" --threads=1,4,8 > run.log 2>&1
        code=$?
        verdicts="$(grep -oE '(PASS\*? spec [0-9.]+%|PASS|FAIL [a-z@0-9]+|n/a)$' run.log | tr '\n' ';')"
        if [ $code -eq 0 ] && [ -n "$verdicts" ] && ! grep -qE 'n/a$' run.log; then
            echo "[ok] $mode $data: $verdicts"
        else
            echo "[FAIL] $mode $data exit=$code: $verdicts"
            tail -15 run.log
            failed=1
        fi
    done
done

if [ "${1:-}" = "tsan" ]; then
    echo "--- ThreadSanitizer"
    if ! TSAN=1 bash scripts/build.sh > tsan_build.log 2>&1; then
        tail -30 tsan_build.log
        echo "[FAIL] ThreadSanitizer build"
        exit 1
    fi
    for mode in $modes; do
        for data in code chat; do
            for evict in 0 2; do
                # setarch -R: WSL 의 "unexpected memory mapping" 시작 실패를 피하려고 주소 무작위화를 끈다
                timeout 900 setarch "$(uname -m)" -R bin/tree_bench_tsan "data/tokens_$data.bin" --mode="$mode" \
                    --sched=phase --threads=4 --b=32 --n_cached=200 --steps=10 --repeat=1 --ref=0 --evict=$evict > tsan.log 2>&1
                code=$?
                warnings=$(grep -c 'WARNING: ThreadSanitizer' tsan.log)
                if [ $code -eq 0 ] && [ "$warnings" -eq 0 ]; then
                    echo "[ok] tsan $mode $data evict=$evict"
                else
                    echo "[FAIL] tsan $mode $data evict=$evict exit=$code warnings=$warnings"
                    grep -m1 -A25 'ThreadSanitizer' tsan.log
                    failed=1
                fi
            done
        done
    done
fi

exit $failed
