#!/bin/bash
# tree_batch 모듈 빌드 (WSL2). 결과: bin/vllm_path/tree_batch*.so
# 필요: ~/venv_arctic (nanobind, numpy 설치됨). 다른 위치면 VENV=경로 bash vllm_path/build.sh
set -e
cd "$(dirname "$0")/.."
VENV="${VENV:-$HOME/venv_arctic}"
source "$VENV/bin/activate"
mkdir -p bin/vllm_build bin/vllm_path
cmake -G Ninja -S vllm_path -B bin/vllm_build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/bin/vllm_path" > /dev/null
cmake --build bin/vllm_build
ls bin/vllm_path
