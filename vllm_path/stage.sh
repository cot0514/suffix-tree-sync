#!/bin/bash
# 진짜 SuffixDecodingCache 를 수정 없이 부르기 위한 임시 폴더를 만든다 (심볼릭 링크만 사용).
# arctic_inference 패키지의 __init__ 이 비어 있어서 torch·vLLM 없이 import 된다.
# 필요: Arctic 소스(~/ArcticInference)와, 그 csrc/suffix_decoding 을 빌드한 폴더(~/arctic_c_build)
set -e
ARCTIC_SRC="${ARCTIC_SRC:-$HOME/ArcticInference}"
ARCTIC_BUILD="${ARCTIC_BUILD:-$HOME/arctic_c_build}"
STAGE="${ARCTIC_STAGE:-$HOME/arctic_stage}"

rm -rf "$STAGE"
mkdir -p "$STAGE/arctic_inference/suffix_decoding"
: > "$STAGE/arctic_inference/__init__.py"
SRC_DIR="$ARCTIC_SRC/arctic_inference/suffix_decoding"
DST_DIR="$STAGE/arctic_inference/suffix_decoding"
ln -s "$SRC_DIR/__init__.py" "$DST_DIR/__init__.py"
ln -s "$SRC_DIR/cache.py" "$DST_DIR/cache.py"
ln -s "$ARCTIC_BUILD"/_C*.so "$DST_DIR/"
echo "staged: $STAGE"
ls -l "$DST_DIR"
