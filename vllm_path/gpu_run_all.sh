#!/bin/bash
for B in "$@"; do
  echo "== B=$B"
  bash ~/suffix-tree-sync/vllm_path/gpu_run.sh Qwen/Qwen2.5-1.5B $B
done
