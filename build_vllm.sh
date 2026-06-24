#!/usr/bin/env bash
set -euo pipefail

# 用于在编译期注入 vllm/v1/attention/ops/triton_unified_attention.py 的默认 BLOCK_M；通过将 BLOCK_M 调到 64 来优化吞吐。
export TRITON_UNIFIED_ATTN_BLOCK_M=64

python setup.py bdist_wheel
