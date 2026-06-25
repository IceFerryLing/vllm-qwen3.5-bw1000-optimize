#!/usr/bin/env bash
set -euo pipefail

# Compile-time default for Triton unified attention BLOCK_M.
export TRITON_UNIFIED_ATTN_BLOCK_M=64

python setup.py bdist_wheel
