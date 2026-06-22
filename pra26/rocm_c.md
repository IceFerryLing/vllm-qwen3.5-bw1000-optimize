# `_rocm_C` quick check

## 结论

- `_rocm_C` 可以用 DCC 编译并链接成功。
- 编译产物可以通过 overlay 注入 vLLM 包并 `import vllm._rocm_C`。
- 带 overlay 的 4K-8K throughput 跑通，观察上比旧 baseline 快。
- 但本轮服务日志显示主 attention backend 是 `TRITON_ATTN`，不能把加速直接归因到
  `_rocm_C.paged_attention`。

## DCC 编译

参考 DUMMA 的 DCC 路径，关键是显式指定 DCC 和 compiler-rt 库目录：

```bash
source /opt/dtk-26.04-DCC2602-0317/env.sh

cmake -S <TEAM_HOME> \
  -B <TEAM_HOME> \
  -DVLLM_TARGET_DEVICE=rocm \
  -DVLLM_PYTHON_EXECUTABLE="$(command -v python3)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/dtk-26.04-DCC2602-0317/dcc/bin/dcc \
  -DCMAKE_CXX_COMPILER=/opt/dtk-26.04-DCC2602-0317/dcc/bin/dcc \
  -DCMAKE_HIP_ARCHITECTURES=gfx936 \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/dtk-26.04-DCC2602-0317/dcc/lib/clang/17.0.0/lib/linux" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/dtk-26.04-DCC2602-0317/dcc/lib/clang/17.0.0/lib/linux"

cmake --build <TEAM_HOME> \
  --target _rocm_C -j2
```

本轮结果：

```text
[100%] Linking HIP shared module _rocm_C.abi3.so
[100%] Built target _rocm_C
```

产物：

```text
<TEAM_HOME>
```

## Overlay 验证

不污染已安装 wheel 的方式：

```bash
overlay=<TEAM_HOME>
mkdir -p "$overlay"
cp -a /usr/local/lib/python3.10/dist-packages/vllm "$overlay/vllm"
cp <TEAM_HOME> \
  "$overlay/vllm/_rocm_C.abi3.so"

export PYTHONPATH="$overlay:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/opt/dtk-26.04-DCC2602-0317/lib:/opt/dtk-26.04-DCC2602-0317/hip/lib:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}

python3 - <<'PY'
import pathlib
import vllm
print("vllm_path", pathlib.Path(vllm.__file__).parent)
import vllm._rocm_C
print("import_overlay_rocm_C=OK")
PY
```

本轮验证输出：

```text
vllm_path <TEAM_HOME>
import_overlay_rocm_C=OK
```

## 4K-8K 观测

带 overlay 启动服务后，4K-8K throughput 跑通：

```text
Successful requests:                     50
Failed requests:                         0
Benchmark duration (s):                  1022.63
Total input tokens:                      315769
Total generated tokens:                  12463
Output token throughput (tok/s):         12.19
Peak output token throughput (tok/s):    15.00
Total token throughput (tok/s):          320.97
P99 TTFT (ms):                           4791.66
P99 TPOT (ms):                           69.12
P99 ITL (ms):                            70.03
P99 E2EL (ms):                           74739.14
```

对应 run 目录：

```text
<TEAM_HOME>
```

租期结束后节点拒绝进入：

```text
Access denied by pam_slurm_adopt: you have no active jobs on this node
```

因此 result 文件未能二次核对，但客户端 stdout 已完整返回 `rc=0`。

## 归因边界

服务日志显示：

```text
Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN'].
```

所以这轮只能说明：

- ROCm/DCU 平台路径有效；
- DCC `_rocm_C` 编译和 import 有效；
- 带 `_rocm_C` overlay 的 4K-8K run 观察上更快。

不能说明：

- 主 attention 已经命中 `_rocm_C.paged_attention`；
- 性能提升一定由 `_rocm_C` kernel 带来。

下一次要坐实归因，做同容器 A/B：

```text
A: 原 wheel，不带 overlay
B: 同参数，带 _rocm_C overlay
```

并给这些入口加一次计数日志：

```text
vllm._custom_ops.paged_attention_rocm
vllm._custom_ops.LLMM1
vllm._custom_ops.wvSplitK
vllm._custom_ops.wvSplitKrc
vllm._custom_ops.wvSplitKQ
```
