---
name: dcu-hipprof-profile
description: Profile vLLM/Qwen inference on国产 DCU/HIP nodes with hipprof. Use when Codex needs to plan, run, or summarize short hipprof profiling slices for vLLM on BW1000/gfx936 or similar DCU environments; identify prefill/decode hot kernels; distinguish Triton, AITER, vLLM ROCm C++ kernels, and framework fallback; or write profiling artifacts and hotpath conclusions.
---

# DCU hipprof Profile

Use this skill to collect low-cost, compliance-safe profiling evidence after a runnable baseline or accuracy check. Performance conclusions must come from DCU/container nodes, not login nodes.

## Container Setup

In each fresh DCU container, create the LLVM shared-library symlink before running `hipprof` so `hipprof` can find `libLLVM-17git.so`:

```bash
mkdir -p /opt/dtk/llvm/lib
ln -sf /opt/dtk-26.04-DCC2602-0317/dcc/lib/libLLVM-17git.so \
  /opt/dtk/llvm/lib/libLLVM-17git.so
ls -l /opt/dtk/llvm/lib/libLLVM-17git.so
```

Run this once after every new container start. The symlink is container-local and may not persist across container restarts.

## Workflow

Follow this order:

```text
dumma_probe -> accuracy -> short profile
```

Do not profile a full baseline by default. Use short representative slices:

- Prefill-heavy: long prompt, small output, such as 16K input and 16-64 output tokens.
- Decode-heavy: medium prompt, longer output, such as 4K-8K input and 512-1024 output tokens.
- Slow bucket sample: one short slice from the slowest bucket, such as 16K-32K.

Each profile run should answer:

- What are the top prefill kernels?
- What are the top decode kernels?
- Is attention using Triton, AITER, vLLM ROCm C++ kernels, or fallback?
- Are GEMM, attention, RoPE, RMSNorm, KV cache copy, allocator, or host API calls the main cost?
- Is there abnormal `hipMemcpy*`, `hipDeviceSynchronize`, allocator, stream, or launch overhead?

## Artifact Layout

Keep artifacts together under:

```text
<TEAM_HOME>
```

Suggested files:

```text
README.md
env.log
vllm_server.log
profile_command.txt
hip_stats*
hip_trace*
kernel_summary.txt
hotpath_conclusion.md
```

Record the exact vLLM command, benchmark command, model path, input/output shape, concurrency, device IDs, relevant env vars, wheel/source revision, and whether the run is prefill-heavy or decode-heavy.

## Proxy Hygiene

Before running localhost benchmark clients, bypass proxies:

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

## Low-Cost Stats First

Start with stats before collecting full timelines:

```bash
mkdir -p <TEAM_HOME>

hipprof --hip-trace --stats --show-pid --devices 0 \
  -o <TEAM_HOME> \
  <short benchmark command>
```

Use `--stats` to identify top kernels/API costs without full trace output. Use `--show-pid` to separate server and client processes. Use `--devices 0` to reduce multi-card noise and trace size unless the question requires multiple devices.

## Timeline Trace

After stats identifies a useful slice, collect a small timeline:

```bash
hipprof --hip-trace --show-pid --devices 0 --output-type 2 \
  -o <TEAM_HOME> \
  <short benchmark command>
```

Add `--hsa-trace` only when HIP trace is insufficient for kernel/HSA behavior. Add `--rccl-trace` only for multi-card communication questions.

## Long-Running vLLM Server

Do not trace a vLLM server from process start to shutdown. Use hipprof session control to capture only the request window:

```bash
hipprof --hip-trace --trace-off --session 936 --show-pid --devices 0 \
  -o <TEAM_HOME> \
  <vllm server command>
```

Around one short benchmark slice:

```bash
hipprof --session-client 936 --start
<run one short request or benchmark slice>
hipprof --session-client 936 --stop
```

## Backend Classification

Classify the observed path from kernel names, logs, and host stack evidence:

- Triton: kernel/module names mention `triton`, or logs/imports point to `aiter.ops.triton.*` or `vllm/third_party/triton_kernels`.
- AITER: kernel/module/log evidence points to AITER ROCm ops.
- vLLM ROCm C++: kernel names or stack traces point to `vllm._rocm_C` or `csrc/rocm/*`.
- Fallback: generic framework kernels dominate, especially many small aten-like elementwise, copy, cast, indexing, softmax, or allocation operations.

If evidence is unclear, write `unclear` and preserve the trace/stat files instead of overclaiming.

## Conclusion Template

Write `hotpath_conclusion.md` like this:

```text
Workload:
- input/output:
- concurrency:
- model:
- vLLM command:

Top kernels:
1. ...
2. ...
3. ...

Backend evidence:
- Triton: yes/no/unclear, evidence ...
- AITER: yes/no/unclear, evidence ...
- _rocm_C: yes/no/unclear, evidence ...
- fallback: yes/no/unclear, evidence ...

Main bottleneck:
- prefill/decode/API overhead/copy/allocator/unknown

Next optimization target:
- ...
```

## Guardrails

- Use profiling to guide mathematically equivalent optimizations only.
- Do not infer performance from login nodes, source smoke checks, or local macOS runs.
- Do not change model weights, tokenizer, prompts, output lengths, benchmark routes, or official statistics.
- Avoid long traces unless a short stats run has already identified a concrete need.
