---
name: scnet-vllm-baseline
description: Run, resume, validate, or summarize clean Qwen3.5/vLLM baselines on SCNet DCU containers. Use when Codex needs to poll Slurm, enter a running SCNet container, install an existing vLLM wheel, start the official vLLM service, run throughput or accuracy scripts, save baseline artifacts, update baseline_index, or check whether a baseline result is valid.
---

# SCNet vLLM Baseline

Use this skill for clean baseline work only. For profiling and kernel hotpath analysis, use `dcu-hipprof-profile`. For AITER, FP8 KV cache, DUMMA, or other experimental paths, keep the experiment clearly separate from the clean baseline run.

## Constants

Use the team home path as the persistent workspace:

```bash
HOMEBASE=<TEAM_HOME>
MODEL=<TEAM_HOME>
```

Do not use `/public/share`. Do not store model weights, build caches, wheel archives, or large logs under `/root`. The official PDF may mention copying the model to `/root`; the team rule overrides that for this repo.

Always bypass local host-key state for E-Shell:

```bash
ssh -F /dev/null <TEMPORARY_LOGIN_COMMAND> -p <COMPETITION_LOGIN_PORT> \
  -o LogLevel=ERROR -o ConnectTimeout=20 \
  -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no \
  c118-team@<COMPETITION_LOGIN_HOST>
```

## Workflow

### 1. Read Current Constraints

Before starting, inspect the repo `AGENTS.md` for current team rules and paths. If it conflicts with `competition.md`, follow `AGENTS.md`.

Check existing baseline state first:

```bash
sed -n '1,220p' "$HOMEBASE/<TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE>" 2>/dev/null || true
find "$HOMEBASE/<TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE>" -maxdepth 3 -type f -name result.json -print 2>/dev/null
```

Do not use these as canonical baseline sources:

```text
latest/test/
baseline_results/
early proxy-failed runs
```

### 2. Poll Slurm Only Until Running

On the E-Shell login node:

```bash
/opt/gridview/slurm/bin/squeue -u "$USER" -o "%.18i %.9P %.40j %.8u %.2t %.10M %.6D %R"
/opt/gridview/slurm/bin/scontrol show job <JOBID>
```

If the job is pending, keep polling at the requested interval. Polling only waits for pending -> running. Do not run vLLM, benchmark, profiler, `hy-smi`, or performance checks on the login node.

When status is `R`, enter the compute node/container. Prefer the container or SSH id shown by the web page; otherwise use the `NODELIST` / `BatchHost` from Slurm:

```bash
ssh <node-or-container-id>
```

The instance metadata normally lives under:

```text
$HOMEBASE/<TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_PLATFORM_HOME>/instance/ssh/<instance_name>/
```

If direct SSH is unavailable, inspect `_dockerlist_<JOBID>` and use `ai_docker exec` only as a fallback. Its wrapper mishandles complex quoting; write complex commands into a script file and run `bash <script>`.

### 3. Prepare Run Directory And Environment

Inside the DCU container or compute node:

```bash
TS=$(date +%Y%m%d_%H%M%S)
RUN=<TEAM_HOME>

export HOMEBASE=<TEAM_HOME>
export MODEL=<TEAM_HOME>
export TMPDIR="$HOMEBASE/<TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE>"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$RUN" "$RUN/logs" "$TMPDIR"

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
```

Save basic context:

```bash
{
  date
  hostname
  pwd
  python --version
  pip show vllm 2>/dev/null || true
  git rev-parse HEAD 2>/dev/null || true
  env | sort
} > "$RUN/env.log" 2>&1
```

### 4. Install Existing Wheel First

Prefer an existing wheel from home. Do not rebuild unless the user explicitly asks for a clean wheel or no usable wheel exists.

```bash
find "$HOMEBASE/<TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE>" -maxdepth 2 -type f -name 'vllm-*.whl' -print 2>/dev/null | sort
find "$HOMEBASE" -path '*/dist/vllm-*.whl' -type f -print 2>/dev/null | sort | tail -20
```

Install with:

```bash
pip install /path/to/vllm-*.whl --no-deps
```

Record the selected wheel in:

```text
$RUN/commands.log
```

### 5. Locate Official Testdata

Use the official `testdata` scripts:

```text
start_vllm.sh
run_throughput.sh
run_accuracy.sh
```

Find them from the persistent workspace or current source tree:

```bash
find "$HOMEBASE" -maxdepth 5 -type f \( -name start_vllm.sh -o -name run_throughput.sh -o -name run_accuracy.sh \) -print
```

If scripts are missing, restore/download testdata into home, not `/root`.

### 6. Start Clean vLLM Service

Clean baseline must not enable experiment env vars such as AITER, FP8 KV cache, custom attention, or profiling hooks.

Start the official service from the testdata directory, or use the local wrapper only if it is known to preserve the official API/statistics contract. Ensure the model path is `"$MODEL"`.

```bash
cd /path/to/testdata
./start_vllm.sh > "$RUN/logs/vllm_server.log" 2>&1 &
echo $! > "$RUN/vllm_server.pid"
```

Wait for the service on `127.0.0.1:8001`, then run a short OpenAI API sanity request before long benchmarks.

### 7. Run Throughput

Run all official buckets for a full clean baseline:

```bash
cd /path/to/testdata
./run_throughput.sh 4-8K
./run_throughput.sh 8-16K
./run_throughput.sh 16-32K
```

If time is short and the user specifies a bucket, run only that bucket. Preserve the raw `result.json` files and copy or symlink them into:

```text
$RUN/testdata/test/4-8K_throughput/result.json
$RUN/testdata/test/8-16K_throughput/result.json
$RUN/testdata/test/16-32K_throughput/result.json
```

Summarize at least:

```text
completed
output_throughput
total_token_throughput
mean_ttft_ms
p99_ttft_ms
mean_tpot_ms
p99_tpot_ms
duration
```

For 16-32K, compare against the current canonical baseline before judging whether the run is clean or polluted.

### 8. Run Accuracy

Run accuracy after throughput unless the user explicitly prioritizes a shorter gate:

```bash
cd /path/to/testdata
./run_accuracy.sh
```

For a quick sanity gate, run the user-specified subset and clearly mark it as partial. Preserve:

```text
$RUN/testdata/accuracy_debug/
$RUN/ACCURACY_RESULT.md
```

Accuracy failures block using the run as a valid optimization baseline.

### 9. Write Baseline Summary

Create:

```text
$RUN/README.md
$RUN/CLEAN_BASELINE_RESULT.md
$RUN/commands.log
$RUN/env.log
$RUN/logs/vllm_server.log
```

The result summary must state:

```text
status: PASS / FAIL / PARTIAL
wheel:
source revision:
model path:
testdata path:
throughput buckets completed:
accuracy status:
known caveats:
```

If a proxy issue, timeout, server crash, non-clean env var, partial run, or non-official script change occurred, mark the result `FAIL` or `PARTIAL`, not canonical.

### 10. Update Canonical Baseline Index

Only after a clean full run passes the required gate, update:

```text
$HOMEBASE/<TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE>
$HOMEBASE/<TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE>
$HOMEBASE/<TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE>
```

`valid_throughput/` should point to or contain only the validated `result.json` files. Do not update the index with experimental AITER/FP8/profile runs or incomplete buckets unless the file explicitly says it is a partial reference.

### 11. Finish Cleanly

Before releasing the job:

```bash
du -sh /root/* /root/.[!.]* 2>/dev/null | sort -h | tail -30
```

Remove accidental large files from `/root` if needed, but do not delete home wheels, model, baseline records, or team artifacts.

Release the job only when the user asks or when the run is complete and no follow-up work needs the DCU:

```bash
scancel <JOBID>
```

## Reporting

In the final user update, include:

- the run directory;
- pass/fail/partial status;
- per-bucket key throughput numbers;
- accuracy status;
- whether `baseline_index` was updated;
- any reason the run should not be used as canonical.
