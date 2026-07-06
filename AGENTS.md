# AGENTS.md

本文件是本 vLLM fork 的队内协作手册。它作为AI Agent 工具在开发时的约束。

## 仓库身份

这是 2026 先导杯「基于国产加速卡的千问大模型推理服务优化」的 vLLM 开发仓库。

## 概念说明
算力节点：是开发时使用的docker容器，需要经过登录节点->计算节点->容器三次跳转。
可以通过运行时环境变量开关某项特性。
测评机/测评平台：是官方提供的吞吐成绩测评系统，其会从git仓库拉取vllm，然后自动
启动容器进行测评。和我们的算力节点不是一个系统。不能向测评机注入运行时自定义环境
变量。可以通过build_vllm.sh引入编译时环境变量，`build_vllm.sh`脚本开发时不
使用。

## 分支语义

- `v0.18.1`：SourceFind/OpenDAS 基线分支，跟踪 `upstream/v0.18.1`。不要直接在此分支
  上做开发提交。
- `main`：稳定比赛提交线。只合入已经过基本验证、能构建 wheel、能解释合规性的改动。
- `develop`：日常开发集成线。默认在这里接收优化 patch、诊断日志和构建修复。
- `competition-ci`：弃用。历史上只用于 CI/CD 把 vLLM 源码同步到 SCNet 无卡通道的共享
  home 或文件夹，不作为评测线。
- `exp/xxx`: 实验分支。exp线的所有代码均不作为正式提交代码。用户可以要求从其上
  cherry-pick 对应的 commit 到远端仓库。
- `submit/vN`: submit的各条线，以重放最小优化patch为主。除非用户明确说明，否则以submit
开头的分支代码禁止修改。


## Remote 规则

- `upstream` 指向 SourceFind/OpenDAS，只用于 fetch，不用于 push。
- 用户要求搜索上游时，应该搜索vllm github，而不是搜索upstream。
- `origin` 应指向队伍自己的 GitHub 仓库。
- `target` 指向队伍的gitlab仓库，任何时候只接受submit开头的分支。
- 禁止由 Agent 向 target 上传代码。必要时必须停止工作并说明 target 必须由用户手动 push。
- 官方自动评测/目标提交仓库如果要求从 GitLab 拉取并只接受 HTTPS 提交，则对应 remote
  必须使用 HTTPS URL；不要配置为 SSH URL，也不要依赖本地 `~/.ssh` key 作为该仓库的提交凭据。

## 构建契约

官方测试调试文档要求 vLLM 修改后能够重新构建 wheel，并通过 wheel 安装路径验证：

```bash
python setup.py bdist_wheel
pip install dist/vllm-*.whl --no-deps
```

如果同容器已经安装过一次版本号相同的vllm，需要使用force install

有效提交不应只靠 `PYTHONPATH`、editable install 或本地未提交文件生效。开发阶段可以用
快速方式调试，但进入 `main` 前必须能走 wheel 构建路径。

构建和运行环境以比赛容器为准：`Python 3.10.12`、`PyTorch 2.10.0`、`transformers
5.5.0`、DTK/HIP/DCU 栈。

每个 fresh DCU 容器或新 shell 中，执行构建、wheel 安装验证、import smoke、Triton JIT
probe、启动 `vllm serve`、跑 benchmark 或 `hipprof` 前，必须先加载 DTK 运行环境，不要依赖
镜像默认环境或继承的 shell 状态：

```bash
set +u
source /opt/dtk/env.sh
set -u
export LD_LIBRARY_PATH=/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/dcc/lib:${LD_LIBRARY_PATH:-}
```

不使用登录节点和本地机器做任何环境检查，不假设本地机器提供相关能力。

GitHub Actions 公共 runner 只运行 `Source Smoke`：检查仓库必要文件和 Python 源码语法。
它不安装 PyTorch、不构建 wheel、不 import vLLM、不编译 DTK/HIP/DCU 扩展，也不能作为
性能结论。正式比赛 wheel 必须仍在官方容器/算力节点里构建和验证。

patch的意思是，对远程工作区恢复到正确状态后，做小的优化改动，然后增量编译wheel。禁止安装wheel
后通过修改site-packages的方式来实现自以为是的patch。

## Qwen3.5 模型结构速查

分析 Qwen3.5-27B 性能、attention、Triton tile、KV cache 或 int8 路径前，必须先按本节模型
结构判断，不要套用普通 dense Qwen/Llama 的 Q/K/V 形状经验。

- 文本侧默认配置见 `vllm/transformers_utils/configs/qwen3_5.py`：`hidden_size=4096`、
  `num_hidden_layers=32`、`num_attention_heads=16`、`num_key_value_heads=4`、
  `head_dim=256`、`intermediate_size=12288`。
- Qwen3.5 是 hybrid decoder。默认 `full_attention_interval=4`，因此 0-based 层号
  `3,7,11,15,19,23,27,31` 是 `full_attention`，其余 24 层是 `linear_attention`
  / Gated Delta Net，不是每层都跑 full self-attention。
- full attention 复用 `Qwen3NextAttention`。它是 GQA：全局 `16` 个 query heads、`4` 个
  KV heads、`head_dim=256`，所以 TP=1 时 `q_size=4096`、`kv_size=1024`、
  `num_queries_per_kv=4`。进入 Triton unified attention 时应按
  `num_query_heads=16/tp`、`num_kv_heads=max(1, 4/tp)`、`HEAD_SIZE=256` 理解。
- full attention 的 `qkv_proj` 默认还带 `attn_output_gate=True`。TP=1 时 packed 输出不是
  简单 `Q+K+V`，而是 `q + gate + k + v = 4096 + 4096 + 1024 + 1024`。代码里对应
  `q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)`，随后
  `q_gate` 再拆成 `q` 和 `gate`。
- linear attention / GDN 路径也不是普通 Q/K/V：默认 `linear_num_key_heads=16`、
  `linear_num_value_heads=32`、`linear_key_head_dim=128`、`linear_value_head_dim=128`，
  因此 TP=1 时 `key_dim=2048`、`value_dim=4096`。Qwen3.5 覆写的
  `create_qkvz_proj` 输出 `[key_dim, key_dim, value_dim, value_dim]`，即
  `q=2048, k=2048, v=4096, z=4096`；`in_proj_ba` 输出 `[num_v_heads, num_v_heads]`。
  这条路径进入 `gdn_attention_core`，不要和 Triton unified full attention 混为一谈。
- 本模型 full attention 的 `head_dim=256`，且比赛路径里 attention/KV page/block 形状可能
  出现 `BLOCK_SIZE=784`。讨论 `triton_unified_attention.py` 的 `TILE_SIZE`、`BLOCK_M`、
  int8 K cache、LDS/stall 或 backend fallback 时，默认先按 `HEAD_SIZE=256` 和实际
  `BLOCK_SIZE=784` 检查，不要假设常见 `head_dim=128` 或 `block_size=16/32/64`。
- 若用户提到 “Triton unified attn” 或 `triton_unified_attention.py`，默认讨论的是 full
  attention 层的 Triton unified attention 路径；不要转到 linear 权重量化、AITER、FlashAttention
  或其它 backend，除非用户明确要求比较。

## 连接容器方式

入口为 `<COMPETITION_LOGIN_HOST>` 的节点称为“登录节点”，之后通过 `squeue` 命令查看正在运行中的
容器。如果 ST 为 PD，启动轮询机制，每 `120` 秒进行一次轮询，直到 ST 变为 R。这一步执行前
需要询问用户，并将 goal 设置为 blocked。nodelist 下形如 `<COMPUTE_NODE>` 的 id 为计算节点 id，
可以通过 ssh 连接。

登录节点的端口为`<COMPETITION_LOGIN_PORT>`，用户名`c118-team`，ssh密钥为 `~/.ssh/<TEMPORARY_CREDENTIAL>`

ssh 连接计算节点后，可以通过 `docker ps` 查看容器列表，`c118` 创建的容器为我们的容器，
可以通过 `docker exec 容器id ...` 来访问。

如果命令过长，必要时编辑 shell 脚本，所有临时的 shell 脚本放置在
`<TEAM_HOME> 下。对于吞吐测试的相关工作文件，放置在
`baseline-runs` 文件夹下；对于 profile 工作，放置在 `profile_runs` 文件夹下。

## 评测与同步边界

比赛算力供给情况：约 `600` 个队伍排队共享约 `144` 张加速卡；每个队伍单次使用额度为
`4` 卡时。

有卡实例时间短。`/public/share` 是公开目录，不再使用；模型、wheel、baseline 记录全部放在
队伍家目录 `<TEAM_HOME>
吞吐 A/B 或快速验证里用户说“跑三档各三组”“三档各 3 组”时，默认含义是跑三档
`4-8K`、`8-16K`、`16-32K`，且每档传给 `run_throughput.sh` 的第二个参数为 `3`
（即 `4-8K3`、`8-16K3`、`16-32K3` 这种短跑），不是每档重复三轮完整数据集。
禁止绕过`run_throughput.sh` 跑自己的吞吐脚本。
任何时候启动 baseline 前先查既有材料，避免重复下载、重复复制、构建或跑长任务：

- 每次开始操作远端 `vllm_cscc` 前，必须先检查工作区状态，检查必须至少包含
  `git status --short`，例如：

  ```bash
  cd <TEAM_HOME>
  git status --short
  git branch --show-current
  git rev-parse HEAD
  ```

  如果发现未预期的本地修改、未跟踪文件、污染 wheel、混入其它实验线路的代码，先停止并询问
  用户如何处理；不要自行 `git reset`、覆盖、删除、继续构建或拿污染源码跑验证。
  如果有正在进行的 goal，要求把 goal 设置为 blocked。
- 远端各工作区 `dist/` 目录下已经存在的所有 `vllm-*.whl` 默认都不可信，不能直接安装、
  提交或作为验证对象。需要 wheel 时，从已确认干净的源码状态重新构建，并把源码 commit、
  `git status --short` 输出、构建命令和 wheel 路径写入本次 run 目录。
  构建wheel后，立刻告知用户wheel的编译路径。
- 模型：`<TEAM_HOME> 已有完整副本；服务和脚本直接使用这个
  home 路径，也不要网络下载。
- **Qwen3.5 权重加载必须提前准备 `runai-model-streamer`**：每个 fresh DCU 容器在启动
  `start_vllm.sh`、warmup、baseline、profile 或任意 `vllm serve` 前，先安装并验证
  `runai-model-streamer`，不要等日志卡在 safetensors shard 读取时再处理。Qwen3.5-27B
  默认应使用 vLLM 的 Run:ai 流式加载路径读取本地 safetensors：

  ```bash
  pip install 'runai-model-streamer[s3,gcs,azure]>=0.15.7'
  python - <<'PY'
  import runai_model_streamer
  print("runai_model_streamer", getattr(runai_model_streamer, "__version__", "unknown"))
  PY
  ```

  启动服务时默认在官方 `vllm serve` 参数中加入：

  ```bash
  --load-format runai_streamer
  ```

  可按容器 CPU/文件系统吞吐 A/B 调整并发，例如：

  ```bash
  --model-loader-extra-config '{"concurrency":16}'
  ```

  这一步是冷启动工作流的一部分，不是异常恢复步骤。若容器不能联网安装，必须提前把
  `runai-model-streamer` wheel 或可复现安装源放到 `<TEAM_HOME>
  包版本、wheel 路径写入本次 run 目录。除非用户明确要求做 `auto` loader 对照实验，否则不要
  因为依赖缺失就静默退回默认 shard 逐片读取。
- 容器 `/root` 不作为持久工作区。不要把模型权重、wheel、源码构建缓存或大日志复制到
  `/root`；构建时把 `TMPDIR`/`TMP`/`TEMP` 指到家目录。若已占用 `/root`，任务结束后清理或
  直接释放容器。
- 节点根分区和默认 `/tmp` 可能已满；所有 build、import smoke、Triton JIT、TorchInductor
  和 hipprof 临时数据都必须显式落到队伍 home，例如：
- 启动服务时，应该把MODEL_DIR设置为`<TEAM_HOME>

  ```bash
  export HOMEBASE=<TEAM_HOME>
  export RUNTIME_CACHE="$HOMEBASE/<TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE><TEAM_WORKSPACE>"
  mkdir -p "$RUNTIME_CACHE/tmp" "$RUNTIME_CACHE/torchinductor" \
    "$RUNTIME_CACHE/triton" "$RUNTIME_CACHE/xdg" "$RUNTIME_CACHE/pycache"
  export TMPDIR="$RUNTIME_CACHE/tmp"
  export TMP="$TMPDIR"
  export TEMP="$TMPDIR"
  export XDG_CACHE_HOME="$RUNTIME_CACHE/xdg"
  export TORCHINDUCTOR_CACHE_DIR="$RUNTIME_CACHE/torchinductor"
  export TRITON_CACHE_DIR="$RUNTIME_CACHE/triton"
  export PYTHONPYCACHEPREFIX="$RUNTIME_CACHE/pycache"
  ```

  跑 `hipprof` 时也要加 `-d "$RUNTIME_CACHE/tmp"`，不要让 hipprof 使用默认 `/tmp`。DTK 26.04
  下 `hipprof --pmc --pmc-type 3 -o <prefix>` 的 PMC 结果可能直接写成 `<prefix>.csv`，不要只
  按旧文档查 `pmc_results_*`。

  只有 profiler 正常退出后的 db 才算有效产物。
  只要存在 -journal / -wal / -shm 或 profiler 非 0 退出，就视为无效 profile。不要对原始未确认状
  态的 db 执行 sqlite3 读取，因为这一步本身就可能触发 recovery。

  整个工作流程开始时，告知用户本次run文件夹的路径。

## profile 工作流的使用

比赛只关心服务期的吞吐。vLLM 的生命周期是：启动 -> warmup2 -> bench1 -> warmup2 -> bench2
-> ...，profile 通常只抓取“服务期”的数据。

vLLM 启动时，如果用户明确要求跑 profile，首先使用 hipprof 包裹 `start_vllm.sh` 或手动构造的
vLLM 启动参数。官方 `run_throughput.sh` 脚本会自动 warmup，在做 profile 前，必须手动
warmup=2，最后开启抓取并发起具体 bench 请求。

- **hipprof 与 session **：

  `--trace-off --session <name>` 方案下，hipprof 将采集数据写入 SQLite db 的 WAL/journal，
  **只有 hipprof 进程正常退出时才会 commit**。如果 hipprof 被 `kill`、崩溃、或 OOM，
  SQLite 回滚所有未提交数据，db 缩回初始空模板大小（如 12KB）。

  **已验证的事实：**
  - `--session-client --exit` 仅对 `--master` 集群模式生效，单节点 session 无视此命令
  - session `--stop` + `--flush` 后 db 文件确实增长（如 90MB），但数据仍在未提交事务中
  - 在 hipprof 存活期间 `cp` 备份的 db 文件，因缺失 journal/WAL 一致性，`hipprof --db` 报
    `not valid trace data`，sqlite3 `.dump` 报 `database disk image is malformed`
  - 备份连同 `-journal` 文件一起复制也无济于事——事务未提交，数据无法恢复

  **正确做法（二选一）：**

  A) **已验证正确做法（20260627，终版）**：hipprof 直接包裹 `start_vllm.sh`，
     不加 `--trace-off`，不加 `--exit-cleanup`。跑完 benchmark 后
     **杀 vLLM（不是 hipprof）**：`kill <vllm_pid>`。vLLM 退出 → bash 退出 →
     hipprof 检测到子进程树结束 → hipprof 自然退出 → commit db → 导出 CSV。
     导出：`rm -f <db>.db-journal && hipprof --kernel-stack --db <db>`。
     **绝对禁止 kill hipprof 进程本身，必须让 hipprof 自行检测子进程退出。**

  B) **保持 vLLM 存活时用 Python 级探针**：如果必须复用已加载权重的 vLLM（不杀进程），
     走 `VLLM_FILL_CALLSITE_PROBE=1` 之类 TorchDispatchMode 探针，在 Python 层拦截
     fill/zero 调用栈。探针结果写入 JSON，不依赖 hipprof db 提交机制。

  **禁止行为：**
  - 禁止对 session 模式下的 hipprof 使用 `kill -9`（SIGKILL 必然回滚）
  - 禁止假设 `--flush`  == 数据已持久化（flush 只写 WAL，不 commit）
  - 禁止 kill hipprof 进程本身（包括 SIGTERM），必须让 hipprof 自行检测子进程退出
  - 禁止在 `--trace-off --session` 跑完后直接重启容器（容器重启 = db 丢失）

- **复用 temp_shell 下已有 .sh 文件前，必须先 Read 确认内容。**

6月29日17:00以前的所有profile数据不完全可信，会混入整个vllm
生命周期。

FillFunctor<int>不是服务期热点。
所有发现 FillFunctor<int> 是第一热点的 profile 文件无效。

本地工作区内没有临时文件夹，禁止在本地项目工作区内创建“temp”
“tmp”等文件夹然后在其中放置临时文件。
