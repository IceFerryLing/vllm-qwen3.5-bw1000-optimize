# AGENTS.md

本文件是本 vLLM fork 的队内协作手册。它适用于所有人在本仓库里的代码修改、AI
辅助开发、CI/CD 同步和提交准备。

## 仓库身份

这是 2026 先导杯「基于国产加速卡的千问大模型推理服务优化」的 vLLM 开发仓库。

当前基线来自官方测试调试 PDF 指定的 SourceFind/OpenDAS 仓库：

```text
upstream: http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git
branch:   v0.18.1
commit:   fa718036b [Arch] Support bmz and nmz
```

## 分支语义

- `v0.18.1`：SourceFind/OpenDAS 基线分支，跟踪 `upstream/v0.18.1`。不要直接在此分支
  上做开发提交。
- `main`：稳定比赛提交线。只合入已经过基本验证、能构建 wheel、能解释合规性的改动。
- `develop`：日常开发集成线。默认在这里接收优化 patch、诊断日志和构建修复。
- `competition-ci`：无卡共享文件夹同步线。它只用于 CI/CD 把 vLLM 源码同步到 SCNet
  无卡通道的共享 home 或文件夹，供后续容器/算力节点取用；它不是评测线。

不要在 `competition-ci` 上引入“只有同步环境需要”的代码行为变化。如果同步脚本需要
路径、host、目标目录或开关，放到脚本参数、CI 变量或文档里，不要硬编码进 vLLM runtime。

## Remote 规则

- `upstream` 指向 SourceFind/OpenDAS，只用于 fetch，不用于 push。
- `origin` 应指向队伍自己的 GitHub 仓库。
- 如果 `upstream` 仍有 push URL，先禁用：

```bash
git remote set-url --push upstream DISABLED
```

GitHub 上创建 `origin` 仓库时不要用 README、LICENSE 或 `.gitignore` 初始化。保持空仓库，
然后从本地 push `main`、`develop`、`competition-ci` 和必要 tag。

## 构建契约

官方测试调试文档要求 vLLM 修改后能够重新构建 wheel，并通过 wheel 安装路径验证：

```bash
python setup.py bdist_wheel
pip install dist/vllm-*.whl --no-deps
```

有效提交不应只靠 `PYTHONPATH`、editable install 或本地未提交文件生效。开发阶段可以用
快速方式调试，但进入 `main` 前必须能走 wheel 构建路径。

构建和运行环境以比赛容器为准：`Python 3.10.12`、`PyTorch 2.10.0`、`transformers
5.5.0`、DTK/HIP/DCU 栈。登录节点或本地 macOS 上的 import/test 只能做工程检查，不能
作为性能或兼容性结论。

GitHub Actions 公共 runner 只运行 `Source Smoke`：检查仓库必要文件和 Python 源码语法。
它不安装 PyTorch、不构建 wheel、不 import vLLM、不编译 DTK/HIP/DCU 扩展，也不能作为
性能结论。正式比赛 wheel 必须仍在官方容器/算力节点里构建和验证。

## 评测与同步边界

比赛算力供给情况：约 `600` 个队伍排队共享约 `100` 张加速卡；每个队伍单次使用额度为
`2` 卡时。

有卡实例时间短。`/public/share` 是公开目录，不再使用；模型、wheel、baseline 记录全部放在
队伍家目录 `<TEAM_HOME> baseline 前先查既有材料，避免重复下载、重复
复制、构建或跑长任务：

- 模型：`<TEAM_HOME> 已有完整副本；新有卡容器首次跑 baseline
  前复制到 `/root/Qwen3.5-27B` 加速读取，服务和脚本使用 `/root/Qwen3.5-27B`。同一容器复制
  完成后不要再复制，也不要网络下载。
- wheel：优先从 `<TEAM_HOME> 或源码 `dist/` 安装已有 wheel，不要重编译。
- baseline：统一入口是 `<TEAM_HOME>
  吞吐只引用 `baseline_index/valid_throughput/*/result.json`，不要直接从 `latest/test/` 取数；
  其中有早期代理失败结果。`baseline_results/` 是早期 root-owned 空目录，不要使用。
- benchmark 客户端连本机 vLLM 前必须设置 `NO_PROXY/no_proxy=127.0.0.1,localhost` 并清掉
  `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY`，否则 localhost 请求可能走 squid 代理得到假
  `503` 结果。
- 新 baseline 的结论、日志摘要、关键命令和结果文件写到
  `<TEAM_HOME>
  `<TEAM_HOME> 和
  `baseline_index/valid_throughput/`。

SCNet 当前已知 SSH 入口是无卡登录/传文件通道：

```text
ssh <TEMPORARY_LOGIN_COMMAND> -p <COMPETITION_LOGIN_PORT> c118-team@<COMPETITION_LOGIN_HOST>
```

该 E-Shell host key 可能随实例变化；登录时始终绕过本地 `known_hosts` 校验，避免动态入口
被旧记录卡住：

```bash
ssh <TEMPORARY_LOGIN_COMMAND> -p <COMPETITION_LOGIN_PORT> \
  -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no \
  c118-team@<COMPETITION_LOGIN_HOST>
```

部分网页容器实例背后由 Slurm/Gridview 调度系统启动。调度入口可以选择个人用户名，也可以
选择队伍用户名；队伍用户名通常是一次性的，可以按需记录，个人用户名不要写入仓库文档、
脚本或提交信息。入口示例：

```text
<TEMPORARY_LOGIN_COMMAND>
```

定位实例对应作业和节点：

```bash
/opt/gridview/slurm/bin/squeue -u "$USER" -o "%.18i %.9P %.40j %.8u %.2t %.10M %.6D %R"
/opt/gridview/slurm/bin/scontrol show job <JOBID>
```

SCNet 官方文档说明：E-Shell 默认进入登录节点；`squeue` 中状态 `R` 表示节点已运行，进入
计算节点可直接 `ssh <节点名>`，容器/SSH 页面给出 id 或别名时优先 `ssh <id>`。轮询只用于
等待实例/容器从 pending 进入 running，不负责后续登录和执行。`scancel <JOBID>` 用于释放
作业。`squeue` 的 `NODELIST`、`scontrol show job` 的 `NodeList` / `BatchHost` 是计算节点。
作业目录通常位于：

```text
<TEAM_HOME>
```

作业目录中的 `_dockerlist_<JOBID>` 记录容器名、容器 ID、DCU 编号、挂载和容器 IP。
`ai_docker exec` 只作为直接 SSH 不可用时的兜底方式，并且它的 wrapper 会拆复杂引号；复杂
命令先写成脚本文件，再执行 `bash <script>`。

该通道只用于传文件、同步源码、保存 wheel、下载模型和汇总日志。不要在该通道上跑：

- vLLM 服务；
- 吞吐或精度 benchmark；
- GPU/DCU profiler；
- DCU kernel 性能测试；
- 任何需要 `hy-smi` / `rocm-smi` / `rocminfo` / 真实 DCU 的结论。

性能结论必须来自有 DCU 的比赛容器或算力节点。官方本地调试入口以 `testdata` 中脚本为准：

```text
start_vllm.sh
run_throughput.sh
run_accuracy.sh
```

吞吐关注 output tokens/s、TTFT P99、TPOT P99。精度关注 LongBench/RULER 相关任务输出
和后处理结果。调试集结果只作为自身 baseline，最终分数以评测机和官方 baseline 为准。

当前比赛卡按 BW1000 / `gfx936` 处理。新的 HIP/DUMMA/`_rocm_C` probe、编译命令和
CMake 目标默认使用 `--offload-arch=gfx936`。
DUMMA probe 若用 DCC 编译，链接时显式加
`-L/opt/dtk-26.04-DCC2602-0317/dcc/lib/clang/17.0.0/lib/linux`。

AITER unified attention 是显式实验路径，不要默认偷偷打开。已知 Qwen3.5 路径要同时使用：

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=1
vllm serve /root/Qwen3.5-27B ... --block-size 16
```

原因：Qwen3.5/mamba 对齐会把 attention page size 设为 `784`；AITER unified 默认覆盖到
`64` 会触发 page-size 不整除，`--block-size 16` 可满足 `784 % 16 == 0`。这可能引入
KV/page 管理开销或 tile 不理想，必须用 accuracy、throughput 和 profiler 记录说明收益与
风险。当前 4-8K full50 记录显示 AITER unified 相对默认 Triton attention 主要改善 TTFT
P99，TPOT 基本持平；它是 backend enablement 证据，不是最终核心优化结论。

## 合规红线

所有优化必须保持官方评测语义不变。禁止：

- 修改模型权重、模型结构、tokenizer、chat template、served model identity 或输出口径。
- 截断输入、降低生成长度、跳过长样本、过滤样本、跳层或动态 early exit。
- prompt/result cache、隐藏测试集特化规则、预生成答案或数据集专用分支。
- 引入 draft model、MTP、投机解码、外挂小模型、自训练预测器或 token predictor。
- 持久化量化权重、剪枝权重、压缩权重、转换后模型文件或可复用模型状态。
- 绕开 OpenAI API 路由、官方测试脚本、统计口径或资源统计路径。
- 最终评测期间依赖外部网络下载包、模型、kernel 或运行材料。

允许的方向包括数学等价的 kernel/tile 调优、算子融合、workspace 预分配、KV cache 和
metadata 开销优化、ROCm/DCU backend 适配、诊断日志，以及赛规允许且精度 gate 通过的
运行时 KV cache 低精度路径。

## Patch 规则

- patch 要小而清晰。每个提交只解决一个问题或一个优化点。
- 性能 patch 必须说明命中的运行路径、形状 gate、回退方式和预期影响。
- 高风险优化要有环境变量或配置开关，能回退到 baseline 行为。
- 不在未确认 hot path 前盲改 kernel。先收集 trace/backend/fallback 证据，再做等价 patch。
- 不把 backend enablement 包装成核心性能贡献。比如补 gfx 宏、修 import、修 dispatch
  只能作为前置修复单列。
- 不提交模型权重、testdata、profile trace、benchmark 原始大日志、wheel、build 目录、
  `.venv`、cache 或本地账号材料。

推荐 commit message 写清楚：

```text
area: concise change summary

- Why: 当前瓶颈或失败现象
- What: 改了哪条路径和形状/开关
- Validation: 构建、单测、吞吐、精度或未验证原因
```

如果使用 AI 辅助生成或修改代码，提交者必须逐行 review，并能解释行为、风险和验证方式。

## 队内交接

涉及远端路径、CI/CD 同步、比赛容器命令、评测日志位置和性能数据时，优先更新文档或脚本
参数，不要只写在聊天记录里。敏感信息不进 Git：SSH 私钥、token、密码、cookie、账号材料
和长期有效下载凭据都必须留在本地或 CI secret。
