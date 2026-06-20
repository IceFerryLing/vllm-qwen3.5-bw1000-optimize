# 先导杯 2026 vLLM 测试调试要点

来源：先导杯2026-大模型推理服务优化-选手测试调试文档.pdf。

本文只整理该 PDF 中的测试调试流程，方便在本 vLLM fork 中查阅。它不是完整赛题规则，
不替代官方赛规、评测说明或最终提交要求。

## 平台与容器

- 登录超算平台：`https://www.scnet.cn`。
- 进入控制台后选择「容器服务」。
- 在「镜像管理」->「镜像仓库」->「核心节点分区一」中搜索并克隆镜像：

```text
qwen3.5-dtk26.04:0509
```

- 创建容器时，选择「核心节点分区一」的 `<COMPETITION_QUEUE>` 队列。
- 开发工具选择 `SSH`。
- 镜像选择刚克隆到「我的镜像」中的比赛镜像。
- 容器创建后通过页面里的 SSH 入口进入容器。

注意：容器实例停止后，非用户家目录的数据会丢失。源码、模型、测试数据、日志等尽量放在
用户家目录下。

## vLLM 基线源码与 wheel 构建

PDF 指定在用户家目录下执行：

```bash
git clone -b v0.18.1 --depth 1 http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git
cd vllm_cscc
python setup.py bdist_wheel
cd dist
pip install vllm-*.whl --no-deps
```

说明：

- 这是比赛基线使用的 vLLM 版本。
- 首次构建约十分钟；后续重新编译约两分钟，实际以容器环境为准。
- 容器实例重新启动后，PDF 提示 vLLM 编译和模型下载相关操作可能需要重新执行。
- 我们对 kernel、调度、算子融合、编译参数等做优化后，最终仍必须能重新构建出
  `dist/vllm-*.whl`，并且能通过 wheel 安装路径生效。
- 不要只依赖 `PYTHONPATH`、editable install 或本地未提交文件。

本仓库的 `main` / `develop` 分支用于源码开发；`competition-ci` 只用于把源码同步到无卡
共享通道，不是评测分支。

## 模型下载与放置

PDF 指定模型为：

```text
Qwen/Qwen3.5-27B
```

下载命令：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3.5-27B --local_dir ./Qwen3.5-27B
```

PDF 建议模型下载到用户家目录；每次启动容器后，启动 vLLM 服务前复制一份到 `/root`，
因为从 `/root` 下加载模型更快：

```bash
cp -r ./Qwen3.5-27B/ /root/Qwen3.5-27B
```

## 测试数据与脚本

在 Linux 服务器中下载测试包：

```bash
curl -f -C - -o testdata.tar.gz \
  https://zzefile.scnet.cn:65011/efile/s/d/c2N5MTE1OTkxMDU1OQ==/a927e65672549b46
mkdir -p ./testdata
tar -xzf testdata.tar.gz -C ./testdata --strip-components=1
cd testdata
chmod +x *.sh
```

PDF 中的 `testdata` 内容包括：

吞吐测试数据集：

- `4-8K_throughput.jsonl`
- `8-16K_throughput.jsonl`
- `16-32K_throughput.jsonl`

精度测试数据集：

- `hotpotqa.jsonl`
- `gov_report.jsonl`
- `retrieval_multi_point.jsonl`
- `aggregation_keyword_aggregation.jsonl`

脚本：

- `start_vllm.sh`
- `run_throughput.sh`
- `run_accuracy.sh`

## 启动 vLLM 服务

进入用户家目录下的 `testdata` 目录执行：

```bash
./start_vllm.sh
```

注意：

- 模型服务启动终端和脚本不能停止或关闭。
- 首次服务启动约十分钟，实际以容器和模型加载情况为准。
- 服务默认监听：

```text
127.0.0.1:8001
```

可用 `hy-smi` 查看 DCU 显卡情况：

```bash
hy-smi
```

另起终端执行单次推理，验证服务是否启动成功：

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-27B",
    "messages": [
      {"role": "user", "content": "你好，简单回复一句话。"}
    ],
    "temperature": 0.0,
    "max_tokens": 64
  }'
```

每次服务关闭或容器关闭后，都需要重新启动该服务脚本。

## 吞吐测试

另起终端进入 `testdata` 执行：

```bash
./run_throughput.sh
./run_throughput.sh all 10
./run_throughput.sh 4-8K 10
./run_throughput.sh 8-16K 20
./run_throughput.sh 16-32K 15
```

含义：

- `./run_throughput.sh`：执行全部吞吐数据集测试。
- `./run_throughput.sh all 10`：所有吞吐数据集都只取前 10 条。
- `./run_throughput.sh 4-8K 10`：`4-8K` 数据集只取前 10 条。
- `./run_throughput.sh 8-16K 20`：`8-16K` 数据集只取前 20 条。
- `./run_throughput.sh 16-32K 15`：`16-32K` 数据集只取前 15 条。

PDF 说明：参赛者可自定义修改验证所需的数据集类型和数量。调试数据集结果只作为自身
baseline，最终分数以评测机具体测试和 baseline 结果为准。

吞吐评测主要关注：

- output tokens/s
- TTFT P99
- TPOT P99

其它指标可以作为调试参考。

## 精度测试

另起终端进入 `~/testdata` 执行：

```bash
./run_accuracy.sh
./run_accuracy.sh hotpotqa 10
./run_accuracy.sh gov_report 10
./run_accuracy.sh retrieval_multi_point 10
./run_accuracy.sh aggregation_keyword_aggregation 10
```

含义：

- `./run_accuracy.sh`：运行全部精度测试集。
- `hotpotqa`：LongBench 问答任务，指标为 F1。
- `gov_report`：LongBench 长文摘要任务，指标为 ROUGE。
- `retrieval_multi_point`：检索任务，脚本会单独重算结果。
- `aggregation_keyword_aggregation`：多答案聚合任务，脚本会单独重算结果。

实时日志：

```bash
tail -f accuracy_debug/opencompass_run.log
```

OpenCompass 输出目录：

```text
testdata/accuracy_debug/output/local_accuracy_qwen35/
```

PDF 说明：`run_accuracy.sh` 会调用 OpenCompass 和脚本内后处理逻辑，评估问答、摘要、
检索、多答案聚合任务。`retrieval_multi_point` 和 `aggregation_keyword_aggregation`
会把模型输出解析成答案列表，再和标准答案列表逐项比对；运行日志中出现 `0` 的指标不一定有
参考意义，应以脚本最终打印结果为准。

## 提交前检查

PDF 建议提交前至少确认：

- vLLM 源码修改后能够重新正常编译通过。
- 安装 wheel 后模型服务能够正常启动。
- curl 单次推理返回正常。
- `run_throughput.sh` 可以正常运行。
- `run_accuracy.sh` 可以正常运行。

## 对本仓库的直接含义

- 所有有效优化都要能落到 wheel 构建路径：`python setup.py bdist_wheel`。
- 无卡登录/共享通道只能用于传源码、模型、wheel 和日志；不能产生 DCU 性能结论。
- GitHub 公共 runner 只适合做 source smoke，不能证明 wheel、DTK/HIP/DCU 扩展或性能。
- 本地调试数据只用于建立自身 baseline；最终性能和精度以官方评测机为准。
- 任何优化都不应改变模型、tokenizer、prompt、采样语义、API 行为或官方统计口径。
