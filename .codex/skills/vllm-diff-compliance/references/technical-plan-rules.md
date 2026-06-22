# Technical Plan Compliance Rules

Source: `misc/2026年全国大学生计算机系统能力大赛-智能计算创新设计赛-基于国产加速卡的千问大模型推理服务优化-技术方案.pdf`, extracted 2026-06-22 from the local repository.

## Competition Baseline

- Use unified model weights, unified container image, vLLM 0.18.1, official Qwen3.5-27B tokenizer, official chat template, official bf16 original weights, and official benchmark tools.
- Final evaluation runs only in the unified container. The host does not inject extra libraries. Downloading dependencies from the external network during evaluation is prohibited.
- Preliminary round evaluates a single-card DCU online serving workload with concurrency fixed at 1.
- Input length buckets are 4K-8K, 8K-16K, and 16K-32K.
- Performance evaluation uses `vllm bench serve`.
- Accuracy evaluation uses OpenCompass for QA, summary, retrieval, and aggregation tasks.
- Throughput means output tokens/second only. Prompt tokens do not count in the numerator.
- TTFT is client end-to-end time from HTTP API request to first streamed token and includes tokenizer encoding plus prefill compute.
- TPOT is time from first token to last token divided by output tokens minus 1; global TPOT P99 is used.

## Allowed Optimization Areas

These are normally allowed when they preserve task semantics:

- Decode-stage scheduling customization.
- KV cache allocation, memory budget control, block management, block granularity/alignment, fragmentation reduction.
- DCU-aware resource organization using memory capacity, bandwidth, on-chip cache hierarchy, parallel execution features, and data movement cost.
- Decode attention and linear operator optimization.
- Operator fusion and custom kernels.
- Reduced Python scheduling overhead, lower kernel launch latency, and optimized memory copy paths.
- Non-persistent operator-level low-precision compute during inference, such as activation dynamic quantization, KV cache quantization, temporary type conversion inside kernels, low-precision matrix multiplication, and attention kernel optimization.

## Hard Prohibitions

Classify any apparent violation here as **BLOCKER** unless the diff clearly proves it is unreachable in evaluation:

- Do not modify sampling/task parameters to avoid real inference cost, including forcing lower `max_tokens`.
- Do not truncate input, skip long samples, filter difficult samples, or skip layers.
- Do not pre-cache test sets, answers, generated token sequences, or pre-generated intermediate results to answer requests.
- Do not bypass the unified service interface, evaluation flow, or resource/statistics path.
- Do not introduce auxiliary models outside the rules, including externally trained small models for speculative sampling.
- Do not modify model structure, replace model weights, change inference semantics, or change output behavior.
- Do not post-train, distill, or fine-tune the model.
- Do not perform persistent quantization, structured pruning, weight reorder/compression, model graph reconstruction, model format conversion, or generate reusable quantized weight caches before or during service initialization/formal inference.
- Do not perform structured pruning, unstructured pruning, dynamic channel skipping, dynamic layer skipping, attention head pruning, token pruning, or early exit.
- Do not generate reusable model files, quantized weight files, pruned weight files, or compressed weight caches during inference.
- Do not write dataset-specific hard-coded if/else rules.
- Do not forge results, hide exceptions, or otherwise violate blind-test fairness.

## Locked Parameters And Interfaces

Treat changes here as **BLOCKER** or **HIGH RISK** unless clearly confined to non-evaluation tooling:

- Model and tokenizer: `model`, `tokenizer`, `tokenizer-mode`, chat template.
- Context and batching: `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, and batch scheduler related semantics. The technical plan states that vLLM internal batch scheduler related code is not allowed to be modified for locked scheduler behavior.
- Generation behavior: `max_tokens`, `temperature=0`, official autoregressive decoding.
- Speculative decoding is prohibited in any form, including draft model, MTP, multi-head prediction, external small model, self-trained predictor, intermediate-layer early-exit draft, or pre-generated token sequence cache.
- Bench and results: benchmark percentile/statistics logic, result saving options, result parser scripts.
- Service interface: `--served-model-name`, OpenAI API path, host/port/platform-fixed route, request format, response format.

## SLA And Accuracy Risks

These do not always make a diff illegal, but they must be called out as validation requirements:

- If TTFT P99 for a length bucket exceeds the corresponding baseline TTFT P99 times 1.5, that bucket's throughput score is zero.
- If global TPOT P99 exceeds baseline global TPOT P99 times 1.5, affected throughput score is zero.
- If service completion rate drops by more than 1%, the corresponding score can be invalidated.
- Accuracy is compared against official baseline output or standard answers. Accuracy drops affect final score and large drops can zero a task class.
- Temperature is fixed at 0.0 for accuracy evaluation.

## Submission And Documentation Requirements

Flag missing evidence as **NEEDS EVIDENCE**:

- Full source and build scripts must compile successfully in the specified evaluation environment.
- Custom environment variables must be documented with variable names, values, purpose, and reason. Undocumented env vars may make the submission incomplete or invalid.
- The optimization plan document must describe methods, technical route, contribution analysis, and optimization summary table. It should include key code paths and performance comparisons.
- Third-party IP, open-source code, public algorithms, third-party libraries, or copied source must be clearly marked in the submitted explanation document and at the top of the project README.

## Common Diff Red Flags

Inspect carefully when a diff touches:

- `sampling_params`, `max_tokens`, `temperature`, `stop`, output length, truncation, prompt preprocessing, chat template rendering.
- `tokenizer`, `model_loader`, checkpoints, weight files, quantized weight caches, safetensors/bin conversion, model graph export.
- `spec_decode`, `draft`, `mtp`, predictor, auxiliary model, early exit, token pruning, layer skipping, head pruning.
- Dataset loaders, benchmark request generation, result parser, percentile code, OpenCompass or LongBench/RULER evaluation scripts.
- OpenAI API route, served model name, request/response schema, streaming behavior, token accounting.
- Persistent files under model, cache, dist, generated weights, quantization outputs, answer caches, prompt/result maps.
- Batch scheduler and locked context/batching flags.
- Environment variable gates that enable any prohibited behavior by default or in evaluation.
