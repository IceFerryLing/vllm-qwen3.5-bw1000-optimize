# Qwen3.5-27B DCU 优化建议稿

## 适用范围

这份建议只针对本仓库当前的 Qwen3.5-27B 路径，不假设额外的模型改造，也不预设某个 backend 一定命中。所有结论都应以仓库代码、profile 和同容器 A/B 为准。

## 先对齐事实

1. Qwen3.5 的文本配置在仓库里是 `num_hidden_layers=32`、`num_attention_heads=16`、`num_key_value_heads=4`、`head_dim=256`。
2. `Qwen3.5` 不是每层都走 full attention，而是默认每 4 层一个 `full_attention`，其余层是 `linear_attention` / Gated DeltaNet。
3. 当前仓库已经存在几条可验证的 attention 路径：
   - `TRITON_ATTN` 的 `kernel_unified_attention_2d`
   - `csrc/rocm/attention.cu` 里的 ROCm paged attention
   - AITER unified attention
   - Qwen3.5 GDN / linear attention 路径
4. 不能先假设某条路径一定是主热点，必须先用 hipprof / 日志确认 backend 和 kernel 名称。

## 当前最值得查的 bank conflict 场景

### 1. `TRITON_ATTN` 的 unified attention 2d

这是目前最明确的 LDS bank conflict 候选。仓库里的 `pra26/lds.md` 已经把它作为已确认问题记录，原因不是“用了 shared memory”本身，而是该 kernel 的 tile layout 和默认 Triton ROCm pipeline 组合后，出现了明显的 `shared memory bank conflict` 和 `LDS stall`。

如果继续优化，这条线优先考虑：

- `QWEN_UA_PROBE_STAGE` 先区分 QK 和 PV 的冲突来源。
- `QWEN_UA_NUM_STAGES` 试 `2/3`，看 double-buffer 是否能打散 stride-aligned 冲突。
- `QWEN_UA_TILE_SIZE` 试 `16/64`，看 tile 粒度变化是否改变 LDS 冲突形态。
- `HEAD_SIZE_PADDED` 做资源/layout 对照；当前 Triton kernel 要求 2 的幂，不能直接实现任意
  `+1` padding。
- `QWEN_UA_BLOCK_M` 作为对照项，不要默认它一定不能动。

这里的原则是：先改 gate，再改默认值；先证实冲突来源，再做更大改动。真正的非 2 次幂
padding 或 XOR swizzle 需要更底层的 Triton layout 改造或自写 HIP/DUMMA 路径。

### 2. `csrc/rocm/attention.cu` 的 ROCm paged attention

这个路径里已经能看到典型的 bank-conflict 规避写法，比如对 shared memory 维度做 `+1` padding。它说明：

- padding 是仓库里已经接受的手段；
- 但 padding 是否值得，必须看具体 kernel 的访问模式和 PMC；
- 不能直接把一个 kernel 的布局经验搬到另一个 kernel。

### 3. Qwen3.5 的 GDN / linear attention

Qwen3.5 的 `linear_attention` 不是简单的 dense attention 替代品，layout 和 fused op 更复杂。这里更可能受益于：

- prefill 侧的拆分和融合；
- 合理的 tile / stage 调参；
- 避免不必要的 LDS staging；
- 先确认热 kernel，再谈 swizzle。

不建议直接把 AITER decode 的经验当成 Qwen3.5 主结论。Qwen3.5 的 flat layout 和 backend capability 还要单独确认。

## 建议的优化顺序

1. 先确认当前负载下的主 backend 和主 kernel。
2. 对 `TRITON_ATTN` 路径做短窗口 profile，拿到 `shared memory bank conflict`、`LDS stall`、`shared memory size`、`VGPR/SGPR/scratch`。
3. 只对已命中的热点 kernel 做 gate 化试验。
4. 优先试数学等价的参数调优，不先动模型语义或缓存口径。
5. 任何收益都必须过同容器 A/B 和 accuracy gate。

## 不建议直接写进建议稿的内容

- 固定写死 Qwen3.5 是 64 层、24 头、128 head_dim。
- 把某个 page size、bank 偏移或 padding 数字当成通用真理。
- 在没有 profile 的情况下断言“主热点一定是某条 attention kernel”。
- 直接把“buffer/page 全部放 LDS”当成默认方案。
- 写 `if Qwen3.5` 这类模型名特判。

## 结论

这份建议应该表达成“可验证的优化路线”，而不是“已经证明的答案”。最稳妥的版本是：先用 profile 锁定热点，再围绕 `num_stages`、`tile size`、`padding`、`layout` 做等价调优；如果是 `TRITON_ATTN` 的 unified attention 2d，就优先检查 LDS bank conflict，如果不是，就按实际 backend 重新分流。
