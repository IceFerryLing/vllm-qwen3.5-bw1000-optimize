# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, next_power_of_2, round_down
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# route the Qwen3.5 full-attention (DECODER) prefill path to
# non-paged flash_attn (gather history KV -> contiguous, per-segment FA + merge),
# bypassing the numerically-broken paged FA kernel on gfx936.
#  Default on; set VLLM_TRITON_FA_PREFILL=0 to fall back to pure Triton.
VLLM_TRITON_FA_PREFILL = int(os.environ.get("VLLM_TRITON_FA_PREFILL", "1"))

# Plain upstream flash_attn (NOT aiter, NOT vllm.vllm_flash_attn). aiter FA is
# import-broken and vllm_flash_attn is unavailable on this ROCm build.
try:
    from flash_attn import flash_attn_varlen_func as _fa_varlen_func
except ImportError:
    _fa_varlen_func = None


# constants
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
# Number of parallel tiled softmax segments for the 3D decode attention kernel.
NUM_PAR_SOFTMAX_SEGMENTS = int(
    os.environ.get("VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS", "256")
)
if NUM_PAR_SOFTMAX_SEGMENTS <= 0 or (
    NUM_PAR_SOFTMAX_SEGMENTS & (NUM_PAR_SOFTMAX_SEGMENTS - 1)
):
    raise ValueError(
        "VLLM_TRITON_ATTN_NUM_PAR_SOFTMAX_SEGMENTS must be a positive "
        f"power of two, got {NUM_PAR_SOFTMAX_SEGMENTS}"
    )


@dataclass
class TritonChunkedContextMetadata:
    """
    Mirrors MLA's ChunkedContextMetadata but for a standard GQA KV cache
    (K and V stored separately, head_size=256), not the MLA latent layout.
    All requests are covered uniformly (context_len = num_computed_tokens);
    requests with no history simply contribute 0 tokens to every chunk.
    """

    # [num_chunks, num_reqs + 1] cumulative token counts per chunk (device)
    cu_seq_lens: torch.Tensor
    # [num_chunks, num_reqs] per-request start offset into its context (device)
    starts: torch.Tensor
    # per-chunk total gathered tokens across the batch
    seq_tot: list[int]
    # per-chunk max single-request context length (for FA max_seqlen_k)
    max_seq_lens: list[int]
    # [num_chunks, max_tokens] maps gathered token -> req idx (device, unused by
    # cp_gather_cache but kept for parity/debug)
    token_to_seq: torch.Tensor
    # gathered-KV workspace, sized [workspace_tokens, num_kv_heads, head_size]
    workspace_k: torch.Tensor
    workspace_v: torch.Tensor


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None

    # FA-prefill path. use_fa_prefill signals the impl to route this step to
    # non-paged flash_attn (this step carries prefill queries). chunked_context
    # describes the gathered history; None when there is no context (fresh
    # prefill on an empty cache).
    use_fa_prefill: bool = False
    chunked_context: TritonChunkedContextMetadata | None = None

    @property
    def mm_prefix_range_tensor(self) -> torch.Tensor | None:
        """Convert mm_prefix_range dict to padded tensor for Triton kernel.

        Returns shape: (num_seqs, max_ranges, 2) with 0-padding for empty ranges.
        Empty ranges have start==end==0, which kernel skips via is_valid check.
        """
        # TODO(Isotr0py): Move to model runner's attention metadata
        # preparation to avoid duplicate computation.
        if self.mm_prefix_range is None:
            return None

        num_seqs = self.seq_lens.shape[0]
        device = self.seq_lens.device

        # Collect ranges, using [(0,0)] for empty sequences to ensure uniform dims
        range_lists = [
            self.mm_prefix_range.get(i, [(0, 0)]) or [(0, 0)] for i in range(num_seqs)
        ]

        # Return None if all ranges are trivial (only (0,0) placeholders)
        if all(r == [(0, 0)] for r in range_lists):
            return None

        # Create 2D tensors with shape (num_ranges, 2) for each sequence
        range_tensors = [
            torch.tensor(r, dtype=torch.int32, device=device).view(-1, 2)
            for r in range_lists
        ]

        return torch.nested.nested_tensor(
            range_tensors, layout=torch.jagged
        ).to_padded_tensor(0)


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        headdim_padded = next_power_of_2(self.headdim)
        self.softmax_segm_output = torch.empty(
            (
                self.seq_threshold_3D,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )

        # FA-prefill (gather+merge) path setup. Only for a standard bf16/fp16 KV
        # cache; fp8 stays on the tuned Triton paged path.
        kv_cache_dtype = kv_cache_spec.dtype
        self.fa_prefill_enabled = (
            VLLM_TRITON_FA_PREFILL != 0
            and _fa_varlen_func is not None
            and current_platform.is_rocm()
            and kv_cache_dtype in (torch.bfloat16, torch.float16)
            and self.headdim in (64, 128, 256)
        )
        self.chunked_prefill_workspace_k = None
        self.chunked_prefill_workspace_v = None
        if self.fa_prefill_enabled:
            model_config = vllm_config.model_config
            scheduler_config = vllm_config.scheduler_config
            cache_config = vllm_config.cache_config
            # Try for 4 pages/request or 4 full-length requests worth of context,
            # capped at 64K tokens to bound memory. Rounded to page multiple.
            ws = min(
                max(
                    4 * model_config.max_model_len,
                    4 * scheduler_config.max_num_seqs * cache_config.block_size,
                ),
                64 * 1024,
            )
            ws = max(ws, scheduler_config.max_num_seqs * cache_config.block_size)
            ws = round_down(ws, self.block_size)
            self.chunked_prefill_workspace_size = ws
            self.chunked_prefill_workspace_k = torch.empty(
                (ws, self.num_heads_kv, self.headdim),
                dtype=kv_cache_dtype,
                device=device,
            )
            self.chunked_prefill_workspace_v = torch.empty(
                (ws, self.num_heads_kv, self.headdim),
                dtype=kv_cache_dtype,
                device=device,
            )

    def _build_chunked_context(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> "TritonChunkedContextMetadata | None":
        """
        The Triton backend does NOT reorder the batch (no decodes-first
        guarantee), so unlike MLA we cannot slice `[num_decodes:]`. Instead every
        request contributes its own context (num_computed_tokens); fresh-prefill
        requests contribute 0 and are harmless. Returns None if no request has
        any context (pure prefill with empty cache).
        """
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        context_lens_cpu = (seq_lens_cpu - query_lens_cpu).to(torch.int32)
        num_reqs = context_lens_cpu.shape[0]
        max_context_len = int(context_lens_cpu.max().item())
        if max_context_len == 0:
            return None

        num_ctx_reqs = int((context_lens_cpu > 0).sum().item())
        max_context_chunk = self.chunked_prefill_workspace_size // num_ctx_reqs
        # cp_gather_cache requires chunk starts aligned to page_size.
        max_context_chunk = round_down(max_context_chunk, self.block_size)
        assert max_context_chunk > 0
        num_chunks = cdiv(max_context_len, max_context_chunk)

        chunk_starts = (
            torch.arange(num_chunks, dtype=torch.int32).unsqueeze(1).expand(-1, num_reqs)
            * max_context_chunk
        )
        chunk_ends = torch.min(
            context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk
        )
        chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

        cu_seq_lens_cpu = torch.zeros(
            num_chunks, num_reqs + 1, dtype=torch.int32, pin_memory=True
        )
        torch.cumsum(
            chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32
        )
        chunk_total_token = cu_seq_lens_cpu[:, -1]

        max_token_num_over_chunk = int(chunk_total_token.max().item())
        token_to_seq_cpu = torch.zeros(
            [num_chunks, max_token_num_over_chunk], dtype=torch.int32
        )
        range_idx = torch.arange(num_reqs, dtype=torch.int32)
        for i in range(num_chunks):
            tts = torch.repeat_interleave(range_idx, chunk_seq_lens[i])
            token_to_seq_cpu[i, : tts.shape[0]] = tts

        return TritonChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens_cpu.to(self.device, non_blocking=True),
            starts=chunk_starts.to(self.device, non_blocking=True),
            seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
            max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
            token_to_seq=token_to_seq_cpu.to(self.device, non_blocking=True),
            workspace_k=self.chunked_prefill_workspace_k,
            workspace_v=self.chunked_prefill_workspace_v,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        # FA-prefill path: active only when this step carries prefill queries
        # (max_query_len > 1). Pure-decode steps keep the tuned Triton path.
        chunked_context = None
        use_fa_prefill = (
            self.fa_prefill_enabled and max_query_len > 1 and not use_cascade
        )
        if use_fa_prefill:
            chunked_context = self._build_chunked_context(common_attn_metadata)

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            seq_threshold_3D=self.seq_threshold_3D,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
            use_fa_prefill=use_fa_prefill,
            chunked_context=chunked_context,
        )
        return attn_metadata


class TritonAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        if include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)

        # (num_blocks, 2, block_size, num_kv_heads, head_size)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonAttentionImpl(AttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.supports_quant_query_input = current_platform.is_cuda()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)
        if self.kv_cache_dtype.startswith("fp8"):
            if key_cache.dtype != self.fp8_dtype:
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        # Directed FA-prefill path: this step carries prefill queries, so route
        # full-attention to non-paged flash_attn (current chunk causal on the
        # contiguous key/value + gathered history non-causal, merged via LSE).
        # Decode-only steps skip this and use the tuned Triton paged kernel.
        if (
            attn_metadata.use_fa_prefill
            and self.attn_type == AttentionType.DECODER
            and not self.kv_cache_dtype.startswith("fp8")
            and attn_metadata.mm_prefix_range is None
            and self.alibi_slopes is None
            and self.sinks is None
            and self.sliding_window == (-1, -1)
            and self.logits_soft_cap == 0
        ):
            self._forward_fa_prefill(
                query=query[:num_actual_tokens],
                key=key[:num_actual_tokens],
                value=value[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                output=output[:num_actual_tokens],
                attn_metadata=attn_metadata,
            )
            return output

        descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        return output

    def _forward_fa_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "TritonAttentionMetadata",
    ) -> None:
        """Non-paged flash_attn for the full-attention prefill path.

        query/key/value: contiguous [num_tokens, heads, head_size] for THIS step
            (key/value are the freshly computed new-token KV, already written to
            the paged cache by the earlier kv-cache-update op).
        key_cache/value_cache: paged cache [num_blocks, block_size, kv_heads, hd].

        Computes, per request: attention over new tokens (causal) merged with
        attention over gathered history (non-causal). Mathematically exact.
        """
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len

        # Current chunk: new tokens attend to themselves, causal.
        return_lse = attn_metadata.chunked_context is not None
        out_new = _fa_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=True,
            return_attn_probs=return_lse,
        )

        if not return_lse:
            # No history: write the new-token attention directly.
            out = out_new[0] if isinstance(out_new, tuple) else out_new
            output.copy_(out.view(output.shape))
            return

        suffix_output, suffix_lse = out_new[0], out_new[1]

        ctx = attn_metadata.chunked_context
        block_table = attn_metadata.block_table
        num_reqs = block_table.shape[0]
        prefix_output = None
        prefix_lse = None
        for i in range(len(ctx.seq_tot)):
            toks = ctx.seq_tot[i]
            if toks == 0:
                continue
            ws_k = ctx.workspace_k[:toks]
            ws_v = ctx.workspace_v[:toks]
            ops.cp_gather_cache(
                src_cache=key_cache,
                dst=ws_k,
                block_table=block_table,
                cu_seq_lens=ctx.cu_seq_lens[i],
                batch_size=num_reqs,
                seq_starts=ctx.starts[i],
            )
            ops.cp_gather_cache(
                src_cache=value_cache,
                dst=ws_v,
                block_table=block_table,
                cu_seq_lens=ctx.cu_seq_lens[i],
                batch_size=num_reqs,
                seq_starts=ctx.starts[i],
            )
            chunk_out, chunk_lse = _fa_varlen_func(
                q=query,
                k=ws_k,
                v=ws_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=ctx.cu_seq_lens[i],
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=ctx.max_seq_lens[i],
                softmax_scale=self.scale,
                causal=False,
                return_attn_probs=True,
            )[:2]
            if prefix_output is None:
                prefix_output = chunk_out
                prefix_lse = chunk_lse
            else:
                merged = torch.empty_like(prefix_output)
                merged_lse = torch.empty_like(prefix_lse)
                merge_attn_states(
                    output=merged,
                    output_lse=merged_lse,
                    prefix_output=prefix_output,
                    prefix_lse=prefix_lse,
                    suffix_output=chunk_out,
                    suffix_lse=chunk_lse,
                )
                prefix_output = merged
                prefix_lse = merged_lse

        if prefix_output is None:
            # Every request happened to have empty context this step.
            output.copy_(suffix_output.view(output.shape))
            return

        merge_attn_states(
            output=output.view(suffix_output.shape),
            prefix_output=prefix_output,
            prefix_lse=prefix_lse,
            suffix_output=suffix_output,
            suffix_lse=suffix_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)

        # Reshape the input keys and values and store them in the cache.
        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            # triton kernel does not support uint8 kv_cache
            #  (because some explicit casts (e.g. float8_e4m3fnuz)
            #   are not supported)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = kv_cache.unbind(1)
        flash_layout = True

        is_fp8_kv_cache = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
