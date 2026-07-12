# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant
from vllm.model_executor.layers.utils import (
    _strided_rows_per_block,
    _use_strided_gemv,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx936, on_gfx950
from vllm.utils.platform_utils import num_compute_units

DTYPES = [torch.bfloat16, torch.float16]
BIAS_MODES = [0, 1, 2]
# Specific (N, K, M) combinations for targeted testing
NKM_FACTORS_LLMM1 = [
    # Small, medium, large cases
    (1, 8, 16),
    (1, 32, 64),
    (1, 128, 256),
    (1, 512, 1024),
    (1, 2048, 4096),
    # Edge cases with specific K sizes
    (1, 6144, 1024),
    (1, 8192, 2048),
    # Very large case
    (1, 4096, 8192),
]

NKM_FACTORS_WVSPLITK = [
    # Different batch sizes with key dimensions
    (1, 32, 16),
    (1, 64, 64),
    (2, 256, 256),
    (3, 1024, 1024),
    (4, 4096, 4096),
    (4, 4096, 4096 + 1),
    (4, 4096 + 16, 4096),
    (4, 4096 + 16, 4096 + 1),
    # Extended K values
    (1, 9216, 512),
    (2, 10240, 1024),
    (4, 16384, 8192),
    (4, 16384 * 2, 8192),
    (4, 16384 * 2, 8192 + 1),
    (4, 16384 * 2 + 16, 8192),
    (4, 16384 * 2 + 16, 8192 + 1),
    # Minimum M constraint validation (m >= 8)
    (1, 64, 8),
    (2, 128, 8),
    (4, 256, 8),
]

N_FACTORS_WVSPLITKRC = [
    13,
    16,
    17,
    25,
    29,
    31,
    32,
    41,
    51,
    64,
    71,
    81,
    91,
    103,
    117,
    128,
]
K_FACTORS_WVSPLITKRC = [2880, 2880 + 8, 3072, 3072 + 8]
M_FACTORS_WVSPLITKRC = [128, 128 + 16, 256, 256 + 16, 640, 640 + 16]
NKM_FACTORS_WVSPLITK_FP8 = [
    # FP8-specific cases with K % 16 == 0
    (1, 16, 16),
    (1, 32, 16 + 16),
    (1, 64, 64),
    (1, 64, 64 + 16),
    (1, 64 + 16, 64),
    (1, 64 + 16, 64 + 16),
    (4, 64, 64),
    (4, 64, 64 + 16),
    (4, 64 + 16, 64),
    (4, 64 + 16, 64 + 16),
    (2, 512, 512),
    (3, 512, 512),
    (3, 512, 512 + 16),
    (4, 512, 512),
    (3, 2048, 2048),
    (3, 2048, 2048 + 16),
    (4, 2048 + 16, 2048),
    (4, 2048 + 16, 2048 + 16),
    (4, 4096, 4096),
    (4, 16400, 2048),
    (4, 16400, 2048 + 16),
    # Extended FP8 dimensions not covered by WVSPLITK
    (1, 14336, 1024),
    (2, 24576, 2048),
    (4, 32768, 28672),
    (4, 32768 * 2, 28672),
    (4, 32768 * 2, 28672 + 16),
    (4, 32768 * 2 + 16, 28672),
    (4, 32768 * 2 + 16, 28672 + 16),
]

SEEDS = [0]


def pad_fp8(weight):
    num_pad = 256 // weight.element_size()
    import torch.nn.functional as F

    return F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]


def test_qwen35_gfx936_bf16_gemv_routing():
    assert _use_strided_gemv(248320, 5120, True)
    assert _strided_rows_per_block(248320, 5120, True) == 2

    # Do not change the generic large-vocabulary behavior on other devices,
    # dtypes, or unrelated shapes.
    assert not _use_strided_gemv(248320, 5120, False)
    assert not _use_strided_gemv(131072, 4096, True)


@pytest.mark.parametrize("xnorm", [False, True])
@pytest.mark.parametrize("n", N_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("k", K_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("m", M_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("padded_a", [False, True])
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx950(), reason="only meant for gfx950")
def test_rocm_wvsplitkrc_kernel(xnorm, n, k, m, dtype, seed, padded_a, bias_mode):
    torch.manual_seed(seed)
    cu_count = num_compute_units()

    # Next ^2 of n
    N_p2 = 1 << (n - 1).bit_length()
    # With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
    # and each working on a 512-shard of K, how many CUs would we need?
    rndup_cus = ((m + 64 - 1) // 64) * ((k + 512 - 1) // 512)
    # How many of 4 waves in a group can work on same 16 Ms at same time?
    # This reduces the Ms each group works on, i.e. increasing the number of CUs needed.
    GrpsShrB = min(N_p2 // 16, 4)
    # Given the above, how many CUs would we need?
    CuNeeded = rndup_cus * GrpsShrB
    # candidate for atomic reduce count splitk?
    fits_wvsplitkrc = (N_p2 * m * ((k + 512 - 1) // 512)) <= 128 * 1024 * 12
    fits_wvsplitkrc &= CuNeeded <= cu_count

    if not fits_wvsplitkrc:
        pytest.skip("Too large for wvSplitKrc")

    xavier = (
        math.sqrt(2 / k) if xnorm else 1
    )  # normalize to avoid large output-bias deltas
    A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    if padded_a:
        A = pad_fp8(A)

    BIAS = None
    if bias_mode == 1:
        BIAS = torch.rand(m, dtype=dtype, device="cuda") * 2 - 1
    elif bias_mode == 2:
        BIAS = torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1

    ref_out = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitKrc(A, B, cu_count, BIAS)

    if xnorm:
        torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-8)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_LLMM1)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows_per_block", [2, 4, 8, 16])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@torch.inference_mode()
def test_rocm_llmm1_kernel(n, k, m, dtype, rows_per_block, seed):
    torch.manual_seed(seed)
    # TODO: Zero-centering the inputs causes errors for LLMM1!
    #      Without that the numbers quickly saturate, and may
    #      be giving false matches.
    A = torch.rand(n, k, dtype=dtype, device="cuda")
    B = torch.rand(m, k, dtype=dtype, device="cuda")

    ref_out = torch.matmul(A, B.t())
    out = ops.LLMM1(B, A, rows_per_block)

    torch.testing.assert_close(out, ref_out, atol=1e-8, rtol=1e-2)


@pytest.mark.parametrize("k,m", [(5120, 6144), (17408, 5120)])
@pytest.mark.parametrize("rows_per_block", [2, 4, 8])
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@torch.inference_mode()
def test_rocm_llmm_strided_k_bf16_accuracy(k, m, rows_per_block):
    torch.manual_seed(0)
    x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
    weight = (
        torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.02
    )

    ref_out = torch.nn.functional.linear(x, weight)
    out = ops.LLMM_StridedK(weight, x, rows_per_block)

    # The gfx936 path accumulates BF16 products in FP32. Keep this threshold
    # tight enough to catch the old pairwise BF16 accumulation regression.
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-3)


@pytest.mark.skipif(not on_gfx936(), reason="kernel is tuned for gfx936")
@torch.inference_mode()
def test_rocm_llmm_strided_k_qwen35_down_rows1_accuracy():
    torch.manual_seed(0)
    k = 17408
    m = 5120
    x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.02

    ref_out = torch.nn.functional.linear(x, weight)
    out = ops.LLMM_StridedK(weight, x, 1)

    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@torch.inference_mode()
def test_rocm_llmm_silu_mul_matches_unfused():
    torch.manual_seed(0)
    k = 5120
    d = 128
    x = torch.randn(1, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(2 * d, k, dtype=torch.bfloat16, device="cuda") * 0.02

    gate_up = ops.LLMM_StridedK(weight, x, 2)
    gate, up = gate_up.float().chunk(2, dim=-1)
    ref_out = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    out = ops.LLMM_SiluMul(weight, x)

    torch.testing.assert_close(out, ref_out, atol=0, rtol=0)


@pytest.mark.skipif(not on_gfx936(), reason="solution index is specific to gfx936")
@torch.inference_mode()
def test_rocblas_bf16_mlp_down_4096_matches_default_solution():
    torch.manual_seed(0)
    x = torch.randn(4096, 17408, dtype=torch.bfloat16, device="cuda") * 0.02
    weight = torch.randn(5120, 17408, dtype=torch.bfloat16, device="cuda") * 0.02

    ref_out = torch.nn.functional.linear(x, weight)
    out = ops.rocblas_bf16_mlp_down_4096(weight, x)

    # Solution 20980 uses the same K reduction and BF16 output rounding as the
    # default gfx936 solution for this shape.
    torch.testing.assert_close(out, ref_out, atol=0, rtol=0)


@pytest.mark.skipif(
    not on_gfx936(), reason="solution index is specific to gfx936"
)
@torch.inference_mode()
def test_rocblas_bf16_mlp_gate_up_4096_matches_default_solution():
    torch.manual_seed(0)
    x = torch.randn(4096, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
    weight = (
        torch.randn(34816, 5120, dtype=torch.bfloat16, device="cuda") * 0.02
    )

    ref_out = torch.nn.functional.linear(x, weight)
    out = ops.rocblas_bf16_mlp_gate_up_4096(weight, x)

    torch.testing.assert_close(out, ref_out, atol=0, rtol=0)


@pytest.mark.parametrize("xnorm", [False, True])
@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.parametrize("padded_a", [False, True])
@pytest.mark.parametrize("padded_b", [False, True])
def test_rocm_wvsplitk_kernel(
    xnorm, n, k, m, dtype, seed, bias_mode, padded_a, padded_b
):
    torch.manual_seed(seed)
    cu_count = num_compute_units()

    xavier = (
        math.sqrt(2 / k) if xnorm else 1
    )  # normalize to avoid large output-bias deltas
    A = (torch.rand(n, k, dtype=dtype, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, dtype=dtype, device="cuda") * 2 - 1) * xavier

    BIAS = None
    if bias_mode == 1:
        BIAS = torch.rand(m, dtype=dtype, device="cuda") * 2 - 1
    elif bias_mode == 2:
        BIAS = torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1

    if padded_a:
        A = pad_fp8(A)
    if padded_b:
        B = pad_fp8(B)

    ref_out = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitK(B, A.view(-1, A.size(-1)), cu_count, BIAS)

    if xnorm:
        assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-8)
    else:
        assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("xnorm", [False, True])
@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK_FP8)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("padded_a", [False, True])
@pytest.mark.parametrize("padded_b", [False, True])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="only test for rocm fp8",
)
def test_rocm_wvsplitk_fp8_kernel(
    xnorm, n, k, m, dtype, seed, padded_a, padded_b, biased
):
    torch.manual_seed(seed)

    xavier = math.sqrt(2 / k) if xnorm else 1  # normalize to avoid large deltas
    A = (torch.rand(n, k, device="cuda") * 2 - 1) * xavier
    B = (torch.rand(m, k, device="cuda") * 2 - 1) * xavier

    A, scale_a = ref_dynamic_per_tensor_fp8_quant(A)
    B, scale_b = ref_dynamic_per_tensor_fp8_quant(B)
    if padded_b:
        B = pad_fp8(B)
    if padded_a:
        A = pad_fp8(A)

    BIAS = None if (not biased) else (torch.rand(m, dtype=dtype, device="cuda") * 2 - 1)

    ref_out = torch._scaled_mm(
        A, B.t(), out_dtype=dtype, scale_a=scale_a, scale_b=scale_b, bias=BIAS
    )
    out = ops.wvSplitKQ(B, A, dtype, scale_a, scale_b, num_compute_units(), BIAS)

    if xnorm:
        torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-8)
    elif k >= 32 * 1024:
        # wider pytrch thresh for large-K & no xnorm
        torch.testing.assert_close(out, ref_out, atol=0.07, rtol=5e-2)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
