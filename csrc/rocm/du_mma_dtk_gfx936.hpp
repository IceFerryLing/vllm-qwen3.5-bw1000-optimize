#if !defined(__DU_MMA_HPP__)
#define __DU_MMA_HPP__
#if defined(__cplusplus) && defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#define __DU_MMA_DEVICE_DECL__ static __device__ __inline__ __attribute__((target("mmop1-insts,mmop-support-lit-lts")))

#ifdef USE_DU_MMA_VOLATILE
#define __DU_MMA_VOLATILE__ volatile
#else
#define __DU_MMA_VOLATILE__
#endif

#if defined(__gfx926__)
#define __DU_MMA_F32_16x16x4F64(a, b, c) \
    __builtin_amdgcn_mmac_f64_16x16x4f64_vstep((a), (b), (c), 0)

#define __DU_MMA_F32_16x16x4F32(a, b, c) \
    __builtin_amdgcn_mmac_f32_16x16x4f32((a), (b), (c))
#elif defined(__gfx928__)
#define __DU_MMA_F32_16x16x4F32(a, b, c) \
    __builtin_amdgcn_mmac_f32_16x16x4f32((a), (b), (c))

#define __DU_MMA_F32_16x16x8TF32(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x8F32(a, b, c) \
    __builtin_hcu_mmac_16x16x8_f32_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x16F16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x16BF16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_I32_16x16x32I8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts((a), (b), (c), false, false, false)

#define __DU_MMA_I32_16x16x32U8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_u8_lit_clamp_lts((a), (b), (c), false, false, false)

#define __DU_MMA_I32_16x16x64I4(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x64_i4_lit_clamp_lts((a), (b), (c), false, false, false)

#define __DU_MMA_I32_16x16x64U4(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x64_u4_lit_clamp_lts((a), (b), (c), false, false, false)

#elif defined(__gfx936__)
#define __DU_MMA_F32_16x16x4F64(a, b, c) \
    __builtin_amdgcn_mmac_f64_16x16x4f64_vstep((a), (b), (c), 0)

#define __DU_MMA_F32_16x16x4F32(a, b, c) \
    __builtin_amdgcn_mmac_f32_16x16x4f32((a), (b), (c))

#define __DU_MMA_F32_16x16x8F32(a, b, c) \
    __builtin_hcu_mmac_16x16x8_f32_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x8TF32(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts((a), (b), (c), false, false)
    
#define __DU_MMA_F32_16x16x16F16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x16BF16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_I32_16x16x32I8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts((a), (b), (c), false, false, false)

#define __DU_MMA_I32_16x16x32U8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_u8_lit_clamp_lts((a), (b), (c), false, false, false)

#define __DU_MMA_I32_16x16x64I4(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x64_i4_lit_clamp_lts((a), (b), (c), false, false, false)

#define __DU_MMA_I32_16x16x64U4(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x64_u4_lit_clamp_lts((a), (b), (c), false, false, false)

#elif defined(__gfx938__)
#define __DU_MMA_F32_16x16x4F32(a, b, c) \
    __builtin_hcu_mmac_16x16x4_f32_vstep_lts((a), (b), (c), 0, false)

#define __DU_MMA_F32_16x16x4F64(a, b, c) \
    __builtin_hcu_mmac_16x16x4_f64_vstep_lts((a), (b), (c), 0, false)

#define __DU_MMA_F32_16x16x8F32(a, b, c) \
    __builtin_hcu_mmac_16x16x8_f32_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x8TF32(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x8_tf32_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x16F16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x16BF16(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_I32_16x16x32I8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_i8_lit_clamp_lts((a), (b), (c), false, false, false)

#define __DU_MMA_I32_16x16x32U8(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x32_u8_lit_clamp_lts((a), (b), (c), false, false, false)

#define __DU_MMA_I32_16x16x64I4(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x64_i4_lit_clamp_lts((a), (b), (c), false, false, false)

#define __DU_MMA_I32_16x16x64U4(a, b, c) \
    __builtin_hcu_mmac_i32_16x16x64_u4_lit_clamp_lts((a), (b), (c), false, false, false)

#define __DU_MMA_F32_16x16x32_FP8_FP8(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x32_BF8_BF8(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x32_bf8_bf8_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x32_FP8_BF8(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x32_fp8_bf8_lit_lts((a), (b), (c), false, false)

#define __DU_MMA_F32_16x16x32_BF8_FP8(a, b, c) \
    __builtin_hcu_mmac_f32_16x16x32_bf8_fp8_lit_lts((a), (b), (c), false, false)
#endif

using doublex4 = __attribute__((ext_vector_type(4))) double; 
using floatx2 = __attribute__((ext_vector_type(2))) float;
using floatx4 = __attribute__((ext_vector_type(4))) float; 
using intx2 = __attribute__((ext_vector_type(2))) int;
using intx4 = __attribute__((ext_vector_type(4))) int;
using shortx4 = __attribute__((ext_vector_type(4))) short;
using halfx4 = __attribute__((ext_vector_type(4))) __fp16;

namespace du
{
namespace dumma
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__) || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) 
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 4, float, row_major> &a, const float *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = __lane_id() >> 4;

    a.x[0] = p[row * ldm + col];
}
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 4, float, col_major> &a, const float *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = __lane_id() >> 4;

    a.x[0] = p[col * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 4, float, row_major> &a, const float *p, unsigned ldm)
{
    unsigned row = __lane_id() >> 4;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[row * ldm + col];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 4, float, col_major> &a, const float *p, unsigned ldm)
{
    unsigned row = __lane_id() >> 4;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[col * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 4, float> &a, const float *p, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        a.x[0] = p[row * ldm + col];
        a.x[1] = p[row * ldm + col + 4];
        a.x[2] = p[row * ldm + col + 8];
        a.x[3] = p[row * ldm + col + 12];
    }
    else
    {
        a.x[0] = p[col * ldm + row];
        a.x[1] = p[(col + 4) * ldm + row];
        a.x[2] = p[(col + 8) * ldm + row];
        a.x[3] = p[(col + 12) * ldm + row];
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 4, double, row_major> &a, const double *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = __lane_id() >> 4;

    a.x[0] = p[row * ldm + col];
}
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 4, double, col_major> &a, const double *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = __lane_id() >> 4;

    a.x[0] = p[col * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 4, double, row_major> &a, const double *p, unsigned ldm)
{
    unsigned row = __lane_id() >> 4;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[row * ldm + col];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 4, double, col_major> &a, const double *p, unsigned ldm)
{
    unsigned row = __lane_id() >> 4;
    unsigned col = __lane_id() & 0xf;
    
    a.x[0] = p[col * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 4, double> &a, const double *p, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        a.x[0] = p[row * ldm + col];
        a.x[1] = p[row * ldm + col + 4];
        a.x[2] = p[row * ldm + col + 8];
        a.x[3] = p[row * ldm + col + 12];
    }
    else
    {
        a.x[0] = p[col * ldm + row];
        a.x[1] = p[(col + 4) * ldm + row];
        a.x[2] = p[(col + 8) * ldm + row];
        a.x[3] = p[(col + 12) * ldm + row];
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx928__)  || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 8, float, row_major> &a, const float *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 1;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col +1];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 8, float, col_major> &a, const float *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 1;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 8, float, row_major> &a, const float *p, unsigned ldm)
{
    unsigned row = (__lane_id() >> 4) << 1;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[(row + 1) * ldm + col];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 8, float, col_major> &a, const float *p, unsigned ldm)
{
    unsigned row = (__lane_id() >> 4) << 1;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[col * ldm + row + 1];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 8, precision::tf32, row_major> &a, const float *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 1;
    
    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col +1];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 8, precision::tf32, col_major> &a, const float *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 1;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 8, precision::tf32, row_major> &a, const float *p, unsigned ldm)
{
    unsigned row = (__lane_id() >> 4) << 1;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[(row + 1) * ldm + col];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 8, precision::tf32, col_major> &a, const float *p, unsigned ldm)
{
    unsigned row = (__lane_id() >> 4) << 1;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[col * ldm + row + 1];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 8, float> &a, const float *p, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        a.x[0] = p[row * ldm + col];
        a.x[1] = p[row * ldm + col + 4];
        a.x[2] = p[row * ldm + col + 8];
        a.x[3] = p[row * ldm + col + 12];
    }
    else
    {
        a.x[0] = p[col * ldm + row];
        a.x[1] = p[(col + 4) * ldm + row];
        a.x[2] = p[(col + 8) * ldm + row];
        a.x[3] = p[(col + 12) * ldm + row];
    }
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 16, __half, row_major> &a, const __half *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 2;
    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col + 1];
    a.x[2] = p[row * ldm + col + 2];
    a.x[3] = p[row * ldm + col + 3];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 16, __half, col_major> &a, const __half *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 2;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
    a.x[2] = p[(col + 2) * ldm + row];
    a.x[3] = p[(col + 3) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 16, __half, row_major> &a, const __half *p, unsigned ldm)
{
    unsigned row = (__lane_id() >> 4) << 2;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[(row + 1) * ldm + col];
    a.x[2] = p[(row + 2) * ldm + col];
    a.x[3] = p[(row + 3) * ldm + col];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 16, __half, col_major> &a, const __half *p, unsigned ldm)
{
    unsigned row = (__lane_id() >> 4) << 2;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[col * ldm + row + 1];
    a.x[2] = p[col * ldm + row + 2];
    a.x[3] = p[col * ldm + row + 3];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 16, __hip_bfloat16, row_major> &a, const __hip_bfloat16 *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 2;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col + 1];
    a.x[2] = p[row * ldm + col + 2];
    a.x[3] = p[row * ldm + col + 3];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 16, __hip_bfloat16, col_major> &a, const __hip_bfloat16 *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 2;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
    a.x[2] = p[(col + 2) * ldm + row];
    a.x[3] = p[(col + 3) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 16, __hip_bfloat16, row_major> &a, const __hip_bfloat16 *p, unsigned ldm)
{
    unsigned row = (__lane_id() >> 4) << 2;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[(row + 1) * ldm + col];
    a.x[2] = p[(row + 2) * ldm + col];
    a.x[3] = p[(row + 3) * ldm + col];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 16, __hip_bfloat16, col_major> &a, const __hip_bfloat16 *p, unsigned ldm)
{
    unsigned row = (__lane_id() >> 4) << 2;
    unsigned col = __lane_id() & 0xf;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[col * ldm + row + 1];
    a.x[2] = p[col * ldm + row + 2];
    a.x[3] = p[col * ldm + row + 3];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 16, float> &a, const float *p, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        a.x[0] = p[row * ldm + col];
        a.x[1] = p[row * ldm + col + 4];
        a.x[2] = p[row * ldm + col + 8];
        a.x[3] = p[row * ldm + col + 12];
    }
    else
    {
        a.x[0] = p[col * ldm + row];
        a.x[1] = p[(col + 4) * ldm + row];
        a.x[2] = p[(col + 8) * ldm + row];
        a.x[3] = p[(col + 12) * ldm + row];
    }
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, signed char, row_major> &a, const signed char *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col + 1];
    a.x[2] = p[row * ldm + col + 2];
    a.x[3] = p[row * ldm + col + 3];
    a.x[4] = p[row * ldm + col + 4];
    a.x[5] = p[row * ldm + col + 5];
    a.x[6] = p[row * ldm + col + 6];
    a.x[7] = p[row * ldm + col + 7];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, signed char, col_major> &a, const signed char *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
    a.x[2] = p[(col + 2) * ldm + row];
    a.x[3] = p[(col + 3) * ldm + row];
    a.x[4] = p[(col + 4) * ldm + row];
    a.x[5] = p[(col + 5) * ldm + row];
    a.x[6] = p[(col + 6) * ldm + row];
    a.x[7] = p[(col + 7) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, signed char, row_major> &a, const signed char *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
    a.x[2] = p[(col + 2) * ldm + row];
    a.x[3] = p[(col + 3) * ldm + row];
    a.x[4] = p[(col + 4) * ldm + row];
    a.x[5] = p[(col + 5) * ldm + row];
    a.x[6] = p[(col + 6) * ldm + row];
    a.x[7] = p[(col + 7) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, signed char, col_major> &a, const signed char *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col + 1];
    a.x[2] = p[row * ldm + col + 2];
    a.x[3] = p[row * ldm + col + 3];
    a.x[4] = p[row * ldm + col + 4];
    a.x[5] = p[row * ldm + col + 5];
    a.x[6] = p[row * ldm + col + 6];
    a.x[7] = p[row * ldm + col + 7];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, unsigned char, row_major> &a, const unsigned char *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col + 1];
    a.x[2] = p[row * ldm + col + 2];
    a.x[3] = p[row * ldm + col + 3];
    a.x[4] = p[row * ldm + col + 4];
    a.x[5] = p[row * ldm + col + 5];
    a.x[6] = p[row * ldm + col + 6];
    a.x[7] = p[row * ldm + col + 7];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, unsigned char, col_major> &a, const unsigned char *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
    a.x[2] = p[(col + 2) * ldm + row];
    a.x[3] = p[(col + 3) * ldm + row];
    a.x[4] = p[(col + 4) * ldm + row];
    a.x[5] = p[(col + 5) * ldm + row];
    a.x[6] = p[(col + 6) * ldm + row];
    a.x[7] = p[(col + 7) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, unsigned char, row_major> &a, const unsigned char *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
    a.x[2] = p[(col + 2) * ldm + row];
    a.x[3] = p[(col + 3) * ldm + row];
    a.x[4] = p[(col + 4) * ldm + row];
    a.x[5] = p[(col + 5) * ldm + row];
    a.x[6] = p[(col + 6) * ldm + row];
    a.x[7] = p[(col + 7) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, unsigned char, col_major> &a, const unsigned char *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col + 1];
    a.x[2] = p[row * ldm + col + 2];
    a.x[3] = p[row * ldm + col + 3];
    a.x[4] = p[row * ldm + col + 4];
    a.x[5] = p[row * ldm + col + 5];
    a.x[6] = p[row * ldm + col + 6];
    a.x[7] = p[row * ldm + col + 7];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 32, int> &a, const int *p, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        a.x[0] = p[row * ldm + col];
        a.x[1] = p[row * ldm + col + 4];
        a.x[2] = p[row * ldm + col + 8];
        a.x[3] = p[row * ldm + col + 12];
    }
    else
    {
        a.x[0] = p[col * ldm + row];
        a.x[1] = p[(col + 4) * ldm + row];
        a.x[2] = p[(col + 8) * ldm + row];
        a.x[3] = p[(col + 12) * ldm + row];
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e4m3, row_major>& a, const __hip_fp8_e4m3* p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col + 1];
    a.x[2] = p[row * ldm + col + 2];
    a.x[3] = p[row * ldm + col + 3];
    a.x[4] = p[row * ldm + col + 4];
    a.x[5] = p[row * ldm + col + 5];
    a.x[6] = p[row * ldm + col + 6];
    a.x[7] = p[row * ldm + col + 7];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e4m3, col_major>& a, const __hip_fp8_e4m3* p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
    a.x[2] = p[(col + 2) * ldm + row];
    a.x[3] = p[(col + 3) * ldm + row];
    a.x[4] = p[(col + 4) * ldm + row];
    a.x[5] = p[(col + 5) * ldm + row];
    a.x[6] = p[(col + 6) * ldm + row];
    a.x[7] = p[(col + 7) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e4m3, row_major>& a, const __hip_fp8_e4m3* p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
    a.x[2] = p[(col + 2) * ldm + row];
    a.x[3] = p[(col + 3) * ldm + row];
    a.x[4] = p[(col + 4) * ldm + row];
    a.x[5] = p[(col + 5) * ldm + row];
    a.x[6] = p[(col + 6) * ldm + row];
    a.x[7] = p[(col + 7) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e4m3, col_major>& a, const __hip_fp8_e4m3* p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col + 1];
    a.x[2] = p[row * ldm + col + 2];
    a.x[3] = p[row * ldm + col + 3];
    a.x[4] = p[row * ldm + col + 4];
    a.x[5] = p[row * ldm + col + 5];
    a.x[6] = p[row * ldm + col + 6];
    a.x[7] = p[row * ldm + col + 7];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e5m2, row_major>& a, const __hip_fp8_e5m2* p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col + 1];
    a.x[2] = p[row * ldm + col + 2];
    a.x[3] = p[row * ldm + col + 3];
    a.x[4] = p[row * ldm + col + 4];
    a.x[5] = p[row * ldm + col + 5];
    a.x[6] = p[row * ldm + col + 6];
    a.x[7] = p[row * ldm + col + 7];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e5m2, col_major>& a, const __hip_fp8_e5m2* p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
    a.x[2] = p[(col + 2) * ldm + row];
    a.x[3] = p[(col + 3) * ldm + row];
    a.x[4] = p[(col + 4) * ldm + row];
    a.x[5] = p[(col + 5) * ldm + row];
    a.x[6] = p[(col + 6) * ldm + row];
    a.x[7] = p[(col + 7) * ldm + row];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e5m2, row_major>& a, const __hip_fp8_e5m2* p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[col * ldm + row];
    a.x[1] = p[(col + 1) * ldm + row];
    a.x[2] = p[(col + 2) * ldm + row];
    a.x[3] = p[(col + 3) * ldm + row];
    a.x[4] = p[(col + 4) * ldm + row];
    a.x[5] = p[(col + 5) * ldm + row];
    a.x[6] = p[(col + 6) * ldm + row];
    a.x[7] = p[(col + 7) * ldm + row];
}
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e5m2, col_major>& a, const __hip_fp8_e5m2* p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = (__lane_id() >> 4) << 3;

    a.x[0] = p[row * ldm + col];
    a.x[1] = p[row * ldm + col + 1];
    a.x[2] = p[row * ldm + col + 2];
    a.x[3] = p[row * ldm + col + 3];
    a.x[4] = p[row * ldm + col + 4];
    a.x[5] = p[row * ldm + col + 5];
    a.x[6] = p[row * ldm + col + 6];
    a.x[7] = p[row * ldm + col + 7];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 32, float>& a, const float* p, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        a.x[0] = p[row * ldm + col];
        a.x[1] = p[row * ldm + col + 4];
        a.x[2] = p[row * ldm + col + 8];
        a.x[3] = p[row * ldm + col + 12];
    }
    else
    {
        a.x[0] = p[col * ldm + row];
        a.x[1] = p[(col + 4) * ldm + row];
        a.x[2] = p[(col + 8) * ldm + row];
        a.x[3] = p[(col + 12) * ldm + row];
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 64, experimental::precision::s4, row_major> &a, const void *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 4) / 2;

    typedef struct
    {
        signed char h : 4;
        signed char l : 4;

    } bitx8;

    bitx8 *sp = (bitx8 *)p;

    union
    {
        intx2 i;
        bitx8 b8[8];
    } tmp;

    tmp.b8[0].l = sp[row * (ldm / 2) + col].l;
    tmp.b8[0].h = sp[row * (ldm / 2) + col].h;
    tmp.b8[1].l = sp[row * (ldm / 2) + col + 1].l;
    tmp.b8[1].h = sp[row * (ldm / 2) + col + 1].h;
    tmp.b8[2].l = sp[row * (ldm / 2) + col + 2].l;
    tmp.b8[2].h = sp[row * (ldm / 2) + col + 2].h;
    tmp.b8[3].l = sp[row * (ldm / 2) + col + 3].l;
    tmp.b8[3].h = sp[row * (ldm / 2) + col + 3].h;

    tmp.b8[4].l = sp[row * (ldm / 2) + col + 4].l;
    tmp.b8[4].h = sp[row * (ldm / 2) + col + 4].h;
    tmp.b8[5].l = sp[row * (ldm / 2) + col + 5].l;
    tmp.b8[5].h = sp[row * (ldm / 2) + col + 5].h;
    tmp.b8[6].l = sp[row * (ldm / 2) + col + 6].l;
    tmp.b8[6].h = sp[row * (ldm / 2) + col + 6].h;
    tmp.b8[7].l = sp[row * (ldm / 2) + col + 7].l;
    tmp.b8[7].h = sp[row * (ldm / 2) + col + 7].h;

    a.x[0] = tmp.i[0];
    a.x[1] = tmp.i[1];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 64, experimental::precision::s4, col_major> &a, const void *p, unsigned ldm)
{
    unsigned row = ((__lane_id() >> 4) << 4) / 2;
    unsigned col = __lane_id() & 0xf;

    typedef struct
    {
        signed char l : 4;
        signed char h : 4;
    } bitx8;

    bitx8 *sp = (bitx8 *)p;

    union
    {
        intx2 i;
        bitx8 b8[8];
    } tmp;

    tmp.b8[0].l = sp[col * (ldm / 2) + row].l;
    tmp.b8[0].h = sp[col * (ldm / 2) + row].h;
    tmp.b8[1].l = sp[col * (ldm / 2) + (row + 1)].l;
    tmp.b8[1].h = sp[col * (ldm / 2) + (row + 1)].h;
    tmp.b8[2].l = sp[col * (ldm / 2) + (row + 2)].l;
    tmp.b8[2].h = sp[col * (ldm / 2) + (row + 2)].h;
    tmp.b8[3].l = sp[col * (ldm / 2) + (row + 3)].l;
    tmp.b8[3].h = sp[col * (ldm / 2) + (row + 3)].h;

    tmp.b8[4].l = sp[col * (ldm / 2) + (row + 4)].l;
    tmp.b8[4].h = sp[col * (ldm / 2) + (row + 4)].h;
    tmp.b8[5].l = sp[col * (ldm / 2) + (row + 5)].l;
    tmp.b8[5].h = sp[col * (ldm / 2) + (row + 5)].h;
    tmp.b8[6].l = sp[col * (ldm / 2) + (row + 6)].l;
    tmp.b8[6].h = sp[col * (ldm / 2) + (row + 6)].h;
    tmp.b8[7].l = sp[col * (ldm / 2) + (row + 7)].l;
    tmp.b8[7].h = sp[col * (ldm / 2) + (row + 7)].h;

    a.x[0] = tmp.i[0];
    a.x[1] = tmp.i[1];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 64, experimental::precision::u4, row_major> &a, const void *p, unsigned ldm)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 4) / 2;

    typedef struct
    {
        signed char l : 4;
        signed char h : 4;
    } bitx8;

    bitx8 *sp = (bitx8 *)p;

    union
    {
        intx2 i;
        bitx8 b8[8];
    } tmp;

    tmp.b8[0].l = sp[row * (ldm / 2) + col].l;
    tmp.b8[0].h = sp[row * (ldm / 2) + col].h;
    tmp.b8[1].l = sp[row * (ldm / 2) + col + 1].l;
    tmp.b8[1].h = sp[row * (ldm / 2) + col + 1].h;
    tmp.b8[2].l = sp[row * (ldm / 2) + col + 2].l;
    tmp.b8[2].h = sp[row * (ldm / 2) + col + 2].h;
    tmp.b8[3].l = sp[row * (ldm / 2) + col + 3].l;
    tmp.b8[3].h = sp[row * (ldm / 2) + col + 3].h;

    tmp.b8[4].l = sp[row * (ldm / 2) + col + 4].l;
    tmp.b8[4].h = sp[row * (ldm / 2) + col + 4].h;
    tmp.b8[5].l = sp[row * (ldm / 2) + col + 5].l;
    tmp.b8[5].h = sp[row * (ldm / 2) + col + 5].h;
    tmp.b8[6].l = sp[row * (ldm / 2) + col + 6].l;
    tmp.b8[6].h = sp[row * (ldm / 2) + col + 6].h;
    tmp.b8[7].l = sp[row * (ldm / 2) + col + 7].l;
    tmp.b8[7].h = sp[row * (ldm / 2) + col + 7].h;

    a.x[0] = tmp.i[0];
    a.x[1] = tmp.i[1];

}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 64, experimental::precision::u4, col_major> &a, const void *p, unsigned ldm)
{
    unsigned row = ((__lane_id() >> 4) << 4) / 2;
    unsigned col = __lane_id() & 0xf;

    typedef struct
    {
        unsigned char l : 4;
        unsigned char h : 4;
    } bitx8;

    bitx8 *sp = (bitx8 *)p;

    union
    {
        intx2 i;
        bitx8 b8[8];
    } tmp;

    tmp.b8[0].l = sp[col * (ldm / 2) + row].l;
    tmp.b8[0].h = sp[col * (ldm / 2) + row].h;
    tmp.b8[1].l = sp[col * (ldm / 2) + (row + 1)].l;
    tmp.b8[1].h = sp[col * (ldm / 2) + (row + 1)].h;
    tmp.b8[2].l = sp[col * (ldm / 2) + (row + 2)].l;
    tmp.b8[2].h = sp[col * (ldm / 2) + (row + 2)].h;
    tmp.b8[3].l = sp[col * (ldm / 2) + (row + 3)].l;
    tmp.b8[3].h = sp[col * (ldm / 2) + (row + 3)].h;

    tmp.b8[4].l = sp[col * (ldm / 2) + (row + 4)].l;
    tmp.b8[4].h = sp[col * (ldm / 2) + (row + 4)].h;
    tmp.b8[5].l = sp[col * (ldm / 2) + (row + 5)].l;
    tmp.b8[5].h = sp[col * (ldm / 2) + (row + 5)].h;
    tmp.b8[6].l = sp[col * (ldm / 2) + (row + 6)].l;
    tmp.b8[6].h = sp[col * (ldm / 2) + (row + 6)].h;
    tmp.b8[7].l = sp[col * (ldm / 2) + (row + 7)].l;
    tmp.b8[7].h = sp[col * (ldm / 2) + (row + 7)].h;

    a.x[0] = tmp.i[0];
    a.x[1] = tmp.i[1];
}

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 64, int> &a, const int *p, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        a.x[0] = p[row * ldm + col];
        a.x[1] = p[row * ldm + col + 4];
        a.x[2] = p[row * ldm + col + 8];
        a.x[3] = p[row * ldm + col + 12];
    }
    else
    {
        a.x[0] = p[col * ldm + row];
        a.x[1] = p[(col + 4) * ldm + row];
        a.x[2] = p[(col + 8) * ldm + row];
        a.x[3] = p[(col + 12) * ldm + row];
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__) || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(float *p, const DUFragment<accumulator, 16, 16, 4, float> &a, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        p[row * ldm + col] = a.x[0];
        p[row * ldm + col + 4] = a.x[1];
        p[row * ldm + col + 8] = a.x[2];
        p[row * ldm + col + 12] = a.x[3];
    }
    else
    {
        p[col * ldm + row] = a.x[0];
        p[(col + 4) * ldm + row] = a.x[1];
        p[(col + 8) * ldm + row] = a.x[2];
        p[(col + 12) * ldm + row] = a.x[3];
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(double *p, const DUFragment<accumulator, 16, 16, 4, double> &a, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        p[row * ldm + col] = a.x[0];
        p[row * ldm + col + 4] = a.x[1];
        p[row * ldm + col + 8] = a.x[2];
        p[row * ldm + col + 12] = a.x[3];
    }
    else
    {
        p[col * ldm + row] = a.x[0];
        p[(col + 4) * ldm + row] = a.x[1];
        p[(col + 8) * ldm + row] = a.x[2];
        p[(col + 12) * ldm + row] = a.x[3];
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx928__)  || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(float *p, const DUFragment<accumulator, 16, 16, 8, float> &a, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        p[row * ldm + col] = a.x[0];
        p[row * ldm + col + 4] = a.x[1];
        p[row * ldm + col + 8] = a.x[2];
        p[row * ldm + col + 12] = a.x[3];
    }
    else
    {
        p[col * ldm + row] = a.x[0];
        p[(col + 4) * ldm + row] = a.x[1];
        p[(col + 8) * ldm + row] = a.x[2];
        p[(col + 12) * ldm + row] = a.x[3];
    }
}

__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(float *p, const DUFragment<accumulator, 16, 16, 16, float> &a, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        p[row * ldm + col] = a.x[0];
        p[row * ldm + col + 4] = a.x[1];
        p[row * ldm + col + 8] = a.x[2];
        p[row * ldm + col + 12] = a.x[3];
    }
    else
    {
        p[col * ldm + row] = a.x[0];
        p[(col + 4) * ldm + row] = a.x[1];
        p[(col + 8) * ldm + row] = a.x[2];
        p[(col + 12) * ldm + row] = a.x[3];
    }
}

__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(int *p, const DUFragment<accumulator, 16, 16, 32, int> &a, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        p[row * ldm + col] = a.x[0];
        p[row * ldm + col + 4] = a.x[1];
        p[row * ldm + col + 8] = a.x[2];
        p[row * ldm + col + 12] = a.x[3];
    }
    else
    {
        p[col * ldm + row] = a.x[0];
        p[(col + 4) * ldm + row] = a.x[1];
        p[(col + 8) * ldm + row] = a.x[2];
        p[(col + 12) * ldm + row] = a.x[3];
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(float *p, const DUFragment<accumulator, 16, 16, 32, float>& a, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        p[row * ldm + col] = a.x[0];
        p[row * ldm + col + 4] = a.x[1];
        p[row * ldm + col + 8] = a.x[2];
        p[row * ldm + col + 12] = a.x[3];
    }
    else
    {
        p[col * ldm + row] = a.x[0];
        p[(col + 4) * ldm + row] = a.x[1];
        p[(col + 8) * ldm + row] = a.x[2];
        p[(col + 12) * ldm + row] = a.x[3];
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(int *p, const DUFragment<accumulator, 16, 16, 64, int> &a, unsigned ldm, layout_t layout)
{
    unsigned row = __lane_id() & 0xf;
    unsigned col = ((__lane_id() >> 4) << 2) / 4;

    if (layout_t::mem_row_major == layout)
    {
        p[row * ldm + col] = a.x[0];
        p[row * ldm + col + 4] = a.x[1];
        p[row * ldm + col + 8] = a.x[2];
        p[row * ldm + col + 12] = a.x[3];
    }
    else
    {
        p[col * ldm + row] = a.x[0];
        p[(col + 4) * ldm + row] = a.x[1];
        p[(col + 8) * ldm + row] = a.x[2];
        p[(col + 12) * ldm + row] = a.x[3];
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__) || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 4, float> &d,
                                        const DUFragment<matrix_a, 16, 16, 4, float, LayoutA> &a,
                                        const DUFragment<matrix_b, 16, 16, 4, float, LayoutB> &b,
                                        const DUFragment<accumulator, 16, 16, 4, float> &c)
{
    const floatx4 cx = *reinterpret_cast<const floatx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_F32_16x16x4F32(a.x[0], b.x[0], cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__) || defined(__gfx936__) || defined(__gfx938__)
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 4, double> &d,
                                        const DUFragment<matrix_a, 16, 16, 4, double, LayoutA> &a,
                                        const DUFragment<matrix_b, 16, 16, 4, double, LayoutB> &b,
                                        const DUFragment<accumulator, 16, 16, 4, double> &c)
{
    const doublex4 cx = *reinterpret_cast<const doublex4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_F32_16x16x4F64(a.x[0], b.x[0], cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx928__)  || defined(__gfx936__) || defined(__gfx938__)
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 8, float> &d,
                                        const DUFragment<matrix_a, 16, 16, 8, float, LayoutA> &a,
                                        const DUFragment<matrix_b, 16, 16, 8, float, LayoutB> &b,
                                        const DUFragment<accumulator, 16, 16, 8, float> &c)
{
    const floatx2 ax = *reinterpret_cast<const floatx2 *>(a.x);
    const floatx2 bx = *reinterpret_cast<const floatx2 *>(b.x);
    const floatx4 cx = *reinterpret_cast<const floatx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_F32_16x16x8F32(ax, bx, cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 8, float> &d,
                                        const DUFragment<matrix_a, 16, 16, 8, precision::tf32, LayoutA> &a,
                                        const DUFragment<matrix_b, 16, 16, 8, precision::tf32, LayoutB> &b,
                                        const DUFragment<accumulator, 16, 16, 8, float> &c)
{
    const floatx2 ax = *reinterpret_cast<const floatx2 *>(a.x);
    const floatx2 bx = *reinterpret_cast<const floatx2 *>(b.x);
    const floatx4 cx = *reinterpret_cast<const floatx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_F32_16x16x8TF32(ax, bx, cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 16, float> &d,
                                        const DUFragment<matrix_a, 16, 16, 16, __half, LayoutA> &a,
                                        const DUFragment<matrix_b, 16, 16, 16, __half, LayoutB> &b,
                                        const DUFragment<accumulator, 16, 16, 16, float> &c)
{
    const halfx4 ax = *reinterpret_cast<const halfx4 *>(a.x);
    const halfx4 bx = *reinterpret_cast<const halfx4 *>(b.x);
    const floatx4 cx = *reinterpret_cast<const floatx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_F32_16x16x16F16(ax, bx, cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 16, float> &d,
                                        const DUFragment<matrix_a, 16, 16, 16, __hip_bfloat16, LayoutA> &a,
                                        const DUFragment<matrix_b, 16, 16, 16, __hip_bfloat16, LayoutB> &b,
                                        const DUFragment<accumulator, 16, 16, 16, float> &c)
{
    const shortx4 ax = *reinterpret_cast<const shortx4 *>(a.x);
    const shortx4 bx = *reinterpret_cast<const shortx4 *>(b.x);
    const floatx4 cx = *reinterpret_cast<const floatx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_F32_16x16x16BF16(ax, bx, cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, int> &d,
                                        const DUFragment<matrix_a, 16, 16, 32, signed char, LayoutA> &a,
                                        const DUFragment<matrix_b, 16, 16, 32, signed char, LayoutB> &b,
                                        const DUFragment<accumulator, 16, 16, 32, int> &c)
{
using scalar_type = intx2;

    const scalar_type ax = *reinterpret_cast<const scalar_type *>(a.x);
    const scalar_type bx = *reinterpret_cast<const scalar_type *>(b.x);
    const intx4 cx = *reinterpret_cast<const intx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_I32_16x16x32I8(ax, bx, cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, int> &d,
                                        const DUFragment<matrix_a, 16, 16, 32, unsigned char, LayoutA> &a,
                                        const DUFragment<matrix_b, 16, 16, 32, unsigned char, LayoutB> &b,
                                        const DUFragment<accumulator, 16, 16, 32, int> &c)
{
using scalar_type = intx2;

    const scalar_type ax = *reinterpret_cast<const scalar_type *>(a.x);
    const scalar_type bx = *reinterpret_cast<const scalar_type *>(b.x);
    const intx4 cx = *reinterpret_cast<const intx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_I32_16x16x32U8(ax, bx, cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e4m3, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e4m3, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 32, float>& c)
{
    const intx2 ax = *reinterpret_cast<const intx2 *>(a.x);
    const intx2 bx = *reinterpret_cast<const intx2 *>(b.x);
    const floatx4 cx = *reinterpret_cast<const floatx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_F32_16x16x32_FP8_FP8(ax, bx, cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e5m2, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e5m2, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 32, float>& c)
{
    const intx2 ax = *reinterpret_cast<const intx2 *>(a.x);
    const intx2 bx = *reinterpret_cast<const intx2 *>(b.x);
    const floatx4 cx = *reinterpret_cast<const floatx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_F32_16x16x32_BF8_BF8(ax, bx, cx);
    
    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e4m3, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e5m2, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 32, float>& c)
{
    const intx2 ax = *reinterpret_cast<const intx2 *>(a.x);
    const intx2 bx = *reinterpret_cast<const intx2 *>(b.x);
    const floatx4 cx = *reinterpret_cast<const floatx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_F32_16x16x32_FP8_BF8(ax, bx, cx);
    
    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e5m2, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e4m3, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 32, float>& c)
{
    const intx2 ax = *reinterpret_cast<const intx2 *>(a.x);
    const intx2 bx = *reinterpret_cast<const intx2 *>(b.x);
    const floatx4 cx = *reinterpret_cast<const floatx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_F32_16x16x32_BF8_FP8(ax, bx, cx);
    
    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 64, int> &d,
                                        const DUFragment<matrix_a, 16, 16, 64, experimental::precision::s4, LayoutA> &a,
                                        const DUFragment<matrix_b, 16, 16, 64, experimental::precision::s4, LayoutB> &b,
                                        const DUFragment<accumulator, 16, 16, 64, int> &c)
{
using scalar_type = intx2;

    const scalar_type ax = *reinterpret_cast<const scalar_type *>(a.x);
    const scalar_type bx = *reinterpret_cast<const scalar_type *>(b.x);
    const intx4 cx = *reinterpret_cast<const intx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_I32_16x16x64I4(ax, bx, cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 64, int> &d,
                                        const DUFragment<matrix_a, 16, 16, 64, experimental::precision::u4, LayoutA> &a,
                                        const DUFragment<matrix_b, 16, 16, 64, experimental::precision::u4, LayoutB> &b,
                                        const DUFragment<accumulator, 16, 16, 64, int> &c)
{
using scalar_type = intx2;

    const scalar_type ax = *reinterpret_cast<const scalar_type *>(a.x);
    const scalar_type bx = *reinterpret_cast<const scalar_type *>(b.x);
    const intx4 cx = *reinterpret_cast<const intx4 *>(c.x);

    __DU_MMA_VOLATILE__ auto temp = __DU_MMA_I32_16x16x64U4(ax, bx, cx);

    d.x[0] = temp[0];
    d.x[1] = temp[1];
    d.x[2] = temp[2];
    d.x[3] = temp[3];
}
#endif
}
}
#endif 
#endif 
