// SPDX-License-Identifier: Apache-2.0
// microbench: unified_attention_2d 的 Q@K 阶段，用 DUMMA m16n16k16 bf16 验证
// 目标：(1) MFMA Q@K 正确性对拍 torch.q@k  (2) 朴素 LDS 基线的 bank conflict
// BLOCK_M 为编译期常量（默认 16，可实例化 64）。BLOCK_M 必须是 16 的倍数。
//
// 编译（算力节点）：
//   source /opt/dtk/env.sh
//   hipcc ua2d_microbench.cu -o ua2d_microbench --offload-arch=gfx936 -O3
//
// 当前范围：单 tile Q@K，不做 softmax/causal/tile 循环。先验证 MFMA + LDS 正确。

#include <hip/hip_runtime.h>
#include "du_mma.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace du::dumma;

#define CHECK_HIP(call) do {                                           \
    hipError_t e_ = (call);                                            \
    if (e_ != hipSuccess) {                                            \
        fprintf(stderr, "HIP error %s at %s:%d\n",                     \
                hipGetErrorString(e_), __FILE__, __LINE__);            \
        std::abort();                                                  \
    }                                                                  \
} while (0)

// ---- 形状常量（对齐 Triton 版 unified_attention_2d） ----
// Qwen3.5: head_dim=256, GQA 16 q-head / 4 kv-head, block_size=784, TILE_SIZE=32
constexpr int HEAD_SIZE = 256;
constexpr int TILE_SIZE = 32;        // K/V 维 tile（对齐 Triton prefill）
constexpr int BLOCK_SIZE = 784;      // KV cache page
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 4;
constexpr int QUERIES_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS;  // 4

// DUMMA m16n16k16: 单条 mma 处理 M=16, N=16, K=16
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;
constexpr int K_LOOP = HEAD_SIZE / MMA_K;  // 256/16 = 16

// ---- LDS 布局：K tile [HEAD_SIZE, TILE_SIZE]，bf16 ----
// 朴素版（无 swizzle）：leading dim = TILE_SIZE 个 bf16 = 64 字节
//   32 bank × 4B = 128B 一行；64B = 半行 → 相邻 2 行落同一组 bank
//   → 16 lane 并发读不同行会撞 bank（预期复现 1.77×）
// 列存：K[row, col] 在 LDS[row * TILE_SIZE + col]
//   这样 lane 读连续 col 是合并的，但不同 row 撞 bank
using k_elem_t = __hip_bfloat16;

template <int BLOCK_M>
__global__ __launch_bounds__(256, 4)
void ua2d_qk_microbench_kernel(
    const __hip_bfloat16* __restrict__ q,       // [num_q_blocks, NUM_Q_HEADS, HEAD_SIZE]
    const __hip_bfloat16* __restrict__ k_cache, // [num_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE]
                                                //   注：这里先用 Triton 的 K 布局 [HEAD_SIZE, TILE_SIZE] 局部
    const int* __restrict__ block_table,        // [num_q_blocks, max_num_blocks]
    float* __restrict__ s_out,                  // [num_q_blocks, BLOCK_M, TILE_SIZE] Q@K 结果
    int max_num_blocks,
    int q_stride_0,    // q.shape[0] stride (elements)
    int kv_head_idx)   // 本 kernel 只算一个 kv head（grid.z 或外部传）
{
    static_assert(BLOCK_M % MMA_M == 0, "BLOCK_M must be multiple of 16");
    constexpr int MMA_M_LOOP = BLOCK_M / MMA_M;  // 1 (BLOCK_M=16) 或 4 (BLOCK_M=64)
    constexpr int TILE_N_LOOP = TILE_SIZE / MMA_N;  // 32/16 = 2

    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpid = tid / 64;
    const int laneid = tid % 64;
    const int num_warps = blockDim.x / 64;  // 256/64 = 4

    // 简化：每个 block 取第一个物理 block 的前 TILE_SIZE 个 token
    // （microbench 不做完整 tile 循环，先验证单 tile）
    const int physical_block = block_table[q_block_idx * max_num_blocks];
    // K 局部 tile [HEAD_SIZE, TILE_SIZE]，从 k_cache 取
    // k_cache 布局: [num_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE]
    //   K[h, t] = k_cache[physical_block * NUM_KV_HEADS*HEAD_SIZE*BLOCK_SIZE
    //                      + kv_head_idx * HEAD_SIZE*BLOCK_SIZE
    //                      + h * BLOCK_SIZE + t]
    const __hip_bfloat16* k_block_base =
        k_cache + physical_block * (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE)
               + kv_head_idx * (HEAD_SIZE * BLOCK_SIZE);

    // ---- K 进 LDS: [HEAD_SIZE, TILE_SIZE] ----
    // 朴素列存布局，leading dim = TILE_SIZE
    __shared__ k_elem_t k_smem[HEAD_SIZE * TILE_SIZE];  // 256*32*2 = 16KB
    // 协作加载：256 threads × 每次 1 个 bf16 → 256*1 = 256 元素/轮
    //   tile 共 256*32 = 8192 元素 → 32 轮
    //   简化：每 thread 直接按 tid 映射（不优化合并，microbench 先跑通）
    for (int i = tid; i < HEAD_SIZE * TILE_SIZE; i += blockDim.x) {
        int h = i / TILE_SIZE;
        int t = i % TILE_SIZE;
        k_smem[h * TILE_SIZE + t] = k_block_base[h * BLOCK_SIZE + t];
    }
    __syncthreads();

    // ---- Q 进寄存器: [BLOCK_M, HEAD_SIZE] ----
    // 简化：BLOCK_M=16 时，4 个 warp 各处理 4 个 query head（GQA）
    //       BLOCK_M=64 时，需要更多 query head 扩展（microbench 先用前 16 head 复制 4 份）
    // 每个 warp 处理 MMA_M=16 行 Q 中的一段。这里先用 warpid 0 做主路径，其它 warp 简化。
    // TODO: BLOCK_M=64 的 Q 加载需要从更多 q_block 取，初版先 BLOCK_M=16 跑通。

    // ---- Q@K 用 DUMMA m16n16k16 ----
    // 结果 S[BLOCK_M, TILE_SIZE] = Q[BLOCK_M, HEAD_SIZE] @ K[HEAD_SIZE, TILE_SIZE]
    // 沿 M 分 MMA_M_LOOP 段，沿 N 分 TILE_N_LOOP 段，沿 K 分 K_LOOP 段
    //
    // DUMMA fragment 布局（m16n16k16 bf16）：
    //   a_frag: matrix_a, 16x16x16, bf16, layout = row_major (Q 是 row-major)
    //   b_frag: matrix_b, 16x16x16, bf16, layout = col_major (K 在 LDS 是 [HEAD_SIZE, TILE_SIZE]，
    //           即 K 行=HEAD_SIZE 维，K 列=TILE_SIZE 维；Q@K 中 K 需要转置→用 col_major 取)
    //   d_frag: accumulator, 16x16x16, float
    //
    // 注意：DUMMA 的 load_matrix_sync 从全局/LDS 指针加载，ldm 是 leading dim。
    //   Q 指针: q + q_block_idx * q_stride_0 + head_offset * HEAD_SIZE
    //   K 指针: k_smem（LDS），leading dim = TILE_SIZE

    // 只 warpid 0 / 第一个 M 段做一次完整 mma，验证正确性
    if (warpid == 0) {
        for (int n_loop = 0; n_loop < TILE_N_LOOP; n_loop++) {
            DUFragment<matrix_a, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> a_frag;
            DUFragment<matrix_b, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> b_frag;
            DUFragment<accumulator, MMA_M, MMA_N, MMA_K, float> c_frag;
            du_fill_fragment(c_frag, 0.0f);

            // K-loop: 累加 16 段。mma d=a*b+c，d==c 别名累加（标准 GEMM 写法）
            for (int kk = 0; kk < K_LOOP; kk++) {
                // Q fragment: Q[0:16, kk*16:(kk+1)*16], row-major, ldm=HEAD_SIZE
                const __hip_bfloat16* q_ptr =
                    q + q_block_idx * q_stride_0 + kk * MMA_K;
                du_load_matrix_sync(a_frag, q_ptr, HEAD_SIZE);

                // K fragment: K[kk*16:(kk+1)*16, n_loop*16:(n_loop+1)*16]
                //   K 在 LDS 是 [HEAD_SIZE, TILE_SIZE] row-major: K[h,t] = k_smem[h*TILE_SIZE+t]
                //   Q@K = Q × K，K 不转置。DUMMA matrix_b 模板参数用 row_major（内存视角）。
                //   b_frag 加载 B[i,j] = K[kk*16+i, n_loop*16+j] = k_smem[(kk*16+i)*TILE_SIZE + (n_loop*16+j)]
                //   起点 = k_smem + (kk*16)*TILE_SIZE + (n_loop*16)，ldm = TILE_SIZE，row_major
                const __hip_bfloat16* k_ptr =
                    k_smem + kk * MMA_K * TILE_SIZE + n_loop * MMA_N;
                du_load_matrix_sync(b_frag, k_ptr, TILE_SIZE);

                du_mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            // store S[0:16, n_loop*16:(n_loop+1)*16]
            float* s_ptr = s_out + q_block_idx * (BLOCK_M * TILE_SIZE)
                                    + n_loop * MMA_N;
            du_store_matrix_sync(s_ptr, c_frag, TILE_SIZE, mem_row_major);
        }
    }
    __syncthreads();
}

// ===================== Q@K global 路径对照（DUMMA，K 不进 LDS）=====================
// 目标：和 LDS 路径对比 PMC，看 DUMMA global_load 的 VMEM/L2 stall
// K 直接从全局内存 DUMMA load（不进 LDS），所有 4 warp 参与，多 tile 循环
// BLOCK_M=64（4 warp × 16），沿序列 TILE_LOOP 个 K tile
template <int BLOCK_M, int TILE_LOOP>
__global__ __launch_bounds__(256, 4)
void ua2d_qk_global_kernel(
    const __hip_bfloat16* __restrict__ q,       // [num_q_blocks, BLOCK_M, HEAD_SIZE]
    const __hip_bfloat16* __restrict__ k_cache, // [num_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE]
    const int* __restrict__ block_table,        // [num_q_blocks, max_num_blocks]
    float* __restrict__ s_out,                  // [num_q_blocks, BLOCK_M, TILE_LOOP*TILE_SIZE]
    int max_num_blocks,
    int q_stride_0,
    int kv_head_idx)
{
    static_assert(BLOCK_M % MMA_M == 0, "BLOCK_M must be multiple of 16");
    constexpr int MMA_M_LOOP = BLOCK_M / MMA_M;     // 4 (BLOCK_M=64)
    constexpr int TILE_N_LOOP = TILE_SIZE / MMA_N;  // 2
    // 每 warp 处理一个 M 段（BLOCK_M=64, 4 warp → 每 warp 16 行）
    static_assert(MMA_M_LOOP == 4, "global path designed for BLOCK_M=64, 4 warps");

    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpid = tid / 64;

    const int* block_table_seq = block_table + q_block_idx * max_num_blocks;
    const __hip_bfloat16* q_base = q + q_block_idx * q_stride_0;

    // 每 warp 处理 M 段 warpid*16 : (warpid+1)*16
    const int m_loop = warpid;  // 0..3

    for (int tile = 0; tile < TILE_LOOP; tile++) {
        // 物理 block：tile 跨多个物理 block（tile_idx 在序列里映射到物理 block）
        // 简化：tile → block_table_seq[tile % max_num_blocks]
        int phys = block_table_seq[tile % max_num_blocks];
        const __hip_bfloat16* k_block_base =
            k_cache + phys * (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE)
                   + kv_head_idx * (HEAD_SIZE * BLOCK_SIZE);
        // K tile [HEAD_SIZE, TILE_SIZE]：K[h, t] = k_block_base[h * BLOCK_SIZE + t]
        //   （HEAD_SIZE 慢变，BLOCK_SIZE 快变；取前 TILE_SIZE=32 个 token）
        // 注意：不进 LDS，直接全局地址传给 DUMMA load

        for (int n_loop = 0; n_loop < TILE_N_LOOP; n_loop++) {
            DUFragment<matrix_a, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> a_frag;
            DUFragment<matrix_b, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> b_frag;
            DUFragment<accumulator, MMA_M, MMA_N, MMA_K, float> c_frag;
            du_fill_fragment(c_frag, 0.0f);

            for (int kk = 0; kk < K_LOOP; kk++) {
                // Q fragment: Q[m_loop*16:(m_loop+1)*16, kk*16:(kk+1)*16]
                //   Q row-major [num_q_blocks, BLOCK_M, HEAD_SIZE]
                //   起点 = q_base + m_loop*16*HEAD_SIZE + kk*MMA_K，ldm=HEAD_SIZE
                const __hip_bfloat16* q_ptr = q_base + m_loop * MMA_M * HEAD_SIZE + kk * MMA_K;
                du_load_matrix_sync(a_frag, q_ptr, HEAD_SIZE);

                // K fragment: K[kk*16:(kk+1)*16, n_loop*16:(n_loop+1)*16]
                //   K[h,t] = k_block_base[h*BLOCK_SIZE + t]
                //   B[i,j] = K[kk*16+i, n_loop*16+j] = k_block_base[(kk*16+i)*BLOCK_SIZE + (n_loop*16+j)]
                //   起点 = k_block_base + (kk*16)*BLOCK_SIZE + (n_loop*16)，ldm=BLOCK_SIZE，row_major
                const __hip_bfloat16* k_ptr =
                    k_block_base + kk * MMA_K * BLOCK_SIZE + n_loop * MMA_N;
                du_load_matrix_sync(b_frag, k_ptr, BLOCK_SIZE);

                du_mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            // store S[m_loop*16:(m_loop+1)*16, tile*TILE_SIZE + n_loop*16 : ...]
            float* s_ptr = s_out + q_block_idx * (BLOCK_M * TILE_LOOP * TILE_SIZE)
                                + m_loop * MMA_M * (TILE_LOOP * TILE_SIZE)
                                + tile * TILE_SIZE + n_loop * MMA_N;
            du_store_matrix_sync(s_ptr, c_frag, TILE_LOOP * TILE_SIZE, mem_row_major);
        }
    }
}

// ===================== Q@K LDS 路径（对照 global，K 进 LDS）=====================
// 和 ua2d_qk_global_kernel 结构完全对齐（BLOCK_M=64, 8 tile, 4 warp），
// 唯一区别：K 协作加载到 __shared__ 再 DUMMA load。PMC 对比看 LDS vs global stall
template <int BLOCK_M, int TILE_LOOP, int K_LDS_PAD = 0, bool SKIP_Q_LOAD = false, bool SKIP_K_LOAD = false, bool HANDWRITE_K_LOAD = false, bool K_SWIZZLE = false, bool K_TRANSPOSE = false>
__global__ __launch_bounds__(256, 4)
void ua2d_qk_lds_kernel(
    const __hip_bfloat16* __restrict__ q,
    const __hip_bfloat16* __restrict__ k_cache,
    const int* __restrict__ block_table,
    float* __restrict__ s_out,
    int max_num_blocks,
    int q_stride_0,
    int kv_head_idx)
{
    static_assert(BLOCK_M % MMA_M == 0, "BLOCK_M must be multiple of 16");
    constexpr int MMA_M_LOOP = BLOCK_M / MMA_M;
    constexpr int TILE_N_LOOP = TILE_SIZE / MMA_N;
    static_assert(MMA_M_LOOP == 4, "lds path designed for BLOCK_M=64, 4 warps");
    // K LDS leading dim：朴素 TILE_SIZE，padding 版 TILE_SIZE+K_LDS_PAD
    //   朴素 TILE_SIZE=32 → 64字节 = 半 bank 行宽，相邻2行撞同 bank
    //   PAD=16 → 48 → 96字节，96%128≠0 打破周期
    constexpr int K_LDS_STRIDE = TILE_SIZE + K_LDS_PAD;

    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpid = tid / 64;

    const int* block_table_seq = block_table + q_block_idx * max_num_blocks;
    const __hip_bfloat16* q_base = q + q_block_idx * q_stride_0;
    const int m_loop = warpid;

    // ===== K LDS 物理布局 =====
    // 非转置: K[h,t] row-major, k_smem[h*STRIDE + t*COL_SCALE], 维度 [HEAD_SIZE, TILE_SIZE]
    //   swizzle：col 间隔2(让16 lane col0-15落16不同bank) + row stride 34(非128倍数+4对齐)
    //     col 0-15 物理地址 0,2,..30 → byte 0,4,..60 → bank 0-15 全不同
    //     row+1 偏移 34*2=68B，68%4=0，68/4=17→bank偏移17
    // 转置: K[t,h] row-major, k_smem[t*KT_STRIDE + h], 维度 [TILE_SIZE, HEAD_SIZE]
    //   matrix_b row_major 读 B[row+i,col]=K[kk*16+row+i, n_loop*16+col]
    //     转置物理 = k_smem[(n_loop*16+col)*KT_STRIDE + (kk*16+row+i)]
    //   每 lane 读 4 个连续 bf16（h 维连续，i=0..3）→ 可 ds_read_b64 一条指令
    //   col 0-15 → 16 个 t 行，间隔 KT_STRIDE*2 字节；KT_STRIDE 非整 bank 周期可分散 bank
    constexpr int K_PHYS_STRIDE = K_SWIZZLE ? (TILE_SIZE * 2 + 2)
                                            : (K_TRANSPOSE ? HEAD_SIZE : K_LDS_STRIDE);
    constexpr int K_COL_SCALE = K_SWIZZLE ? 2 : 1;
    // 转置版物理布局：[TILE_SIZE, HEAD_SIZE]，KT_STRIDE 行步进
    //   朴素 KT_STRIDE=256 → bank偏移0 → 16 col 全同 bank（BANK_CF=499712）
    //   pad KT_STRIDE=258 → bank偏移1 → 相邻col共享1 bank（ds_read2_b32 8B跨2bank）→ 16384
    //   pad KT_STRIDE=260 → bank偏移2 → 16 col 全不同 + 相邻不共享 → 8192
    //   pad KT_STRIDE=262 → bank偏移3 → gcd(3,32)=1 周期32 → 试是否到 0
    //   规律：每 pad+2 conflict 减半
    constexpr int KT_STRIDE = K_TRANSPOSE ? (HEAD_SIZE + 6) : HEAD_SIZE;  // 262
    // k_smem 大小：转置 [TILE_SIZE, KT_STRIDE]=32*258=8256 bf16≈16.5KB
    //              非转置 [HEAD_SIZE, K_PHYS_STRIDE]
    constexpr int K_SMEM_SIZE = K_TRANSPOSE ? (TILE_SIZE * KT_STRIDE)
                                            : (HEAD_SIZE * K_PHYS_STRIDE);
    __shared__ __hip_bfloat16 k_smem[K_SMEM_SIZE];

    for (int tile = 0; tile < TILE_LOOP; tile++) {
        int phys = block_table_seq[tile % max_num_blocks];
        const __hip_bfloat16* k_block_base =
            k_cache + phys * (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE)
                   + kv_head_idx * (HEAD_SIZE * BLOCK_SIZE);

        // 协作加载 K tile [HEAD_SIZE, TILE_SIZE] 到 LDS（按物理布局写）
        if (K_TRANSPOSE) {
            // 转置写：k_smem[t*KT_STRIDE + h] = k_block_base[h*BLOCK_SIZE + t]
            // K[t,h] row-major，行步进 KT_STRIDE=HEAD_SIZE
            for (int i = tid; i < HEAD_SIZE * TILE_SIZE; i += blockDim.x) {
                int h = i / TILE_SIZE;
                int t = i % TILE_SIZE;
                k_smem[t * KT_STRIDE + h] = k_block_base[h * BLOCK_SIZE + t];
            }
        } else {
            for (int i = tid; i < HEAD_SIZE * TILE_SIZE; i += blockDim.x) {
                int h = i / TILE_SIZE;
                int t = i % TILE_SIZE;
                k_smem[h * K_PHYS_STRIDE + t * K_COL_SCALE] = k_block_base[h * BLOCK_SIZE + t];
            }
        }
        __syncthreads();

        for (int n_loop = 0; n_loop < TILE_N_LOOP; n_loop++) {
            DUFragment<matrix_a, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> a_frag;
            DUFragment<matrix_b, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> b_frag;
            DUFragment<accumulator, MMA_M, MMA_N, MMA_K, float> c_frag;
            du_fill_fragment(c_frag, 0.0f);

            for (int kk = 0; kk < K_LOOP; kk++) {
                const __hip_bfloat16* q_ptr = q_base + m_loop * MMA_M * HEAD_SIZE + kk * MMA_K;
                if (SKIP_Q_LOAD) {
                    du_fill_fragment(a_frag, __float2bfloat16(0.0f));  // 跳过 Q load，量 bank conflict 来自哪
                } else {
                    du_load_matrix_sync(a_frag, q_ptr, HEAD_SIZE);
                }

                // K 从 LDS load: B[i,j] = K[kk*16+i, n_loop*16+j]（K_logical[h,t]）
                //   朴素/pad DUMMA load: k_smem + (kk*16)*K_LDS_STRIDE + n_loop*16，ldm=K_LDS_STRIDE
                //   手写/swizzle(非转置): 按 matrix_b lane 映射 + 物理地址(K_PHYS_STRIDE, col*K_COL_SCALE)
                //   转置: K[t,h] row-major, B[row+i,col]=K_logical[kk*16+row+i,n_loop*16+col]
                //         = k_smem[(n_loop*16+col)*KT_STRIDE + (kk*16+row+i)]
                //         每 lane 读 4 个连续 bf16（h 维连续）→ 可 ds_read_b64
                if (K_TRANSPOSE) {
                    unsigned lane = __lane_id();
                    unsigned row = (lane >> 4) << 2;   // K h 维（kk*16 方向）
                    unsigned col = lane & 0xf;          // K t 维（n_loop*16 方向）
                    unsigned base = (n_loop * MMA_N + col) * KT_STRIDE
                                  + (kk * MMA_K + row);
                    b_frag.x[0] = k_smem[base + 0];
                    b_frag.x[1] = k_smem[base + 1];
                    b_frag.x[2] = k_smem[base + 2];
                    b_frag.x[3] = k_smem[base + 3];
                } else {
                    const __hip_bfloat16* k_ptr =
                        k_smem + kk * MMA_K * K_PHYS_STRIDE + n_loop * MMA_N * K_COL_SCALE;
                    if (SKIP_K_LOAD) {
                        du_fill_fragment(b_frag, __float2bfloat16(0.0f));
                    } else if (HANDWRITE_K_LOAD || K_SWIZZLE) {
                        // 手写 load：matrix_b row_major lane 映射
                        //   row = (laneid>>4)<<2, col = laneid&0xf, 每 lane 读 4 行同列
                        //   swizzle 时物理地址用 K_PHYS_STRIDE 行步进 + col*K_COL_SCALE 列步进
                        unsigned lane = __lane_id();
                        unsigned row = (lane >> 4) << 2;
                        unsigned col = lane & 0xf;
                        unsigned base = (kk * MMA_K + row) * K_PHYS_STRIDE
                                      + (n_loop * MMA_N + col) * K_COL_SCALE;
                        b_frag.x[0] = k_smem[base + 0 * K_PHYS_STRIDE];
                        b_frag.x[1] = k_smem[base + 1 * K_PHYS_STRIDE];
                        b_frag.x[2] = k_smem[base + 2 * K_PHYS_STRIDE];
                        b_frag.x[3] = k_smem[base + 3 * K_PHYS_STRIDE];
                    } else {
                        du_load_matrix_sync(b_frag, k_ptr, K_PHYS_STRIDE);
                    }
                }

                du_mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            float* s_ptr = s_out + q_block_idx * (BLOCK_M * TILE_LOOP * TILE_SIZE)
                                + m_loop * MMA_M * (TILE_LOOP * TILE_SIZE)
                                + tile * TILE_SIZE + n_loop * MMA_N;
            du_store_matrix_sync(s_ptr, c_frag, TILE_LOOP * TILE_SIZE, mem_row_major);
        }
        __syncthreads();  // 下个 tile 覆盖 k_smem 前同步
    }
}

// ===================== P@V microbench =====================
// 验证 P@V 的 V 进 LDS 朴素布局的 bank conflict（预期高，复现 Triton 1.77× 主因）
// P [BLOCK_M, TILE_SIZE] (float, 来自 Q@K 结果) → 转 bf16
// V [TILE_SIZE, HEAD_SIZE] 朴素 LDS
// out [BLOCK_M, HEAD_SIZE] = P @ V
template <int BLOCK_M, int V_LDS_PAD>   // V_LDS_PAD: V leading dim 的 padding 列数
__global__ __launch_bounds__(256, 4)
void ua2d_pv_microbench_kernel(
    const __hip_bfloat16* __restrict__ p_in,    // [num_q_blocks, BLOCK_M, TILE_SIZE] bf16 (P)
    const __hip_bfloat16* __restrict__ v_cache, // [num_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE]
    const int* __restrict__ block_table,
    float* __restrict__ out,                    // [num_q_blocks, BLOCK_M, HEAD_SIZE]
    int max_num_blocks,
    int kv_head_idx)
{
    static_assert(BLOCK_M % MMA_M == 0, "BLOCK_M must be multiple of 16");
    constexpr int MMA_M_LOOP = BLOCK_M / MMA_M;
    constexpr int N_LOOP = HEAD_SIZE / MMA_N;     // 256/16 = 16
    constexpr int K_LOOP_PV = TILE_SIZE / MMA_K;  // 32/16 = 2
    // V LDS leading dim = HEAD_SIZE + V_LDS_PAD (bf16 元素数)
    constexpr int V_LDS_STRIDE = HEAD_SIZE + V_LDS_PAD;

    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpid = tid / 64;

    const int physical_block = block_table[q_block_idx * max_num_blocks];
    const __hip_bfloat16* v_block_base =
        v_cache + physical_block * (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE)
               + kv_head_idx * (HEAD_SIZE * BLOCK_SIZE);
    // V 局部 tile [TILE_SIZE, HEAD_SIZE]: V[t, h] = v_block_base[t * BLOCK_SIZE + h]
    //   注: v_cache 布局 [num_blocks, kv_heads, HEAD_SIZE, BLOCK_SIZE]，
    //       所以 V[t,h] 在全局是 v_block_base[h * BLOCK_SIZE + t]
    //       （HEAD_SIZE 维是慢变，BLOCK_SIZE(token) 维是快变）
    //   → V[t,h] 全局 = v_block_base[h * BLOCK_SIZE + t]

    // ---- V 进 LDS: [TILE_SIZE, HEAD_SIZE]，朴素 row-major (leading dim = V_LDS_STRIDE) ----
    //   V[t, h] 在 LDS = v_smem[t * V_LDS_STRIDE + h]
    //   朴素版 V_LDS_PAD=0 时 leading dim=256，512 字节 = 32 bank 周期 → 相邻 t 撞同 bank
    __shared__ __hip_bfloat16 v_smem[TILE_SIZE * V_LDS_STRIDE];
    for (int i = tid; i < TILE_SIZE * HEAD_SIZE; i += blockDim.x) {
        int t = i / HEAD_SIZE;
        int h = i % HEAD_SIZE;
        // 全局 V[t,h] = v_block_base[h * BLOCK_SIZE + t]
        v_smem[t * V_LDS_STRIDE + h] = v_block_base[h * BLOCK_SIZE + t];
    }
    __syncthreads();

    // ---- P@V 用 DUMMA m16n16k16 ----
    // out[BLOCK_M, HEAD_SIZE] = P[BLOCK_M, TILE_SIZE] @ V[TILE_SIZE, HEAD_SIZE]
    // 沿 M 分 MMA_M_LOOP, 沿 N 分 N_LOOP, 沿 K 分 K_LOOP_PV
    if (warpid == 0) {
        for (int m_loop = 0; m_loop < MMA_M_LOOP; m_loop++) {
            for (int n_loop = 0; n_loop < N_LOOP; n_loop++) {
                DUFragment<matrix_a, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> a_frag;
                DUFragment<matrix_b, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> b_frag;
                DUFragment<accumulator, MMA_M, MMA_N, MMA_K, float> c_frag;
                du_fill_fragment(c_frag, 0.0f);

                for (int kk = 0; kk < K_LOOP_PV; kk++) {
                    // P fragment: P[m_loop*16:(m_loop+1)*16, kk*16:(kk+1)*16]
                    //   P 在全局 row-major [num_q_blocks, BLOCK_M, TILE_SIZE]
                    //   起点 = p_in + q_block_idx*BLOCK_M*TILE_SIZE + m_loop*MMA_M*TILE_SIZE + kk*MMA_K
                    //   ldm = TILE_SIZE
                    const __hip_bfloat16* p_ptr =
                        p_in + q_block_idx * (BLOCK_M * TILE_SIZE)
                             + m_loop * MMA_M * TILE_SIZE + kk * MMA_K;
                    du_load_matrix_sync(a_frag, p_ptr, TILE_SIZE);

                    // V fragment: V[kk*16:(kk+1)*16, n_loop*16:(n_loop+1)*16]
                    //   V 在 LDS row-major [TILE_SIZE, V_LDS_STRIDE]
                    //   B[i,j] = V[kk*16+i, n_loop*16+j] = v_smem[(kk*16+i)*V_LDS_STRIDE + (n_loop*16+j)]
                    //   起点 = v_smem + (kk*16)*V_LDS_STRIDE + (n_loop*16)，ldm = V_LDS_STRIDE
                    const __hip_bfloat16* v_ptr =
                        v_smem + kk * MMA_K * V_LDS_STRIDE + n_loop * MMA_N;
                    du_load_matrix_sync(b_frag, v_ptr, V_LDS_STRIDE);

                    du_mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                // store out[m_loop*16:(m_loop+1)*16, n_loop*16:(n_loop+1)*16]
                float* o_ptr = out + q_block_idx * (BLOCK_M * HEAD_SIZE)
                                    + m_loop * MMA_M * HEAD_SIZE + n_loop * MMA_N;
                du_store_matrix_sync(o_ptr, c_frag, HEAD_SIZE, mem_row_major);
            }
        }
    }
    __syncthreads();
}

// ===================== host 端：初始化 + launch + 对拍 =====================

// bf16 随机填充（用 float 转 bf16）
static void fill_random_bf16(__hip_bfloat16* p, int n, unsigned seed) {
    std::srand(seed);
    for (int i = 0; i < n; i++) {
        float v = (std::rand() / float(RAND_MAX) - 0.5f) * 2.0f;  // [-1, 1)
        p[i] = __float2bfloat16(v);
    }
}

// CPU 参考：S_ref[q_block, m, t] = sum_h Q[q_block, m, h] * K[phys_block, h, t]
//   K[phys_block, h, t] = k_cache[phys_block * (NUM_KV_HEADS*HEAD_SIZE*BLOCK_SIZE)
//                                  + kv_head_idx * (HEAD_SIZE*BLOCK_SIZE)
//                                  + h * BLOCK_SIZE + t]
static void cpu_ref_qk(
    const __hip_bfloat16* q,           // [num_q_blocks, BLOCK_M, HEAD_SIZE]
    const __hip_bfloat16* k_cache,     // [num_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE]
    const int* block_table,            // [num_q_blocks, max_num_blocks]
    float* s_ref,                      // [num_q_blocks, BLOCK_M, TILE_SIZE]
    int num_q_blocks, int BLOCK_M, int max_num_blocks, int kv_head_idx)
{
    for (int qb = 0; qb < num_q_blocks; qb++) {
        int phys = block_table[qb * max_num_blocks];
        for (int m = 0; m < BLOCK_M; m++) {
            for (int t = 0; t < TILE_SIZE; t++) {
                float acc = 0.0f;
                for (int h = 0; h < HEAD_SIZE; h++) {
                    float qv = __bfloat162float(q[qb * BLOCK_M * HEAD_SIZE + m * HEAD_SIZE + h]);
                    float kv = __bfloat162float(
                        k_cache[phys * (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE)
                              + kv_head_idx * (HEAD_SIZE * BLOCK_SIZE)
                              + h * BLOCK_SIZE + t]);
                    acc += qv * kv;
                }
                s_ref[qb * BLOCK_M * TILE_SIZE + m * TILE_SIZE + t] = acc;
            }
        }
    }
}

// CPU 参考：P@V  out[qb, m, h] = sum_t P[qb, m, t] * V[phys, t, h]
//   V[phys, t, h] = k_cache...[h * BLOCK_SIZE + t]  (v_cache 布局 [kv_heads, HEAD_SIZE, BLOCK_SIZE])
static void cpu_ref_pv(
    const __hip_bfloat16* p_in,         // [num_q_blocks, BLOCK_M, TILE_SIZE]
    const __hip_bfloat16* v_cache,      // [num_blocks, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE]
    const int* block_table,
    float* out_ref,                     // [num_q_blocks, BLOCK_M, HEAD_SIZE]
    int num_q_blocks, int BLOCK_M, int max_num_blocks, int kv_head_idx)
{
    for (int qb = 0; qb < num_q_blocks; qb++) {
        int phys = block_table[qb * max_num_blocks];
        const __hip_bfloat16* v_base =
            v_cache + phys * (NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE)
                   + kv_head_idx * (HEAD_SIZE * BLOCK_SIZE);
        for (int m = 0; m < BLOCK_M; m++) {
            for (int h = 0; h < HEAD_SIZE; h++) {
                float acc = 0.0f;
                for (int t = 0; t < TILE_SIZE; t++) {
                    float pv = __bfloat162float(
                        p_in[qb * BLOCK_M * TILE_SIZE + m * TILE_SIZE + t]);
                    float vv = __bfloat162float(v_base[h * BLOCK_SIZE + t]);
                    acc += pv * vv;
                }
                out_ref[qb * BLOCK_M * HEAD_SIZE + m * HEAD_SIZE + h] = acc;
            }
        }
    }
}

int main() {
    constexpr int BLOCK_M = 16;        // 初版朴素先用 16
    const int num_q_blocks = 4;
    const int num_blocks = 8;          // 物理 block 数
    const int max_num_blocks = 4;
    const int kv_head_idx = 0;

    // Q: [num_q_blocks, BLOCK_M, HEAD_SIZE]
    const int q_cnt = num_q_blocks * BLOCK_M * HEAD_SIZE;
    const int k_cnt = num_blocks * NUM_KV_HEADS * HEAD_SIZE * BLOCK_SIZE;
    const int bt_cnt = num_q_blocks * max_num_blocks;
    const int s_cnt = num_q_blocks * BLOCK_M * TILE_SIZE;

    std::vector<__hip_bfloat16> h_q(q_cnt), h_k(k_cnt);
    std::vector<int> h_bt(bt_cnt);
    std::vector<float> h_s(s_cnt, 0.0f), h_s_ref(s_cnt, 0.0f);

    fill_random_bf16(h_q.data(), q_cnt, 42);
    fill_random_bf16(h_k.data(), k_cnt, 7);
    for (int i = 0; i < bt_cnt; i++) h_bt[i] = i % num_blocks;  // 简单映射

    __hip_bfloat16 *d_q, *d_k;
    int *d_bt;
    float *d_s;
    CHECK_HIP(hipMalloc(&d_q, q_cnt * sizeof(__hip_bfloat16)));
    CHECK_HIP(hipMalloc(&d_k, k_cnt * sizeof(__hip_bfloat16)));
    CHECK_HIP(hipMalloc(&d_bt, bt_cnt * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_s, s_cnt * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_q, h_q.data(), q_cnt * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_k, h_k.data(), k_cnt * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_bt, h_bt.data(), bt_cnt * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_s, 0, s_cnt * sizeof(float)));

    // Q stride_0 = BLOCK_M * HEAD_SIZE（Q row-major [num_q_blocks, BLOCK_M, HEAD_SIZE]）
    const int q_stride_0 = BLOCK_M * HEAD_SIZE;

    dim3 grid(num_q_blocks, 1);          // (q_blocks, kv_heads) — kv_head 固定 0，简化
    dim3 block(256);
    ua2d_qk_microbench_kernel<BLOCK_M><<<grid, block>>>(
        d_q, d_k, d_bt, d_s, max_num_blocks, q_stride_0, kv_head_idx);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_s.data(), d_s, s_cnt * sizeof(float), hipMemcpyDeviceToHost));

    // CPU 参考
    cpu_ref_qk(h_q.data(), h_k.data(), h_bt.data(), h_s_ref.data(),
               num_q_blocks, BLOCK_M, max_num_blocks, kv_head_idx);

    // 对拍
    float max_err = 0.0f, max_abs = 0.0f;
    for (int i = 0; i < s_cnt; i++) {
        float diff = std::fabs(h_s[i] - h_s_ref[i]);
        if (diff > max_err) max_err = diff;
        if (std::fabs(h_s_ref[i]) > max_abs) max_abs = std::fabs(h_s_ref[i]);
    }
    printf("=== ua2d Q@K microbench (BLOCK_M=%d, TILE_SIZE=%d, HEAD_SIZE=%d) ===\n",
           BLOCK_M, TILE_SIZE, HEAD_SIZE);
    printf("num_q_blocks=%d, s_cnt=%d\n", num_q_blocks, s_cnt);
    printf("max_abs_ref = %.4f\n", max_abs);
    printf("max_err     = %.6f\n", max_err);
    printf("rel_err     = %.6f%%\n", max_abs > 0 ? (max_err / max_abs) * 100 : 0);
    // 打印前几个值对比
    printf("\nfirst 8 (gpu vs ref):\n");
    for (int i = 0; i < 8 && i < s_cnt; i++) {
        printf("  [%d] gpu=%.5f  ref=%.5f  diff=%.6f\n",
               i, h_s[i], h_s_ref[i], std::fabs(h_s[i] - h_s_ref[i]));
    }
    bool ok = (max_err < 0.05f) || (max_abs > 0 && max_err / max_abs < 0.01f);
    printf("\nRESULT: %s\n", ok ? "PASS" : "FAIL");

    // ============================================================
    // Q@K global 路径对照（DUMMA，K 不进 LDS，多 tile，所有 warp）
    // 目标：hipprof --pmc 抓这个 kernel 的 VMEM/L2 stall，和 LDS 路径对比
    // 不严格对拍（mma 逻辑同 Q@K，只换 K 地址来源），只验证不崩
    // ============================================================
    printf("\n=== ua2d Q@K global path (DUMMA no-LDS, multi-tile) ===\n");
    {
        constexpr int TILE_LOOP = 8;  // 多 tile 循环
        constexpr int BM = 64;        // global kernel 要 4 warp，BLOCK_M=64
        const int gs_cnt = num_q_blocks * BM * TILE_LOOP * TILE_SIZE;
        // Q 要 BM=64 行，但原 Q 是 BLOCK_M=16。补一个 BM=64 的 Q（复制 4 份）
        const int q64_cnt = num_q_blocks * BM * HEAD_SIZE;
        std::vector<__hip_bfloat16> h_q64(q64_cnt);
        for (int qb = 0; qb < num_q_blocks; qb++)
            for (int m4 = 0; m4 < 4; m4++)
                for (int m = 0; m < 16; m++)
                    for (int h = 0; h < HEAD_SIZE; h++)
                        h_q64[qb*BM*HEAD_SIZE + (m4*16+m)*HEAD_SIZE + h] =
                            h_q[qb*BLOCK_M*HEAD_SIZE + m*HEAD_SIZE + h];
        __hip_bfloat16* d_q64;
        float* d_sg;
        CHECK_HIP(hipMalloc(&d_q64, q64_cnt * sizeof(__hip_bfloat16)));
        CHECK_HIP(hipMalloc(&d_sg, gs_cnt * sizeof(float)));
        CHECK_HIP(hipMemcpy(d_q64, h_q64.data(), q64_cnt * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemset(d_sg, 0, gs_cnt * sizeof(float)));
        const int q64_stride_0 = BM * HEAD_SIZE;
        ua2d_qk_global_kernel<BM, TILE_LOOP><<<dim3(num_q_blocks,1), dim3(256)>>>(
            d_q64, d_k, d_bt, d_sg, max_num_blocks, q64_stride_0, kv_head_idx);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        // 不对拍，只确认跑通。打印前几个值确认非 NaN
        std::vector<float> h_sg(gs_cnt);
        CHECK_HIP(hipMemcpy(h_sg.data(), d_sg, gs_cnt * sizeof(float), hipMemcpyDeviceToHost));
        printf("global path ran. first vals: %.4f %.4f %.4f (nan=%d)\n",
               h_sg[0], h_sg[1], h_sg[2], std::isnan(h_sg[0]));

        // LDS 路径对照（同 workload，K 进 LDS）
        float* d_sl;
        CHECK_HIP(hipMalloc(&d_sl, gs_cnt * sizeof(float)));
        CHECK_HIP(hipMemset(d_sl, 0, gs_cnt * sizeof(float)));
        ua2d_qk_lds_kernel<BM, TILE_LOOP><<<dim3(num_q_blocks,1), dim3(256)>>>(
            d_q64, d_k, d_bt, d_sl, max_num_blocks, q64_stride_0, kv_head_idx);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<float> h_sl(gs_cnt);
        CHECK_HIP(hipMemcpy(h_sl.data(), d_sl, gs_cnt * sizeof(float), hipMemcpyDeviceToHost));
        printf("lds path ran. first vals: %.4f %.4f %.4f (nan=%d)\n",
               h_sl[0], h_sl[1], h_sl[2], std::isnan(h_sl[0]));
        // 两个路径数值应一致（同 Q@K，只差 K 来源）
        float cmp_err = 0.0f;
        for (int i = 0; i < gs_cnt; i++) {
            float d = std::fabs(h_sg[i] - h_sl[i]);
            if (d > cmp_err) cmp_err = d;
        }
        printf("global vs lds max diff = %.6f (应≈0)\n", cmp_err);

        // padding 版 LDS（K_LDS_PAD=16，leading dim 32→48 打破 bank 周期）
        float* d_sp;
        CHECK_HIP(hipMalloc(&d_sp, gs_cnt * sizeof(float)));
        CHECK_HIP(hipMemset(d_sp, 0, gs_cnt * sizeof(float)));
        ua2d_qk_lds_kernel<BM, TILE_LOOP, 16><<<dim3(num_q_blocks,1), dim3(256)>>>(
            d_q64, d_k, d_bt, d_sp, max_num_blocks, q64_stride_0, kv_head_idx);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<float> h_sp(gs_cnt);
        CHECK_HIP(hipMemcpy(h_sp.data(), d_sp, gs_cnt * sizeof(float), hipMemcpyDeviceToHost));
        float pad_err = 0.0f;
        for (int i = 0; i < gs_cnt; i++) {
            float d = std::fabs(h_sg[i] - h_sp[i]);
            if (d > pad_err) pad_err = d;
        }
        printf("lds PAD=16 ran. vs global max diff = %.6f (应≈0，验证 padding 不破坏数值)\n", pad_err);

        // SKIP_Q_LOAD 版：Q 用 fill(0) 跳过 load，量 bank conflict 来自 Q load 还是 K 协作加载
        // 结果数值不对，只用于 hipprof --pmc 抓 BANK_CONFLICT
        float* d_sq;
        CHECK_HIP(hipMalloc(&d_sq, gs_cnt * sizeof(float)));
        CHECK_HIP(hipMemset(d_sq, 0, gs_cnt * sizeof(float)));
        ua2d_qk_lds_kernel<BM, TILE_LOOP, 0, true><<<dim3(num_q_blocks,1), dim3(256)>>>(
            d_q64, d_k, d_bt, d_sq, max_num_blocks, q64_stride_0, kv_head_idx);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        printf("SKIP_Q_LOAD 版 ran (Q填0，只用于 PMC)\n");

        // SKIP_QK 版：Q 和 K 都跳过 load，量 bank conflict 是否来自协作加载写入
        float* d_skq;
        CHECK_HIP(hipMalloc(&d_skq, gs_cnt * sizeof(float)));
        CHECK_HIP(hipMemset(d_skq, 0, gs_cnt * sizeof(float)));
        ua2d_qk_lds_kernel<BM, TILE_LOOP, 0, true, true><<<dim3(num_q_blocks,1), dim3(256)>>>(
            d_q64, d_k, d_bt, d_skq, max_num_blocks, q64_stride_0, kv_head_idx);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        printf("SKIP_QK 版 ran (Q+K都填0，只用于 PMC)\n");

        // 手写 K load 版（朴素，验证能和 DUMMA mma 正确配合）
        float* d_hw;
        CHECK_HIP(hipMalloc(&d_hw, gs_cnt * sizeof(float)));
        CHECK_HIP(hipMemset(d_hw, 0, gs_cnt * sizeof(float)));
        ua2d_qk_lds_kernel<BM, TILE_LOOP, 0, false, false, true><<<dim3(num_q_blocks,1), dim3(256)>>>(
            d_q64, d_k, d_bt, d_hw, max_num_blocks, q64_stride_0, kv_head_idx);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<float> h_hw(gs_cnt);
        CHECK_HIP(hipMemcpy(h_hw.data(), d_hw, gs_cnt * sizeof(float), hipMemcpyDeviceToHost));
        float hw_err = 0.0f;
        for (int i = 0; i < gs_cnt; i++) {
            float d = std::fabs(h_sg[i] - h_hw[i]);
            if (d > hw_err) hw_err = d;
        }
        printf("手写 K load 版 vs global max diff = %.6f (应≈0，验证手写load能配DUMMA mma)\n", hw_err);

        // swizzle 版：K 物理布局 col 间隔2，手写 load 按间隔地址读，让16 lane落16不同bank
        float* d_sw;
        CHECK_HIP(hipMalloc(&d_sw, gs_cnt * sizeof(float)));
        CHECK_HIP(hipMemset(d_sw, 0, gs_cnt * sizeof(float)));
        ua2d_qk_lds_kernel<BM, TILE_LOOP, 0, false, false, false, true><<<dim3(num_q_blocks,1), dim3(256)>>>(
            d_q64, d_k, d_bt, d_sw, max_num_blocks, q64_stride_0, kv_head_idx);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<float> h_sw(gs_cnt);
        CHECK_HIP(hipMemcpy(h_sw.data(), d_sw, gs_cnt * sizeof(float), hipMemcpyDeviceToHost));
        float sw_err = 0.0f;
        for (int i = 0; i < gs_cnt; i++) {
            float d = std::fabs(h_sg[i] - h_sw[i]);
            if (d > sw_err) sw_err = d;
        }
        printf("swizzle版 vs global max diff = %.6f (应≈0)\n", sw_err);

        // K 转置版：K 在 LDS 存成 [TILE_SIZE, HEAD_SIZE] row-major
        //   matrix_b row_major 读 B[row+i,col] → 转置物理 k_smem[(n_loop*16+col)*HEAD_SIZE + (kk*16+row+i)]
        //   每 lane 读 4 个连续 bf16（h 维连续）→ 可 ds_read_b64，改变 conflict 模式
        //   目标：看 BANK_CF 是否从 32768 下降
        float* d_kt;
        CHECK_HIP(hipMalloc(&d_kt, gs_cnt * sizeof(float)));
        CHECK_HIP(hipMemset(d_kt, 0, gs_cnt * sizeof(float)));
        ua2d_qk_lds_kernel<BM, TILE_LOOP, 0, false, false, false, false, true>
            <<<dim3(num_q_blocks,1), dim3(256)>>>(
                d_q64, d_k, d_bt, d_kt, max_num_blocks, q64_stride_0, kv_head_idx);
        CHECK_HIP(hipGetLastError());
        CHECK_HIP(hipDeviceSynchronize());
        std::vector<float> h_kt(gs_cnt);
        CHECK_HIP(hipMemcpy(h_kt.data(), d_kt, gs_cnt * sizeof(float), hipMemcpyDeviceToHost));
        float kt_err = 0.0f;
        for (int i = 0; i < gs_cnt; i++) {
            float d = std::fabs(h_sg[i] - h_kt[i]);
            if (d > kt_err) kt_err = d;
        }
        printf("K转置版 vs global max diff = %.6f (应≈0，验证转置布局数值正确)\n", kt_err);
        CHECK_HIP(hipFree(d_kt));

        // ============================================================
        // 单调用耗时对比：朴素 DUMMA vs K转置260（同 workload，重复 N 次取中位数）
        //   扫不同 TILE_LOOP（单 dispatch 内 tile 数）看加速比随 workload 变长是否拉开
        //   TILE_LOOP 越大 = 序列越长（长 prefill），bank conflict 累积越多
        // ============================================================
        // 计时宏：TL 是 TILE_LOOP 字面量
        #define BENCH_TL(TL) \
        do { \
            const int gs_tl = num_q_blocks * BM * (TL) * TILE_SIZE; \
            float* d_naive_out; \
            CHECK_HIP(hipMalloc(&d_naive_out, BIG_GRID * gs_tl * sizeof(float))); \
            float* d_kt_out; \
            CHECK_HIP(hipMalloc(&d_kt_out, BIG_GRID * gs_tl * sizeof(float))); \
            /* warmup */ \
            for (int g = 0; g < BIG_GRID; g++) \
                ua2d_qk_lds_kernel<BM, (TL)><<<dim3(num_q_blocks,1), dim3(256)>>>( \
                    d_q64, d_k, d_bt, d_naive_out + g * gs_tl, max_num_blocks, q64_stride_0, kv_head_idx); \
            for (int g = 0; g < BIG_GRID; g++) \
                ua2d_qk_lds_kernel<BM, (TL), 0, false, false, false, false, true><<<dim3(num_q_blocks,1), dim3(256)>>>( \
                    d_q64, d_k, d_bt, d_kt_out + g * gs_tl, max_num_blocks, q64_stride_0, kv_head_idx); \
            CHECK_HIP(hipDeviceSynchronize()); \
            /* 朴素计时 */ \
            CHECK_HIP(hipEventRecord(start)); \
            for (int r = 0; r < N_REPEAT; r++) \
                for (int g = 0; g < BIG_GRID; g++) \
                    ua2d_qk_lds_kernel<BM, (TL)><<<dim3(num_q_blocks,1), dim3(256)>>>( \
                        d_q64, d_k, d_bt, d_naive_out + g * gs_tl, max_num_blocks, q64_stride_0, kv_head_idx); \
            CHECK_HIP(hipEventRecord(stop)); \
            CHECK_HIP(hipEventSynchronize(stop)); \
            float ms_naive = 0; \
            CHECK_HIP(hipEventElapsedTime(&ms_naive, start, stop)); \
            /* 转置260计时 */ \
            CHECK_HIP(hipEventRecord(start)); \
            for (int r = 0; r < N_REPEAT; r++) \
                for (int g = 0; g < BIG_GRID; g++) \
                    ua2d_qk_lds_kernel<BM, (TL), 0, false, false, false, false, true><<<dim3(num_q_blocks,1), dim3(256)>>>( \
                        d_q64, d_k, d_bt, d_kt_out + g * gs_tl, max_num_blocks, q64_stride_0, kv_head_idx); \
            CHECK_HIP(hipEventRecord(stop)); \
            CHECK_HIP(hipEventSynchronize(stop)); \
            float ms_kt = 0; \
            CHECK_HIP(hipEventElapsedTime(&ms_kt, start, stop)); \
            float pc_naive = ms_naive / (N_REPEAT * BIG_GRID) * 1000.0f; \
            float pc_kt = ms_kt / (N_REPEAT * BIG_GRID) * 1000.0f; \
            printf("TILE_LOOP=%-4d  朴素=%.3fus  转置260=%.3fus  加速=%.3fx\n", \
                   (TL), pc_naive, pc_kt, ms_naive / ms_kt); \
            CHECK_HIP(hipFree(d_naive_out)); \
            CHECK_HIP(hipFree(d_kt_out)); \
        } while(0)

        printf("\n=== 单调用耗时 vs TILE_LOOP（朴素 vs K转置260，BM=64）===\n");
        {
            constexpr int N_REPEAT = 100;
            constexpr int BIG_GRID = 8;
            hipEvent_t start, stop;
            CHECK_HIP(hipEventCreate(&start));
            CHECK_HIP(hipEventCreate(&stop));
            // 扫 TILE_LOOP = 8, 64, 256, 512（模拟短→长 prefill）
            BENCH_TL(8);
            BENCH_TL(64);
            BENCH_TL(256);
            BENCH_TL(512);
            CHECK_HIP(hipEventDestroy(start));
            CHECK_HIP(hipEventDestroy(stop));
        }
        #undef BENCH_TL

        CHECK_HIP(hipFree(d_sl));
        CHECK_HIP(hipFree(d_sp));
        CHECK_HIP(hipFree(d_sq));
        CHECK_HIP(hipFree(d_skq));
        CHECK_HIP(hipFree(d_hw));
        CHECK_HIP(hipFree(d_sw));
        CHECK_HIP(hipFree(d_q64));
        CHECK_HIP(hipFree(d_sg));
    }

    CHECK_HIP(hipFree(d_q));
    CHECK_HIP(hipFree(d_k));
    CHECK_HIP(hipFree(d_bt));
    CHECK_HIP(hipFree(d_s));

    // ============================================================
    // P@V microbench：朴素 V_LDS_PAD=0 vs padding V_LDS_PAD=16
    // 目标：对比 V 朴素布局 vs padding 布局的正确性 + bank conflict
    // P 用独立随机 bf16（不走 Q@K→softmax），隔离测 P@V
    // ============================================================
    printf("\n=== ua2d P@V microbench (V 朴素 vs padding) ===\n");
    const int p_cnt = num_q_blocks * BLOCK_M * TILE_SIZE;
    const int o_cnt = num_q_blocks * BLOCK_M * HEAD_SIZE;
    std::vector<__hip_bfloat16> h_p(p_cnt);
    fill_random_bf16(h_p.data(), p_cnt, 99);

    // V cache 复用 h_k（同布局 [num_blocks, kv_heads, HEAD_SIZE, BLOCK_SIZE]）
    __hip_bfloat16 *d_p;
    float *d_o;
    CHECK_HIP(hipMalloc(&d_p, p_cnt * sizeof(__hip_bfloat16)));
    CHECK_HIP(hipMalloc(&d_o, o_cnt * sizeof(float)));
    CHECK_HIP(hipMemcpy(d_p, h_p.data(), p_cnt * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    // 重新分配 d_k（上面已 free）+ 重新拷贝
    CHECK_HIP(hipMalloc(&d_k, k_cnt * sizeof(__hip_bfloat16)));
    CHECK_HIP(hipMemcpy(d_k, h_k.data(), k_cnt * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    CHECK_HIP(hipMalloc(&d_bt, bt_cnt * sizeof(int)));
    CHECK_HIP(hipMemcpy(d_bt, h_bt.data(), bt_cnt * sizeof(int), hipMemcpyHostToDevice));

    std::vector<float> h_o(o_cnt, 0.0f), h_o_ref(o_cnt, 0.0f);

    // CPU 参考
    cpu_ref_pv(h_p.data(), h_k.data(), h_bt.data(), h_o_ref.data(),
               num_q_blocks, BLOCK_M, max_num_blocks, kv_head_idx);

    // --- 朴素 V_LDS_PAD=0 ---
    CHECK_HIP(hipMemset(d_o, 0, o_cnt * sizeof(float)));
    ua2d_pv_microbench_kernel<BLOCK_M, 0><<<grid, block>>>(
        d_p, d_k, d_bt, d_o, max_num_blocks, kv_head_idx);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_o.data(), d_o, o_cnt * sizeof(float), hipMemcpyDeviceToHost));
    float pv_err_plain = 0.0f, pv_abs_plain = 0.0f;
    for (int i = 0; i < o_cnt; i++) {
        float diff = std::fabs(h_o[i] - h_o_ref[i]);
        if (diff > pv_err_plain) pv_err_plain = diff;
        if (std::fabs(h_o_ref[i]) > pv_abs_plain) pv_abs_plain = std::fabs(h_o_ref[i]);
    }
    printf("[V_LDS_PAD=0 朴素] max_err=%.6f rel_err=%.6f%% (max_abs=%.4f)\n",
           pv_err_plain, pv_abs_plain > 0 ? pv_err_plain / pv_abs_plain * 100 : 0, pv_abs_plain);

    // --- padding V_LDS_PAD=16 (leading dim 256→272, 272%64=16≠0 打破 bank 周期) ---
    CHECK_HIP(hipMemset(d_o, 0, o_cnt * sizeof(float)));
    ua2d_pv_microbench_kernel<BLOCK_M, 16><<<grid, block>>>(
        d_p, d_k, d_bt, d_o, max_num_blocks, kv_head_idx);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_o.data(), d_o, o_cnt * sizeof(float), hipMemcpyDeviceToHost));
    float pv_err_pad = 0.0f;
    for (int i = 0; i < o_cnt; i++) {
        float diff = std::fabs(h_o[i] - h_o_ref[i]);
        if (diff > pv_err_pad) pv_err_pad = diff;
    }
    printf("[V_LDS_PAD=16 padding] max_err=%.6f rel_err=%.6f%%\n",
           pv_err_pad, pv_abs_plain > 0 ? pv_err_pad / pv_abs_plain * 100 : 0);

    bool pv_ok = (pv_err_plain < 0.05f) && (pv_err_pad < 0.05f);
    printf("\nP@V RESULT: %s (朴素与 padding 数值应一致且对拍通过)\n", pv_ok ? "PASS" : "FAIL");
    printf("注: bank conflict 对比需 hipprof --pmc --pmc-type 3 单独抓两个 kernel\n");

    CHECK_HIP(hipFree(d_p));
    CHECK_HIP(hipFree(d_o));
    CHECK_HIP(hipFree(d_k));
    CHECK_HIP(hipFree(d_bt));
    return (ok && pv_ok) ? 0 : 1;
}
