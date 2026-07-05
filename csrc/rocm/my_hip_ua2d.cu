// my_hip_ua2d.cu
// 自定义 HIP unified attention 2d kernel（prefill 主热点）
// 核心优化：K 在 LDS 转置存储 [TILE_SIZE, HEAD_SIZE] + KT_STRIDE=260
//   让 matrix_b 的「4行同列」读取变成「1行4连续列」→ ds_read2_b32 向量化 + bank conflict 4×↓
// 参考 microbench 验证：BANK_CF 32768→8192，Q@K 段单调用 1.69× 加速（稳定不随序列变长）
//
// 当前版本（完整 attention，bf16 路径）：
//   - bf16 Q/K/V（int8/fp8 KV cache descale 后续补）
//   - causal mask
//   - online softmax（FlashAttention 风格，寄存器 accumulator + __shfl_xor 行 reduce）
//   - K 转置 260 布局，V 朴素
//   - BLOCK_SIZE=784 通用路径（seq_offset % BLOCK_SIZE，支持任意 block_size）
//   - BLOCK_M=96，TILE_SIZE=32，HEAD_SIZE=256
//     q_per_kv=6 时每个 q_block 覆盖 16 个完整 query token
//
// 接口对齐 vllm triton_unified_attention.kernel_unified_attention_2d（子集）

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <limits>

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)
#include "du_mma_dtk_gfx936.h"
#define MY_UA_ENABLE_DUMMA 1
#else
#define MY_UA_ENABLE_DUMMA 0
#endif

#ifndef MY_UA_STANDALONE
// 集成 vllm 时启用 torch 头 + launch 函数；裸编验证时用 -DMY_UA_STANDALONE 跳过
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#ifdef USE_ROCM
#include <c10/hip/HIPException.h>
#else
#include <c10/cuda/CUDAException.h>
#endif
#endif

// ===== 常量（Qwen3.5 full attention）=====
//   HEAD_SIZE=256, num_attention_heads=16, num_key_value_heads=4, head_dim=256
//   BLOCK_SIZE=784（KV cache 物理 block），TILE_SIZE=32（attention tile）
//   full attention 层：0-based 3,7,11,15,19,23,27,31
static constexpr int HEAD_SIZE = 256;
static constexpr int TILE_SIZE = 32;          // attention tile（K 的 t 维 tile）
static constexpr int BLOCK_M = 96;            // q_per_kv=6 时 16 token × 6 q_head
static constexpr int MMA_M = 16, MMA_N = 16, MMA_K = 16;
static constexpr int K_LOOP = HEAD_SIZE / MMA_K;          // 16（Q@K 的 kk 循环）
static constexpr int N_LOOP = TILE_SIZE / MMA_N;          // 2（Q@K 的 n_loop）
static constexpr int K_LOOP_PV = TILE_SIZE / MMA_K;       // 2（P@V 的 kk 循环）
static constexpr int N_LOOP_PV = HEAD_SIZE / MMA_N;       // 16（P@V 的 n_loop）
static constexpr int MMA_M_LOOP = BLOCK_M / MMA_M;        // 6（= wavefront 数）
static constexpr int NUM_THREADS = MMA_M_LOOP * 64;

#if MY_UA_ENABLE_DUMMA

using namespace du::dumma;

// K 转置开关：默认开，保留 1.69× Q@K 单调用收益路径。
//   如需 A/B 朴素布局，可编译时加 MY_UA_DISABLE_K_TRANSPOSE。
//   关闭时 K 回朴素布局，LDS 从 33KB 降到 32KB，换取 occupancy 1→2 block/CU。
//   tradeoff：K 转置降 bank conflict 但 BCF 未转化成 stall（WAIT_LDS 低）；
//   撤转置让 BCF 回升但 occupancy 翻倍，global load latency hiding 改善。
//   A/B 用 26 tile case wall-clock 对比。
//   注：K 转置布局 [t, h]（stride KT_STRIDE=260）vs 朴素布局 [h, t]（stride TILE_SIZE=32）
//       两者物理形状都是 [32, 256] 等价元素数，但行/列含义互换，stride 不同。
#ifndef MY_UA_DISABLE_K_TRANSPOSE
static constexpr int KT_STRIDE = HEAD_SIZE + 4;           // 260（转置：t 行 h 列）
#else
static constexpr int KT_STRIDE = TILE_SIZE;               // 32（朴素：h 行 t 列，stride=t 维宽）
#endif

// matrix_b row_major lane 映射（du_mma.hpp）：
//   row = (lane_id >> 4) << 2, col = lane_id & 0xf, 每 lane 读 4 行同列
// 转置后物理 K[t,h]：B[row+i, col] = K_logical[kk*16+row+i, n_loop*16+col]
//   = k_smem[(n_loop*16+col)*KT_STRIDE + (kk*16+row+i)]，i=0..3 连续 bf16
#ifndef MY_UA_DISABLE_K_TRANSPOSE
__device__ __forceinline__ void load_k_frag_transpose(
    DUFragment<matrix_b, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> &b_frag,
    const __hip_bfloat16* __restrict__ k_smem,
    int n_loop, int kk)
{
    unsigned lane = __lane_id();
    unsigned row = (lane >> 4) << 2;   // K h 维（kk*16 方向）
    unsigned col = lane & 0xf;          // K t 维（n_loop*16 方向）
    unsigned base = (n_loop * MMA_N + col) * KT_STRIDE + (kk * MMA_K + row);
    b_frag.x[0] = k_smem[base + 0];
    b_frag.x[1] = k_smem[base + 1];
    b_frag.x[2] = k_smem[base + 2];
    b_frag.x[3] = k_smem[base + 3];
}
#else
// K 朴素 load（不转置，k_smem[h, t] 行主序，h 行 t 列，stride=KT_STRIDE=TILE_SIZE=32）
//   B[row+i, col] = K_logical[kk*16+row+i, n_loop*16+col] = k_smem[(kk*16+row+i)*KT_STRIDE + (n_loop*16+col)]
//   用 du_load_matrix_sync 黑盒 load（物理布局与 V 朴素同构，仅 stride 不同）
__device__ __forceinline__ void load_k_frag_transpose(
    DUFragment<matrix_b, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> &b_frag,
    const __hip_bfloat16* __restrict__ k_smem,
    int n_loop, int kk)
{
    const __hip_bfloat16* k_ptr =
        k_smem + kk * MMA_K * KT_STRIDE + n_loop * MMA_N;
    du_load_matrix_sync(b_frag, k_ptr, KT_STRIDE);
}
#endif

// V 朴素 load（V 不转置，记忆 v-transpose-not-worth-it：V BANK_CF=1024/WAIT_LDS=0 不是瓶颈）
//   V[t,h] 物理，B[row+i,col]=V[kk*16+row+i, n_loop*16+col]=v_smem[(kk*16+row+i)*V_STRIDE+(n_loop*16+col)]
__device__ __forceinline__ void load_v_frag(
    DUFragment<matrix_b, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> &b_frag,
    const __hip_bfloat16* __restrict__ v_smem,
    int n_loop, int kk, int v_stride)
{
    const __hip_bfloat16* v_ptr =
        v_smem + kk * MMA_K * v_stride + n_loop * MMA_N;
    du_load_matrix_sync(b_frag, v_ptr, v_stride);
}

// ===== 主 kernel：my_hip_unified_attention_2d =====
//   grid: (num_q_blocks, num_kv_heads)
//   block: 384（6 wavefront × 64 lane），每个 wavefront 算 16 行
//   BLOCK_M=96。q_per_kv=6 时 BLOCK_Q=16，正好启用 96 行。
//     offs_m = warpid*16 + (lane&0xf)
//     query_pos = q_block_local*BLOCK_Q + offs_m // num_queries_per_kv
//     q_head = kv_head_idx*num_queries_per_kv + offs_m % num_queries_per_kv
//
// DUMMA accumulator (16x16,k=16) lane 映射（du_mma.hpp store 反推）：
//   row = lane_id & 0xf（0-15），col_grp = lane_id >> 4（0-3）
//   x[i] 对应列 col_grp + i*4（即 col_grp, col_grp+4, col_grp+8, col_grp+12）
//   同一行的 4 个 lane（row 同，col_grp 0-3）需 reduce → __shfl_xor mask 16/32
// LDS 占用（gfx936 LDS 64KB/CU）：
//   K 转置：k_smem 32×260×2=16640B + v_smem 32×256×2=16384B = 33024B → 1 block/CU（放不下 2）
//   K 朴素：k_smem 256×32×2=16384B + v_smem 32×256×2=16384B = 32768B → 2 block/CU（临界可放）
//   launch_bounds 第二参数随开关给：转置=1（务实，避免编译器按 2 分寄存器导致 spill），
//   朴素=2（让编译器按 2 block/CU 分寄存器，目标 occupancy 翻倍）。
#ifndef MY_UA_DISABLE_K_TRANSPOSE
__global__ __launch_bounds__(NUM_THREADS, 1)
#else
__global__ __launch_bounds__(NUM_THREADS, 2)
#endif
void my_hip_unified_attention_2d_kernel(
    __hip_bfloat16* __restrict__ output,        // [num_tokens, num_q_heads, head_size]
    const __hip_bfloat16* __restrict__ query,   // [num_tokens, num_q_heads, head_size]
    const __hip_bfloat16* __restrict__ key_cache,
    const __hip_bfloat16* __restrict__ value_cache,
    const int* __restrict__ block_table,        // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,           // [num_seqs]
    const int* __restrict__ query_start_len,    // [num_seqs+1]
    float scale,
    int num_q_heads,
    int num_kv_heads,
    int num_queries_per_kv,
    int block_table_stride,
    int query_stride_0,
    int query_stride_1,
    int output_stride_0,
    int output_stride_1,
    int BLOCK_SIZE,                             // 784
    int num_seqs,
    long long stride_k_cache_0,
    long long stride_k_cache_1,
    long long stride_k_cache_2,
    long long stride_v_cache_0,
    long long stride_v_cache_1,
    long long stride_v_cache_2)
{
    const int q_block_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int warpid = tid / 64;
    // BLOCK_Q = 每个 q_block 覆盖的 query token 数（不是 BLOCK_M！）
    //   BLOCK_M=96 是 mma 行数；BLOCK_Q 向下取整，保证每个 q_block 只覆盖完整 query token。
    //   q_per_kv=6 时 BLOCK_Q=16，active_m_count=96。
    const int BLOCK_Q = BLOCK_M / num_queries_per_kv;
    const int active_m_count = BLOCK_Q * num_queries_per_kv;


    // ---- find_seq_idx（线性查，对齐 Triton q_block 模式）----
    //   q_block 模式：seq s 的 q_block 范围是 [qsl[s]//BLOCK_Q + s, qsl[s+1]//BLOCK_Q + (s+1))
    //   即把 query_start_len 转成 q_block 单位并加 seq 偏移（每 seq 的 q_block 局部编号从 0 开始）
    //   host 用上界 floor(total_q/BLOCK_Q)+num_seqs 启动，可能多 launch 落在最后 seq 的 padding 区，
    //   这种 block 会在下面 q_block_local_idx 检查处 return。s 越界保护：到 num_seqs 仍未找到则 return。
    int seq_idx = 0, q_start = 0, q_stop = 0;
    bool found = false;
    for (int s = 0; s <= num_seqs && !found; s++) {
        if (s >= num_seqs) break;  // 防止读 query_start_len[num_seqs+1] / seq_lens[num_seqs] 越界
        int start = query_start_len[s];
        int stop = query_start_len[s + 1];
        int qb_start = start / BLOCK_Q + s;
        int qb_stop = stop / BLOCK_Q + (s + 1);
        if (q_block_idx >= qb_start && q_block_idx < qb_stop) {
            seq_idx = s; q_start = start; q_stop = stop; found = true;
        }
    }
    if (!found) return;
    const int cur_batch_query_len = q_stop - q_start;
    const int q_block_local_idx = q_block_idx - (q_start / BLOCK_Q + seq_idx);
    if (q_block_local_idx * BLOCK_Q >= cur_batch_query_len) return;

    const int seq_len = seq_lens[seq_idx];
    int context_len = seq_len - cur_batch_query_len;
    if (context_len < 0) context_len = 0;


    const int m_base = warpid * MMA_M;  // 该 wavefront 负责的 offs_m 起始

    // ---- LDS 分配 ----
    // ---- LDS 分配 ----
    //   转置：k_smem[t * KT_STRIDE + h]，KT_STRIDE=260，形状 [32, 260]，8320 bf16=16640B
    //   朴素：k_smem[h * KT_STRIDE + t]，KT_STRIDE=TILE_SIZE=32，形状 [256, 32]，8192 bf16=16384B
    //   两者元素数接近（8320 vs 8192），朴素省 256B（4 元素 pad），总 LDS 33024→32768B → 2 block/CU
#ifndef MY_UA_DISABLE_K_TRANSPOSE
    __shared__ __hip_bfloat16 k_smem[TILE_SIZE * (HEAD_SIZE + 4)];  // 转置 [32, 260]
#else
    __shared__ __hip_bfloat16 k_smem[HEAD_SIZE * TILE_SIZE];        // 朴素 [256, 32]
#endif
    __shared__ __hip_bfloat16 v_smem[TILE_SIZE * HEAD_SIZE];        // 朴素 [32, 256]

    // ---- causal num_tiles 剪枝（对齐 Triton max_seq_prefix_len 语义）----
    //   只需覆盖当前 q_block 内最远 query token 能看到的 K 前缀：
    //   context_len + q_block_local_idx*BLOCK_Q + BLOCK_Q
    //   并上界到 seq_len（防止越界读 K cache）。远端 tile 由 mask 全置 -inf，跳过省算。
    int max_seq_prefix_len = context_len
        + q_block_local_idx * BLOCK_Q
        + BLOCK_Q;
    if (max_seq_prefix_len > seq_len) max_seq_prefix_len = seq_len;
    if (max_seq_prefix_len < 0) max_seq_prefix_len = 0;
    const int num_tiles = (max_seq_prefix_len + TILE_SIZE - 1) / TILE_SIZE;

    // ---- P@V accumulator（寄存器）：16 个 c_frag ----
    DUFragment<accumulator, MMA_M, MMA_N, MMA_K, float> acc_frag[N_LOOP_PV];
    #pragma unroll
    for (int n = 0; n < N_LOOP_PV; n++) du_fill_fragment(acc_frag[n], 0.0f);

    // ---- online softmax 状态（每 lane 1 float，对应自己的 row）----
    float M_row = -INFINITY;
    float L_row = 0.0f;

    // ---- Q 一次性加载进寄存器 ----
    // Q 只与 q_block/kv_head 相关，不随 K/V tile 变化。放在 tile 循环外，避免长 prefill
    // 下按 num_tiles 重复读 global Q。
    const unsigned lane = __lane_id();
    const unsigned row = lane & 0xf;
    const unsigned col_grp = lane >> 4;
    const unsigned q_col = col_grp << 2;
    const int offs_m = m_base + row;
    const bool active_m = offs_m < active_m_count;
    const int q_pos = q_block_local_idx * BLOCK_Q + offs_m / num_queries_per_kv;
    const int q_head = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv;
    const int q_token = q_start + q_pos;
    const bool q_valid = active_m && (q_pos < cur_batch_query_len) &&
                         (q_token < q_stop) && (q_head < num_q_heads);
    const int query_abs_pos = context_len + q_pos;
    const int safe_q_token = q_valid ? q_token : 0;
    const int safe_q_head = q_valid ? q_head : 0;

    DUFragment<matrix_a, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> q_frag[K_LOOP];
    #pragma unroll
    for (int kk = 0; kk < K_LOOP; kk++) {
        const __hip_bfloat16* q_p =
            query + (long long)safe_q_token * query_stride_0
                  + (long long)safe_q_head * query_stride_1
                  + (kk * MMA_K + q_col);
        q_frag[kk].x[0] = q_valid ? q_p[0] : __float2bfloat16(0.0f);
        q_frag[kk].x[1] = q_valid ? q_p[1] : __float2bfloat16(0.0f);
        q_frag[kk].x[2] = q_valid ? q_p[2] : __float2bfloat16(0.0f);
        q_frag[kk].x[3] = q_valid ? q_p[3] : __float2bfloat16(0.0f);
    }

    // ---- 主 tile 循环 ----
    for (int j = 0; j < num_tiles; j++) {
        const int seq_offset_base = j * TILE_SIZE;
        const int page_bt_idx = seq_offset_base / BLOCK_SIZE;
        const int page_blk_off = seq_offset_base - page_bt_idx * BLOCK_SIZE;
        const bool page_local = page_blk_off + TILE_SIZE <= BLOCK_SIZE;
        int page_phys_blk = 0;
        if (page_local && seq_offset_base < max_seq_prefix_len) {
            page_phys_blk = block_table[seq_idx * block_table_stride + page_bt_idx];
        }

        // ---- 协作加载 K/V tile（page-local fastpath + 跨 page 通用路径）----
        for (int i = tid; i < HEAD_SIZE * TILE_SIZE; i += blockDim.x) {
            int h = i / TILE_SIZE;
            int t = i % TILE_SIZE;
            int seq_off = seq_offset_base + t;
            __hip_bfloat16 k_val = __float2bfloat16(0.0f);
            __hip_bfloat16 v_val = __float2bfloat16(0.0f);
            if (seq_off < max_seq_prefix_len) {
                int blk_off;
                int phys_blk;
                if (page_local) {
                    blk_off = page_blk_off + t;
                    phys_blk = page_phys_blk;
                } else {
                    int bt_idx = seq_off / BLOCK_SIZE;
                    blk_off = seq_off - bt_idx * BLOCK_SIZE;
                    phys_blk = block_table[seq_idx * block_table_stride + bt_idx];
                }
                k_val = *(key_cache + (long long)phys_blk * stride_k_cache_0
                                    + (long long)blk_off * stride_k_cache_1
                                    + (long long)kv_head_idx * stride_k_cache_2 + h);
                v_val = *(value_cache + (long long)phys_blk * stride_v_cache_0
                                    + (long long)blk_off * stride_v_cache_1
                                    + (long long)kv_head_idx * stride_v_cache_2 + h);
            }
#ifndef MY_UA_DISABLE_K_TRANSPOSE
            k_smem[t * KT_STRIDE + h] = k_val;       // 转置写：k_smem[t, h]
#else
            k_smem[h * KT_STRIDE + t] = k_val;       // 朴素写：k_smem[h, t]（KT_STRIDE=32）
#endif
            v_smem[t * HEAD_SIZE + h] = v_val;       // 朴素写
        }
        __syncthreads();

        // ---- Q@K → S（2 个 n_loop，寄存器 accumulator）----
        DUFragment<accumulator, MMA_M, MMA_N, MMA_K, float> s_frag[N_LOOP];
        du_fill_fragment(s_frag[0], 0.0f);
        du_fill_fragment(s_frag[1], 0.0f);

        for (int kk = 0; kk < K_LOOP; kk++) {  // 16
            DUFragment<matrix_b, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> b_frag;
            #pragma unroll
            for (int n = 0; n < N_LOOP; n++) {
                load_k_frag_transpose(b_frag, k_smem, n, kk);
                du_mma_sync(s_frag[n], q_frag[kk], b_frag, s_frag[n]);
            }
        }

        // ---- S = scale*Q@K + causal/seq mask ----
        //   s_frag[n].x[i] 对应 S[row, n*16 + col_grp + i*4]
        {
            #pragma unroll
            for (int n = 0; n < N_LOOP; n++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int col = n * MMA_N + col_grp + i * 4;
                    int seq_off = seq_offset_base + col;
                    float s_val = s_frag[n].x[i] * scale;
#ifdef MY_UA_QK_ONLY
                    // Q@K 诊断模式：不做 causal mask，但保留 active_m 防止 partial head 重复写。
                    (void)query_abs_pos;
                    s_frag[n].x[i] = q_valid ? s_val : -INFINITY;
#else
                    if (!q_valid || seq_off > query_abs_pos ||
                        seq_off >= max_seq_prefix_len) {
                        s_val = -INFINITY;
                    }
                    s_frag[n].x[i] = s_val;
#endif
                }
            }
        }

#ifdef MY_UA_QK_ONLY
        // 诊断模式：store S（Q@K*scale，无 softmax/P@V）到 output
        {
            if (q_valid) {
                #pragma unroll
                for (int n = 0; n < N_LOOP; n++) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int h = n * MMA_N + col_grp + i * 4;
                        if (h < TILE_SIZE) {
                            __hip_bfloat16* o_ptr =
                                output + (long long)q_token * output_stride_0
                                       + (long long)q_head * output_stride_1 + h;
                            *o_ptr = __float2bfloat16(s_frag[n].x[i]);
                        }
                    }
                }
            }
        }
        __syncthreads();
        continue;  // 跳过 softmax/P@V
#else

        // ---- online softmax: row_max + 跨 col_grp reduce ----
        float row_max = -INFINITY;
        #pragma unroll
        for (int n = 0; n < N_LOOP; n++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
                row_max = fmaxf(row_max, s_frag[n].x[i]);
        // 同 row 的 4 lane（col_grp 0-3）reduce：lane = row | (col_grp<<4)
        row_max = fmaxf(row_max, __shfl_xor(row_max, 16));
        row_max = fmaxf(row_max, __shfl_xor(row_max, 32));
        float m_j = fmaxf(M_row, row_max);
        if (m_j == -INFINITY) m_j = 0.0f;  // 全 mask 行防 NaN

        // ---- P = exp(S - m_j)，alpha = exp(M - m_j) ----
        float alpha = (M_row == -INFINITY) ? 0.0f : expf(M_row - m_j);
        #pragma unroll
        for (int n = 0; n < N_LOOP; n++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float s_val = s_frag[n].x[i];
                s_frag[n].x[i] = (s_val == -INFINITY) ? 0.0f : expf(s_val - m_j);
            }
        }

        // ---- l_j = rowsum(P) + reduce ----
        float row_sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < N_LOOP; n++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
                row_sum += s_frag[n].x[i];
        row_sum += __shfl_xor(row_sum, 16);
        row_sum += __shfl_xor(row_sum, 32);

        // ---- acc *= alpha（rescale 历史 P@V）----
        #pragma unroll
        for (int n = 0; n < N_LOOP_PV; n++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
                acc_frag[n].x[i] *= alpha;

        M_row = m_j;
        L_row = L_row * alpha + row_sum;

        // ---- P@V → acc ----
        //   P[row, kk*16+col] 需 matrix_a layout（col 连续）
        //   s_frag[kk].x[i_s] 列 = col_grp + i_s*4（不连续）
        //   P[row, col_grp*4+i] 的值在 lane (row | (i<<4)) 的 s_frag[kk].x[col_grp]
        for (int kk = 0; kk < K_LOOP_PV; kk++) {  // 2
            DUFragment<matrix_a, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> p_frag;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned src_lane = row | (i << 4);  // col_grp' = i 的同 row lane
                float p_val = __shfl(s_frag[kk].x[col_grp], src_lane, 64);
                p_frag.x[i] = __float2bfloat16(p_val);
            }

            DUFragment<matrix_b, MMA_M, MMA_N, MMA_K, __hip_bfloat16, row_major> v_frag;
            #pragma unroll
            for (int n = 0; n < N_LOOP_PV; n++) {
                load_v_frag(v_frag, v_smem, n, kk, HEAD_SIZE);
                du_mma_sync(acc_frag[n], p_frag, v_frag, acc_frag[n]);
            }
        }

        __syncthreads();
#endif  // MY_UA_QK_ONLY
    }

    // ---- epilogue：acc /= L，store output ----
    //   acc_frag[n].x[i] → output[row, n*16 + col_grp + i*4]
#ifndef MY_UA_QK_ONLY
    float inv_L = (L_row > 0.0f) ? (1.0f / L_row) : 0.0f;
    {
        if (q_valid) {
            #pragma unroll
            for (int n = 0; n < N_LOOP_PV; n++) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int h = n * MMA_N + col_grp + i * 4;
                    float val = acc_frag[n].x[i] * inv_L;
                    __hip_bfloat16* o_ptr =
                        output + (long long)q_token * output_stride_0
                               + (long long)q_head * output_stride_1 + h;
                    *o_ptr = __float2bfloat16(val);
                }
            }
        }
    }
#endif  // !MY_UA_QK_ONLY
}

#else

__global__ void my_hip_unified_attention_2d_kernel(
    __hip_bfloat16* __restrict__ output,
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key_cache,
    const __hip_bfloat16* __restrict__ value_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_len,
    float scale,
    int num_q_heads,
    int num_kv_heads,
    int num_queries_per_kv,
    int block_table_stride,
    int query_stride_0,
    int query_stride_1,
    int output_stride_0,
    int output_stride_1,
    int BLOCK_SIZE,
    int num_seqs,
    long long stride_k_cache_0,
    long long stride_k_cache_1,
    long long stride_k_cache_2,
    long long stride_v_cache_0,
    long long stride_v_cache_1,
    long long stride_v_cache_2)
{
}

#endif  // MY_UA_ENABLE_DUMMA

// ===== host 端 launch =====
#ifndef MY_UA_STANDALONE
void my_hip_unified_attention_2d(
    torch::Tensor& output,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_table,
    torch::Tensor& seq_lens,
    torch::Tensor& query_start_len,
    double scale,
    int64_t block_size)
{
    TORCH_CHECK(output.is_cuda() && query.is_cuda() && key_cache.is_cuda() &&
                    value_cache.is_cuda() && block_table.is_cuda() &&
                    seq_lens.is_cuda() && query_start_len.is_cuda(),
                "my_hip_unified_attention_2d: all tensors must be CUDA tensors");
    const c10::cuda::CUDAGuard g(query.device());

    TORCH_CHECK(output.device() == query.device() &&
                    key_cache.device() == query.device() &&
                    value_cache.device() == query.device() &&
                    block_table.device() == query.device() &&
                    seq_lens.device() == query.device() &&
                    query_start_len.device() == query.device(),
                "my_hip_unified_attention_2d: all tensors must be on the same CUDA device");
    TORCH_CHECK(output.scalar_type() == torch::kBFloat16 &&
                    query.scalar_type() == torch::kBFloat16 &&
                    key_cache.scalar_type() == torch::kBFloat16 &&
                    value_cache.scalar_type() == torch::kBFloat16,
                "my_hip_unified_attention_2d: output, query, key_cache, and value_cache must be bf16");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                    seq_lens.scalar_type() == torch::kInt32 &&
                    query_start_len.scalar_type() == torch::kInt32,
                "my_hip_unified_attention_2d: block_table, seq_lens, and query_start_len must be int32");
    TORCH_CHECK(output.dim() == 3,
                "my_hip_unified_attention_2d: output must be 3D [num_tokens,num_q_heads,head_size]");
    TORCH_CHECK(query.dim() == 3,
                "my_hip_unified_attention_2d: query must be 3D [num_tokens,num_q_heads,head_size]");
    TORCH_CHECK(output.size(0) == query.size(0),
                "my_hip_unified_attention_2d: output and query token dimensions must match");
    TORCH_CHECK(output.size(1) == query.size(1),
                "my_hip_unified_attention_2d: output and query head dimensions must match");
    TORCH_CHECK(output.size(2) == HEAD_SIZE && query.size(2) == HEAD_SIZE,
                "my_hip_unified_attention_2d: head_size must be 256");
    TORCH_CHECK(output.stride(2) == 1 && query.stride(2) == 1,
                "my_hip_unified_attention_2d: output/query head_size dim must be contiguous");
    TORCH_CHECK(query.stride(0) <= std::numeric_limits<int>::max() &&
                    query.stride(1) <= std::numeric_limits<int>::max() &&
                    output.stride(0) <= std::numeric_limits<int>::max() &&
                    output.stride(1) <= std::numeric_limits<int>::max(),
                "my_hip_unified_attention_2d: query/output strides must fit int");

    // kernel 用 +h 直接索引 head_size 维，要求该维 contiguous。
    //   bf16 KV cache 是 4D [num_blks, block_size, num_kv_heads, head_size]，
    //   head_size 维 = dim 3，stride(3) 应为 1。int8/5D 布局不支持（会读错），这里挡住。
    TORCH_CHECK(key_cache.dim() == 4,
        "my_hip_unified_attention_2d: key_cache must be 4D bf16 [num_blks,block_size,num_kv_heads,head_size]");
    TORCH_CHECK(value_cache.dim() == 4,
        "my_hip_unified_attention_2d: value_cache must be 4D bf16 [num_blks,block_size,num_kv_heads,head_size]");
    TORCH_CHECK(key_cache.size(0) == value_cache.size(0) &&
                    key_cache.size(1) == value_cache.size(1) &&
                    key_cache.size(2) == value_cache.size(2) &&
                    key_cache.size(3) == value_cache.size(3),
                "my_hip_unified_attention_2d: key_cache and value_cache shapes must match");
    TORCH_CHECK(key_cache.size(1) == block_size,
                "my_hip_unified_attention_2d: block_size must match key_cache.size(1)");
    TORCH_CHECK(key_cache.size(3) == HEAD_SIZE,
                "my_hip_unified_attention_2d: KV cache head_size must be 256");
    TORCH_CHECK(value_cache.size(3) == HEAD_SIZE,
                "my_hip_unified_attention_2d: value_cache head_size must be 256");
    TORCH_CHECK(key_cache.stride(3) == 1,
        "my_hip_unified_attention_2d: key_cache head_size dim must be contiguous (stride(3)==1)");
    TORCH_CHECK(value_cache.stride(3) == 1,
        "my_hip_unified_attention_2d: value_cache head_size dim must be contiguous (stride(3)==1)");
    TORCH_CHECK(block_table.dim() == 2,
                "my_hip_unified_attention_2d: block_table must be 2D [num_seqs,max_num_blocks_per_seq]");
    TORCH_CHECK(seq_lens.dim() == 1,
                "my_hip_unified_attention_2d: seq_lens must be 1D [num_seqs]");
    TORCH_CHECK(query_start_len.dim() == 1,
                "my_hip_unified_attention_2d: query_start_len must be 1D [num_seqs+1]");

    const int num_kv_heads = key_cache.size(2);
    const int num_q_heads = output.size(1);
    TORCH_CHECK(num_kv_heads > 0,
                "my_hip_unified_attention_2d: num_kv_heads must be positive");
    TORCH_CHECK(num_q_heads > 0 && num_q_heads % num_kv_heads == 0,
                "my_hip_unified_attention_2d: output/query num_q_heads must be a positive multiple of key_cache.size(2)");
    const int num_queries_per_kv = num_q_heads / num_kv_heads;
    TORCH_CHECK(num_queries_per_kv > 0 && num_queries_per_kv <= BLOCK_M,
                "my_hip_unified_attention_2d: num_queries_per_kv must be in (0, BLOCK_M]");
    const int BLOCK_Q_host = BLOCK_M / num_queries_per_kv;
    const int num_seqs = seq_lens.size(0);
    TORCH_CHECK(block_table.size(0) >= num_seqs,
                "my_hip_unified_attention_2d: block_table rows must cover seq_lens");
    TORCH_CHECK(block_table.stride(0) > 0 && block_table.stride(0) <= std::numeric_limits<int>::max(),
                "my_hip_unified_attention_2d: block_table stride(0) must fit int");
    TORCH_CHECK(query_start_len.size(0) >= num_seqs + 1,
                "my_hip_unified_attention_2d: query_start_len must have at least num_seqs + 1 entries");
    TORCH_CHECK(block_size > 0 && block_size <= std::numeric_limits<int>::max(),
                "my_hip_unified_attention_2d: block_size must fit int");

    // Triton 上界：floor(total_q / BLOCK_Q) + num_seqs
    //   （对齐 triton_unified_attention.py:994）
    const int num_q_blocks = output.size(0) / BLOCK_Q_host + num_seqs;

    dim3 grid(num_q_blocks, num_kv_heads);
    dim3 block(NUM_THREADS);
    my_hip_unified_attention_2d_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<__hip_bfloat16*>(output.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(key_cache.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(value_cache.data_ptr()),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        query_start_len.data_ptr<int>(),
        static_cast<float>(scale),
        num_q_heads, num_kv_heads, num_queries_per_kv,
        static_cast<int>(block_table.stride(0)),
        static_cast<int>(query.stride(0)), static_cast<int>(query.stride(1)),
        static_cast<int>(output.stride(0)), static_cast<int>(output.stride(1)),
        static_cast<int>(block_size),
        num_seqs,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2));
#ifdef USE_ROCM
    C10_HIP_KERNEL_LAUNCH_CHECK();
#else
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}
#endif  // MY_UA_STANDALONE
