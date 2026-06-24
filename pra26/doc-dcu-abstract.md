# DCU 硬件与编程参考

本文档面向小队内部开发和性能分析使用，整理 DCU/HIP/DTK 文档中与内核优化直接相关的硬件事实、编程模型和验证方法。

## 适用范围

当前比赛环境按国产 DCU、DTK 26.04、HIP 编程栈处理。仓库内已有工程约定把目标卡按 `BW1000 / gfx936` 处理，因此新写 HIP/DCC/DUMMA probe、CMake 架构参数和反汇编检查时，默认使用 `gfx936`。

本文中的硬件事实优先来自以下本地资料：

- `dtk-pdf/DTK 26.04 HIP C++编程指南.pdf`
- `dtk-pdf/DTK 26.04 HIP最佳实践手册.pdf`
- `dtk-pdf/DTK 26.04 DUMMA使用手册.pdf`
- `dtk-pdf/DTK 26.04 hipprof使用手册.pdf`

## 执行模型

HIP kernel 由主机端发起，在 DCU 设备端执行。kernel launch 后，主机端通常不会等待 kernel 执行完成，而是继续向下执行；需要显式同步时可使用 `hipDeviceSynchronize`、流同步、事件同步，或会触发隐式同步的 API。`hipMemcpy` 是同步拷贝，会阻塞主机等待相关操作完成；需要隐藏传输和执行开销时，应优先考虑 `hipMemcpyAsync` 配合 stream。

线程组织分为 grid、block、thread 三层。block 可以在逻辑上是一维、二维或三维，但硬件执行时会被组织为一维线程束。一个线程块只会调度到一个 CU 上，并在该 CU 上常驻直到执行完成。CU 是 DCU 的核心执行单元，寄存器和共享内存/LDS 都是 CU 上的关键稀缺资源。

DCU 的线程束大小为 64。线程块大小如果不是 64 的整数倍，末尾不足 64 个线程也会占用一个完整线程束的调度和存储资源，造成浪费。常规 kernel 的 block 配置应遵守以下原则：

- block 线程数保持为 64 的倍数。
- 避免过小 block；通用起点通常选 128 或 256 线程。
- block 数量应多于 CU 数量，以提供足够并行度。
- 高寄存器、高 LDS、高线程数会共同限制每个 CU 的常驻 block/wavefront 数量。

同一线程束内以 SIMT 方式执行。若同一线程束中的 lane 走不同分支，会发生线程束分化：一个分支执行时，未命中该分支的 lane 被禁用等待；之后再执行其他分支。因此，热路径中应尽量避免 wavefront 内高度发散的条件分支，尤其是循环内分支。

线程块内同步使用 `__syncthreads()`。它只同步同一 block 内线程，不跨 block。涉及 LDS 数据生产/消费、块内规约、双缓冲阶段切换时需要明确同步点；但同步本身会形成等待，应通过合理分块、流水化和常驻 wavefront 数隐藏等待。

## 存储层次

DCU 程序通常涉及以下存储资源：

- 全局内存：容量大、延迟高、带宽依赖访问合并程度。
- 寄存器：线程私有，片上资源，延迟最低，适合保存复用数据和累加器。
- LDS/shared memory：block 内共享，位于 CU 上，适合 tile staging、块内复用和规约。
- 主机内存：CPU 侧内存，通过 PCIe 与设备交互。
- 页锁定主机内存：用于高效异步 H2D/D2H 数据传输。

全局内存访问的基本原则是让连续线程访问连续地址。若相邻线程访问相邻、对齐的地址，硬件更容易合并访存，带宽利用率更高；若 stride 变大或访问离散，带宽会明显下降。优化 kernel 时，先检查热路径中每个 wavefront 的 global load/store 地址模式，再谈更复杂的计算优化。

寄存器适合承载短期复用。以 GEMM 为例，A/B tile 数据和 C 的部分和应尽量留在寄存器中，减少反复访问全局内存。循环展开能增加指令调度空间、减少循环控制开销，并为预取提供无依赖指令窗口。但寄存器使用过多会降低 occupancy，甚至触发 spill；因此寄存器复用需要用 hipprof 或编译产物确认 VGPR/SGPR/scratch 变化。

LDS 适合把跨线程、跨输出元素复用的数据从 global memory 搬到片上，再由各线程重复读取。典型路径是：

```text
global memory -> LDS -> register -> compute
```

相比每个线程直接反复读 global memory，LDS staging 能降低全局访存压力。但 LDS 不是免费资源：占用过大降低常驻 block 数；访问模式不当会产生 bank conflict；额外的 `__syncthreads()` 也可能成为等待点。只有当数据确实被多个线程或多个计算步骤复用时，LDS 才值得引入。

## LDS Bank Conflict

DTK 最佳实践文档明确指出，LDS 使用时需要特别关注 bank conflict。同一个 wavefront 内，如果多个线程同时访问同一个 bank 中的不同地址，访问会被串行化，降低 LDS 访存效率。

文档示例按 32 个 LDS bank 进行分析。常见冲突来源包括：

- wavefront 内 lane 的 LDS 地址 stride 正好落在 bank 数的周期上。
- 二维 tile 的 leading dimension 是 bank 数或其倍数。
- 每个线程连续读取多个元素，但不同 lane 的访问组映射到重复 bank。
- 为了转置或重排数据，把 global memory 数据搬到 LDS 后按不友好的模式读取。

降低 bank conflict 的常见方法：

- 改变 LDS 数据布局，使 wavefront 内访问分散到更多 bank。
- 对二维 LDS tile 增加 padding，避免 leading dimension 与 bank 周期强对齐。
- 改变每个线程读取顺序，例如把连续 4 个元素改为分两段读取。
- 使用 swizzle 或 XOR 类布局打散 bank 映射。
- 比较不同 tile shape、`num_warps`、`num_stages`、访问 order 对 LDS 指标的影响。

判断 LDS 优化是否成立，应看硬件指标，而不是只看代码里是否用了 `__shared__`。hipprof PMC 可提供 `shared memory operation`、`shared memory bank conflict`、`L1 cache unit is stalled` 等指标；资源项中也可看到 shared memory size、VGPR、SGPR、scratch 等信息。

## GEMM 与矩阵计算

GEMM 优化的核心是分块和复用。朴素 GEMM 会重复读取 A/B 矩阵元素，访存量巨大；分块后可以在 M/N 方向复用 A/B 数据，在 K 方向把部分和保留在寄存器中，显著降低内存访问次数。

通用优化顺序：

1. 先做 M/N/K 分块，让输出 tile 的计算有数据复用。
2. 把 A/B 小片段和 C 累加器放入寄存器。
3. 通过循环展开增加指令级并行。
4. 用预取把下一步 global/LDS load 与当前计算重叠。
5. 对跨线程复用的数据使用 LDS staging。
6. 用 LDS 双缓冲把 `global -> LDS` 和 `LDS -> register -> compute` 形成流水。
7. 检查 bank conflict、寄存器压力、scratch、occupancy 和实际 kernel time。

数据预取的基本思想是：在当前数据参与计算之前，提前发起下一组无依赖数据的加载，使访存操作和计算单元工作重叠。预取通常需要更多寄存器或 LDS 空间，因此必须和 occupancy 权衡。

## DUMMA 能力边界

DUMMA 是 DTK 提供的矩阵乘加编程接口，用于以线程束为单位调用 DCU 上的 Tensor Core 类硬件能力。它的接口围绕 fragment 展开：

- `du_fill_fragment`：填充 fragment。
- `du_load_matrix_sync`：把矩阵数据加载到 fragment。
- `du_mma_sync`：执行矩阵乘加。
- `du_store_matrix_sync`：把结果写回矩阵。

DUMMA 适合做小 tile 矩阵乘加和 roofline/microbench 对照，但它不是完整 GEMM 库，也不会自动处理 vLLM attention 中的 paged KV、mask、online softmax、layout 转换和 cache 索引。把 DUMMA 接入生产 attention 通常意味着手写完整 fused attention kernel，而不是简单替换一次 GEMM 调用。

DTK 26.04 DUMMA 类型表中，`gfx936` 相关能力包括：

| A/B 类型 | Accumulator | Tile | 支持架构 |
|---|---|---:|---|
| `float` | `float` | `m16n16k4` | `GFX[926,928,936,938]` |
| `float` | `float` | `m16n16k8` | `GFX[928,936,938]` |
| `precision::tf32` | `float` | `m16n16k8` | `GFX[928,936,938]` |
| `__half` | `float` | `m16n16k16` | `GFX[928,936,938]` |
| `__hip_bfloat16` | `float` | `m16n16k16` | `GFX[928,936,938]` |
| `signed char` | `int` | `m16n16k32` | `GFX[928,936,938]` |
| `unsigned char` | `int` | `m16n16k32` | `GFX[928,936,938]` |
| `precision::s4` | `int` | `m16n16k64` | `GFX[936,938]` |
| `precision::u4` | `int` | `m16n16k64` | `GFX[936,938]` |
| `double` | `double` | `m16n16k4` | `GFX[926,936,938]` |
| `__hip_fp8_e4m3` | `float` | `m16n16k32` | `GFX[938]` |
| `__hip_fp8_e5m2` | `float` | `m16n16k32` | `GFX[938]` |

对当前 `gfx936` 目标，DUMMA 支持 BF16/FP16/TF32/int8/int4 等矩阵乘加路径，但 DUMMA FP8 MMA 只列在 `GFX[938]`，不能作为 `gfx936` FP8 矩阵计算的默认方案。若需要 FP8 GEMM roofline，应优先评估 hipBLASLt 或其他明确支持当前卡的库路径。

DUMMA 示例编译需要指定目标架构，例如：

```bash
source dtk/env.sh
hipcc test.cpp -o test --offload-arch=gfx936
```

如果同一源码需要同时覆盖多个架构，可显式追加多个 `--offload-arch`。

## HIP 数据传输与并发

设备内存使用 `hipMalloc` 分配，使用 `hipFree` 释放。主机与设备之间的数据传输使用 `hipMemcpy` 或 `hipMemcpyAsync`。同步 `hipMemcpy` 会阻塞主机端；异步拷贝需要配合 stream，并且通常需要页锁定主机内存才能充分发挥异步传输能力。

stream 表示按序执行队列。不同 stream 中的 kernel 和 memcpy 有机会并发，但是否真正重叠取决于硬件资源、依赖关系、内存带宽和 runtime 调度。事件可用于计时和跨 stream 同步。

hipGraph 适用于大量轻量 kernel 或固定执行图的场景。流程一般是创建 graph、添加 memcpy/memset/kernel node、实例化 graph、重复 launch、最后销毁 graph。对 vLLM 这类高频推理服务，graph 的价值主要在减少 CPU launch overhead 和稳定重复执行路径，但动态 shape、动态 batch、内存地址变化会增加 graph 使用复杂度。

## 多 DCU 与通信

HIP 提供多设备管理 API：

- `hipGetDeviceCount`：查询设备数量。
- `hipGetDeviceProperties`：查询设备属性，包括 `warpSize`、`sharedMemPerBlock`、`totalGlobalMem` 等。
- `hipSetDevice`：设置当前设备。
- `hipDeviceCanAccessPeer`：检查 P2P 访问能力。
- `hipDeviceEnablePeerAccess`：启用对等设备访问。
- `hipMemcpyPeerAsync`：设备间异步拷贝。

多 DCU 场景下，设备间通信效率取决于拓扑。若没有 P2P，设备间数据传输可能需要经过主机内存中转，带宽和延迟都会变差。若设备共享合适的 PCIe 根节点并支持 P2P，可直接走设备间路径，降低传输开销。

单卡比赛路径中，多 DCU 通信通常不是主瓶颈；但调试环境、profile 工具和提交脚本仍可能暴露多进程/多设备信息。做性能结论时，应明确采样的是哪张卡、哪个进程和哪个设备 ID。

## Profiling 指标

hipprof 是 DTK 的性能分析工具，可用于 HIP API、memory copy、kernel timeline、HSA、RCCL、ROCTX、PMC 硬件计数器、泄漏检测和寄存器压力分析。

常用模式：

```bash
hipprof --hip-trace --stats --show-pid --devices 0 -o <out> <command>
hipprof --hip-trace --show-pid --devices 0 --output-type 2 -o <out> <command>
hipprof --hip-trace --kernel-stack --show-pid --devices 0 -o <out> <command>
hipprof --pmc --pmc-type 3 --kernel-name "<kernel>" <command>
hipprof --pmc-read --pmc-type 3 --kernel-name "<kernel>" <command>
hipprof --pmc-write --pmc-type 3 --kernel-name "<kernel>" <command>
```

长驻服务不应从进程启动到退出全程 trace。更合适的方式是用 session 控制采样窗口：

```bash
hipprof --hip-trace --trace-off --session 936 --show-pid --devices 0 -o <out> <server-command>
hipprof --session-client 936 --start
# run one benchmark slice
hipprof --session-client 936 --stop
hipprof --session-client 936 --flush
```

PMC 三组关注点不同：

- `--pmc`：kernel time、ALU 指令、Gflops、LDS 操作、bank conflict、L1/L2 相关指标。
- `--pmc-read`：L2 读请求和读流量相关指标。
- `--pmc-write`：L2 写请求和写流量相关指标。

PMC 输出中对 kernel 优化最关键的资源/行为项：

- `kernel time`
- `work group size`
- `shared memory size`
- `scratch size`
- `vgpr count`
- `sgpr count`
- `processed ALU instructions`
- `performance`
- `shared memory operation`
- `shared memory bank conflict`
- `L1 cache unit is active`
- `L1 cache unit is stalled`
- L2 read/write request 与带宽相关指标

`--kernel-stack` 用于定位某个热点 kernel 是从哪条主机端路径发起的。对 PyTorch fallback、allocator、fill/copy、Triton kernel、AITER kernel、vLLM ROCm C++ kernel 的归因，优先使用 kernel 名称、主机栈和库路径共同判断。

## vLLM 优化检查清单

做 DCU 相关改动时，应先判断瓶颈类型，再决定优化方向：

- 如果热点是 GEMM：确认是否已走 rocBLAS/hipBLASLt/Triton MFMA；用真实 M/N/K 做 roofline 对照，再考虑替换。
- 如果热点是 attention：检查 tile shape、global load 合并、LDS 使用、bank conflict、mask 分支、paged KV gather 和 launch 次数。
- 如果热点是 fill/copy/indexing：优先减少不必要的初始化、拷贝、dtype 转换和 host/device 同步。
- 如果热点是 HIP API：检查 `hipMemcpy`、`hipDeviceSynchronize`、event sync、allocator、stream 依赖和 CPU launch overhead。
- 如果热点是 LDS：用 PMC 证明 bank conflict 或 LDS stall，再尝试 padding、swizzle、tile/order/num_stages 改动。
- 如果热点是寄存器或 occupancy：检查 VGPR/SGPR/scratch/shared memory size，避免单纯加大 tile 导致常驻 wavefront 下降。

任何性能结论都应来自真实 DCU 节点或容器上的短 profile/benchmark。登录节点、本地 macOS、源码阅读和 import smoke 只能证明工程可行性，不能证明 DCU 性能。

## 术语对照

| 术语 | 含义 |
|---|---|
| DCU | 国产异构加速设备，在 HIP 中作为 device 使用 |
| DTK | DCU ToolKit，提供 HIP、DCC、hipprof、数学库等工具链 |
| HIP | 面向异构设备的 C++ 编程接口 |
| CU | Compute Unit，DCU 上调度 block/wavefront 的核心计算单元 |
| wavefront/线程束 | DCU 上 SIMT 执行的线程组，大小为 64 |
| block/线程块 | HIP kernel 的线程块，一个 block 调度到一个 CU 上执行 |
| LDS/shared memory | CU 上的 block 内共享片上存储 |
| VGPR/SGPR | 向量/标量寄存器资源 |
| scratch | 寄存器溢出或临时存储带来的额外内存资源 |
| DUMMA | DTK 的线程束级矩阵乘加 fragment API |
| PMC | Performance Monitoring Counter，hipprof 硬件计数器 |
