# DCU 编程指南

本文档面向小队内部开发和性能分析使用，整理 DCU/HIP/DTK 编程模型、硬件相关约束、常用优化方法和验证手段。内容以 DCU 开发者指南和 DTK 26.04 文档为依据，保留对 vLLM 内核优化有直接价值的事实与方法。

## 资料范围

当前比赛环境按国产 DCU、DTK 26.04、HIP 编程栈处理。仓库工程约定把目标卡按 `BW1000 / gfx936` 处理；新写 HIP/DCC/DUMMA probe、CMake 架构参数和反汇编检查时，默认使用 `gfx936`。

主要参考资料：

- `DTK 26.04 HIP C++编程指南`
- `DTK 26.04 HIP最佳实践手册`
- `DTK 26.04 DUMMA使用手册`
- `DTK 26.04 hipprof使用手册`

## 第一个 DCU 程序

DCU 程序的基本结构与 CPU 程序不同。CPU 程序通常直接在主机内存中循环处理数据；DCU 程序需要显式管理设备内存，把数据从 CPU 侧传到设备侧，在设备上启动 kernel，再把结果传回主机侧。

最小 DCU 程序通常包含四步：

1. 在 CPU 侧准备输入和输出缓冲区。
2. 使用 `hipMalloc` 在 DCU 显存上分配设备缓冲区。
3. 使用 `hipMemcpy` 或 `hipMemcpyAsync` 进行 H2D/D2H 数据传输。
4. 使用 `<<<grid, block>>>` 或 `hipLaunchKernelGGL` 启动 `__global__` kernel。

示例结构：

```cpp
#include <hip/hip_runtime.h>

__global__ void add(float* a, float* b, float* c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int n = 10000;
    size_t bytes = n * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);

    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    add<<<grid, block>>>(d_a, d_b, d_c, n);

    hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}
```

编译：

```bash
hipcc vector_add_dcu.cpp -o vector_add_dcu
```

如果需要指定当前比赛目标架构：

```bash
hipcc vector_add_dcu.cpp -o vector_add_dcu --offload-arch=gfx936
```

运行时可用 `rocm-smi` 或 DCU2 环境中的 `hy-smi` 查看设备利用率。该类工具只能说明程序确实在设备上活动；性能结论仍需来自 benchmark 和 profiler。

## 主机端与设备端

HIP 程序有主机端和设备端两个执行域：

- 主机端：CPU 执行普通 C/C++ 代码，负责分配内存、准备数据、启动 kernel、同步和收集结果。
- 设备端：DCU 执行 `__global__` kernel 和 `__device__` 函数。

常见函数修饰符：

| 修饰符 | 含义 |
|---|---|
| `__global__` | kernel 入口；主机端调用，设备端执行 |
| `__device__` | 设备端函数；只能从设备端调用 |
| `__host__` | 主机端函数；默认函数属性 |
| `__host__ __device__` | 同一函数可同时生成主机端和设备端版本 |

kernel launch 通常是异步的。主机端发起 kernel 后会继续执行，除非遇到显式同步或同步拷贝。`hipMemcpy` 是同步拷贝；`hipMemcpyAsync` 是异步拷贝，需要 stream 和合适的主机内存条件配合。

常用运行时 API：

| API | 用途 |
|---|---|
| `hipGetDeviceCount` | 获取可用设备数量 |
| `hipGetDeviceProperties` | 查询设备属性 |
| `hipSetDevice` | 设置当前设备 |
| `hipMalloc` | 分配设备内存 |
| `hipFree` | 释放设备内存 |
| `hipHostMalloc` | 分配页锁定主机内存 |
| `hipMemcpy` | 同步拷贝 |
| `hipMemcpyAsync` | 异步拷贝 |
| `hipStreamCreate` | 创建 stream |
| `hipStreamSynchronize` | 等待 stream 完成 |
| `hipDeviceSynchronize` | 等待当前设备所有已提交工作完成 |

## 线程执行模型

HIP kernel 由大量线程并行执行。线程组织层级为：

```text
grid -> block -> thread
```

`gridDim` 表示 grid 中 block 的数量，`blockDim` 表示每个 block 的线程数量，`blockIdx` 表示当前 block 坐标，`threadIdx` 表示当前线程在 block 内的坐标。

常见一维全局线程编号：

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

二维 block 或 grid 最终仍可映射到一维物理线程序列。多维组织主要是为了让索引表达贴合矩阵、图像或张量布局。使用多维 `dim3` 时，没有显式指定的维度默认是 1。

```cpp
dim3 grid(90, 90);
dim3 block(64, 1);
kernel<<<grid, block>>>(...);
```

### 线程束

DCU 的线程束大小为 64。一个线程块会被划分为若干线程束执行。若 block 线程数不是 64 的整数倍，最后不足 64 个线程仍会占用完整线程束的调度和存储资源。

性能配置原则：

- block 线程数保持为 64 的倍数。
- 通用 kernel 起点可选 128 或 256 线程。
- grid 中 block 数应多于 CU 数量，避免并行度不足。
- 避免为了增大单个 block 工作量而过度消耗寄存器和 LDS。

### CU 调度

一个 block 只能调度到一个 CU 上，并在该 CU 上常驻直到执行完成。CU 上的寄存器、LDS/shared memory、可常驻 block 数和可常驻 wavefront 数共同决定 occupancy。

当一个 wavefront 因访存或同步等待而阻塞时，CU 可以切换到同一 CU 上其他可运行 wavefront。足够的活跃 wavefront 有助于隐藏访存延迟和同步等待；但高 VGPR、高 SGPR、高 LDS 或过大的 workgroup 会降低常驻数量。

### 线程束分化

同一个 wavefront 内执行不同分支时，会发生线程束分化。硬件会分阶段执行不同分支，未命中当前分支的 lane 被禁用等待。

应尽量避免：

- wavefront 内 lane 条件高度不一致。
- 热循环中有复杂 `if/else`。
- 不同 lane 走不同长度的循环。
- 分支后访问完全不同的内存区域。

不可避免的边界检查应尽量放在外层或通过 mask 处理，让主体计算保持规整。

### 同步

同步分为主机/设备同步和 block 内同步。

主机/设备同步：

- `hipDeviceSynchronize()`：等待当前设备上所有已提交工作完成。
- `hipStreamSynchronize(stream)`：等待指定 stream 完成。
- `hipEventSynchronize(event)`：等待事件完成。
- 同步 `hipMemcpy`：隐式等待相关拷贝完成。

block 内同步：

```cpp
__syncthreads();
```

`__syncthreads()` 只同步同一 block 内线程，不跨 block。使用 LDS 做生产/消费、块内规约、双缓冲切换时，必须明确同步点。同步不当会造成数据竞争；同步过多会造成等待开销。

## 存储模型

DCU kernel 常见存储资源：

| 存储 | 作用 | 性能特征 |
|---|---|---|
| 全局内存 | 设备显存，大容量数据 | 延迟高，带宽依赖访问模式 |
| 寄存器 | 线程私有临时数据 | 延迟低，数量有限 |
| LDS/shared memory | block 内共享片上存储 | 低延迟，容量有限，需避免 bank conflict |
| 主机内存 | CPU 侧内存 | 需经 PCIe 与设备交互 |
| 页锁定主机内存 | 支持高效异步拷贝 | 分配成本高，不宜滥用 |

### 全局内存

全局内存适合保存大规模输入、输出、权重、KV cache、临时张量等。访问性能高度依赖地址模式。

基本原则：

- 连续线程访问连续地址。
- 尽量对齐访问。
- 避免大 stride、随机 gather 和非合并 store。
- 对重复使用的数据，考虑寄存器或 LDS 复用。
- 对只读参数和小表，确认编译器或库是否已有缓存友好路径。

典型好的访问模式：

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
float x = input[tid];
output[tid] = x * scale;
```

典型差的访问模式：

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
float x = input[tid * stride];  // stride 大时合并访问差
```

对 vLLM 类 workload，paged KV gather、block table indirection、mask 边界、不同请求长度都会破坏规整访存。优化前应先用 profiler 或反汇编确认瓶颈是 global load、LDS、计算还是 launch/API 开销。

### 寄存器

寄存器适合保存：

- 矩阵 tile 中短期复用的 A/B 元素。
- GEMM/attention 的累加器。
- 循环展开后的预取数据。
- 局部索引和小型常量。

寄存器优化的收益来自减少全局内存/LDS 重读和增加指令级并行。但寄存器使用过多会降低 occupancy，甚至产生 scratch spill。需要关注：

- `vgpr count`
- `sgpr count`
- `scratch size`
- kernel time 是否实际下降
- occupancy 是否明显降低

### LDS/shared memory

LDS 是 CU 上的 block 内共享存储，适合存放跨线程复用的数据 tile。典型路径：

```text
global memory -> LDS -> register -> compute
```

适合使用 LDS 的场景：

- 多个线程会重复读取同一 global memory tile。
- 需要重排、转置或规整化访问以便后续计算。
- 块内规约需要共享中间结果。
- global memory 访问可以通过 staging 变成更规整的访问。

不适合使用 LDS 的场景：

- 数据只被单线程使用一次。
- LDS staging 增加了同步但没有减少全局访存。
- LDS 占用使常驻 block 数明显下降。
- 访问模式产生严重 bank conflict。

使用 LDS 时必须考虑：

- `__syncthreads()` 是否正确且必要。
- LDS 容量是否限制 occupancy。
- wavefront 内地址是否产生 bank conflict。
- 是否需要 padding、swizzle 或改变读取顺序。
- 是否可通过双缓冲隐藏 `global -> LDS` 等待。

## 全局内存管理与数据传输

设备内存分配：

```cpp
float* d_x = nullptr;
hipMalloc(&d_x, n * sizeof(float));
hipFree(d_x);
```

同步拷贝：

```cpp
hipMemcpy(d_x, h_x, bytes, hipMemcpyHostToDevice);
hipMemcpy(h_y, d_y, bytes, hipMemcpyDeviceToHost);
```

异步拷贝：

```cpp
hipStream_t stream;
hipStreamCreate(&stream);
hipMemcpyAsync(d_x, h_x, bytes, hipMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_x, d_y, n);
hipMemcpyAsync(h_y, d_y, bytes, hipMemcpyDeviceToHost, stream);
hipStreamSynchronize(stream);
hipStreamDestroy(stream);
```

页锁定主机内存：

```cpp
float* h_x = nullptr;
hipHostMalloc(&h_x, bytes);
hipHostFree(h_x);
```

页锁定内存有助于异步 H2D/D2H 拷贝，但它占用系统资源，分配释放成本较高。长生命周期缓冲区适合使用；短小临时对象不宜频繁申请。

数据传输优化原则：

- 减少 H2D/D2H 次数和总字节数。
- 避免热路径中同步 `hipMemcpy`。
- 使用 stream 让拷贝与计算重叠。
- 合并小拷贝，避免大量细碎传输。
- 对长驻服务，避免每个请求重复分配和释放大缓冲区。
- 用 profiler 检查 `hipMemcpy*`、`hipDeviceSynchronize`、event sync 和 allocator 开销。

## HIP Stream 与事件

stream 是设备工作队列。单个 stream 内操作按提交顺序执行；不同 stream 的操作有机会并发执行。

常用 API：

| API | 用途 |
|---|---|
| `hipStreamCreate` | 创建 stream |
| `hipStreamDestroy` | 销毁 stream |
| `hipStreamSynchronize` | 等待 stream 完成 |
| `hipEventCreate` | 创建事件 |
| `hipEventRecord` | 在 stream 中记录事件 |
| `hipEventSynchronize` | 等待事件 |
| `hipEventElapsedTime` | 计算两个事件间耗时 |

事件计时示例：

```cpp
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

hipEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
hipEventRecord(stop, stream);
hipEventSynchronize(stop);

float ms = 0.0f;
hipEventElapsedTime(&ms, start, stop);
```

并发执行注意事项：

- 不同 stream 不等于一定并发；实际并发受资源、依赖、拷贝引擎和调度影响。
- 同步 API 会打断并发。
- 默认 stream 语义可能引入隐式依赖，工程中应明确 stream 使用策略。
- 对小 kernel，大量 launch overhead 可能比 kernel 本身更显著。

## hipGraph

hipGraph 用于把一组固定执行关系的 memcpy、memset、kernel 等操作捕获或构造成图，实例化后可重复 launch，减少 CPU 端提交开销。

典型流程：

1. `hipGraphCreate` 创建 graph。
2. `hipGraphAddMemcpyNode` / `hipGraphAddMemsetNode` / `hipGraphAddKernelNode` 添加节点。
3. 设置节点依赖。
4. `hipGraphInstantiate` 实例化。
5. `hipGraphLaunch` 重复执行。
6. `hipGraphExecDestroy` / `hipGraphDestroy` 释放资源。

适合场景：

- 大量轻量 kernel。
- 执行图结构稳定。
- 输入输出地址、shape、依赖关系可复用或可更新。
- CPU launch overhead 明显。

对 vLLM 的影响：

- decode 阶段若 shape 和地址稳定，graph 可能减少 launch overhead。
- 动态 batch、动态请求长度、动态 KV block 分配会增加 graph 复用难度。
- graph 优化前应先确认 HIP API/launch overhead 是瓶颈。

## 多 DCU 编程

多 DCU 编程用于单卡内存不够、任务需要跨卡并发或通信计算重叠的场景。单卡比赛场景通常不把多 DCU 通信作为主线，但理解 API 有助于排查环境、进程和设备绑定问题。

设备查询：

```cpp
int count = 0;
hipGetDeviceCount(&count);

for (int i = 0; i < count; ++i) {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, i);
}
```

设置设备：

```cpp
hipSetDevice(device_id);
```

P2P 能力检查：

```cpp
int can_access = 0;
hipDeviceCanAccessPeer(&can_access, device, peer);
if (can_access) {
    hipSetDevice(device);
    hipDeviceEnablePeerAccess(peer, 0);
}
```

设备间拷贝：

```cpp
hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream);
```

多 DCU 注意事项：

- `hipSetDevice` 是低开销异步设置，不会等待之前设备工作完成。
- 每个设备应有明确的 stream、event 和内存归属。
- 无 P2P 时，设备间数据可能经主机内存中转，延迟和带宽都会变差。
- 拓扑会影响 P2P 性能；不能只凭 API 成功判断通信效率。
- profiling 时要明确 `--devices`、PID、rank 和设备 ID。

## GEMM 优化

GEMM 是深度学习模型中的核心计算。朴素 GEMM 访存量大、数据复用差；高性能 GEMM 依赖分块、寄存器复用、预取、LDS staging 和矩阵硬件单元。

朴素伪代码：

```cpp
for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            C[m][n] += A[m][k] * B[k][n];
        }
    }
}
```

问题：

- A/B 元素被重复从全局内存读取。
- C 的部分和可能反复读写。
- 算术强度低，容易被访存限制。

### 分块

分块目标是提高数据复用：

- 在 M/N 方向对 C 分块。
- 在 K 方向对 A/B 分块。
- 一个输出 tile 对应一段 A tile 和 B tile。
- A/B tile 被多个输出元素复用。

优化后的思路：

```text
for each C tile:
  keep C partial sums in registers
  for each K tile:
    load A tile
    load B tile
    accumulate C tile
  write C tile once
```

### 寄存器复用

把 A/B 小片段和 C 累加器放在寄存器中，可以减少 global memory 反复访问。循环展开能减少控制开销，并给编译器更多调度空间。

风险：

- 累加器 tile 太大导致 VGPR 增加。
- 预取寄存器太多导致 occupancy 降低。
- spill 到 scratch 可能抵消所有收益。

验证：

- hipprof PMC 看 `vgpr count`、`sgpr count`、`scratch size`。
- 对比 kernel time，而不是只看理论访存减少。

### 预取

预取利用无依赖指令间隔隐藏访存延迟。典型做法是在当前数据计算前，提前加载下一步需要的数据。

```text
load tile 0
for each tile i:
  load tile i+1
  compute tile i
compute last tile
```

预取可以发生在：

- global memory -> register
- global memory -> LDS
- LDS -> register

预取收益依赖指令调度、寄存器资源和访存延迟。过度预取会增加寄存器压力。

### LDS Staging

使用 LDS 时，数据路径变为：

```text
global memory -> LDS -> register -> compute
```

收益：

- global memory tile 只加载一次，被多个线程复用。
- 可把后续访问改成更规整的 LDS 访问。
- 有机会降低全局访存带宽压力。

成本：

- LDS 容量占用。
- 额外同步。
- bank conflict 风险。
- 更复杂的地址和边界逻辑。

### LDS 双缓冲

双缓冲把 LDS 分为两个区域，一个用于当前计算读取，一个用于下一 tile 写入。循环中交替切换读写缓冲区。

```text
buffer 0: compute tile i
buffer 1: load tile i+1
swap
```

作用：

- 将 `global -> LDS` 与 `LDS -> register -> compute` 重叠。
- 减少每轮同步等待暴露时间。
- 提高长 K 循环中的流水化程度。

风险：

- LDS 使用量约增加一倍。
- 地址切换和同步更复杂。
- 若 occupancy 降低过多，收益可能消失。

## LDS Bank Conflict

LDS bank conflict 是 DCU 深度优化中的重点。一个 wavefront 内多个线程同时访问同一个 bank 的不同地址时，访问会串行化。

DTK 最佳实践文档按 32 个 LDS bank 进行示例分析。常见问题模式：

- leading dimension 是 32 或其倍数。
- stride 正好和 bank 周期对齐。
- 每个线程连续读取多个元素，不同线程组映射到相同 bank。
- LDS 中保存转置 tile，但读取方向导致周期性冲突。

常见修法：

| 方法 | 思路 | 代价 |
|---|---|---|
| padding | 给二维 LDS tile 增加额外列，打破 bank 周期 | 增加 LDS 占用 |
| swizzle | 改变逻辑索引到物理地址的映射 | 地址计算更复杂 |
| 改读取顺序 | 把连续读取拆成分段读取，减少同 bank 重叠 | 代码复杂，可能影响 coalescing |
| 改 tile shape | 让 tile leading dimension 避开 bank 倍数 | 可能影响计算复用 |
| 改 `num_stages`/调度 | 影响编译器生成的 staging 和访问相位 | 需实测 |

判断依据：

- `shared memory bank conflict`
- `shared memory operation`
- `L1 cache unit is stalled`
- kernel time
- shared memory size
- occupancy 相关资源指标

不能只凭代码使用 `__shared__` 推断有 bank conflict，也不能只凭 bank conflict 指标存在就说明它是主瓶颈。需要结合该 kernel 的总耗时占比、LDS 操作量和 stall 指标判断。

## 并行规约

规约是把一组数据通过满足结合律/交换律的操作变成较少结果，例如 sum、max、min。DCU 上规约常见于 softmax、norm、top-k、统计量计算和块内部分和。

朴素相邻规约可能存在：

- 线程束分化。
- 非合并 global load/store。
- 多轮同步。
- 低活跃线程比例。

更好的规约一般使用：

- 交错配对，减少不友好访问。
- 块内 LDS 保存部分和。
- wavefront 内 shuffle 或专用集合函数减少 LDS 和同步。
- 最后一个 wavefront 内展开，减少多余同步。

优化规约时要同时看：

- 分支分化。
- global memory 访问模式。
- LDS bank conflict。
- 同步次数。
- active lane 利用率。

## 全局内存合并访问

全局内存读写是许多 kernel 的性能关键。带宽测试通常能看到：连续线程访问连续地址时吞吐最高；stride 增大时吞吐下降。

好的访问：

```text
thread 0 -> a[0]
thread 1 -> a[1]
thread 2 -> a[2]
...
```

差的访问：

```text
thread 0 -> a[0]
thread 1 -> a[16]
thread 2 -> a[32]
...
```

优化策略：

- 调整张量 layout，使最内层维度对应连续线程。
- 合并小向量 load/store，例如使用 `float2`、`float4` 或等价向量化类型，前提是对齐可靠。
- 对 gather 场景，先判断能否重排数据或批量化访问。
- 对 store 场景，避免同一 cache line 被离散线程反复写。
- 关注边界 mask 对 coalescing 的影响。

## DUMMA

DUMMA 是 DTK 提供的线程束级矩阵乘加接口，用于利用 DCU 上 Tensor Core 类硬件能力。它类似 fragment API，不是完整 GEMM 库。

核心 API：

| API | 作用 |
|---|---|
| `du_fill_fragment` | 填充 fragment |
| `du_load_matrix_sync` | 从矩阵加载 fragment |
| `du_mma_sync` | 执行矩阵乘加 |
| `du_store_matrix_sync` | 写回矩阵 |

支持类型和 tile：

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

对 `gfx936` 的结论：

- BF16/FP16/TF32/int8/int4 等 DUMMA MMA 路径可用于 microbench 或实验性 kernel。
- DUMMA FP8 MMA 只列在 `GFX[938]`，不能作为 `gfx936` FP8 矩阵计算默认方案。
- 若要做 FP8 GEMM roofline，应优先看 hipBLASLt 或其他明确支持当前卡的库路径。

编译：

```bash
source dtk/env.sh
hipcc test.cpp -o test --offload-arch=gfx936
```

使用边界：

- DUMMA 是 device-side fragment API。
- DUMMA 不自动处理 paged KV、mask、online softmax、layout transform、block table。
- 把 DUMMA 接入 attention 主路径通常意味着手写完整 fused attention kernel。
- DUMMA 更适合作为矩阵硬件能力上限和 tile scheduling 对照，而不是直接替换 vLLM 中已有 GEMM 库调用。

## hipprof

hipprof 是 DTK 性能分析工具，支持 HIP API、memory copy、kernel timeline、HSA、RCCL、ROCTX、PMC 硬件计数器、泄漏检测和寄存器压力分析。

低成本统计：

```bash
hipprof --hip-trace --stats --show-pid --devices 0 \
  -o <out> <command>
```

timeline：

```bash
hipprof --hip-trace --show-pid --devices 0 --output-type 2 \
  -o <out> <command>
```

kernel 主机栈：

```bash
hipprof --hip-trace --kernel-stack --show-pid --devices 0 \
  -o <out> <command>
```

PMC：

```bash
hipprof --pmc --pmc-type 3 --kernel-name "<kernel>" <command>
hipprof --pmc-read --pmc-type 3 --kernel-name "<kernel>" <command>
hipprof --pmc-write --pmc-type 3 --kernel-name "<kernel>" <command>
```

长驻服务建议使用 session 控制采样窗口：

```bash
hipprof --hip-trace --trace-off --session 936 --show-pid --devices 0 \
  -o <out> <server-command>

hipprof --session-client 936 --start
# run one benchmark slice
hipprof --session-client 936 --stop
hipprof --session-client 936 --flush
```

PMC 指标分组：

| 命令 | 关注点 |
|---|---|
| `--pmc` | kernel time、ALU、Gflops、LDS、bank conflict、L1/L2 基础指标 |
| `--pmc-read` | L2 read request、读流量、读路径 stall |
| `--pmc-write` | L2 write request、写流量、写路径 stall |

关键资源指标：

- `work group size`
- `shared memory size`
- `scratch size`
- `vgpr count`
- `sgpr count`
- `fbarrier count`

关键行为指标：

- `kernel time`
- `processed ALU instructions`
- `performance`
- `shared memory operation`
- `shared memory bank conflict`
- `L1 cache unit is active`
- `L1 cache unit is stalled`
- L2 read/write requests
- L2 read/write bytes 或等价带宽指标

使用原则：

- 先用 `--stats` 找热点，再收 timeline。
- 多进程服务用 `--show-pid` 区分 server/client。
- 多卡环境用 `--devices` 限定设备，减少噪声。
- 不要长时间 trace 整个服务生命周期。
- `--kernel-stack` 用于定位 PyTorch fallback、Triton、AITER、vLLM C++/ROCm kernel 的发起路径。
- PMC 三组硬件计数器资源有限，不能假设一次 run 得到所有指标。

## vLLM 内核优化检查清单

做 DCU 相关改动前，先把瓶颈归类。

### GEMM 热点

检查：

- 是否已走 rocBLAS、hipBLASLt、Triton MFMA 或其他高性能路径。
- 当前 M/N/K 是否碎片化。
- 是否有额外 transpose、copy、cast、slice。
- 是否存在小 GEMM launch 过多。
- 是否可用库自带 bench 做真实 shape roofline。

不要只因为存在 DUMMA 就直接替换 GEMM。已有库通常已经做了 shape tuning、分块、prefetch 和 epilogue 管理。

### Attention 热点

检查：

- Q/K/V layout 是否有合并访问。
- paged KV gather 是否造成随机访问。
- block table 和 slot mapping 是否引入额外小 kernel。
- online softmax 规约是否高效。
- LDS 使用量、bank conflict 和 wait 是否明显。
- tile shape 是否导致 occupancy 下降。
- mask/边界条件是否造成 wavefront 分化。

### Fill/Copy/Indexing 热点

检查：

- 是否有整段初始化但 kernel 只读前缀。
- 是否有重复 `torch.full`、`fill_`、`zero_`。
- 是否有 CPU/GPU 来回拷贝。
- 是否有不必要 dtype 转换。
- 是否有小 tensor 操作触发大量 PyTorch elementwise kernel。

这类热点优先做数学等价的生命周期和边界缩减，不要先写复杂自定义 kernel。

### HIP API 热点

检查：

- `hipMemcpy` 是否在热路径同步。
- `hipDeviceSynchronize` 是否可移除或缩小范围。
- event sync 是否过密。
- allocator 是否频繁申请释放。
- kernel launch 数是否过多。
- stream 间是否存在隐式依赖。

### LDS 热点

检查：

- `shared memory operation` 是否高。
- `shared memory bank conflict` 是否高。
- `L1 cache unit is stalled` 是否高。
- shared memory size 是否限制 occupancy。
- 改 padding/swizzle/tile shape 后 kernel time 是否下降。

### 寄存器/Occupancy 热点

检查：

- VGPR/SGPR 是否过高。
- scratch 是否非零。
- tile 增大是否导致常驻 wavefront 降低。
- 预取是否带来寄存器压力。
- 循环展开是否过度。

## 常用命令速查

编译 HIP：

```bash
hipcc kernel.cpp -o kernel --offload-arch=gfx936
```

编译 DCC/HIP 源码时指定架构：

```bash
dcc -x hip kernel.cpp -O3 --offload-arch=gfx936 -o kernel
```

设备状态：

```bash
rocm-smi
hy-smi
```

查看设备属性可写小程序调用：

```cpp
hipGetDeviceCount(&count);
hipGetDeviceProperties(&prop, device);
```

hipprof stats：

```bash
hipprof --hip-trace --stats --show-pid --devices 0 -o <out> <command>
```

hipprof session：

```bash
hipprof --hip-trace --trace-off --session 936 --show-pid --devices 0 -o <out> <server>
hipprof --session-client 936 --start
hipprof --session-client 936 --stop
hipprof --session-client 936 --flush
```

PMC CSV：

```bash
hipprof --pmc --pmc-type 3 --kernel-name "<kernel>" <command>
```

## 术语表

| 术语 | 含义 |
|---|---|
| DCU | 国产异构加速设备，在 HIP 中作为 device 使用 |
| DTK | DCU ToolKit，包含 HIP、DCC、数学库、调试和 profiling 工具 |
| HIP | 异构编程接口，提供 kernel、runtime API、内存、stream、event 等能力 |
| DCC | DTK 编译器工具链 |
| CU | Compute Unit，DCU 上调度 block/wavefront 的核心计算单元 |
| wavefront/线程束 | DCU SIMT 执行单元，大小为 64 线程 |
| block/线程块 | HIP kernel 的线程块，一个 block 调度到一个 CU 上执行 |
| grid | 一次 kernel launch 中的 block 集合 |
| LDS/shared memory | CU 上 block 内共享片上存储 |
| VGPR | 向量通用寄存器 |
| SGPR | 标量通用寄存器 |
| scratch | 寄存器溢出或临时内存导致的额外存储 |
| occupancy | CU 上活跃 wavefront 数与最大可活跃 wavefront 数的比例 |
| DUMMA | DTK 的线程束级矩阵乘加 fragment API |
| PMC | Performance Monitoring Counter，硬件性能计数器 |
