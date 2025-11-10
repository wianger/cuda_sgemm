# 并行与分布式计算实验 3 报告

## 1. 实验环境
- 硬件: Intel Xeon Silver 4210, NVIDIA GeForce RTX 2080 Ti (11 GB GDDR6)
- 软件: Ubuntu 22.04, NVIDIA Driver 525.89, CUDA 12.0, nvcc 12.0
- 默认矩阵规模: `N = 4096`
- 编译命令: `nvcc -O3 -arch=sm_75 main.cu -o build/main`

## 2. 理论峰值分析
- **算力**: RTX 2080 Ti 共 68 个 SM, 每个 SM 含 64 个 FP32 CUDA Core, 双发射 FMA。
	- 理论峰值 FLOPS: `68 × 64 × 2 × 2.1 GHz ≈ 18.3 TFLOPS`
- **显存带宽**: 14 Gbps 有效速率, 352-bit 总线。
	- 理论峰值带宽: `14 × 10^9 × 352 / 8 ≈ 616 GB/s`
- 作为对比, CPU 单线程约 50 GFLOPS, 多核约 0.6 TFLOPS, 远低于 GPU。

## 3. 实现与优化步骤
1. **基准版本**: 直接的三重循环 SGEMM GPU kernel, 仅使用 global memory。
2. **共享内存分块**: 采用 64×64×32 的 tile, 将 A/B 子块缓存在 shared memory 中, 减少重复访存。
3. **双缓冲加载**: shared memory 采用 stage[2] 结构, 在计算当前 K 分块时预取下一分块, 降低同步开销。
4. **寄存器分块**: 每个线程同时累计 8×4 个 C 元素, 提升算术强度, 减少写回次数。
5. **CUDA events 计时 & 热身**: 添加预热迭代和事件计时, 获得稳定的平均耗时。
6. **Tensor Core (WMMA) 路径**: 在 sm_75 设备上利用 FP16×FP16→FP32 的 WMMA API, 保留 FP32 累加, 精度可控。
7. **验证机制**: `N ≤ 1024` 时进行全量 CPU 对照, 否则随机抽样 128 个位置计算残差; Tensor Core 路径放宽阈值至 5%。

## 4. 实验结果
| N      | Kernel            | 时间 / ms | GFLOPS/s | GB/s  |
|--------|-------------------|----------:|---------:|------:|
| 1024   | Shared-memory     | 0.77      | 2.78     | 16.3  |
| 2048   | Shared-memory     | 3.43      | 5.01     | 14.7  |
| 2048   | Tensor Core (FP16)| 3.35      | 5.12     | 10.0  |
| 4096   | Shared-memory     | 21.23     | 6.47     | 9.48  |
| 4096   | Tensor Core (FP16)| 18.18     | 7.56     | 7.38  |

**说明**:
- FLOPS 计算公式: `2 × N^3 / 时间`
- 带宽估算: 按照读取 A/B、写入 C 计算数据量。
- 所有结果均为 10 次 kernel 重复的平均值, GPU 端同步后记录。
- 验证环节的误差全部保持在设定阈值内, 未出现错误。

## 5. 结果分析
1. **性能趋势**: 随矩阵规模增大, shared-memory kernel 性能提升但逐渐被显存带宽限制; Tensor Core 在大规模矩阵上拥有更高的 FLOPS, 但同样受限于内存流量。
2. **与理论峰值比较**: Shared-memory 路径达到约 35% 的 FP32 峰值; Tensor Core 路径约为 41%, 主要瓶颈来自 global memory 带宽与数据复用不足。
3. **带宽利用**: 4096 规模下带宽仅 ~9.5 GB/s (shared) 与 7.4 GB/s (Tensor Core), 远低于 616 GB/s 峰值, 表明 kernel 在 compute-bound 区域, 进一步的算术强度提升能带来收益。
4. **精度**: Tensor Core 版本采用 FP16 输入 + FP32 累加, 与 FP32 基准最大相对误差约 0.004 < 0.05 容差, 可接受。

## 6. 遇到的问题与解决方案
- **shared memory 占用超限**: TILE_K=64 时超出 48 KB 限制, 将 TILE_K 调整为 32 保持较高并行度。
- **大规模验证耗时**: 对 N=4096 进行全量 CPU 校验开销过大, 引入随机抽样验证以平衡正确性与耗时。
- **Tensor Core 精度**: 初始阈值过严导致误报, 通过调整容差并保留 FP32 累加解决。

## 7. 结论与展望
- 通过共享内存分块、寄存器分块、双缓冲加载等优化, FP32 SGEMM 达到 6.47 TFLOPS。
- Tensor Core 版本进一步提升到 7.56 TFLOPS, 误差保持在可接受范围内。
- 与 18.3 TFLOPS 理论峰值的差距主要来自算术强度与内存访问混合效率, 后续可在 Ampere+ 设备上尝试 `cp.async` 管线、自动调参以及 WMMA 混合精度修正, 继续逼近峰值性能。

---
**附录: 运行方式**
```bash
make
./build/main
```
程序默认运行 Shared-memory 与 Tensor Core 两条路径, 输出性能与精度验证情况。
