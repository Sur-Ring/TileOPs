# Max Pooling 实现计划

## Operator Description

实现 Max Pooling 前向算子。

对于输入张量 `x ∈ R^{N × S × D}`，输出张量 `y ∈ R^{N × S_out × D}` 定义为：

```
y[n, s_out, d] = max(x[n, s_out * stride + k, d])  for k in [0, kernel_size)
```

其中：
```
S_out = (S + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
```

**输入形状：** `(N, S, D)` = `(batch, in_seq_len, dim)`
**输出形状：** `(N, S_out, D)`

本 issue 跟踪标准 Max Pooling forward 算子，作为 conv & pooling 算子家族的一部分（#402）。

## Dtype Support Matrix

| Op / API | Input dtypes | Output dtype | PyTorch / reference semantic baseline |
| -------- | ------------ | ------------ | ------------------------------------ |
| MaxPoolingFwdOp | fp16, bf16 | same as input | `torch.nn.functional.max_pool1d` |

## Related Files

- `tileops/kernels/pooling/max_pooling_fwd.py` — **add**: TileLang kernel 实现
- `tileops/ops/pooling/max_pooling_fwd.py` — **add**: Op wrapper 和公共接口
- `tileops/ops/__init__.py` — **modify**: 导出新的 Conv2d op 类
- `tileops/kernels/__init__.py` — **modify**: 导出新的 MaxPoolingFwdKernel 类
- `tests/ops/test_max_pooling_fwd.py` — **add**: 正确性测试
- `benchmarks/ops/bench_max_pooling_fwd.py` — **add**: 延迟/吞吐/带宽基准测试

## Implementation Plan

### 1. Kernel Implementation (L1)

在 `tileops/kernels/pooling/` 下实现 TileLang forward kernel。

- **Kernel**: 在 `tileops/kernels/pooling/max_pooling_fwd.py` 中实现 forward kernel
  - 参考 `tileops/kernels/deepseek_nsa/mean_pooling_fwd.py` 的结构
  - 使用 `tilelang.jit` 装饰器
  - 实现 `torch.library.custom_op` 包装
  - 支持 SM80+ 架构
- **Support**: 标准 forward Max Pooling semantics，支持可配置的 kernel_size, stride, padding, dilation
- **Exclude**: 本 issue 不包含 backward 算子
- **Verification**: 通过功能正确性检查

### 2. Op Definition (L2)

添加算子包装器、测试覆盖和基准测试入口。

- **Interface**: 在 `tileops/ops/pooling/max_pooling_fwd.py` 定义 Op wrapper
- **Update**: 更新 `tileops/ops/__init__.py` 中的包导出
- **Unit Tests**: 在 `tests/ops/test_max_pooling_fwd.py` 实现测试
  - FP16 (rtol=1e-3, atol=1e-3)
  - BF16 (rtol=1.6e-2, atol=1.6e-2)
  - 覆盖 kernel_size, stride, padding, dilation 组合
  - 在真实 CUDA 机器上运行测试
- **Benchmarks**: 在 `benchmarks/ops/bench_max_pooling_fwd.py` 实现基准测试
  - 报告 latency, TFLOPS, DRAM bandwidth
  - 在代表性 workload 上比较性能

## Goal

| 指标 | 值 |
| --- | --- |
| Data types | FP16, BF16 |
| Maximum relative error (FP16) | 1e-3 |
| Maximum relative error (BF16) | 1.6e-2 |
| Maximum absolute error (FP16) | 1e-3 |
| Maximum absolute error (BF16) | 1.6e-2 |
| Accuracy reference | PyTorch `torch.nn.functional.max_pool1d` |

## Acceptance Criteria

- [ ] AC-1: MaxPoolingFwdKernel 在 `tileops/kernels/pooling/max_pooling_fwd.py` 中实现，支持 kernel_size, stride, padding, dilation 配置
- [ ] AC-2: MaxPoolingFwdOp 在 `tileops/ops/pooling/max_pooling_fwd.py` 中实现并通过 `tileops/ops/__init__.py` 导出
- [ ] AC-3: Kernel 正确性在 FP16 和 BF16 上验证，误差在确认的相对/绝对误差范围内
- [ ] AC-4: `tests/ops/test_max_pooling_fwd.py` 覆盖约定的参数组合并通过 PyTorch 参考实现
- [ ] AC-5: `benchmarks/ops/bench_max_pooling_fwd.py` 报告 latency, TFLOPS, DRAM bandwidth，并包含与 PyTorch 基线的比较

## Constraints

- 必须遵循 Op 基类模式 — 不引入新模式
- 不能更改任何现有的公共 API 或破坏现有测试
- Kernel 必须用 Tile-lang 编写
- CUDA only；CPU 支持不在范围内

## Reference

- PyTorch MaxPool1d API: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
- TileOPs 家族跟踪: [FEAT][CONV] implement conv & pooling operator family (16 ops) #402
