# Max Pooling 实现计划

## Description

实现 Max Pooling 前向算子，支持 1D、2D 和 3D 输入。

### MaxPool1d

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

### MaxPool2d

对于输入张量 `x ∈ R^{N × C × H × W}`，输出张量 `y ∈ R^{N × C × H_out × W_out}` 定义为：

```
y[n, c, h_out, w_out] = max_{kh, kw} x[n, c, h_out * stride_h + kh - pad_h, w_out * stride_w + kw - pad_w]
```

其中：

```
H_out = (H + 2 * pad_h - dilation * (kernel_size_h - 1) - 1) // stride_h + 1
W_out = (W + 2 * pad_w - dilation * (kernel_size_w - 1) - 1) // stride_w + 1
```

**输入形状：** `(N, C, H, W)` = `(batch, channels, height, width)`
**输出形状：** `(N, C, H_out, W_out)`

### MaxPool3d

对于输入张量 `x ∈ R^{N × C × D × H × W}`，输出张量 `y ∈ R^{N × C × D_out × H_out × W_out}` 定义为：

```
y[n, c, d_out, h_out, w_out] = max_{kd, kh, kw} x[n, c, d_out * stride_d + kd - pad_d, h_out * stride_h + kh - pad_h, w_out * stride_w + kw - pad_w]
```

其中：

```
D_out = (D + 2 * pad_d - dilation * (kernel_size_d - 1) - 1) // stride_d + 1
H_out = (H + 2 * pad_h - dilation * (kernel_size_h - 1) - 1) // stride_h + 1
W_out = (W + 2 * pad_w - dilation * (kernel_size_w - 1) - 1) // stride_w + 1
```

**输入形状：** `(N, C, D, H, W)` = `(batch, channels, depth, height, width)`
**输出形状：** `(N, C, D_out, H_out, W_out)`

______________________________________________________________________

本 issue 跟踪标准 Max Pooling forward 算子，作为 conv & pooling 算子家族的一部分（#402）。

## Dtype Support Matrix

| Op / API    | Input dtypes | Output dtype  | PyTorch / reference semantic baseline |
| ----------- | ------------ | ------------- | ------------------------------------- |
| MaxPool1dOp | fp16, bf16   | same as input | `torch.nn.functional.max_pool1d`      |
| MaxPool2dOp | fp16, bf16   | same as input | `torch.nn.functional.max_pool2d`      |
| MaxPool3dOp | fp16, bf16   | same as input | `torch.nn.functional.max_pool3d`      |

## Related Files

### 1D MaxPool1d

- `tileops/kernels/pooling/max_pool1d.py` — **add**: TileLang kernel 实现
- `tileops/ops/pooling/max_pool1d.py` — **add**: Op wrapper
- `tests/ops/test_max_pool1d.py` — **add**: 正确性测试
- `benchmarks/ops/bench_max_pool1d.py` — **add**: 基准测试

### 2D MaxPool2d

- `tileops/kernels/pooling/max_pool2d.py` — **add**: TileLang kernel 实现
- `tileops/ops/pooling/max_pool2d.py` — **add**: Op wrapper
- `tests/ops/test_max_pool2d.py` — **add**: 正确性测试
- `benchmarks/ops/bench_max_pool2d.py` — **add**: 基准测试

### 3D MaxPool3d

- `tileops/kernels/pooling/max_pool3d.py` — **add**: TileLang kernel 实现
- `tileops/ops/pooling/max_pool3d.py` — **add**: Op wrapper
- `tests/ops/test_max_pool3d.py` — **add**: 正确性测试
- `benchmarks/ops/bench_max_pool3d.py` — **add**: 基准测试

### 包级别导出

- `tileops/ops/__init__.py` — **modify**: 导出 MaxPool1dOp, MaxPool2dOp, MaxPool3dOp
- `tileops/kernels/__init__.py` — **modify**: 导出对应的 Kernel 类

## Plan

### Phase 1: 1D MaxPool1d

#### 1.1 Kernel Implementation (L1)

- [ ] **Kernel**: 在 `tileops/kernels/pooling/max_pool1d.py` 中实现 forward kernel
  - 参考 `tileops/kernels/deepseek_nsa/mean_pooling_fwd.py` 的结构
  - 使用 `tilelang.jit` 装饰器
  - 实现 `torch.library.custom_op` 包装
  - 支持 SM80+ 架构

#### 1.2 Op Definition (L2)

- [ ] **Interface**: 在 `tileops/ops/pooling/max_pool1d.py` 定义 Op wrapper
- [ ] **Update**: 更新 `tileops/ops/__init__.py` 中的包导出
- [ ] **Unit Tests**: 在 `tests/ops/test_max_pool1d.py` 实现测试
  - [ ] FP16 (rtol=1e-3, atol=1e-3)
  - [ ] BF16 (rtol=1.6e-2, atol=1.6e-2)
  - [ ] 覆盖 kernel_size, stride, padding, dilation 组合
  - [ ] 在真实 CUDA 机器上运行测试
- [ ] **Benchmarks**: 在 `benchmarks/ops/bench_max_pool1d.py` 实现基准测试
  - 报告 latency, TFLOPS, DRAM bandwidth
  - 在代表性 workload 上比较性能

### Phase 2: 2D MaxPool2d

#### 2.1 Kernel Implementation (L1)

- [ ] **Kernel**: 在 `tileops/kernels/pooling/max_pool2d.py` 中实现 forward kernel
  - 2D pooling 在 H 和 W 两个维度上进行
  - 支持方形和矩形 kernel

#### 2.2 Op Definition (L2)

- [ ] **Interface**: 在 `tileops/ops/pooling/max_pool2d.py` 定义 Op wrapper
- [ ] **Unit Tests**: 在 `tests/ops/test_max_pool2d.py` 实现测试
  - [ ] FP16 (rtol=1e-3, atol=1e-3)
  - [ ] BF16 (rtol=1.6e-2, atol=1.6e-2)
  - [ ] 覆盖 kernel_size (h, w), stride (h, w), padding (h, w), dilation (h, w)
- [ ] **Benchmarks**: 在 `benchmarks/ops/bench_max_pool2d.py` 实现基准测试
  - 报告 latency, TFLOPS, DRAM bandwidth
  - 在代表性 workload 上比较性能

### Phase 3: 3D MaxPool3d

#### 3.1 Kernel Implementation (L1)

- [ ] **Kernel**: 在 `tileops/kernels/pooling/max_pool3d.py` 中实现 forward kernel
  - 3D pooling 在 D, H 和 W 三个维度上进行

#### 3.2 Op Definition (L2)

- [ ] **Interface**: 在 `tileops/ops/pooling/max_pool3d.py` 定义 Op wrapper
- [ ] **Unit Tests**: 在 `tests/ops/test_max_pool3d.py` 实现测试
  - [ ] FP16 (rtol=1e-3, atol=1e-3)
  - [ ] BF16 (rtol=1.6e-2, atol=1.6e-2)
- [ ] **Benchmarks**: 在 `benchmarks/ops/bench_max_pool3d.py` 实现基准测试
  - 报告 latency, TFLOPS, DRAM bandwidth
  - 在代表性 workload 上比较性能

## Directory Structure

```
tileops/kernels/pooling/
├── __init__.py
├── max_pool1d.py             # MaxPool1d Kernel
├── max_pool2d.py             # MaxPool2d Kernel
└── max_pool3d.py             # MaxPool3d Kernel

tileops/ops/pooling/
├── __init__.py
├── max_pool1d.py             # MaxPool1d Op
├── max_pool2d.py             # MaxPool2d Op
└── max_pool3d.py             # MaxPool3d Op

tests/ops/
├── test_max_pool1d.py
├── test_max_pool2d.py
└── test_max_pool3d.py

benchmarks/ops/
├── bench_max_pool1d.py
├── bench_max_pool2d.py
└── bench_max_pool3d.py
```

## Goal

| 指标                          | 值                                       |
| ----------------------------- | ---------------------------------------- |
| Data types                    | FP16, BF16                               |
| Maximum relative error (FP16) | 1e-3                                     |
| Maximum relative error (BF16) | 1.6e-2                                   |
| Maximum absolute error (FP16) | 1e-3                                     |
| Maximum absolute error (BF16) | 1.6e-2                                   |
| Accuracy reference (1D)       | PyTorch `torch.nn.functional.max_pool1d` |
| Accuracy reference (2D)       | PyTorch `torch.nn.functional.max_pool2d` |
| Accuracy reference (3D)       | PyTorch `torch.nn.functional.max_pool3d` |

## Acceptance Criteria

### 1D MaxPool1d

- [ ] AC-1: MaxPool1dKernel 在 `tileops/kernels/pooling/max_pool1d.py` 中实现
- [ ] AC-2: MaxPool1dOp 在 `tileops/ops/pooling/max_pool1d.py` 中实现并导出
- [ ] AC-3: `tests/ops/test_max_pool1d.py` 通过 PyTorch 参考验证

### 2D MaxPool2d

- [ ] AC-4: MaxPool2dKernel 在 `tileops/kernels/pooling/max_pool2d.py` 中实现
- [ ] AC-5: MaxPool2dOp 在 `tileops/ops/pooling/max_pool2d.py` 中实现并导出
- [ ] AC-6: `tests/ops/test_max_pool2d.py` 通过 PyTorch 参考验证

### 3D MaxPool3d

- [ ] AC-7: MaxPool3dKernel 在 `tileops/kernels/pooling/max_pool3d.py` 中实现
- [ ] AC-8: MaxPool3dOp 在 `tileops/ops/pooling/max_pool3d.py` 中实现并导出
- [ ] AC-9: `tests/ops/test_max_pool3d.py` 通过 PyTorch 参考验证

### Benchmarks

- [ ] AC-10: 所有三个算子的 benchmark 报告 latency, TFLOPS, DRAM bandwidth

## Constraints

- 必须遵循 Op 基类模式 — 不引入新模式
- 不能更改任何现有的公共 API 或破坏现有测试
- Kernel 必须用 Tile-lang 编写
- CUDA only；CPU 支持不在范围内

## Reference

- PyTorch MaxPool1d API: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
- PyTorch MaxPool2d API: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
- PyTorch MaxPool3d API: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html
- TileOPs 家族跟踪: [FEAT][CONV] implement conv & pooling operator family (16 ops) #402
