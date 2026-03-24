---
type: FEAT
component: Pooling
labels: [feat, pooling]
target_repo: Sur-Ring/TileOPs
---

# [FEAT][Pooling] Implement Average Pooling (AvgPool) Forward Operator

## Description

### Symptom / Motivation
The conv & pooling operator family tracking issue #402 requires implementing Average Pooling operators alongside the existing Max Pooling implementation. Currently only MaxPool1d/2d/3d forward kernels and MaxPool2dOp are implemented. AvgPool operators are missing but needed for completeness of the pooling operator family.

### Root Cause Analysis
N/A - new feature implementation

### Related Files

**Reference files (existing)**:
- `tileops/kernels/pooling/max_pool1d.py` — 1D max pooling kernel (pattern reference)
- `tileops/kernels/pooling/max_pool2d.py` — 2D max pooling kernel (pattern reference)
- `tileops/kernels/pooling/max_pool3d.py` — 3D max pooling kernel (pattern reference)
- `tileops/ops/pooling/max_pool2d.py` — 2D max pooling op (pattern reference)
- `tileops/ops/__init__.py` — existing op exports (for adding new exports)
- `tileops/kernels/__init__.py` — existing kernel exports (for adding new exports)
- `tileops/kernels/pooling/__init__.py` — existing pooling kernel exports (for adding new exports)
- `tileops/ops/pooling/__init__.py` — existing pooling op exports (for adding new exports)

**Files to create**:
- `tileops/kernels/pooling/avg_pool1d.py` — **new**: 1D avg pooling kernel
- `tileops/kernels/pooling/avg_pool2d.py` — **new**: 2D avg pooling kernel
- `tileops/kernels/pooling/avg_pool3d.py` — **new**: 3D avg pooling kernel
- `tileops/ops/pooling/avg_pool1d.py` — **new**: 1D avg pooling op
- `tileops/ops/pooling/avg_pool2d.py` — **new**: 2D avg pooling op
- `tileops/ops/pooling/avg_pool3d.py` — **new**: 3D avg pooling op
- `tests/ops/test_avg_pool1d.py` — **new**: 1D avg pooling tests
- `tests/ops/test_avg_pool2d.py` — **new**: 2D avg pooling tests
- `tests/ops/test_avg_pool3d.py` — **new**: 3D avg pooling tests
- `benchmarks/ops/bench_avg_pool1d.py` — **new**: 1D avg pooling benchmarks
- `benchmarks/ops/bench_avg_pool2d.py` — **new**: 2D avg pooling benchmarks
- `benchmarks/ops/bench_avg_pool3d.py` — **new**: 3D avg pooling benchmarks

## Goal

Implement AvgPool1d, AvgPool2d, and AvgPool3d forward operators following the same L1 Kernel → L2 Op architecture pattern as MaxPooling, with TileLang-based kernels and PyTorch reference validation.

## Plan

**Plan type: proposal**

### Phase 1: AvgPool1d

1. **Kernel**: Create `tileops/kernels/pooling/avg_pool1d.py` implementing `AvgPooling1dFwdKernel`
   - Reference `tileops/kernels/pooling/max_pool1d.py` structure
   - Use `tilelang.jit` decorator and `torch.library.custom_op`
   - Replace max reduction with sum + divide (mean) reduction
   - Support SM80+ architectures

2. **Op**: Create `tileops/ops/pooling/avg_pool1d.py` implementing `AvgPooling1dFwdOp`
   - Reference `tileops/ops/pooling/max_pool2d.py` pattern
   - Expose kernel_size, stride, padding, dilation parameters

3. **Tests**: Create `tests/ops/test_avg_pool1d.py`
   - FP16 (rtol=1e-3, atol=1e-3)
   - BF16 (rtol=1.6e-2, atol=1.6e-2)
   - Reference: `torch.nn.functional.avg_pool1d`

4. **Benchmarks**: Create `benchmarks/ops/bench_avg_pool1d.py`
   - Report latency, TFLOPS, DRAM bandwidth
   - Compare against `torch.nn.functional.avg_pool1d` baseline

### Phase 2: AvgPool2d

1. **Kernel**: Create `tileops/kernels/pooling/avg_pool2d.py` implementing `AvgPooling2dFwdKernel`
   - Reference `tileops/kernels/pooling/max_pool2d.py` structure
   - Replace max reduction with sum + divide (mean) reduction

2. **Op**: Create `tileops/ops/pooling/avg_pool2d.py` implementing `AvgPooling2dFwdOp`

3. **Tests**: Create `tests/ops/test_avg_pool2d.py`
   - Reference: `torch.nn.functional.avg_pool2d`

4. **Benchmarks**: Create `benchmarks/ops/bench_avg_pool2d.py`

### Phase 3: AvgPool3d

1. **Kernel**: Create `tileops/kernels/pooling/avg_pool3d.py` implementing `AvgPooling3dFwdKernel`
   - Reference `tileops/kernels/pooling/max_pool3d.py` structure

2. **Op**: Create `tileops/ops/pooling/avg_pool3d.py` implementing `AvgPooling3dFwdOp`

3. **Tests**: Create `tests/ops/test_avg_pool3d.py`
   - Reference: `torch.nn.functional.avg_pool3d`

4. **Benchmarks**: Create `benchmarks/ops/bench_avg_pool3d.py`

### Package Updates

- Update `tileops/ops/pooling/__init__.py` to export `AvgPooling1dFwdOp`, `AvgPooling2dFwdOp`, `AvgPooling3dFwdOp`
- Update `tileops/ops/__init__.py` to export the new AvgPool ops
- Update `tileops/kernels/pooling/__init__.py` to export the new Kernel classes

## Constraints

- Must follow existing L1 Kernel → L2 Op architecture pattern (no new patterns)
- Must not change any existing public APIs or break existing tests
- Kernels must be written in TileLang
- CUDA only; CPU support is out of scope
- Must pass existing linting and code style checks

## Acceptance Criteria

### AvgPool1d
- [ ] AC-1: AvgPooling1dFwdKernel implemented in `tileops/kernels/pooling/avg_pool1d.py`
- [ ] AC-2: AvgPooling1dFwdOp implemented in `tileops/ops/pooling/avg_pool1d.py` and exported
- [ ] AC-3: `tests/ops/test_avg_pool1d.py` passes with PyTorch reference validation

### AvgPool2d
- [ ] AC-4: AvgPooling2dFwdKernel implemented in `tileops/kernels/pooling/avg_pool2d.py`
- [ ] AC-5: AvgPooling2dFwdOp implemented in `tileops/ops/pooling/avg_pool2d.py` and exported
- [ ] AC-6: `tests/ops/test_avg_pool2d.py` passes with PyTorch reference validation

### AvgPool3d
- [ ] AC-7: AvgPooling3dFwdKernel implemented in `tileops/kernels/pooling/avg_pool3d.py`
- [ ] AC-8: AvgPooling3dFwdOp implemented in `tileops/ops/pooling/avg_pool3d.py` and exported
- [ ] AC-9: `tests/ops/test_avg_pool3d.py` passes with PyTorch reference validation

### Benchmarks
- [ ] AC-10: All three operators report latency, TFLOPS, DRAM bandwidth in benchmarks
- [ ] AC-11: Benchmarks compare against `torch` baseline
