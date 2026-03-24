# TileOPs 待办事项

本文档记录项目中已知但尚未完成的工作。

---

## 高优先级

### 1. MoE Op 未在顶层导出

**位置**: [tileops/ops/__init__.py](tileops/ops/__init__.py)

MoE 子模块 `tileops/ops/moe/__init__.py` 导出了 4 个 op：
- `FusedTopKOp`
- `MoePermuteOp`
- `MoePermuteAlignOp`
- `MoeUnpermuteOp`

但只有 `MoePermuteAlignOp` 在 `tileops.ops` 层被导入和导出（第 30 行），其他三个只能通过 `from tileops.ops.moe import FusedTopKOp` 访问。

---

### 2. qk_norm 是空实现

**位置**: [tileops/kernels/norm/qk_norm/](tileops/kernels/norm/qk_norm/)

该目录存在但只包含空的 `__all__: list[str] = []`，没有任何实际的内核实现。也没有对应的 benchmark（`bench_qk_norm.py`）或测试（`test_qk_norm.py`）。

---

### 3. GEMM 优化未完成

**位置**: [tileops/kernels/gemm/gemm.py:194](tileops/kernels/gemm/gemm.py#L194)

```python
# TODO: add persistent, split-k, stream-k...
```

---

### 4. MHA Decode Causal 分支未处理

**位置**:
- [tileops/kernels/flash_decode/mha_decode.py:114,190](tileops/kernels/flash_decode/mha_decode.py#L114)
- [tileops/kernels/flash_decode/mha_decode_paged.py:122](tileops/kernels/flash_decode/mha_decode_paged.py#L122)

```python
# TODO: Handle causal split case
```

---

## 中优先级

### 5. GQA/MHA 不支持 s_q != s_kv

**位置**:
- [tileops/ops/gqa.py:37](tileops/ops/gqa.py#L37)
- [tileops/ops/mha.py:35](tileops/ops/mha.py#L35)

```python
# TODO: support s_q != s_kv
```

---

### 6. DeepSeek MLA 多缓冲优化未实现

**位置**:
- [tileops/kernels/deepseek_mla/deepseek_mla_decode.py:404,629](tileops/kernels/deepseek_mla/deepseek_mla_decode.py#L404)
- [tileops/kernels/deepseek_mla/deepseek_dsa_decode.py:195](tileops/kernels/deepseek_mla/deepseek_dsa_decode.py#L195)

```python
# TODO: Multi buffer
```

用于流水线内存加载的性能优化尚未实现。

---

### 7. Reduction Op 计划但未实现

**位置**: [tileops/ops/reduction/__init__.py:25-26,62-66](tileops/ops/reduction/__init__.py#L25)

```python
# from .cummax import CummaxOp
# from .cummin import CumminOp
...
# "ReduceMaxOp",
# "ReduceMeanOp",
# "ReduceMinOp",
# "ReduceProdOp",
# "ReduceSumOp",
```

这些 reduction 别名被注释掉，内核不存在。

CHECK：这些用了其他名字
---

### 8. InstanceNorm Kernel 未导出

**位置**: [tileops/kernels/norm/__init__.py](tileops/kernels/norm/__init__.py)

`tileops/ops/norm/__init__.py` 导出了 `InstanceNormOp`，但 `tileops/kernels/norm/__init__.py` 没有导出 `InstanceNormKernel`。`instance_norm` 子目录只是用 `G=C` 重用了 `GroupNormKernel`：

```python
# InstanceNorm reuses GroupNormKernel with G=C.
# No dedicated kernel is needed; see tileops/kernels/norm/group_norm/fwd.py.
__all__: list[str] = []
```

---

## 其他

### 9. 变长序列 Op 无法计算 FLOPs/内存

**位置**: [tileops/ops/gqa_sliding_window_varlen_fwd.py:186-194](tileops/ops/gqa_sliding_window_varlen_fwd.py#L186)

```python
@property
def total_flops(self) -> int:
    raise NotImplementedError(
        "total_flops is not defined for varlen ops; "
        "compute per-sample from cu_seqlens at call time.")

@property
def total_memory(self) -> int:
    raise NotImplementedError(
        "total_memory is not defined for varlen ops; "
        "compute per-sample from cu_seqlens at call time.")
```

这意味着 varlen op 无法使用标准 BenchmarkReport 框架进行正确的 FLOPs 或内存指标基准测试。

---

### 10. mhc_pre T.copy 问题

**位置**: [tileops/kernels/mhc/mhc_pre.py:62](tileops/kernels/mhc/mhc_pre.py#L62)

```python
# TODO: <*important> try to figure out why "T.copy" does not work...
```

T.copy 与 mbarrier 不兼容，导致效率较低的 Parallel loop fallback。

---

### 11. NSA Padding 问题

**位置**: [tileops/kernels/deepseek_nsa/nsa_fwd.py:111](tileops/kernels/deepseek_nsa/nsa_fwd.py#L111)

```python
# TODO(TileOPs): may have some padding issues
```

---

---

## 来自 Weekly Report (2026-03-20) 的建议

### MoE 管道 (2/6 → 目标 5/6)
- [ ] Land moe_unpermute (#590) 和 fused_topk (#588) — 完成数据移动层
- [ ] 开始 fused_moe Qwen3 变体 (#591) — 最高优先级的端到端内核

### Linear Attention + SSM (linear_attention 4/8, ssm 0/2)
- [ ] 实现 deltanet chunkwise + recurrence (#405) — 重用 GLA chunk 基础设施
- [ ] 开始 Mamba-2 SSD ops (#596) — ssd_chunk_scan / ssd_chunk_state / ssd_state_passing

### 质量与稳定化
- [ ] 关闭 elementwise 技术债务: fp8 e5m2 perf regression (#559), where fp8 gap (#558), independent ops validation (#606)
- [ ] 合并文件命名重构 PRs (#574–#580) — 120+ 文件排队中
- [ ] CI: 修复 packaging wheel moe header (#612), compile cache eviction policy (#486)

### Op 交付进度 (116/186 = 62%)

| Category | Completed | Total | Progress |
|----------|-----------|-------|----------|
| elementwise | 72 | 72 | 100% |
| reduce | 20 | 20 | 100% |
| norm | 9 | 10 | 90% |
| flash_attention | 7 | 16 | 44% |
| linear_attention | 4 | 8 | 50% |
| moe | 1 | 6 | 17% |
| gemm | 3 | 19 | 16% |
| quantize | 0 | 10 | 0% |
| sampling | 0 | 7 | 0% |
| conv & pooling | 0 | 16 | 0% |
| ssm | 0 | 2 | 0% |

---

## 影响文件汇总

### 缺少正确导出
- `tileops/ops/__init__.py` (MoE ops 未重新导出)
- `tileops/kernels/norm/__init__.py` (缺少 instance_norm, qk_norm)

### 包含 TODOs 的文件
- `tileops/kernels/mhc/mhc_pre.py`
- `tileops/kernels/gemm/gemm.py`
- `tileops/kernels/flash_decode/mha_decode.py`
- `tileops/kernels/flash_decode/mha_decode_paged.py`
- `tileops/kernels/deepseek_mla/deepseek_mla_decode.py`
- `tileops/kernels/deepseek_mla/deepseek_dsa_decode.py`
- `tileops/kernels/deepseek_nsa/nsa_fwd.py`
- `tileops/ops/gqa.py`
- `tileops/ops/mha.py`

### 空/占位符实现
- `tileops/kernels/norm/qk_norm/` (空)
- `tileops/kernels/norm/instance_norm/__init__.py` (空 `__all__`)
