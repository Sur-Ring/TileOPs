# TileOPs 开发指南

本文档概述了 TileOPs 项目的软件工程标准、架构和开发流程。所有贡献者必须遵守这些准则，以确保代码质量、可维护性和性能。

## 1. 架构概述

TileOPs 遵循严格的**双层分层架构**。这种关注点分离确保了硬件特定优化（Kernel）与用户面向 API（Op）的解耦。

| 层级 |   名称   |    类比    | 描述                                                                                                                                         |
| :--: | :------: | :--------: | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **L2** | **Op** | `torch.ops` | **无状态调度器**：硬件无关的入口点。调度到特定内核。支持 **CUDA-Graph** 和 **torch.compile**。                                                   |
| **L1** | **Kernel** | TileLang   | **实现层**：使用 TileLang 编写的原始内核，针对特定硬件（如 Hopper、Ampere）优化。                                                              |

______________________________________________________________________

## 2. 开发流程

开发新算子采用自底向上的方法，从 Kernel 实现逐步过渡到 Op 抽象。

### 第 0 步：创建跟踪 Issue

- **操作**：使用 **"New Operator Request"** 模板创建新 issue。
- **目标**：定义范围并跟踪双层的进度。
- **任务分解**：对于新算子，**将检查清单项目分解为详细的子 issue**（即 **Kernel 实现**、**Op 实现**、**基准测试结果**）。这使得新贡献者能够承担更小、定义更明确的任务并提交更小的 PR。
- **完成定义**：当算子完全实现并验证后，issue 关闭。

### 第 1 步：Kernel 实现（L1）

- **位置**：`tileops/kernels/{算子名称}/`
- **目标**：使用 TileLang 实现核心逻辑。
- **文档字符串**：详细描述参数和返回值。
- **完成定义**：内核正确编译并运行。

### 第 2 步：Op 定义与验证（L2）

- **位置**：`tileops/ops/{算子名称}.py`
- **职责**：
  - 将内核封装为 Python 函数。
  - **文档字符串**：Google 风格（Args、Returns、Example）。
  - **单元测试**：与纯 PyTorch 参考实现比较输出（必需）。
  - **dtype 契约**：明确定义支持的输入 dtype、输出 dtype 和拒绝的 dtype。
  - **参数契约**：在 Op/API 边界验证用户提供的标量参数与有效内核 dtype 的对比。无效值必须在任何 TIR/代码生成步骤之前抛出用户可读的 `ValueError`。
  - **基准测试**：测量延迟、TFLOPS（必需）和 DRAM 带宽（必需）。
- **标准**：
  - 使用 `torch.testing.assert_close` 进行浮点数验证。
    - **FP16**：`rtol=1e-3`，`atol=1e-3`
    - **BF16**：`rtol=1.6e-2`，`atol=1.6e-2`
  - 对非浮点输出（如 `bool`、掩码和索引张量）使用精确比较（`torch.equal`）。
  - 当输出 dtype 与输入 dtype 不同时，测试必须断言输出 dtype。
  - GPU 依赖的单元测试必须在具有主机可见 CUDA 设备的真实机器上运行。不要将沙盒环境的结果作为最终正确性证据。
  - 基准测试结果必须可复现。
  - 不要用内核本地的 lowering 变通方法修复接口契约 bug。如果失败是由无效用户参数引起的，应先修复验证边界，只有在语义是合法且有文档记录的情况下才修改内核。
- **完成定义**：Op 在单元测试中验证通过，基准测试正确运行。

### 第 3 步：基准测试结果

- **位置**：`benchmarks/ops/bench_{算子名称}.py`
- **目标**：测量延迟、TFLOPS（必需）和 DRAM 带宽（必需）。
- **执行**：`pytest benchmarks/` 自动生成 `profile_run.log`。
- **必需顺序**：在报告基准测试数字之前，先在同一台真实 GPU 机器上运行目标正确性套件。
- **必需形状**：基准测试表必须包含具有代表性的小、中、大形状，除非 issue 明确定义了不同的基准测试矩阵。
- **完成定义**：对算子进行基准测试并将结果放入 issue。

### 第 4 步：PR 验收包

每个添加新 op、扩展 dtype 覆盖范围或更改语义行为的 PR 都必须在 PR 描述中包含以下内容：

1. 一个 dtype 支持矩阵，说明实际支持的输入 dtype、输出 dtype 以及 PyTorch 或文档化的基准语义。
2. 一个具体的验收检查清单（`AC-1`、`AC-2`、...），涵盖实现、正确性测试、dtype 契约检查和基准测试交付物。
3. 一个包含真实测量数据和环境元数据的基准测试比较表。

在 PR 正文中使用以下基准测试表格式：

| Shape / Params | dtype | Op         | TileOPs (ms) | Baseline (ms) | Ratio | Notes |
| -------------- | ----- | ---------- | ------------ | ------------- | ----- | ----- |
| example        | fp16  | example_op | ...          | ...           | ...   | ...   |

PR 正文应保持简洁：仅包含摘要、dtype 矩阵、验收检查清单和基准测试表。除非审查者明确要求，否则不要将冗长的验证命令输出粘贴到 PR 描述中。

如果实现是正确性优先且性能后续跟进被推迟，在 PR 正文中明确说明并链接后续 issue。不要将基准测试部分留空。

______________________________________________________________________

## 3. 代码规范

我们对代码质量和一致性有高标准要求。

### Python 代码风格

- **风格指南**：**[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)**。
  - 我们严格遵循 Google 风格的格式和文档字符串规范。
- **格式化/检查工具**：`ruff`
- **文档字符串**：所有公共函数和类必须使用 **Google 风格的文档字符串**。

### 改进与类型安全

- **类型提示**：所有函数签名（输入和输出）必须有类型提示。
- **严格类型检查** *（计划中）*：L2（Op）API 将在未来版本中使用 `mypy` 严格模式进行检查。

______________________________________________________________________

## 4. 测试与基准测试策略

测试和基准测试**按关注点分离**：`pytest tests/` 仅验证正确性，`pytest benchmarks/` 仅运行性能分析并自动生成 markdown 报告（`profile_run.log`）。

### 核心抽象类

| 类名              | 位置                        | 职责                                                                                                                           |
| ----------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `FixtureBase`     | `tests/test_base.py`        | 基于元类的 `@decorator`，从 `PARAMS` 类属性应用 `pytest.mark.parametrize`。                                                       |
| `TestBase`        | `tests/test_base.py`       | ABC，包含 `gen_inputs()`、`ref_program()`、`check()`、`check_fn()`。每个 op 子类继承此基类。                                      |
| `BenchmarkBase`   | `benchmarks/benchmark.py`  | 封装 `TestBase` 实例的 ABC。子类实现 `calculate_flops()` 和 `calculate_memory()`。提供 `profile()` 方法。                             |
| `BenchmarkReport` | `benchmarks/benchmark.py`  | 静态收集器——`record()` 存储结果，`dump()` 写入 markdown，`clear()` 重置。                                                          |

### 模式示例

```python
# tests/ops/test_mha.py
class MhaFwdFixture(FixtureBase):
    PARAMS = [("batch, seq_len, heads, dim, causal, dtype, tune", [...])]


class MhaFwdTest(TestBase):
    def gen_inputs(self): ...
    def ref_program(self, q, k, v): ...


@MhaFwdFixture
def test_mha_fwd(batch, seq_len, heads, dim, causal, dtype, tune):
    test = MhaFwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionFwdOp(...)
    test.check(op, *test.gen_inputs())


# benchmarks/ops/bench_mha.py
class MhaFwdBenchmark(BenchmarkBase):
    def calculate_flops(self): ...
    def calculate_memory(self): ...


@MhaFwdFixture  # 重用相同的参数化装饰器
def test_mha_fwd_bench(batch, seq_len, heads, dim, causal, dtype, tune):
    test = MhaFwdTest(batch, heads, seq_len, dim, causal, dtype)
    bm = MhaFwdBenchmark(test)
    inputs = test.gen_inputs()
    op = MultiHeadAttentionFwdOp(...)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("mha_fwd", locals(), result, tag="tileops")
```

### 单元测试

- **框架**：`pytest`
- **位置**：`tests/`
- **要求**：
  - 每个 op 在 `tests/ops/` 中定义一个 `TestBase` 子类，包含 `gen_inputs()` 和 `ref_program()`。
  - 测试必须覆盖 `FP16` 和 `BF16` 数据类型。
  - 测试必须对常见形状（批量大小、头数、序列长度）进行参数化。
  - 测试必须明确编码 dtype 契约：覆盖支持的 dtype，拒绝不支持的 dtype，并断言输出 dtype。
  - 对共享测试基础设施（如 `tests/test_base.py`、公共 fixtures 或共享比较器）的更改必须保留现有默认语义，除非迁移计划在同一 PR 中更新所有受影响的测试。
  - 在声称实现已就绪之前，在真实 GPU 机器上运行目标算子族的完整目标测试文件，而不仅仅是拒绝路径或 smoke 子集。
  - 如果 PR 触及共享测试基础设施，还要在合并前在真实机器上运行更广泛的 `pytest -m smoke` 通过以捕获目标算子族之外的回归。

### 基准测试

- **框架**：`benchmarks.benchmark.BenchmarkBase`
- **位置**：`benchmarks/ops/`
- **执行**：`pytest benchmarks/`——自动生成 `profile_run.log`（markdown 格式）。
- **指标**：
  - 延迟（ms）
  - TFLOPS（每秒万亿次浮点运算）
  - DRAM 带宽（GB/s）
- **报告规则**：
  - PR 和 issue 中报告的基准测试数字必须来自真实 GPU 机器，不能是没有直接 CUDA 可见性的沙盒环境。
  - 报告具有代表性的小、中、大形状。
  - 不要只挑选有利形状；如果具有代表性的大形状结果相对于基准出现回归，如实报告。

______________________________________________________________________

## 5. 目录结构参考

```text
TileOPs/
├── tileops/
│   ├── kernels/   # L1: TileLang 内核
│   ├── ops/       # L2: Op + 调度器
│   └── utils/     # 工具函数
├── tests/         # 单元测试
├── benchmarks/    # 基准测试和性能脚本
└── docs/          # 项目文档
```

## 6. Pull Request 流程

### 提交 PR 前

1. **格式化代码**：运行 pre-commit hooks 以确保代码风格合规。
   ```bash
   pre-commit run --all-files
   ```
1. **运行测试**：确保所有相关单元测试在本地通过。
   ```bash
   PYTHONPATH="$PWD" python -m pytest tests/ops/test_<op_name>.py
   ```

### CI/CD 检查

当你打开 PR 时，将运行以下自动化检查：

- **Lint**：检查代码风格（Google Style）、导入排序和拼写。
- **Test**：在 GPU runner 上运行单元测试和基准测试。
- **Build**：验证包构建成功。

**注意**：

- 所有 CI 检查通过后才能合并。
- **审批**：遵循 **2+1 策略**（2 位同行 + 1 位导师）。参见 **[CONTRIBUTING.md](./CONTRIBUTING.md)**。
