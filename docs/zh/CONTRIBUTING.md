# TileOPs 协作指南

本文档介绍了向 TileOPs 贡献代码的标准工作流程。我们遵循严格的 **"2+1 Review"** 策略，以确保代码质量和知识共享。

## 1. 开发生命周期

我们遵循标准的 **Issue -> Fork -> PR** 工作流程。**请勿**直接在主仓库创建分支。

### 步骤 1：Issue（选取或创建任务）

- **新算子**：如果您代表一个新算子，请**创建新的 Issue**，使用 "New Operator Request" 模板。
- **现有任务**：浏览 [Issues](https://github.com/tile-ai/TileOPs/issues) 并评论以认领任务。
- **分解任务**：对于 "New Operator" 类型的 epic，创建**子任务 Issue**（例如 "Implement Kernel"），使用相应模板。

### 步骤 2：Fork 与分支（开始工作）

- **Fork**：将仓库 Fork 到您的 GitHub 账户。
- **克隆**：在本地克隆您的 Fork（`git clone ...`）。
- **设置**：运行 `make install` 以一步安装开发依赖项和 pre-commit hooks。
- **创建分支**：在您的 Fork 上创建新分支。
  - **基础分支**：在创建分支前与上游 `main` 同步。
  - **命名**：`type/scope/description`（规范前缀见 `.claude/conventions/types.sh`），例如：
    - `feat/flash-attn/fwd-kernel`
    - `fix/mla/parsing-error`
    - `doc/readme/add-examples`
    - `test/gemv/add-edge-cases`
    - `perf/mha/improve-bandwidth`
    - `bench/gemm/add-triton-baseline`

### 步骤 3：提交（保存工作）

我们遵循 **TileLang Commit Convention**：`[Type] Description` 或 `[Type][Scope] Description`。

**常用类型**（规范列表见 `.claude/conventions/types.sh`）：

- `[Feat]`：新功能或算子。
- `[BugFix]`：Bug 修复。
- `[Fix]`：非 bug 修正。
- `[Refactor]`：代码重构，不改变行为。
- `[Enhancement]`：现有功能的改进。
- `[Doc]`：文档更新。
- `[Chore]`：构建系统或工作流程变更。
- `[Bench]`：基准测试更新。
- `[CI]`：CI/CD 变更。
- `[Test]`：测试变更。
- `[Perf]`：性能改进。

**示例**：

- `[Feat] Add multi-head attention forward op`
- `[BugFix] Fix index out of bounds in reduction kernel`
- `[Refactor] Reformat code to adhere to Google Style`
- `[Enhancement][MHA] Improve multi-head attention forward op performance on Hopper`
- `[Doc] Update README.md`
- `[Chore][CI] Update CUDA version to 12.9`
- `[Bench][MHA] Add Triton baseline for multi-head attention forward op`

> **提示**：在推送之前，运行 `pre-commit run --all-files` 并修复所有问题！
>
> **提交作用域规则**：保持提交信息简洁。不要在提交正文中放入冗长的验证部分、
> 基准测试表格或命令记录；这些内容应放在 PR 描述和跟踪 Issue 中。

### 步骤 4：Pull Request（提交代码）

- **标题**：与您的提交信息一致，例如 `[Feat] Add multi-head attention forward op`。
- **模板**：完整填写 PR 模板检查清单。
- **描述**：提供变更的详细描述，包括任何相关的上下文或背景信息。您可以利用 `gemini-code-assist` 来生成 PR 描述摘要。
- **基准测试报告**：对于新算子或语义/性能敏感变更，报告在可见 CUDA 机器上的实际测量数据。包括具有代表性的小、中、大形状。
- **验证顺序**：在发布基准测试数据之前，先在同一台机器上运行针对性的正确性测试套件。
- **CI**：确保所有 GitHub Actions（Lint/Test/Build）通过。

## 2. "2+1" Review 策略

我们使用 2+1 审核流程来规范开发流程，同时保持代码质量。

### 同行评审（"2"）

- **目标**：完整性检查、代码风格、逻辑验证和知识共享。
- **操作**：邀请**其他 2 位开发者**审核您的 PR。
- **要求**：在进入第二阶段之前，您必须获得 **2 位同行的 Approvals**。
  - *同行应验证：流程（Issue-Fork-PR）、逻辑正确性、测试覆盖率、CI 通过、格式、命名规范。*

### 导师评审（"1"）

- **目标**：架构验证、安全检查和最终把关。
- **操作**：获得 2 位同行 Approvals 后，向**导师团队**（`@tile-ai/tileops-review`）请求审核。
- **要求**：您必须获得**导师团队的 1 位 Approval**。
  - *导师验证：架构适配性、性能影响、破坏性变更。*

### 合并

只有满足 **2 位同行 Approvals + 1 位导师 Approval + CI 通过** 后，代码才能被合并。

## 3. 审核清单

### 开发者（同行评审）

**流程与风格**

- [ ] **工作流程**：PR 是否关联了 Issue？是否来自 Fork？
- [ ] **CI/CD**：所有自动化检查（Lint、Test、Build）是否通过？
- [ ] **格式**：是否严格遵循 Google Python Style（导入、命名）？
- [ ] **文档字符串**：所有公共元素是否有 Google 风格的文档字符串？
- [ ] **文档**：文档是否清晰、完整、无拼写错误？

**正确性与测试**

- [ ] **逻辑**：算法是否正确？是否有明显的 bug？
- [ ] **单元测试**：`tests/` 中是否有与代码匹配的测试？它们是否通过？
- [ ] **共享测试框架**：如果 PR 修改了共享测试基础设施（如 `tests/test_base.py` 或公共 fixtures），是否保留了现有默认语义，或在同一 PR 中更新了所有受影响的测试？
- [ ] **边界情况**：是否处理了空输入或边界形状？
- [ ] **错误处理**：输入是否通过信息性错误消息进行验证？
- [ ] **Smoke 范围**：如果共享测试基础设施变更，除了针对性测试套件外，是否在真实机器上完成了更广泛的 `pytest -m smoke` 运行？

**基准测试**

- [ ] **基准测试**：`benchmarks/` 中是否提供了基准测试脚本？
- [ ] **结果**：Issue 和 PR 描述中是否包含了延迟 / TFLOPS / 带宽数据？
- [ ] **运行时**：GPU 依赖的测试和基准测试是否在真实的 CUDA 可见机器上运行，而非沙箱环境？
- [ ] **覆盖率**：基准测试表格是否包括具有代表性的小、中、大形状？
- [ ] **诚实性**：基准测试表格是否如实报告具有代表性的性能退化，而非省略它们？

### 导师（导师评审）

**架构与设计**

- [ ] **两层架构准备**：是否清晰分离了 `Kernel`（L1）-> `Op`（L2）？
- [ ] **API 设计**：L2 Op 接口是否符合 Python 规范？
- [ ] **兼容性**：L2 Op 是否与 `torch.compile` 和 CUDA Graphs 兼容？

**维护**

- [ ] **破坏性变更**：这是否破坏了现有 API？（如果是，是否必要/有文档说明？）
- [ ] **文档**：文档是否清晰、完整、无拼写错误？
