# CLAUDE.md

## Project Overview

TileOPs is a high-performance LLM operator library built on TileLang. The goal is to provide efficient, modular, and maintainable AI workload implementations.

## Development Environment

1. Clone repository: `git clone https://github.com/tile-ai/TileOPs && cd TileOPs`
1. Create and activate a virtual environment (venv, conda, etc.)
1. Install dependencies and pre-commit hooks: `make install`

## Key References

- [DEVELOPMENT.md](docs/DEVELOPMENT.md) — architecture (2-layer stack), development workflow, coding standards, testing strategy, and PR process
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) — "2+1 Review" policy, branch naming, and commit message conventions

## Collaboration Rules for Claude

- Prefer minimal, targeted changes and avoid unrelated refactoring.
- After code changes, run the most relevant tests first.
- If unrelated failures appear, report them but do not fix them in the same task.
- Add necessary docs and tests when introducing files/interfaces.
- Response should include: change summary, affected paths, validation steps, and next suggestions.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

TileOPs 是基于 TileLang 的高性能 LLM 算子库，提供高效、模块化、可维护的 AI 工作负载实现。

## 常用命令

```bash
make install      # 安装依赖和 pre-commit hooks
make lint         # 运行代码风格检查
make test         # 运行测试套件
make test-smoke   # 仅运行 smoke 级测试
make test-full    # 运行 smoke + full 级测试
make test-nightly # 运行 smoke + full + nightly 级测试
make bench        # 运行基准测试
```

单测试运行：
```bash
PYTHONPATH="$PWD" python -m pytest tests/ops/test_<op_name>.py
```

## 架构：2 层分层

| 层级 | 名称 | 说明 |
|:---:|:---:|:---|
| L2 | **Op** | 无状态调度器，兼容 CUDA-Graph 和 torch.compile |
| L1 | **Kernel** | TileLang 内核，针对特定硬件优化 |

开发流程：**Kernel (L1) → Op (L2)**，先实现内核，再封装 Op 接口。

## 代码规范要点

- **TIR API**: 使用 `T.Tensor(shape, dtype)` 和 `T.reinterpret(value, dtype)`，而非旧版 `T.Buffer` 和 `T.reinterpret(dtype, value)`
- **导入规则**: 包内用相对导入 (`from .op import Op`)，跨包用绝对导入 (`tileops.ops`)
- **Kernel 子包**: `tileops/kernels/*` 每个子包必须有 `__init__.py`，包含显式 `__all__` 和 `from .module import Symbol` 重新导出

## 基准测试规范

- `BenchmarkReport.record()` 第一个参数必须是 Op 对象，禁止用字符串
- 必须记录至少一个非 `"tileops"` 标签的基线（如 `"torch"`、`"FA3"`）
- `calculate_flops()` 和 `calculate_memory()` 必须返回非 None 值

## 关键参考文档

- [DEVELOPMENT.md](docs/DEVELOPMENT.md) — 架构、开发流程、编码标准、测试策略、PR 流程
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) — "2+1 Review" 策略、分支命名规范、commit 规范
- [.claude/rules/](.claude/rules/) — 代码风格、安全、基准测试规则
