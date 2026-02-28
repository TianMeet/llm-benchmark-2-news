# 量化新闻 LLM 测试中台（MVP）

本仓库已实现一个本地可运行的评测闭环：多模型、多任务、多步 workflow、结果落盘与报告生成。

## 目录

- `docs/system_architecture.md`：当前实现版系统架构图与执行时序图
- `docs/eval_data_contract.md`：通用数据评测契约（提示词、标签、指标）
- `docs/design/unified_answer_contract.md`：统一答案抽象与回归评测设计
- `docs/design/migration_eval_to_bench.md`：`eval/` 到 `bench/` 的迁移完成记录

- `llm_core/`：已有 LLM 调用基础包（保持兼容）
- `bench/`：评测中台核心模块
  - `contracts/`：跨模块统一数据契约（结果行、错误分类、常量）
  - `execution/`：模型调用与 task/workflow 执行器
  - `io/`：运行产物存储、缓存、prompt store
  - `reporting/`：报告生成抽象与 Markdown 实现
  - `registry.py`：模型注册表 + 成本估算
  - `tasks/`：任务契约与示例任务（`ie_json`/`stock_score`/`news_dedup`）
  - `workflow.py`：轻量 workflow 解析
  - `cli/runner.py`：CLI 执行入口
  - `metrics/aggregate.py`：聚合统计
- `datasets/demo_news.jsonl`：demo 数据集（10 条）
- `workflows/news_pipeline.yaml`：多步协作 workflow 示例
- `runs/`：每次评测输出目录

## 环境变量

所有 API Key 必须通过环境变量提供，不要写死。

当前 `configs/llm_providers.yaml` 示例使用：

- `ANTHROPIC_AUTH_TOKEN`

示例：

```bash
export ANTHROPIC_AUTH_TOKEN="your_api_key"
```

也支持在仓库根目录使用 `.env`（推荐本地开发）：

```bash
cp .env.example .env
# 然后编辑 .env 填入真实 key
```

程序会自动读取 `.env`；`.env` 已在 `.gitignore` 中默认忽略，不会被提交。

说明：请以你本地 `configs/llm_providers.yaml` 中 `api_key: "${...}"` 的变量名为准。

## 安装依赖

```bash
pip install -r requirements-dev.txt
```

## 运行方式

### 1) 单任务多模型对比

```bash
python -m bench.cli.runner --task ie_json --dataset datasets/demo_news.jsonl --models deepseek-chat,gpt-4o-mini --out runs/
```

说明：`--models` 需填写你 `configs/llm_providers.yaml` 中实际存在的 `model_id`。

### 2) workflow 多模型协作评测

```bash
python -m bench.cli.runner --workflow workflows/news_pipeline.yaml --dataset datasets/demo_news.jsonl --out runs/
```

可选并发（样本级并发，样本内 step 仍按顺序）：

```bash
python -m bench.cli.runner --workflow workflows/news_pipeline.yaml --dataset datasets/demo_news.jsonl --out runs/ --workflow-concurrency 4
```

### 3) 单独生成报告

```bash
python -m eval.reporting.report --run_dir runs/{run_id}
```

### 4) 通用数据（带真实标签）评测

核心做法：在数据集中放 `gt_*` 标签字段，在任务 YAML 的 `metrics` 中配置预测字段与标签字段的对比规则。  
完整契约与模板见 `docs/eval_data_contract.md`。

## 输出结构

每次执行生成 `runs/{run_id}/`：

- `config.json`：本次 run 配置（模型、参数、prompt 版本、时间、git hash）
- `run_meta.json`：运行时元数据（git dirty、python/platform、schema 版本）
- `dataset_fingerprint.json`：数据集路径、行数、SHA256 指纹
- `model_snapshot.json`：本次实际使用模型参数快照（脱敏）
- `results.jsonl`：逐样本逐模型（/逐 step）结果
- `summary.csv`：聚合指标
- `report.md`：可读报告（含对比表 + 失败 case）

版本化字段：

- `config.json`：`scorer_version`、`result_schema_version`、`summary_schema_version`
- `results.jsonl`：`schema_version=result_row.v1`
- `summary.csv`：`schema_version=summary_row.v1`

## 支持指标

- `request_count` / `success_count` / `success_rate`
- `avg_latency_ms` / `p95_latency_ms`
- `avg_total_tokens` / `p95_total_tokens`
- `parse_rate`
- `retry_count`
- `cost_estimate`（存在 `price_config` 时）

## 复现与缓存

- 固定输入数据 + 固定参数 + 输出落盘
- 缓存键：`hash(model_id + params + messages + sample_id)`
- 默认缓存目录：`.eval_cache/`
- 可通过 `--no-cache` 禁用
- 支持最小断点续跑：`--resume-run-dir runs/{run_id}`

## 测试

```bash
pytest -q
```

当前最小测试覆盖：schema 解析、聚合统计、cache、workflow 加载、报告生成。

当前实现补充：

- task 模式支持按模型并发（`--concurrency`）。
- workflow 模式支持样本级并发（`--workflow-concurrency`），并使用流式回收避免全量 `gather` 带来的 OOM 风险。
- `LLMGateway` 对相同 `model_id + params_override` 复用 client，减少重复建连开销。
- `BaseLLMClient` 重试退避已加入 jitter，缓解限流场景下的惊群效应。
- 任务 YAML 支持 `default_params`（例如 `response_format: {type: json_object}`）并会透传到模型调用。

## 本地离线 Smoke Run（无 API 调用）

当本机没有安装 `openai/pyyaml` 或暂未配置 API Key 时，可先跑离线闭环验证：

```bash
python -m bench.cli.runner --task ie_json --dataset datasets/demo_news.jsonl --models deepseek-v3,doubao-seed --model-registry configs/llm_providers.json --out runs/ --mock --max-samples 3
python -m bench.cli.runner --workflow workflows/news_pipeline.json --dataset datasets/demo_news.jsonl --model-registry configs/llm_providers.json --out runs/ --mock --max-samples 3
```
