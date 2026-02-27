# 量化新闻 LLM 测试中台（MVP）

本仓库已实现一个本地可运行的评测闭环：多模型、多任务、多步 workflow、结果落盘与报告生成。

## 目录

- `llm_core/`：已有 LLM 调用基础包（保持兼容）
- `eval/`：评测中台核心模块
  - `registry.py`：模型注册表 + 成本估算
  - `tasks/`：任务契约与示例任务（`ie_json`/`stock_score`/`news_dedup`）
  - `workflow.py`：轻量 workflow 解析
  - `runner.py`：CLI 执行入口
  - `metrics.py`：聚合统计
  - `report.py`：Markdown 报告生成
  - `cache.py`：本地缓存
- `datasets/demo_news.jsonl`：demo 数据集（10 条）
- `workflows/news_pipeline.yaml`：多步协作 workflow 示例
- `runs/`：每次评测输出目录

## 环境变量

所有 API Key 必须通过环境变量提供，不要写死。

当前 `configs/llm_providers.yaml` 使用：

- `CAIJJ_API_KEY`

示例：

```bash
export CAIJJ_API_KEY="your_api_key"
```

## 安装依赖

```bash
pip install -U openai pyyaml jinja2 pytest
```

## 运行方式

### 1) 单任务多模型对比

```bash
python -m eval.runner --task ie_json --dataset datasets/demo_news.jsonl --models deepseek-chat,gpt-4o-mini --out runs/
```

说明：`--models` 需填写你 `configs/llm_providers.yaml` 中实际存在的 `model_id`。

### 2) workflow 多模型协作评测

```bash
python -m eval.runner --workflow workflows/news_pipeline.yaml --dataset datasets/demo_news.jsonl --out runs/
```

### 3) 单独生成报告

```bash
python -m eval.report --run_dir runs/{run_id}
```

## 输出结构

每次执行生成 `runs/{run_id}/`：

- `config.json`：本次 run 配置（模型、参数、prompt 版本、时间、git hash）
- `results.jsonl`：逐样本逐模型（/逐 step）结果
- `summary.csv`：聚合指标
- `report.md`：可读报告（含对比表 + 失败 case）

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

## 测试

```bash
pytest -q
```

当前最小测试覆盖：schema 解析、聚合统计、cache、workflow 加载、报告生成。

## 本地离线 Smoke Run（无 API 调用）

当本机没有安装 `openai/pyyaml` 或暂未配置 API Key 时，可先跑离线闭环验证：

```bash
python -m eval.runner --task ie_json --dataset datasets/demo_news.jsonl --models deepseek-v3,doubao-seed --model-registry configs/llm_providers.json --out runs/ --mock --max-samples 3
python -m eval.runner --workflow workflows/news_pipeline.json --dataset datasets/demo_news.jsonl --model-registry configs/llm_providers.json --out runs/ --mock --max-samples 3
```
