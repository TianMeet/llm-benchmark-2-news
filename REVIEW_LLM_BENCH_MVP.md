# LLM 基准评测系统（MVP 启动期）Review

> 历史说明：本文是 2026-02-28 的评审快照，后续已有多项建议落地（如 `run_meta/dataset_fingerprint/model_snapshot`、结构化错误字段、`--resume-run-dir`、CI smoke）。阅读时请以当前代码与 README 为准。

## 1) 一页总结（10行以内）
1. 当前项目已经具备可跑闭环：`dataset -> task/workflow runner -> parse -> metrics -> aggregate -> report`，并已能离线 `--mock` 跑通。  
2. 我已实跑 smoke：task 与 workflow 两条链路都成功落盘（`runs/20260228_040343_560604`、`runs/20260228_040343_560603`）。  
3. 我已执行测试：`pytest -q`，结果 `67 passed`。   
4. 最大风险不是“跑不通”，而是“复现实验细节不够可追溯”（配置快照不完整、数据版本未指纹化）。  
5. 第二风险是失败样本仅 `error` 文本，缺少统一错误类型，后续规模化分析会困难。  
6. 第三风险是 CI 缺少 smoke eval 任务，回归只测单元，不测真实执行链路。  
7. 最优先改动 #1：冻结 run 协议并落盘 `run_meta.json + config_snapshot.json + dataset_fingerprint.json`。  
8. 最优先改动 #2：给结果行增加结构化错误字段（`error_type/error_stage/error_code`）。  
9. 最优先改动 #3：在 CI 增加离线 smoke（task+workflow），保证主链路不会悄悄坏。  
10. 实际执行命令：`python3 -m eval.cli.runner --task ie_json ... --mock --max-samples 3`、`python3 -m eval.cli.runner --workflow workflows/news_pipeline.json ... --mock --max-samples 3`、`pytest -q`。  

## 2) 地基清单建议（P0/P1/P2）

当前评测链路（文字流程图）：  
`datasets/*.jsonl` -> `eval.cli.runner` -> `run_task_mode/run_workflow_mode` -> `LLMGateway(call + cache)` -> `task.parse + task.metrics` -> `aggregate_records` -> `RunStore(config/results/summary)` -> `MarkdownReporter(report.md)`。

### P0-1 复现协议快照不完整（B/E）
问题现象/证据：`eval/cli/runner.py:51-66` 仅写了 `params` 和基础元信息；没有完整冻结每个 model 的最终参数、prompt 原文、workflow/task 文件内容指纹；`eval/execution/utils.py:28-41` 读数据但无数据哈希。  
为什么重要：同一 run 很难做到“未来可精确重放”，尤其是多人协作和跨机器回归。  
推荐做法：新增 `run_meta.json`，记录 `git_commit`、`dirty`、`dataset_sha256`、`workflow_sha256`、`task_yaml_sha256`、`prompt_sha256`、`resolved_model_params`、`python_version`。  
改动位置：`eval/cli/runner.py`、`eval/execution/utils.py`、`eval/io/store.py`。  
验证方法（命令+预期）：  
`python3 -m eval.cli.runner --task ie_json --dataset datasets/demo_news.jsonl --models deepseek-v3 --model-registry configs/llm_providers.json --mock --max-samples 2 --out runs/`；预期 `runs/<run_id>/run_meta.json` 存在且含 `dataset_sha256`、`resolved_model_params`。  

### P0-2 失败样本缺少统一错误分类（D）
问题现象/证据：`eval/execution/task_runner.py:58-130` 与 `eval/execution/workflow_runner.py:73-237` 只拼接 `error` 字符串（如 `gateway_error:`、`parse_error:`），缺少结构化字段。  
为什么重要：后续看板、告警、失败归因（超时/拒答/解析失败/格式不符）无法稳定聚合。  
推荐做法：结果行新增 `error_type`（timeout/refusal/parse_failure/schema_mismatch/upstream_missing/unknown）、`error_stage`（build_prompt/gateway/parse/metrics/upstream）、`error_detail`。  
改动位置：`eval/execution/task_runner.py`、`eval/execution/workflow_runner.py`、`eval/contracts/result.py`、`eval/reporting/report.py`。  
验证方法（命令+预期）：  
`pytest -q tests/test_runner.py`；预期失败 case 断言可检查 `error_type`，报告里按 `error_type` 聚合统计。  

### P0-3 CI 缺少 smoke eval 主链路回归（F）
问题现象/证据：`.github/workflows/ci.yml:32-35` 仅跑 pytest，未跑 `python -m eval.cli.runner`。  
为什么重要：单测通过不代表 CLI 主流程可执行；早期项目最怕入口断裂。  
推荐做法：CI 新增 `smoke-eval` job，固定 `--mock --max-samples 2`，覆盖 task 与 workflow 两条命令，并检查产物文件存在。  
改动位置：`.github/workflows/ci.yml`。  
验证方法（命令+预期）：  
本地先跑  
`python3 -m eval.cli.runner --task ie_json --dataset datasets/demo_news.jsonl --models deepseek-v3 --model-registry configs/llm_providers.json --mock --max-samples 2 --out /tmp/runs_smoke`  
`python3 -m eval.cli.runner --workflow workflows/news_pipeline.json --dataset datasets/demo_news.jsonl --model-registry configs/llm_providers.json --mock --max-samples 2 --out /tmp/runs_smoke`；预期两次都生成 `config.json/results.jsonl/summary.csv/report.md`。  

### P1-1 目录边界可再收敛，降低后续扩展摩擦（A）
问题现象/证据：现在是 `eval/execution`、`eval/io`、`eval/reporting`、`eval/tasks` + 根级 `datasets/`、`workflows/`，可用但“评测域对象”与“执行细节”仍有耦合。  
为什么重要：加新任务、新 scorer、新报告格式时容易跨层改动。  
推荐做法：保留现有实现，先做“逻辑分层目录”轻迁移（见第 3 节），不做大重写。  
改动位置：新增目录并逐步挪动，保持旧入口兼容。  
验证方法（命令+预期）：`pytest -q` + 两条 smoke 命令通过；导入路径不破坏。  

### P1-2 断点续跑尚未最小闭环（C）
问题现象/证据：有 cache（`eval/execution/gateway.py:85-127`），但没有 `--resume-run-id`，中断后无法按结果文件快速跳过已完成样本。  
为什么重要：真实评测常中断；只靠 cache 不够（parse/metrics/report 阶段也要续）。  
推荐做法：增加 `--resume-run-dir`，启动时读取既有 `results.jsonl`，构建 `(sample_id,task_name,step_id,model_id,repeat_idx)` 完成集合，执行前跳过。  
改动位置：`eval/cli/runner.py`、`eval/io/store.py`、`eval/execution/task_runner.py`、`eval/execution/workflow_runner.py`。  
验证方法（命令+预期）：先跑 `--max-samples 2`，再同 run 续跑全量；预期第二次只新增未完成样本，最终总条数与一次性全量一致。  

### P2-1 依赖锁定缺失（B/F）
问题现象/证据：仓库根目录无 `pyproject.toml/requirements*.txt/uv.lock`；依赖当前仅在 README 与 CI 里散落声明。  
为什么重要：跨机重现会因依赖版本漂移导致结果不稳定。  
推荐做法：最小落地 `requirements.txt`（运行）+ `requirements-dev.txt`（测试/lint），并在 README 固定 Python 版本区间。  
改动位置：仓库根目录 + README。  
验证方法（命令+预期）：`python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt && pytest -q` 通过。  

## 3) 建议的最小目录结构（MVP 可扩展版）

```text
llm-benchmark-2-news/
  bench/
    cli/                    # 命令入口与参数解析
    orchestration/          # task/workflow 调度与并发
    contracts/              # run/result/error 的 schema 与 dataclass
    tasks/                  # 任务注册与 task yaml
    prompts/                # prompt 模板
    scoring/                # task metrics + aggregate
    io/                     # run store / cache / dataset loader
    reporting/              # markdown/html/json 报告
  configs/
    llm_providers.json|yaml
    eval_contract/
  datasets/                 # 评测数据 + 版本说明
  workflows/                # workflow 定义
  runs/                     # 每次 run 产物
  tests/                    # 单测 + smoke 测试
  README.md
```

目录职责说明：  
`cli` 只负责入参与 run 生命周期；`orchestration` 只做执行；`contracts` 固定协议；`io` 管数据/缓存/落盘；`scoring` 只算分；`reporting` 只展示。这样 task/scorer/report 独立演进，互不拖拽。

## 4) 最小可用规范（MVP Spec）

### 4.1 一次评测 run 必须保存的文件
1. `config_snapshot.json`：CLI 原始参数 + 解析后的运行参数。  
2. `run_meta.json`：`run_id`、`created_at`、`git_commit`、`git_dirty`、`python_version`、`platform`、`scorer_version`、`schema_versions`。  
3. `dataset_fingerprint.json`：`dataset_path`、`row_count`、`sha256`、可选 `dataset_version`。  
4. `model_snapshot.json`：本次实际用到的 model 配置（脱敏后）与 `resolved_params`。  
5. `results.jsonl`：逐样本逐步骤原始输出（含 `output_text`、`parsed`、`error_*`、`from_cache`）。  
6. `scores.jsonl`（可选与 results 同文件，但建议分离）：逐样本分数字段。  
7. `summary.csv`：聚合指标。  
8. `report.md`：人类可读报告。  

### 4.2 失败样本统一 schema（建议）
```json
{
  "run_id": "20260228_040343_560603",
  "sample_id": "S001",
  "mode": "workflow_step",
  "task_name": "ie_json",
  "step_id": "step2_extract",
  "model_id": "kimi-personal",
  "success": false,
  "error_type": "parse_failure",
  "error_stage": "parse",
  "error_code": "json_decode_error",
  "error_detail": "Expecting value: line 1 column 1 (char 0)",
  "raw_output_excerpt": "NOT VALID JSON ...",
  "latency_ms": 123.4,
  "retry_count": 2,
  "timestamp": "2026-02-28T04:03:43.590675+00:00"
}
```

## 5) 关键补丁建议（最划算的 5 处）

### 补丁 1：统一 run 元数据与指纹落盘
文件路径：`eval/execution/utils.py`
```python
import hashlib
from pathlib import Path

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def dataset_fingerprint(path: str | Path) -> dict[str, object]:
    row_count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row_count += 1
    return {"dataset_path": str(path), "row_count": row_count, "sha256": sha256_file(path)}
```

### 补丁 2：RunStore 增加标准产物写接口
文件路径：`eval/io/store.py`
```python
def write_run_meta(self, payload: dict[str, Any]) -> None:
    (self.run_dir / "run_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def write_dataset_fingerprint(self, payload: dict[str, Any]) -> None:
    (self.run_dir / "dataset_fingerprint.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
```

### 补丁 3：结果行结构化错误分类
文件路径：`eval/execution/task_runner.py`、`eval/execution/workflow_runner.py`
```python
def classify_error(err: str) -> tuple[str, str]:
    s = (err or "").lower()
    if "timeout" in s:
        return "timeout", "gateway"
    if "refusal" in s or "safety" in s:
        return "refusal", "gateway"
    if "parse_error" in s:
        return "parse_failure", "parse"
    if "build_prompt_error" in s:
        return "prompt_build_failure", "build_prompt"
    if "upstream_" in s:
        return "upstream_missing", "upstream"
    return "unknown", "unknown"

# 组装 row 时新增字段
error_type, error_stage = classify_error(error_message)
row["error_type"] = error_type
row["error_stage"] = error_stage
row["error_detail"] = error_message or ""
```

### 补丁 4：添加最小断点续跑
文件路径：`eval/cli/runner.py`、`eval/io/store.py`
```python
# runner CLI
parser.add_argument("--resume-run-dir", default=None, help="Resume from an existing run directory")

# store helper
def load_completed_keys(self) -> set[tuple[str, str, str, str, int]]:
    keys = set()
    if not self.results_path.exists():
        return keys
    for line in self.results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((
            str(row.get("sample_id", "")),
            str(row.get("task_name", "")),
            str(row.get("step_id", "")),
            str(row.get("model_id", "")),
            int(row.get("repeat_idx", 0)),
        ))
    return keys
```

### 补丁 5：CI 增加离线 smoke
文件路径：`.github/workflows/ci.yml`
```yaml
  smoke-eval:
    name: smoke eval (mock)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install openai pyyaml jinja2
      - run: python -m eval.cli.runner --task ie_json --dataset datasets/demo_news.jsonl --models deepseek-v3 --model-registry configs/llm_providers.json --mock --max-samples 2 --out runs/
      - run: python -m eval.cli.runner --workflow workflows/news_pipeline.json --dataset datasets/demo_news.jsonl --model-registry configs/llm_providers.json --mock --max-samples 2 --out runs/
```

## 6) README 需要补的章节模板

### Quickstart（3 条命令内跑通 smoke eval）
```text
## Quickstart
pip install -r requirements.txt
python -m eval.cli.runner --task ie_json --dataset datasets/demo_news.jsonl --models deepseek-v3 --model-registry configs/llm_providers.json --mock --max-samples 3 --out runs/
python -m eval.cli.runner --workflow workflows/news_pipeline.json --dataset datasets/demo_news.jsonl --model-registry configs/llm_providers.json --mock --max-samples 3 --out runs/
```

### How to add a new task
```text
## How to add a new task
1. 在 `eval/tasks/` 新建 `<task_name>.yaml`（定义 parse_schema/metrics/mock_response）。
2. 在 `eval/prompts/` 新建同名 prompt 模板并设置 `version`。
3. 准备最小数据样本（建议 5 条）并执行：
   `python -m eval.cli.runner --task <task_name> --dataset <dataset>.jsonl --models <model_id> --mock --max-samples 5`
4. 为该任务补至少 1 个单测（parse/metrics）。
```

### Reproducibility（固定协议与版本）
```text
## Reproducibility
- 固定模型协议：`seed/temperature/top_p/max_tokens/stop/system_prompt/few_shot` 全量记录在 `model_snapshot.json`。
- 固定输入版本：`dataset_fingerprint.json` 记录 `sha256 + row_count`。
- 固定代码版本：`run_meta.json` 记录 `git_commit + git_dirty + python_version`。
- 固定输出协议：`results.jsonl` 与 `summary.csv` 需包含 schema_version。
```
