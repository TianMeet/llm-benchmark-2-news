# 通用数据评测契约（Prompt 驱动）

目标：让任意业务数据都能走同一条评测链路  
`提示词定义任务 -> LLM 生成结构化预测 -> 与真实标签对比 -> 输出可汇总指标`

补充设计（统一答案抽象、思维过程评估、回归门禁）见：
`docs/design/unified_answer_contract.md`

## 1. 数据集约定（JSONL）

每行一个样本，推荐包含三类字段：

1. 元信息字段：`sample_id` 等唯一标识。
2. 模型输入字段：你希望喂给提示词的内容（如 `ticker`、`news_items`、`question`）。
3. 真实标签字段：用于评测的 ground truth（建议统一前缀 `gt_`）。

示例：

```json
{
  "sample_id": "S001",
  "ticker": "AAPL",
  "news_items": [{"title": "..."}, {"title": "..."}],
  "gt_sentiment": "positive",
  "gt_impact_score": 2,
  "gt_keywords": ["ai", "earnings"]
}
```

## 2. 任务契约（YAML）

在 `bench/tasks/<task>.yaml` 定义：

1. `prompt_template`：提示词模板名（对应 `bench/prompts/<name>.yaml`）。
2. `parse_schema`：LLM 输出 JSON 的结构与校验规则。
3. `metrics`：预测值与真实值的对比规则。
4. `default_params`（可选）：任务级模型参数（例如 `response_format`、`temperature`）。

备注：当输出字段是对象列表（例如 `claims=[{"text":"...","evidence":[...]}]`）时，`parse_schema` 建议用 `type: list_raw` 保留原始结构，便于后续指标计算。

### 指标类型（当前支持）

1. `field_completeness`：输出字段完整度。
2. `reference_rouge`：摘要类文本与参考答案相似度。
3. `custom_ratio`：通用比值指标。
4. `exact_match`：分类/标签准确率（0/1）。
5. `numeric_error`：数值误差 MAE，支持容差命中率。
6. `list_overlap`：列表重叠 precision/recall/f1。
7. `reasoning_consistency`：`final_answer` 与步骤/结论文本一致性。
8. `claim_evidence_overlap`：claim 引用证据与标注证据重叠（precision/recall/f1）。

示例：

```yaml
name: event_eval
version: v1
prompt_template: ie_json
default_params:
  response_format:
    type: json_object
serialize_fields:
  - news_items

parse_schema:
  - field: sentiment
    type: enum
    values: [positive, neutral, negative]
    default: neutral
  - field: impact_score
    type: int
    default: 0
    lo: -5
    hi: 5
  - field: keywords
    type: list
    default: []

metrics:
  - type: exact_match
    name: sentiment_acc
    pred_field: sentiment
    label_field: gt_sentiment
  - type: numeric_error
    name: impact
    pred_field: impact_score
    label_field: gt_impact_score
    tolerance: 1
  - type: list_overlap
    name: keyword
    pred_field: keywords
    label_field: gt_keywords
```

## 3. 运行方式

```bash
python -m bench.cli.runner \
  --task event_eval \
  --dataset datasets/your_data.jsonl \
  --models deepseek-v3 \
  --model-registry configs/llm_providers.json \
  --out runs/
```

输出：

1. `results.jsonl`：逐样本预测与解析结果。
2. `summary.csv`：聚合指标（含 `tm_*` 指标列）。
3. `report.md`：可读报告。
4. `run_meta.json` / `dataset_fingerprint.json` / `model_snapshot.json`：复现与审计辅助产物。

## 4. 推荐实践

1. 真实标签字段统一 `gt_` 前缀，减少任务迁移成本。
2. 每次改提示词都提升 `prompt` 版本号，方便回归。
3. 先用 `--mock --max-samples 3` 验证流程，再切真实模型。
4. 数据字段要与任务契约匹配：`ie_json` 依赖 `news_items` 数组；如果数据是 `news_title/news_body`，建议使用 `news_analysis` 任务或新增映射任务。
