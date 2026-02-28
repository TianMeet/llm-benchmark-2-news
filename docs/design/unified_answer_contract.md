# 统一答案抽象与回归评测设计（v1）

目标：把不同答案形态（选项、数值、列表、文本、思维过程）抽象为统一结构，保证跨任务可回归、可解释、可扩展。

## 1. 设计原则

1. 先评估 `final_answer`，再评估 `reasoning`。
2. 统一“答案对象”与“评分对象”，避免每个任务单独造轮子。
3. 所有比较都先做标准化（大小写、空白、单位、同义词）。
4. 指标版本化（`schema_version` / `scorer_version`），确保历史结果可复现。

## 2. 统一答案对象（CanonicalAnswer）

```json
{
  "task_id": "event_eval",
  "sample_id": "S001",
  "answer_type": "choice",
  "prediction": {},
  "ground_truth": {},
  "meta": {
    "schema_version": "v1",
    "prompt_version": "ie_json.v1",
    "scorer_version": "v1"
  }
}
```

字段说明：

1. `answer_type`：`choice | number | list | text | reasoning`
2. `prediction`：模型输出的标准化结构
3. `ground_truth`：标注答案的标准化结构
4. `meta`：版本与追踪信息

## 3. 类型化答案子结构

1. `choice`
```json
{"value":"B","aliases":["option_b","b"]}
```

2. `number`
```json
{"value":12.5,"unit":"%","tolerance_abs":0.5,"tolerance_rel":0.02}
```

3. `list`
```json
{"items":["ai","earnings"],"ordered":false,"unique":true}
```

4. `text`
```json
{"value":"短摘要文本","normalize":"basic"}
```

5. `reasoning`
```json
{
  "final_answer":{"value":"positive"},
  "steps":[{"id":"s1","text":"..."}],
  "claims":[{"text":"营收增长","evidence":["news_1"]}]
}
```

`reasoning` 的主评分对象仍是 `final_answer`，`steps/claims` 只做辅助解释分。

## 4. 统一评分输出（ScoreCard）

```json
{
  "primary_score": 0.0,
  "pass": false,
  "sub_scores": {},
  "error_tags": [],
  "explain": ""
}
```

1. `primary_score`：回归主KPI（0~1）
2. `sub_scores`：分项指标（如 `precision`、`mae`、`within_tol`）
3. `error_tags`：失败原因标签（如 `wrong_unit`、`missing_item`）

## 5. 各类型评分策略

1. `choice`：`exact_match`（支持同义映射）
2. `number`：`mae` + `within_tolerance`
3. `list`：`precision/recall/f1`（可选顺序分）
4. `text`：关键词覆盖 + 可选 rouge/语义分
5. `reasoning`：
   1. 主分：`final_answer` 的准确性
   2. 辅分：`claims` 覆盖率、证据引用率、步骤完整度

## 6. 与当前仓库的映射

当前 `GenericTask` 已支持：

1. `exact_match`（适配 `choice`）
2. `numeric_error`（适配 `number`）
3. `list_overlap`（适配 `list`）
4. `reference_rouge`（部分适配 `text`）
5. `reasoning_consistency`（`final_answer` 与推理文本一致性）
6. `claim_evidence_overlap`（claim 证据重叠）

建议新增（下一阶段）：

1. `reasoning_step_order`：步骤顺序与逻辑链完整度。
2. `claim_verifiability`：claim 可验证率与证据充分性。

## 7. 回归机制设计

1. 基线：固定 `run_id` 作为 baseline。
2. 对比粒度：`task + model + prompt_version + scorer_version`。
3. 门禁：
   1. `primary_score` 不低于基线阈值。
   2. 关键 `sub_scores`（如 `within_tol`、`f1`）不退化超过容忍区间。
4. 失败输出：
   1. TopN 失败样本
   2. `error_tags` 统计
   3. 指标漂移报告

## 8. 数据与版本规范

1. 样本标签字段统一前缀：`gt_*`
2. 预测字段统一前缀：`pred_*`（可选，或由 `parsed` 提供）
3. 契约版本：
   1. `schema_version`：答案结构版本
   2. `scorer_version`：评分逻辑版本
   3. `prompt_version`：提示词版本

## 9. 最小落地路径

1. 数据集加入 `gt_*` 标签字段。
2. 在任务 YAML 配置 `exact_match/numeric_error/list_overlap`。
3. 在 run 产物里记录 `scorer_version`（当前实现已写入 `config.json`）。
4. 用固定 baseline run 做回归比较。
