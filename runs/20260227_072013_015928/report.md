# LLM Eval Report

Run Dir: `runs/20260227_072013_015928`

## Model x Metric Summary

| task_name | step_id | model_id | request_count | success_rate | parse_rate | avg_latency_ms | p95_latency_ms | avg_total_tokens | cost_estimate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ie_json | step2_extract | doubao-seed | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 30.0 | 0.0 |
| news_dedup | step1_dedup | deepseek-v3 | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 30.0 | 0.0 |
| news_pipeline | __e2e__ | deepseek-v3+doubao-seed+qwen3-235b | 3 | 1.0 | 1.0 | 3.0 | 3.0 | 90.0 | 0.0 |
| stock_score | step3_score | qwen3-235b | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 30.0 | 0.0 |

## Top Error Cases (max 3)

No error cases found.