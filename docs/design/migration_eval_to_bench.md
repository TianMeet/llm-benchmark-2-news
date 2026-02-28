# eval -> bench 迁移完成记录（MVP）

目标：避免 `eval/` 与 `bench/` 长期并存导致的二义性，收敛到单一命名空间。

## 当前状态

1. 评测实现已全部迁移至 `bench/`。
2. 运行入口为 `python -m bench.cli.runner`。
3. `eval/` 目录已下线，不再兼容旧引用。

## 执行摘要

1. 完整代码树迁移：`eval/* -> bench/*`。
2. 全仓引用切换：`eval.*` 导入与命令入口替换为 `bench.*`。
3. CI、README、系统架构文档已同步。

## 迁移验收标准

1. CI 全绿（单测 + smoke eval）。
2. 同一数据/模型配置下，迁移前后 `summary.csv` 关键指标差异在容忍阈值内。
3. `results.jsonl` schema 兼容，`run_meta.json` 可追溯字段不减少。
