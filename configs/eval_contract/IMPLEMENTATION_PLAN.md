# Eval Contract 落地改造计划（v2 — 修订版）

> 修订说明：v1 方案存在过度设计（不必要的 pydantic 依赖、冗余路由层、
> 与现有 `bench/contracts/` 包结构冲突、全局可变单例反模式等），此版本
> 从实际代码出发，以**最小侵入成本**完成设计文档与代码的对齐。

---

## 一、问题概述

`configs/eval_contract/` 目录下的设计文档与 `bench/` 实际代码完全脱节。
设计文档定义了"统一答案抽象 + 配置驱动评分"体系，但代码层面仍硬编码扁平化实现。

---

## 二、现状分析

### 2.1 设计文档（当前未被使用）

| 文件 | 定义内容 |
|------|---------|
| `canonical_answer.schema.json` | 统一答案结构：`task_id / sample_id / answer_type / prediction / ground_truth / meta` |
| `scoring_profile.template.yaml` | 按 `answer_type` 映射主指标 + 子指标配置 |
| `docs/design/unified_answer_contract.md` | `CanonicalAnswer` / `ScoreCard` / 回归门禁设计蓝图 |

### 2.2 实际代码（当前工作方式）

```
GenericTask.__init__()
  └─ self._metrics_cfg = cfg["metrics"]   # 从任务 YAML 直读 list[dict]

GenericTask.metrics()
  └─ for rule in self._metrics_cfg:
       compute_metric(mtype, rule, sample, parsed)
         └─ _CALC_REGISTRY[mtype](rule, sample, parsed)
```

- `_CALC_REGISTRY` 已包含 8 种计算器，功能完备
- 各计算器接口统一：`(rule, sample, parsed) -> dict[str, Any]`
- 但**没有** `answer_type`、`ScoringProfile`、`ScoreCard` 概念参与

### 2.3 Gap 对比

| 设计意图 | 实际实现 | 根因 |
|---------|---------|------|
| `answer_type` 决定评分策略 | `metrics[].type` 直接指定计算器 | 缺少"预设配置"层 |
| `scoring_profile.yaml` 配置驱动 | 任务 YAML 内联 metrics 列表 | profile 从未被加载 |
| `CanonicalAnswer` 统一结构 | 裸 dict 直传 | 无类型约束 |
| `ScoreCard` 统一评分输出 | 计算器返回散装 dict | 无主分/通过标记 |

---

## 三、改造原则

1. **零新依赖**：使用 `dataclasses` + `TypedDict`（项目已有风格），不引入 pydantic
2. **最小侵入**：不新建 `bench/eval_contract/` 包；模型放入已有 `bench/contracts/`，加载/路由放入 `bench/task_metrics/`
3. **渐进迁移**：`answer_type` 是可选字段——配了走 profile 预设，没配走原始 metrics 列表
4. **配置合并**：`scoring_profile.yaml` 提供默认指标集→ 任务 YAML `metrics` 可覆盖/追加
5. **可测可回退**：所有新逻辑由纯函数实现，不依赖全局状态

---

## 四、改造方案

### Phase 1：数据模型（扩展 `bench/contracts/`）

**`bench/contracts/scoring.py`** — 新增文件

```python
"""统一评分契约模型。

与 canonical_answer.schema.json / scoring_profile.template.yaml 对齐，
使用 dataclasses 保持项目风格一致（无额外依赖）。
"""
from __future__ import annotations

__all__ = [
    "AnswerType",
    "CanonicalAnswer",
    "ScoreCard",
    "MetricRule",
    "ScorerPreset",
    "ScoringProfile",
]

from dataclasses import dataclass, field
from typing import Any, Literal

AnswerType = Literal["choice", "number", "list", "text", "reasoning"]


@dataclass(frozen=True, slots=True)
class CanonicalAnswer:
    """统一答案对象 — 仅在需要跨任务回归或 ScoreCard 输出时构建。"""
    task_id: str
    sample_id: str
    answer_type: AnswerType
    prediction: dict[str, Any]
    ground_truth: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreCard:
    """统一评分输出 — 与 docs/design/unified_answer_contract.md §4 对齐。"""
    primary_metric: str            # 主指标键名
    primary_score: float           # 主指标值 (0~1)
    passed: bool                   # 是否通过门禁
    sub_scores: dict[str, Any]     # 全部指标明细
    error_tags: list[str] = field(default_factory=list)
    explain: str = ""


@dataclass(frozen=True, slots=True)
class MetricRule:
    """单条指标规则 — scoring_profile 中 metrics 列表的每一项。"""
    type: str
    name: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_rule_dict(self) -> dict[str, Any]:
        """转为 _CALC_REGISTRY 计算器所需的 rule dict。"""
        d: dict[str, Any] = {"type": self.type}
        if self.name:
            d["name"] = self.name
        d.update(self.extras)
        return d


@dataclass(frozen=True, slots=True)
class ScorerPreset:
    """某个 answer_type 对应的评分预设。"""
    primary_metric: str
    pass_threshold: float          # 主分 >= 此值视为通过
    metrics: tuple[MetricRule, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScorerPreset:
        threshold = float(data.get("pass_threshold", 0.5))
        metrics = []
        for m in data.get("metrics", []):
            mtype = m.pop("type")
            mname = m.pop("name", "")
            metrics.append(MetricRule(type=mtype, name=mname, extras=m))
        return cls(
            primary_metric=data["primary_metric"],
            pass_threshold=threshold,
            metrics=tuple(metrics),
        )


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """标准化配置。"""
    lowercase: bool = True
    trim_whitespace: bool = True
    collapse_spaces: bool = True
    unit_aliases: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoringProfile:
    """评分配置文件的完整模型。"""
    version: str
    normalization: NormalizationConfig
    scorers: dict[str, ScorerPreset]   # key = answer_type

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ScoringProfile:
        norm_raw = raw.get("normalization", {})
        norm = NormalizationConfig(
            lowercase=norm_raw.get("lowercase", True),
            trim_whitespace=norm_raw.get("trim_whitespace", True),
            collapse_spaces=norm_raw.get("collapse_spaces", True),
            unit_aliases=norm_raw.get("unit_aliases", {}),
        )
        scorers = {}
        for atype, scorer_data in raw.get("scorers", {}).items():
            # deep copy to avoid mutating raw
            import copy
            scorers[atype] = ScorerPreset.from_dict(copy.deepcopy(scorer_data))
        return cls(
            version=raw.get("version", "v1"),
            normalization=norm,
            scorers=scorers,
        )

    def get_preset(self, answer_type: str) -> ScorerPreset | None:
        return self.scorers.get(answer_type)
```

**改进要点 vs v1：**
- `dataclasses` 替代 `pydantic`：零依赖增加
- `frozen=True` + `slots=True`：不可变、内存高效
- `ScorerPreset` 包含 `pass_threshold`：支持可配置门禁
- `NormalizationConfig`：设计文档中定义的标准化配置得到建模
- `MetricRule.to_rule_dict()`：直接桥接到现有 `_CALC_REGISTRY` 计算器签名
- `ScoringProfile.from_dict()`：显式工厂方法，不依赖 pydantic 魔法

### Phase 2：配置加载（扩展 `bench/task_metrics/`）

**`bench/task_metrics/profile.py`** — 新增文件

```python
"""ScoringProfile 加载与指标解析。

职责：
1. 从 YAML 加载 ScoringProfile
2. 根据 answer_type 解析出 metrics 规则列表（profile 预设 + 任务级覆盖）
3. 验证所有 metric type 在 _CALC_REGISTRY 中有注册
"""
from __future__ import annotations

import logging
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any

from bench.contracts.scoring import ScoringProfile, ScorerPreset

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "configs" / "eval_contract" / "scoring_profile.template.yaml"
)


@lru_cache(maxsize=4)
def load_profile(path: str | None = None) -> ScoringProfile:
    """加载并缓存 ScoringProfile。

    使用 lru_cache 替代全局可变单例 —— 线程安全、无副作用、可用
    load_profile.cache_clear() 在测试中重置。
    """
    resolved = Path(path) if path else _DEFAULT_PROFILE_PATH
    with resolved.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    profile = ScoringProfile.from_dict(raw)
    _validate_profile(profile)
    return profile


def _validate_profile(profile: ScoringProfile) -> None:
    """启动时校验 profile 中的 metric type 是否都已注册。"""
    from bench.task_metrics.calculators import _CALC_REGISTRY

    for atype, preset in profile.scorers.items():
        for m in preset.metrics:
            if m.type not in _CALC_REGISTRY:
                logger.warning(
                    "ScoringProfile: answer_type '%s' 引用了未注册的指标类型 '%s'",
                    atype, m.type,
                )


def resolve_metrics(
    answer_type: str | None,
    task_metrics_cfg: list[dict[str, Any]],
    profile_path: str | None = None,
) -> list[dict[str, Any]]:
    """解析最终生效的 metrics 列表。

    策略（三层合并）：
    1. answer_type 为 None → 直接返回 task_metrics_cfg（向后兼容）
    2. answer_type 有值、task_metrics_cfg 非空 → task 级配置覆盖 profile 预设
    3. answer_type 有值、task_metrics_cfg 为空 → 使用 profile 预设
    """
    if answer_type is None:
        return task_metrics_cfg

    profile = load_profile(profile_path)
    preset = profile.get_preset(answer_type)
    if preset is None:
        logger.warning("answer_type '%s' 在 scoring_profile 中无预设，回退到任务级配置", answer_type)
        return task_metrics_cfg

    # 任务级显式配置优先
    if task_metrics_cfg:
        return task_metrics_cfg

    # 使用 profile 预设
    return [m.to_rule_dict() for m in preset.metrics]
```

**改进要点 vs v1：**
- `lru_cache` 替代全局 `_default_profile` 可变单例：线程安全、测试友好
- `_validate_profile()`：**加载时**即验证配置与注册表一致性，早发现早报错
- `resolve_metrics()` 三层合并策略：简洁、可预测
- 不新建 package：放入已有 `bench/task_metrics/` 目录

### Phase 3：ScoreCard 组装（扩展 `bench/task_metrics/`）

**`bench/task_metrics/scoring.py`** — 新增文件

```python
"""ScoreCard 组装逻辑 — 将散装指标 dict 包装为统一 ScoreCard。

仅在需要 ScoreCard 输出（如回归门禁比较）时调用。
GenericTask.metrics() 的默认返回仍为 dict（保持向后兼容）。
"""
from __future__ import annotations

from typing import Any

from bench.contracts.scoring import (
    AnswerType,
    CanonicalAnswer,
    ScoreCard,
    ScorerPreset,
)
from bench.task_metrics.profile import load_profile


def build_score_card(
    raw_scores: dict[str, Any],
    answer_type: str,
    profile_path: str | None = None,
) -> ScoreCard:
    """将计算器返回的散装 dict 包装为 ScoreCard。"""
    profile = load_profile(profile_path)
    preset = profile.get_preset(answer_type)

    if preset is None:
        # 无预设 → 取第一个指标作为 primary
        first_key = next(iter(raw_scores), "")
        return ScoreCard(
            primary_metric=first_key,
            primary_score=float(raw_scores.get(first_key, 0.0) or 0.0),
            passed=False,
            sub_scores=raw_scores,
        )

    primary_score = _find_primary_score(raw_scores, preset)
    return ScoreCard(
        primary_metric=preset.primary_metric,
        primary_score=primary_score,
        passed=primary_score >= preset.pass_threshold,
        sub_scores=raw_scores,
    )


def _find_primary_score(
    raw_scores: dict[str, Any],
    preset: ScorerPreset,
) -> float:
    """在散装指标 dict 中定位主指标值。

    支持两种查找方式：
    1. 精确匹配 primary_metric 键名
    2. 模糊匹配 — 键名以 primary_metric 结尾（适配 {name}_{metric} 命名模式）
    3. 名称反查 — 查找 preset 中 type == primary_metric 的 metric，用其 name 做查找
    """
    key = preset.primary_metric
    if key in raw_scores:
        return float(raw_scores[key] or 0.0)

    # 模糊匹配：如 primary_metric="within_tol" → 匹配 "value_within_tol"
    for k, v in raw_scores.items():
        if k.endswith(f"_{key}") or k.endswith(key):
            return float(v or 0.0)

    # 名称反查：primary_metric="exact_match"(type) → metric.name="choice_acc"(output key)
    for m in preset.metrics:
        if m.type == key and m.name and m.name in raw_scores:
            return float(raw_scores[m.name] or 0.0)

    return 0.0
```

**改进要点 vs v1：**
- ScoreCard 组装与路由解耦：不再把"指标计算"和"包装为 ScoreCard"混在一起
- `_find_primary_score()` 模糊匹配：解决 `primary_metric: value_within_tol` 与实际输出键 `{name}_within_tol` 的命名错位问题
- ScoreCard 是可选产出（不强制每次 metrics 调用都包装）

### Phase 4：最小改造 GenericTask（修改现有文件）

**`bench/tasks/generic.py`** — 仅改动 `__init__` 和 `metrics`

```python
class GenericTask(EvalTask):
    def __init__(self, task_config_path: str | Path) -> None:
        # ... 现有代码不变 ...

        # ── 新增：answer_type 支持（可选）──
        self._answer_type: str | None = cfg.get("answer_type")

        # 解析最终生效的 metrics 列表：
        # - 有 answer_type 且无显式 metrics → 从 scoring_profile 取预设
        # - 有 answer_type 且有显式 metrics → 显式配置优先
        # - 无 answer_type → 直接使用 YAML 中的 metrics 列表（原行为）
        from bench.task_metrics.profile import resolve_metrics
        self._metrics_cfg = resolve_metrics(
            answer_type=self._answer_type,
            task_metrics_cfg=self._metrics_cfg,
        )

    # metrics() 方法无需分叉！统一走原来的 for loop
    def metrics(
        self,
        sample: dict[str, Any],
        output: str,
        parsed: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """按 metrics 配置计算指标 — 统一路径，无新旧分叉。"""
        sample = self._resolve_sample(sample)
        result: dict[str, Any] = {}
        for rule in self._metrics_cfg:
            mtype = rule.get("type", "")
            result.update(compute_metric(mtype, rule, sample, parsed, task_name=self.name))
        return result
```

**关键改进 vs v1：**
- **`metrics()` 方法零改动！** 所有"路由"在 `__init__` 阶段通过 `resolve_metrics()` 完成
- 无 `if self._use_contract:` 分叉 → 只有一条执行路径 → 测试/维护成本大幅降低
- `CanonicalAnswer` 不在热路径构造 → 零额外运行时开销

### Phase 5：回归门禁支持（可选增强）

**`bench/task_metrics/regression.py`** — 新增文件（Phase 2 后续迭代）

```python
"""回归门禁：对比当前 run 与 baseline run 的 ScoreCard。"""
from __future__ import annotations

from dataclasses import dataclass
from bench.contracts.scoring import ScoreCard


@dataclass
class RegressionResult:
    baseline_score: float
    current_score: float
    delta: float
    regressed: bool
    details: str


def check_regression(
    current: ScoreCard,
    baseline: ScoreCard,
    tolerance: float = 0.02,
) -> RegressionResult:
    """判断主指标是否退化超出容忍范围。"""
    delta = current.primary_score - baseline.primary_score
    regressed = delta < -tolerance
    return RegressionResult(
        baseline_score=baseline.primary_score,
        current_score=current.primary_score,
        delta=round(delta, 6),
        regressed=regressed,
        details=(
            f"REGRESSION: {current.primary_metric} dropped by {abs(delta):.4f} "
            f"(tolerance={tolerance})"
            if regressed else "OK"
        ),
    )
```

---

## 五、任务 YAML 格式

### 旧格式（完全兼容，无需任何改动）：

```yaml
name: ie_json
version: v1
prompt_template: ie_json_v1
parse_schema:
  - field: sentiment
    type: enum
    values: [positive, negative, neutral]
metrics:
  - type: exact_match
    pred_field: sentiment
    label_field: gt_sentiment
```

### 新格式（推荐，metrics 自动从 profile 加载）：

```yaml
name: ie_json
version: v1
prompt_template: ie_json_v1
answer_type: choice          # 声明答案类型 → 从 scoring_profile 取预设
parse_schema:
  - field: sentiment
    type: enum
    values: [positive, negative, neutral]
# metrics 不写 → 自动使用 profile 中 choice 类型的预设
```

### 混合格式（显式覆盖 profile 预设）：

```yaml
name: ie_json_custom
version: v1
prompt_template: ie_json_v1
answer_type: choice          # 声明类型（用于 ScoreCard 组装时确定 primary_metric）
parse_schema:
  - field: sentiment
    type: enum
    values: [positive, negative, neutral]
metrics:                     # 显式写 → 覆盖 profile 预设
  - type: exact_match
    name: sentiment_acc
    pred_field: sentiment
    label_field: gt_sentiment
  - type: field_completeness
    fields: [sentiment]
```

---

## 六、文件变更清单

### 新增文件（3 个）

| 文件 | 职责 | 行数估计 |
|------|------|---------|
| `bench/contracts/scoring.py` | dataclass 模型定义 | ~120 |
| `bench/task_metrics/profile.py` | profile 加载 + metrics 解析 | ~80 |
| `bench/task_metrics/scoring.py` | ScoreCard 组装 | ~60 |

### 修改文件（2 个）

| 文件 | 改动 | 范围 |
|------|------|------|
| `bench/tasks/generic.py` | `__init__` 新增 3 行（`answer_type` + `resolve_metrics` 调用） | 极小 |
| `bench/contracts/__init__.py` | 补充 `scoring` 模块的懒加载声明 | 极小 |

### 不改动

| 文件 | 原因 |
|------|------|
| `bench/task_metrics/calculators.py` | 计算器接口不变 |
| `bench/tasks/base.py` | 抽象接口不变 |
| `requirements.txt` | 无新依赖 |

### 可选后续新增

| 文件 | 触发条件 |
|------|---------|
| `bench/task_metrics/regression.py` | 需要回归门禁功能时 |

---

## 七、配置文件改进

### `scoring_profile.template.yaml` 改进：

1. **每个 scorer 新增 `pass_threshold`** — 门禁阈值可配置
2. **`number` 的 `primary_metric` 改为 `within_tol`** — 匹配实际计算器输出键后缀
3. **`list` 的 `primary_metric` 改为 `f1`** — 同理

### `canonical_answer.schema.json` 改进：

1. **`meta` 中 `scorer_version` 改为可选** — 实际构造时未必已知
2. **新增 `meta.pass_threshold`** — 记录门禁阈值，便于回归比较

---

## 八、验收标准

- [ ] `ScoringProfile.from_dict()` 能从 YAML 正确加载并校验所有 answer_type
- [ ] `_validate_profile()` 能在 metric type 未注册时输出 WARNING
- [ ] `resolve_metrics(answer_type="choice", task_metrics_cfg=[])` 返回 profile 预设
- [ ] `resolve_metrics(answer_type="choice", task_metrics_cfg=[...])` 返回任务级配置
- [ ] `resolve_metrics(answer_type=None, ...)` 返回原始 task_metrics_cfg
- [ ] 现有所有任务 YAML 运行结果不变（向后兼容零回归）
- [ ] `build_score_card()` 能正确识别 primary_score（含模糊匹配）
- [ ] 单元测试覆盖：
  - `bench/contracts/scoring.py` 所有模型的创建 / 序列化
  - `bench/task_metrics/profile.py` 加载 / 校验 / 三层合并
  - `bench/task_metrics/scoring.py` ScoreCard 组装 + 模糊匹配
  - `bench/tasks/generic.py` 新旧 YAML 格式均可运行

---

## 九、与 v1 方案对比

| 维度 | v1 方案 | v2 方案（本版） |
|------|--------|----------------|
| 新依赖 | pydantic（~15MB） | 无 |
| 新包 | `bench/eval_contract/`（4 文件） | `bench/contracts/scoring.py` + `bench/task_metrics/` 2 文件 |
| GenericTask 改动 | `metrics()` 双路径分叉 | `__init__` +3 行，`metrics()` 零改动 |
| 运行时开销 | 每次 metrics 调用创建 CanonicalAnswer | 仅 `__init__` 阶段一次 resolve |
| 全局状态 | `_default_profile` 可变单例 | `lru_cache` 纯函数缓存 |
| ScoreCard | 强制每次产出 | 按需调用 `build_score_card()`（可选） |
| normalization 配置 | 未使用 | 建模为 NormalizationConfig |
| profile 校验 | 无 | 加载时自动校验 metric type 注册 |
| pass_threshold | 硬编码 0.5 | 可配置，per answer_type |
| primary_metric 匹配 | 精确匹配（经常匹配失败） | 精确 + 后缀模糊匹配 |
| 向后兼容 | 需 `if self._use_contract` 分叉 | `resolve_metrics()` 透明合并，调用方无感知 |

---

## 十、风险提示

1. **`resolve_metrics()` 在 `__init__` 中加载 YAML**：首次加载有 I/O 开销，`lru_cache` 确保后续调用零开销
2. **模糊匹配 `_find_primary_score()`**：若指标命名冲突可能选错键，需 WARNING 日志
3. **向后兼容**：旧格式保持原行为，`answer_type` 是纯增量字段

---

## 十一、实施顺序

```
Phase 1 → 2 → 3 → 4 → 测试 → 可选 Phase 5（回归门禁）
  │         │       │       │
  │         │       │       └─ GenericTask 极小改动（3 行）
  │         │       └─ ScoreCard 组装（60 行纯函数）
  │         └─ Profile 加载 + 合并（80 行纯函数）
  └─ 数据模型（120 行 dataclass）
```

预计总改动量：~260 行新增代码 + 3 行修改，可在一个 PR 内完成。

---

## 十二、参考文档

- 统一答案契约设计：`docs/design/unified_answer_contract.md`
- 评分配置模板：`configs/eval_contract/scoring_profile.template.yaml`
- 答案 schema：`configs/eval_contract/canonical_answer.schema.json`
- 现有指标实现：`bench/task_metrics/calculators.py`
- 现有契约模型：`bench/contracts/result.py`（dataclass 风格参考）
