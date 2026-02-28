# 鲁棒 JSON 解析增强设计（v1）

## 目标
- 提高模型输出 JSON 的解析成功率。
- 对失败样本给出可操作的失败分类（空输出、截断、非法格式）。
- 保持现有 `parse_llm_json(text) -> dict | None` 兼容。

## 方案
在 `llm_core/response_parser.py` 中新增多策略解析管线：

1. 全文解析：
- 先做标准 `json.loads`
- 再做容错修复后重试（全角符号、注释、尾逗号）
- 再用 `ast.literal_eval` 兼容单引号 Python 字面量

2. 代码块提取：
- 提取首个 markdown fenced block 内容后重复步骤 1

3. 平衡提取：
- 使用括号平衡状态机提取首个完整 JSON（对象/数组）
- 提取结果重复步骤 1

新增元信息接口：
- `parse_llm_json_with_meta(text) -> JsonParseMeta`
  - `data: dict | None`
  - `reason: ok | empty | truncated | invalid_json | non_string`
  - `strategy: full_text | fenced_block | balanced_extract | none`

兼容接口：
- `parse_llm_json(text)` 保持不变，内部复用 `parse_llm_json_with_meta`。

## 失败分类语义
- `empty`：输出为空白或空字符串
- `truncated`：检测到 JSON 起始但未闭合（典型半截输出）
- `invalid_json`：格式不合法且无法通过修复策略恢复
- `non_string`：调用方传入非字符串

## 已覆盖场景
- Markdown 代码块包裹 JSON
- 尾逗号（`{"a":1,}` / `[1,2,]`）
- 注释（`//` / `/* */`）
- 全角符号与智能引号
- 单引号 Python 字面量对象

## 明确边界
- 空输出与严重截断无法在解析器层恢复为正确业务结果。
- 复杂非标准语法（混合多对象、深度语义错误）仍可能失败，需要上游模型输出约束与参数治理配合。

## 测试
- 新增 `tests/test_response_parser.py`，覆盖 7 类异常格式与失败分类。
- 全量测试通过。
