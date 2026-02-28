"""展示选股宝 API 原始返回数据示例"""
import json
import sqlite3

import requests

# 从 DB 找 PTA 板块 ID
conn = sqlite3.connect("data/xuangubao.db")
cur = conn.execute("SELECT id, name FROM plates WHERE name='PTA'")
row = cur.fetchone()
print(f"板块: {row[1]}, ID: {row[0]}")
plate_id = row[0]
conn.close()

# 调用选股宝 API
resp = requests.get(
    f"https://flash-api.xuangubao.com.cn/api/plate/plate_set?id={plate_id}",
    timeout=15,
)
data = resp.json()
d = data["data"]

print("\n=== API 原始返回结构 ===")
print(f"code: {data['code']}")
print(f"message: {data['message']}")
print(f"\ndata 顶层字段: {list(d.keys())}")

print("\n--- 板块基本信息 ---")
for k in ["id", "name", "desc", "subject_id", "create_name", "update_name"]:
    print(f"  {k}: {d.get(k)}")

print("\n--- 行情统计 ---")
for k in [
    "avg_pcp", "core_avg_pcp", "rise_count", "fall_count",
    "stay_count", "fund_flow", "has_north",
    "core_stocks_count", "hang_ye_long_tou_stocks_count",
]:
    print(f"  {k}: {d.get(k)}")

stocks = d.get("stocks", [])
print(f"\n--- 成分股列表 (共{len(stocks)}只, 展示前3只) ---")
for i, s in enumerate(stocks[:3]):
    print(f"\n  股票[{i}]:")
    print(json.dumps(s, ensure_ascii=False, indent=4))

chains = d.get("industrial_chains", [])
print(f"\n--- 产业链 (共{len(chains)}条) ---")
for c in chains[:5]:
    print(f"  {json.dumps(c, ensure_ascii=False)}")

qas = d.get("question_answers", [])
print(f"\n--- 问答 (共{len(qas)}条) ---")
for q in qas:
    ans = (q.get("html_answer") or "")[:100]
    print(f"  Q: {q.get('question')}")
    print(f"  A: {ans}...")
    print()
