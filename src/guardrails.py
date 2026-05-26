"""输出侧护栏(Day3):在答案返回用户前做安全检查。

两项检查:
  1) PII 泄露扫描:答案里不该出现结构化账户类 PII。
  2) 引用强制:非拒答答案必须带 [n] 内联引用,否则视为"不接地"——拦下。
任一违规 → 用安全兜底话术替换答案,绝不把可疑内容吐给用户。
"""

from __future__ import annotations

import re

from . import config, pii

_CITATION_RE = re.compile(r"\[\d+\]")


def check_output(result: dict) -> dict:
    """对 rag.answer() 的结果做输出护栏,原地补上 result['guardrail']。"""
    # 拒答本身是安全输出(且没有引用很正常),直接放行。
    if result.get("refused"):
        result["guardrail"] = {"ok": True, "violations": []}
        return result

    answer = result["answer"]
    violations: list[str] = []

    # 1) PII 泄露扫描(只查结构化 PII,不查 PERSON,避免误拦合规文本里的人名)
    _, leaked = pii.mask_pii(answer, entities=pii.LEAK_ENTITIES)
    if leaked:
        violations.append(f"PII leak: {leaked}")

    # 2) 引用强制:接地的答案必须能溯源
    if not _CITATION_RE.search(answer):
        violations.append("no inline citation (ungrounded)")

    if violations:
        result["answer"] = config.GUARDRAIL_BLOCK_MSG   # 安全兜底
        result["guardrail"] = {"ok": False, "violations": violations}
    else:
        result["guardrail"] = {"ok": True, "violations": []}
    return result
