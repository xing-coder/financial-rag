"""完整管线(Day3):把输入护栏 + 接地 RAG + 输出护栏串成一个安全入口。

    用户问题
      → [输入护栏] PII 检测+脱敏        (pii.mask_pii)
      → [接地 RAG] 阈值闸+接地prompt+引用 (rag.answer)
      → [输出护栏] PII泄露扫描+引用强制   (guardrails.check_output)
      → 返回

这是对外应该调用的函数 safe_answer();rag.answer() 是中间层,ask.py 是 Day1 baseline。
"""

from __future__ import annotations

import sys

from . import guardrails, pii
from .rag import answer as _rag_answer


def safe_answer(question: str) -> dict:
    # 1) 入口护栏:脱敏后才进入后续所有环节(日志/检索/LLM 只看脱敏版)
    masked_q, pii_in = pii.mask_pii(question)

    # 2) 接地 RAG
    result = _rag_answer(masked_q)

    # 3) 出口护栏
    result = guardrails.check_output(result)

    result["pii_in_query"] = pii_in
    result["masked_query"] = masked_q
    return result


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or (
        "My card is 4111 1111 1111 1111 and my Emirates ID is 784-1990-1234567-1. "
        "What is the cash advance fee on FAB credit cards?"
    )
    r = safe_answer(q)
    print("原始问题: ", q)
    print("脱敏后:   ", r["masked_query"])
    print("入口PII:  ", r["pii_in_query"])
    print()
    print("答案:", r["answer"])
    print(f"\n[refused={r['refused']} | top_score={r['top_score']:.3f} | "
          f"guardrail_ok={r['guardrail']['ok']} | violations={r['guardrail']['violations']}]")
    if r["sources"]:
        print("--- 来源 ---")
        for i, (doc, page) in enumerate(r["sources"], 1):
            print(f"  [{i}] {doc} p.{page}")
