"""Day2 接地版 RAG:阈值闸 + 接地 prompt + 内联引用。

与 Day1 的 ask.py 区别(面试对比点):
  Day1: 检索 → 直接塞进宽松 prompt → 生成(拒答靠模型自觉)
  Day2: 检索(带分数)→ [闸1] 分太低直接拒答 → 接地 prompt(强制只用上下文 +
        内联 [n] 引用 + [闸2] 上下文不足也拒答)→ 生成
两道闸 = 纵深防御:闸1 便宜确定地挡掉"根本没召回到",闸2 兜住"召回了但不够"。
"""

from __future__ import annotations

import sys

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_postgres import PGVector

from . import config
from .build_index import get_embeddings

# 接地 prompt:三条硬规则——只用上下文、内联引用、不足则用固定话术拒答。
_PROMPT = ChatPromptTemplate.from_template(
    "You are a compliance assistant for a bank. Answer the question using ONLY "
    "the numbered context passages below.\n\n"
    "Rules:\n"
    "- Use ONLY information found in the context. Do NOT use any outside knowledge.\n"
    "- After each claim, cite the passage number(s) you used, e.g. [1] or [2][3].\n"
    "- If the context does not contain enough information to answer, reply EXACTLY "
    f'with: "{config.REFUSAL_MSG}" and nothing else.\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata['doc']} p.{d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def _get_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        connection=config.PG_CONNECTION,
        use_jsonb=True,
    )


def retrieve(question: str, k: int | None = None) -> list[tuple[Document, float]]:
    """检索并带"相关性分数"(0~1,越高越相似)。

    用 with_relevance_scores 而不是 with_score:后者返回的是 cosine 距离
    (越小越近),不直观;前者把它换算成 0~1 的相关性,好设阈值。
    """
    store = _get_store()
    return store.similarity_search_with_relevance_scores(question, k=k or config.TOP_K)


def answer(question: str) -> dict:
    """返回 {answer, refused, reason, top_score, sources}。"""
    scored = retrieve(question)
    docs = [d for d, _ in scored]
    top_score = scored[0][1] if scored else 0.0

    # —— 闸1:相似度阈值。最高分都太低,说明知识库里大概率没有,直接拒答,省一次 LLM。
    if top_score < config.RELEVANCE_THRESHOLD:
        return {
            "answer": config.REFUSAL_MSG,
            "refused": True,
            "reason": f"top_score {top_score:.3f} < 阈值 {config.RELEVANCE_THRESHOLD}",
            "top_score": top_score,
            "sources": [],
        }

    # —— 闸2:接地 prompt(LLM 仍可能输出拒答话术,如果上下文虽相关但不足以回答)。
    llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0)
    messages = _PROMPT.format_messages(context=_format_docs(docs), question=question)
    text = llm.invoke(messages).content.strip()

    refused = text == config.REFUSAL_MSG
    return {
        "answer": text,
        "refused": refused,
        "reason": "LLM 判定上下文不足" if refused else "ok",
        "top_score": top_score,
        "sources": [] if refused else [(d.metadata["doc"], d.metadata["page"]) for d in docs],
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the cash advance fee on FAB credit cards?"
    print(f"Q: {q}\n")
    r = answer(q)
    print("A:", r["answer"])
    print(f"\n[refused={r['refused']} | top_score={r['top_score']:.3f} | {r['reason']}]")
    if r["sources"]:
        print("--- 来源 ---")
        for i, (doc, page) in enumerate(r["sources"], 1):
            print(f"  [{i}] {doc} p.{page}")
