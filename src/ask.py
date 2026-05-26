"""Day1 baseline RAG:retrieve → stuff → generate(本地 Ollama)。

刻意"朴素":只把检索到的上下文塞进 prompt,不做强制接地/拒答/引用校验
(那些是 Day2 的活)。先有一个会"基于上下文回答"的最小闭环,Day2 再加固,
这样 grounding 带来的提升是可对比的。
"""

from __future__ import annotations

import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_postgres import PGVector

from . import config
from .build_index import get_embeddings

# Day1 的朴素 prompt:让模型"参考上下文"作答,但没强制"只能用上下文/查不到就拒答"。
_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant for banking and compliance questions.\n"
    "Use the following context to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _format_docs(docs) -> str:
    """把检索到的块拼成带编号 + 出处的上下文串。"""
    return "\n\n".join(
        f"[{i}] ({d.metadata['doc']} p.{d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def get_retriever():
    store = PGVector(
        embeddings=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        connection=config.PG_CONNECTION,
        use_jsonb=True,
    )
    # as_retriever 内部就是 embed(query) → pgvector ANN 搜 top-k → 返回 Document。
    return store.as_retriever(search_kwargs={"k": config.TOP_K})


def ask(question: str):
    retriever = get_retriever()
    llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0)  # 金融场景要确定性,temp=0

    docs = retriever.invoke(question)                # 1) 检索
    context = _format_docs(docs)                     # 2) 拼上下文
    messages = _PROMPT.format_messages(context=context, question=question)
    answer = llm.invoke(messages).content            # 3) 生成

    return answer, docs


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is a bank's customer acceptance policy?"
    print(f"Q: {q}\n")
    answer, docs = ask(q)
    print("A:", answer)
    print("\n--- 来源 ---")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.metadata['doc']} p.{d.metadata['page']}")
