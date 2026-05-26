"""评估 harness(Day4)。

分开量两件事(面试核心诊断思路):
  - 检索好不好:hit-rate@k、MRR(用 gold_docs 标注)
  - 系统行为对不对:拒答准确率(库内不该拒、库外该拒)
  - 生成忠不忠于来源:faithfulness(本地 qwen 当 judge)
外加 ablation:扫 top-k 和阈值,看指标怎么变。

效率:每题只检索一次(k=10)缓存,小 k 切片、阈值比对 top 分都从缓存算,不重复 embedding。
"""

from __future__ import annotations

import functools

from langchain_ollama import ChatOllama

from . import config
from .eval_data import TESTSET
from .rag import _PROMPT, _format_docs, retrieve

_MAX_K = 10


@functools.lru_cache(maxsize=1)
def _retrieval_cache() -> dict:
    """每题检索一次到 k=10,缓存 [(doc, score), ...]。"""
    return {t["question"]: retrieve(t["question"], k=_MAX_K) for t in TESTSET}


def _first_gold_rank(scored, gold_docs, k) -> int:
    """gold 文档在 top-k 里第一次出现的名次(1-based);没命中返回 0。"""
    for rank, (doc, _) in enumerate(scored[:k], 1):
        if doc.metadata["doc"] in gold_docs:
            return rank
    return 0


# ---------- 检索指标 ----------
def retrieval_metrics(k: int) -> tuple[float, float]:
    cache = _retrieval_cache()
    incorpus = [t for t in TESTSET if t["type"] == "in"]
    hits = rr = 0.0
    for t in incorpus:
        rank = _first_gold_rank(cache[t["question"]], t["gold_docs"], k)
        if rank:
            hits += 1
            rr += 1 / rank
    n = len(incorpus)
    return hits / n, rr / n  # hit@k, MRR


# ---------- 拒答准确率(gate1 阈值,纯分数,不调 LLM,适合扫描)----------
def refusal_accuracy(threshold: float) -> float:
    cache = _retrieval_cache()
    correct = 0
    for t in TESTSET:
        top = cache[t["question"]][0][1] if cache[t["question"]] else 0.0
        refused = top < threshold
        if refused == (t["type"] == "out"):
            correct += 1
    return correct / len(TESTSET)


# ---------- 生成 faithfulness(本地 qwen 当 judge)----------
def faithfulness() -> tuple[float, int]:
    """对库内题:生成答案 → 让 judge 判"每条主张是否被上下文支撑"。"""
    judge = ChatOllama(model=config.OLLAMA_MODEL, temperature=0)
    gen = ChatOllama(model=config.OLLAMA_MODEL, temperature=0)
    cache = _retrieval_cache()
    incorpus = [t for t in TESTSET if t["type"] == "in"]

    supported = answered = 0
    for t in incorpus:
        docs = [d for d, _ in cache[t["question"]][:config.TOP_K]]
        context = _format_docs(docs)
        ans = gen.invoke(_PROMPT.format_messages(
            context=context, question=t["question"])).content.strip()
        if ans == config.REFUSAL_MSG:
            continue  # 库内被拒答的不计入 faithfulness 分母
        answered += 1
        verdict = judge.invoke(
            f"Context:\n{context}\n\nAnswer:\n{ans}\n\n"
            "Is every factual claim in the Answer directly supported by the Context "
            "above? Reply with exactly one word: YES or NO."
        ).content.strip().upper()
        if verdict.startswith("YES"):
            supported += 1
    return (supported / answered if answered else 0.0), answered


# ---------- ablation 扫描 ----------
def sweep() -> None:
    print("\n=== Ablation: top-k 对检索的影响 (阈值固定) ===")
    print(f"{'k':>4} | {'hit@k':>7} | {'MRR':>6}")
    for k in (3, 5, 10):
        hit, mrr = retrieval_metrics(k)
        print(f"{k:>4} | {hit:>7.2%} | {mrr:>6.3f}")

    print("\n=== Ablation: 阈值对拒答准确率的影响 ===")
    print(f"{'threshold':>9} | {'refusal_acc':>11}")
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        print(f"{thr:>9.2f} | {refusal_accuracy(thr):>11.2%}")


def main() -> None:
    print(f"测试集:{len(TESTSET)} 题 "
          f"({sum(t['type']=='in' for t in TESTSET)} 库内 / "
          f"{sum(t['type']=='out' for t in TESTSET)} 库外)")

    hit, mrr = retrieval_metrics(config.TOP_K)
    print(f"\n[检索] hit@{config.TOP_K} = {hit:.2%} | MRR = {mrr:.3f}")
    print(f"[拒答] 准确率 @ threshold={config.RELEVANCE_THRESHOLD} = "
          f"{refusal_accuracy(config.RELEVANCE_THRESHOLD):.2%}")

    print("\n[生成] 计算 faithfulness(逐题调 LLM,稍慢)...")
    faith, n = faithfulness()
    print(f"[生成] faithfulness = {faith:.2%}(在 {n} 道被作答的库内题上)")

    sweep()


if __name__ == "__main__":
    main()
