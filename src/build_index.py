"""建库:load → split → embed(BGE)→ store(pgvector)。

这是 RAG 的"离线链路"。跑一次,知识库就建好了,在线问答只读不写。
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

from . import config
from .load import load_pdfs


def get_embeddings() -> HuggingFaceEmbeddings:
    """BGE embedding。两个关键设置面试都会问:

    1) normalize_embeddings=True:把向量归一化成单位长度。我们用 cosine 距离,
       归一化后 cosine 等价于点积,数值稳定、可比。不归一化 + cosine 会出错。
    2) bge-*-en-v1.5 对 query 的指令前缀('Represent this sentence...')依赖
       已经很小(官方说 v1.5 弱化了对 instruction 的依赖),所以这里省掉前缀
       保持简单;若做对比实验可加 query_instruction 看是否提升。
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build() -> None:
    # 1) 加载:PDF → 逐页 Document
    pages = load_pdfs()

    # 2) 切分:用 embedding 模型"自己的" tokenizer 按 token 切。
    #    为什么不用字符数 / tiktoken?——因为最终是 bge 来编码,用它的 tokenizer
    #    计长度才能保证每块真的不超过 512 的模型上限(否则被截断丢内容)。
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=config.EMBED_MODEL,
        tokens_per_chunk=config.CHUNK_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(f"切分:{len(pages)} 页 → {len(chunks)} 块 "
          f"(每块 {config.CHUNK_TOKENS} token / 重叠 {config.CHUNK_OVERLAP})")

    # 3) embedding + 4) 灌库,一步到位。
    #    pre_delete_collection=True:每次重建先清空旧 collection,
    #    这样反复调试不会堆叠重复数据(幂等)。
    #    use_jsonb=True:元数据用 JSONB 存,后面按 doc/page 做元数据过滤更快。
    PGVector.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        connection=config.PG_CONNECTION,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    print(f"✅ 已灌入 pgvector:collection='{config.COLLECTION_NAME}'")


if __name__ == "__main__":
    build()
