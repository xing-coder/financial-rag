"""把 data/raw/docs/ 下的 PDF 读成 LangChain Document(逐页 + 元数据)。

为什么逐页?——页码是天然的"引用粒度"。金融场景要求 audit-ready:答案能
精确到"哪份文件第几页",这个 page 元数据后面直接变成引用 [doc p.X]。
"""

from __future__ import annotations

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from . import config


def load_pdfs() -> list[Document]:
    """读取 DOCS_DIR 下所有 PDF,返回逐页 Document 列表。"""
    pdf_paths = sorted(config.DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"{config.DOCS_DIR} 里没有 PDF")

    docs: list[Document] = []
    for path in pdf_paths:
        # PyPDFLoader.load() 每页产出一个 Document,
        # metadata 自带 {'source': 全路径, 'page': 0基页码}。
        pages = PyPDFLoader(str(path)).load()
        for d in pages:
            # 把 source 精简成文件名(全路径太长,引用里不好看),
            # 并加一个干净的 doc 标题字段。
            d.metadata["source"] = path.name
            d.metadata["doc"] = path.stem
        docs.extend(pages)
        print(f"  ✓ {path.name}: {len(pages)} 页")

    print(f"共加载 {len(docs)} 页,来自 {len(pdf_paths)} 个 PDF")
    return docs


if __name__ == "__main__":
    load_pdfs()
